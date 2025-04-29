import json
import os
import re
import shutil
import sys
import tempfile

from fastapi import APIRouter, Request
import orjson
from streaming_form_data import StreamingFormDataParser
import streaming_form_data
from streaming_form_data.targets import FileTarget
from streaming_form_data.validators import MaxSizeValidator
from starlette.requests import ClientDisconnect

sys.path.insert(0, "./")
from utils.logging import logger
from preprocessing.block import Block
from retrieval.server_utils import exempt_from_rate_limit_when, limiter, templates


MAX_FILE_SIZE = 1024 * 1024 * 1024 * 4  # 4GB
CHUNK_SIZE = 1024 * 1024  # 1MB chunk size
ALLOWED_EXTENSIONS = {".jsonl"}
upload_router = APIRouter()


# Custom exception for body size validation
class MaxBodySizeException(Exception):
    def __init__(self, body_len: str):
        self.body_len = body_len


class MaxBodySizeValidator:
    def __init__(self, max_size: int):
        self.body_len = 0
        self.max_size = max_size

    def __call__(self, chunk: bytes):
        self.body_len += len(chunk)
        if self.body_len > self.max_size:
            raise MaxBodySizeException(body_len=self.body_len)


def is_allowed_file(filename: str) -> bool:
    """Check if the file has an allowed extension."""
    _, ext = os.path.splitext(filename)
    return ext.lower() in ALLOWED_EXTENSIONS


@upload_router.post("/upload_collection", include_in_schema=False)
@limiter.limit("4/minute", exempt_when=exempt_from_rate_limit_when)
async def upload_collection_file(request: Request):
    # create uploads directory if it doesn't exist
    os.makedirs("workdir/uploads", exist_ok=True)
    # Prepare for streaming the file
    body_validator = MaxBodySizeValidator(MAX_FILE_SIZE)
    temp_file = tempfile.NamedTemporaryFile(dir="workdir/uploads", delete=False)
    temp_file_path = temp_file.name
    file_target = FileTarget(temp_file_path, validator=MaxSizeValidator(MAX_FILE_SIZE))
    parser = StreamingFormDataParser(headers=request.headers)
    parser.register("file", file_target)

    try:
        try:
            # Stream the file in chunks
            async for chunk in request.stream():
                body_validator(chunk)
                parser.data_received(chunk)
            logger.info("File upload complete.Will now validate the file.")
        except ClientDisconnect:
            return render_error_template(request, "Client disconnected during upload.")
        except MaxBodySizeException:
            return render_error_template(
                request,
                f"File size exceeds the limit. Maximum size is {MAX_FILE_SIZE / (1024 * 1024)} MB.",
                status_code=413,
            )
        except streaming_form_data.validators.ValidationError:
            return render_error_template(
                request,
                f"File size exceeds the limit. Maximum size is {MAX_FILE_SIZE / (1024 * 1024)} MB.",
                status_code=413,
            )
        except Exception as e:
            return render_error_template(
                request, f"File upload failed: {str(e)}", status_code=500
            )

        file_name = file_target.multipart_filename
        if not is_allowed_file(file_name):
            return render_error_template(
                request,
                f"Invalid file extension. Allowed extensions are {ALLOWED_EXTENSIONS}.",
                status_code=400,
            )
        # Validate JSONL file format and metadata
        try:
            validate_jsonl_file(temp_file_path)
        except ValueError as e:
            return render_error_template(request, str(e), status_code=400)

        # Generate a secure file name and move the file to the final location
        try:
            final_file_path = move_file_to_final_location(file_name, temp_file_path)
        except Exception as e:
            return render_error_template(
                request, f"File upload failed: {str(e)}", status_code=500
            )

        # Return success response
        return templates.TemplateResponse(
            "templates/upload.jinja2",
            {
                "title": "Upload Collection",
                "request": request,
                "index_path": os.path.splitext(final_file_path)[
                    0
                ],  # Remove the file suffix
            },
        )
    finally:
        # Ensure the temporary file is removed
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def render_error_template(request: Request, error_message: str, status_code: int = 400):
    """Helper function to render error templates."""
    return templates.TemplateResponse(
        "templates/upload.jinja2",
        {
            "title": "Upload Collection",
            "request": request,
            "error": error_message,
        },
        status_code=status_code,
    )


def validate_jsonl_file(file_path: str) -> tuple:
    """Validate the JSONL file format and metadata consistency."""
    metadata_fields = []
    metadata_types = []
    blocks = []
    with open(file_path, "r") as f:
        for row_idx, line in enumerate(f):
            try:
                row = orjson.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSONL file format at row {row_idx}: {str(e)}"
                )

            block = Block(**row)  # Validate the block format
            blocks.append(block)
            if row_idx == 0:
                metadata_fields, metadata_types = block.get_metadata_fields_and_types()
            else:
                if block.get_metadata_fields_and_types() != (
                    metadata_fields,
                    metadata_types,
                ):
                    raise ValueError(
                        f"Metadata fields and types in row {row_idx} should match the first row of the file. "
                        f"Expected ({metadata_fields}, {metadata_types})\n but found {block.get_metadata_fields_and_types()}"
                    )

    # Check the block lengths
    # We do this separately so that we can use batch tokenization, which significantly speeds up the process for large files
    Block.batch_set_num_tokens(blocks)
    for b in blocks:
        num_tokens = b.num_tokens
        if num_tokens < 5:
            raise ValueError("`content` must be at least 5 tokens long")
        if num_tokens > 8000:
            raise ValueError("`content` must be at most 8000 tokens long")


def move_file_to_final_location(filename: str, temp_file_path: str) -> str:
    """Move the temporary file to the final location with a secure name."""
    secure_filename = generate_secure_filename(filename)
    file_location = f"workdir/uploads/{secure_filename}"

    # Ensure the upload directory exists
    os.makedirs("workdir/uploads", exist_ok=True)

    # Handle file name conflicts by appending version numbers
    file_duplicate_index = 2
    while os.path.exists(file_location):
        logger.info(f"File {file_location} already exists.")
        file_location = file_location.replace(
            ".jsonl", f"_v{file_duplicate_index}.jsonl"
        )
        file_duplicate_index += 1

    # Move the file to the final location
    shutil.move(temp_file_path, file_location)
    return file_location


def generate_secure_filename(filename: str) -> str:
    """Generate a secure filename by removing unsafe characters."""
    return re.sub(r"[/\\?%*:|\"<>\x7F\x00-\x1F]", "-", filename).replace(" ", "_")
