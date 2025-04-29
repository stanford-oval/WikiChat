from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
import re
import markdown
from slowapi import Limiter
from utils.logging import logger

markdown_converter = None


def convert_custom_markdown_table_to_html(markdown_table_string: str) -> str:
    """Converts the custom Markdown-like table string from _triplet_serialize to an HTML table."""
    # Remove the outer <Table> tags and strip whitespace
    content = (
        markdown_table_string.strip()
        .removeprefix("<Table>")
        .removesuffix("</Table>")
        .strip()
    )
    lines = content.split("\n")

    if len(lines) < 2:
        # Not enough lines for a header and separator
        return "<table><!-- Malformed custom table input --></table>"

    header_line = lines[0]
    # The separator line (lines[1]) is specific to Markdown and not needed for HTML
    data_lines = lines[2:]

    # --- Process Header ---
    # Use regex to handle potential empty cells or extra spacing
    header_cells_match = re.findall(r"\|\s*(.*?)\s*(?=\|)", header_line)
    if not header_cells_match:
        return "<table><!-- Invalid header format --></table>"
    header_cells = header_cells_match  # Already stripped by the regex group
    num_cols = len(header_cells)

    html_header = "<thead>\n  <tr>\n"
    for header in header_cells:
        html_header += f"    <th>{header}</th>\n"  # No strip needed
    html_header += "  </tr>\n</thead>"

    # --- Process Data Rows ---
    html_body = "<tbody>\n"
    for row_line in data_lines:
        row_line = row_line.strip()
        if not row_line:
            continue

        html_body += "  <tr>\n"
        # Check if it's a simplified "key: value" row (doesn't start with '|')
        if ":" in row_line and not row_line.startswith("|"):
            parts = row_line.split(":", 1)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ""
            html_body += f"    <td>{key}</td>\n"
            # Use colspan for the value if the original table had more than 1 column
            colspan_attr = f' colspan="{num_cols - 1}"' if num_cols > 1 else ""
            # Ensure at least one value cell even if num_cols is 1
            if num_cols > 1 or colspan_attr == "":
                html_body += f"    <td{colspan_attr}>{value}</td>\n"

        # Check if it's a standard Markdown table row
        elif row_line.startswith("|") and row_line.endswith("|"):
            # Extract cells using regex to handle potential empty cells correctly
            row_cells_match = re.findall(r"\|\s*(.*?)\s*(?=\|)", row_line)
            row_cells = row_cells_match  # Already stripped
            # Pad row with empty cells if it has fewer columns than the header
            row_cells.extend([""] * (num_cols - len(row_cells)))
            # Ensure we don't create more cells than header columns
            for i in range(num_cols):
                cell_content = row_cells[i] if i < len(row_cells) else ""
                html_body += f"    <td>{cell_content}</td>\n"
        else:
            # Handle unexpected row format, maybe treat as a single cell spanning the row
            html_body += f'    <td colspan="{num_cols}">{row_line}</td>\n'

        html_body += "  </tr>\n"
    html_body += "</tbody>"

    # --- Combine and Return ---
    html_table = f"<table>\n{html_header}\n{html_body}\n</table>"
    return html_table


def markdown_to_html(markdown_string: str) -> str:
    """
    Converts a Markdown string, potentially containing custom <Table> blocks, to HTML.

    Args:
        markdown_string: The input Markdown string.

    Returns:
        The converted HTML string.
    """

    def replace_table_match(match):
        """Helper function to pass to re.sub"""
        custom_table_block = match.group(0)  # The full <Table>...</Table> block
        return convert_custom_markdown_table_to_html(custom_table_block)

    # Find and replace all custom table blocks with their HTML equivalent
    # Use re.DOTALL so '.' matches newline characters within the table block
    # Use non-greedy matching '.*?' to handle multiple tables correctly
    processed_markdown = re.sub(
        r"<Table>.*?</Table>", replace_table_match, markdown_string, flags=re.DOTALL
    )

    # Convert the rest of the Markdown (including the inserted HTML tables) to HTML
    # The markdown library usually treats existing HTML tags as raw HTML.
    html_output = markdown.markdown(processed_markdown, extensions=["extra", "tables"])

    return html_output


limiter = Limiter(key_func=get_remote_address, storage_uri="redis://localhost:6379")
templates = Jinja2Templates(directory="public")


# Custom middleware to increase the maximum request size
class MaxSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_body_size: int):
        super().__init__(app)
        self.max_body_size = max_body_size

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            # Check the content length of the request
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_body_size:
                return JSONResponse({"error": "File too large"}, status_code=413)
        return await call_next(request)


def exempt_from_rate_limit_when(request: Request):
    sender_ip = request.client.host
    logger.info(f"Request from IP: {sender_ip}")
    if sender_ip in [
        "127.0.0.1",  # Exempt requests from the front-end
        "20.83.187.209",
        "testclient",  # Exempt requests when testing
    ]:
        logger.debug("Exempt from rate limit")
        return True
    return False
