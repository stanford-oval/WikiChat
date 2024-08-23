import argparse
import asyncio
import atexit
import sys
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from async_lru import alru_cache
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

sys.path.insert(0, "./")
from pipelines.utils import get_logger
from retrieval.qdrant_index import QdrantIndex, SearchResult

limiter = Limiter(key_func=get_remote_address, storage_uri="redis://localhost:6379")

app = FastAPI(
    title="Wikipedia Search API",
    description="An API for retrieving information from 10 Wikipedia languages from the Wikipedia dump of Feb 20, 2024.",
    version="1.0.0",
    docs_url="/search/docs",
    redoc_url="/search/redoc",
    openapi_url="/search/openapi.json",
    contact={
        "name": "Stanford Open Virtual Assistant Lab",
        "url": "https://oval.cs.stanford.edu",
        "email": "genie@cs.stanford.edu",
    },
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

logger = get_logger(__name__)


class Language(str, Enum):
    ENGLISH = "en"
    SPANISH = "es"
    PERSIAN = "fa"
    GERMAN = "de"
    ITALIAN = "it"
    CHINESE = "zh"
    FRENCH = "fr"
    RUSSIAN = "ru"
    PORTUGUESE = "pt"
    JAPANESE = "ja"


class BlockType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    INFOBOX = "infobox"


class QueryData(BaseModel):
    query: Union[list[str], str] = Field(
        ...,  # This means the field is required
        description="Query or a list of queries in natural language",
        json_schema_extra={"example": ["What is GPT-4?", "What is LLaMA-3?"]},
    )
    num_blocks: int = Field(
        ...,
        description="Number of results to return",
        gt=0,
        le=50,
        json_schema_extra={"example": 3},
    )
    languages: Optional[Union[list[Language], Language]] = Field(
        default=None, description="The language codes of the results you want"
    )
    block_types: Optional[Union[list[BlockType], BlockType]] = Field(
        default=None, description="Can be zero, one or more of the defined block types"
    )

    @field_validator("query")
    def validate_query(cls, value):
        if isinstance(value, str):
            if len(value) < 1:
                raise ValueError("String query must be non-empty")
        elif isinstance(value, list):
            if len(value) < 1:
                raise ValueError("List query must contain at least one item")
            if len(value) > 100:
                raise ValueError("List query must contain at most 100 items")
            for item in value:
                if not isinstance(item, str) or len(item) < 1:
                    raise ValueError("Each item in the list must be a non-empty string")
        else:
            raise TypeError("Query must be either a string or a list of strings")
        return value

    model_config = ConfigDict(
        use_enum_values=True  # This allows the API to accept string values for enums
    )


qdrant_index = None


@alru_cache(maxsize=10000)
async def qdrant_search(
    queries: tuple[str],
    k: int,
    languages: tuple[str] = None,
    block_types: tuple[str] = None,
) -> list[SearchResult]:
    filters = {}
    if languages:
        filters["language"] = languages
    if block_types:
        filters["block_types"] = block_types
    return await qdrant_index.search(queries, k, filters=filters)


def exempt_from_rate_limit_when(request: Request):
    sender_ip = request.client.host
    if sender_ip in [
        "127.0.0.1",
        "20.83.187.209",
    ]:  # Exempt requests from the front-end
        logger.debug("Exempt from rate limit")
        return True
    return False


@app.post(
    "/search",
    description=Path("docs/search_api.md").read_text(),
    summary="High-quality Wikipedia Search",
    response_model=list[SearchResult],
)
@limiter.limit("20/minute", exempt_when=exempt_from_rate_limit_when)
async def search(request: Request, query_data: QueryData):
    queries = query_data.query
    if isinstance(queries, str):
        queries = [queries]
    evi_num = query_data.num_blocks
    languages = query_data.languages
    if isinstance(languages, str):
        languages = [languages]
    block_types = query_data.block_types
    if isinstance(block_types, str):
        block_types = [block_types]

    try:
        results = await qdrant_search(
            tuple(queries),
            evi_num,
            languages=tuple(languages) if languages else None,
            block_types=tuple(block_types) if block_types else None,
        )
    except Exception as e:
        logger.error(str(e))
        return Response(status_code=500)

    return results


def init():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--embedding_model_name",
        type=str,
        required=True,
        help="Path or the HuggingFace model name for the model used to encode the query.",
    )
    arg_parser.add_argument(
        "--collection_name", type=str, help="The name of the Qdrant collection"
    )
    arg_parser.add_argument(
        "--use_onnx", default=False, type=lambda x: (str(x).lower() == "true")
    )

    args = arg_parser.parse_args()

    global qdrant_index

    if args.use_onnx:
        logger.info("Using ONNX for the embedding model.")
    qdrant_index = QdrantIndex(
        embedding_model_name=args.embedding_model_name,
        collection_name=args.collection_name,
        use_onnx=args.use_onnx,
    )

    # logger.info("Running a test query.")
    # asyncio.run(
    #     search(
    #         QueryData(
    #             **{
    #                 "query": [
    #                     "Tell me about Haruki Murakami.",
    #                     "who is the current monarch of the UK?",
    #                 ],
    #                 "num_blocks": 5,
    #                 "languages": [],
    #                 "block_types": [],
    #             }
    #         )
    #     )
    # )

    logger.info("Retrieval server is ready.")

    return args


@atexit.register
def cleanup():
    # Cleanup qdrant_index here
    if qdrant_index:
        asyncio.run(qdrant_index.qdrant_client.close())


# used as a way to pass commandline arguments to gunicorn
def gunicorn_app(*args, **kwargs):
    # Gunicorn CLI args are useless.
    # https://stackoverflow.com/questions/8495367/
    #
    # Start the application in modified environment.
    # https://stackoverflow.com/questions/18668947/
    #

    sys.argv = ["--gunicorn"]
    for k in kwargs:
        sys.argv.append("--" + k)
        sys.argv.extend(kwargs[k].split())  # split to support list arguments
    init()
    return app


if __name__ == "__main__":
    args = init()
    app.run(port=args.port, debug=False, use_reloader=False)
