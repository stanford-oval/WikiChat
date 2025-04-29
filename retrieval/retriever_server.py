import argparse
import sys
from typing import Optional

from async_lru import alru_cache
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from jinja2 import Template
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import langcodes
from corpora import all_corpus_objects, get_public_indices
from retrieval.llm_reranker import batch_llm_rerank, initialize_llm_reranker
from retrieval.retrieval_commons import QueryResult
from retrieval.search_query import SearchQuery
from retrieval.upload_collection import upload_router

sys.path.insert(0, "./")

from retrieval.qdrant_index import AsyncQdrantVectorDB
from utils.logging import logger

from .server_utils import (
    exempt_from_rate_limit_when,
    limiter,
    markdown_to_html,
    templates,
)

vector_db = None


@alru_cache(ttl=60)  # get newly added collections from the vector database every minute
async def get_available_collections():
    assert vector_db
    return await vector_db.list_collections()


@asynccontextmanager
async def init(app: FastAPI):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        required=True,
        help="Path or the HuggingFace model name for the model used to encode the query.",
    )
    parser.add_argument(
        "--use_onnx", default=False, type=lambda x: (str(x).lower() == "true")
    )
    parser.add_argument(
        "--vector_db_type",
        type=str,
        choices=["qdrant"],
        required=True,
        help="The type of vector database to use for indexing.",
    )
    parser.add_argument(
        "--reranker_engine",
        type=str,
        required=True,
        help="The LLM to use for reranking.",
    )
    parser.add_argument("--vector_db_url", type=str, default="http://localhost")

    args = parser.parse_args()

    global vector_db

    if args.vector_db_type == "qdrant":
        vector_db_class = AsyncQdrantVectorDB
    else:
        raise ValueError("Invalid vector_db_type")

    # Initialize the vector database asynchronously
    vector_db = vector_db_class(
        vector_db_url=args.vector_db_url,
        embedding_model_name=args.embedding_model_name,
        use_onnx=args.use_onnx,
    )

    available_collections = await vector_db.list_collections()
    logger.info(f"All available collections: {available_collections}")
    logger.info(f"Out of which, these will be public: {get_public_indices()[0]}")

    initialize_llm_reranker(engine=args.reranker_engine, reranker_type="pointwise")

    if available_collections:
        logger.info("Running a test query.")
        test_queries = [
            "Tell me about Haruki Murakami.",
            "who is the current monarch of the UK?",
            "شهرستان مرند کجاست؟",  # Query in Farsi, to test multilingual support
        ]
        results = await vector_db.search(
            collection_name=available_collections[-1],
            search_query=SearchQuery(query=test_queries, num_blocks=3),
        )
        for q, r in zip(test_queries, results):
            logger.info(f"Test Query: {q}")
            logger.info(f"Results: {r.model_dump_json(indent=2)}")

    logger.info("Retrieval server is ready.")

    # Yield control back to FastAPI (this is where the app will start serving requests)
    yield

    # Shutdown logic (cleanup)
    if vector_db:
        await vector_db.close_connections()
        logger.info("Vector DB connections closed.")


app = FastAPI(
    title="Genie Search API",
    description="""An API for high-quality search in knowledge corpora. This API is a hosted version of [WikiChat](https://github.com/stanford-oval/WikiChat)'s search subsystem.
Specifically, it uses state-of-the-art multilingual vector embedding models and LLM reranking for highest quality search results.""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "Stanford Open Virtual Assistant Lab",
        "url": "https://oval.cs.stanford.edu",
        "email": "genie@cs.stanford.edu",
    },
    lifespan=init,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore
app.mount("/public", StaticFiles(directory="public"), name="public")

# app.add_middleware(MaxSizeLimitMiddleware, max_body_size=MAX_FILE_SIZE)
app.include_router(upload_router)


search_api_docs = Template(open("public/templates/search_api.jinja2").read()).render(
    all_corpora=all_corpus_objects
)


@app.get("/upload_collection", include_in_schema=False)
async def upload_collection(request: Request):
    return templates.TemplateResponse(
        "templates/upload.jinja2", {"title": "Upload Collection", "request": request}
    )


@app.post(
    "/{collection_name}",
    description=search_api_docs,
    summary="A High-Quality Multilingual LLM-Based Search Endpoint",
    response_model=list[QueryResult],
)
@limiter.limit("50/minute", exempt_when=exempt_from_rate_limit_when)
async def search(request: Request, collection_name: str, search_query: SearchQuery):
    assert vector_db is not None
    try:
        results: list[QueryResult] = await vector_db.search(
            collection_name=collection_name,
            search_query=search_query,
        )
        assert len(results) == len(search_query.query)
        if search_query.rerank:
            results = await batch_llm_rerank(search_query.query, results)
            assert len(results) == len(search_query.query)
            for r in results:
                r.results = r.results[
                    : search_query.num_blocks
                ]  # truncate to the desired number of blocks
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

    assert len(results) == len(search_query.query)

    return results


@app.get("/{collection_name}", include_in_schema=False)
async def search_via_frontend_redirect(
    request: Request, collection_name: str, query: Optional[str] = None
):
    assert vector_db
    if not await vector_db.collection_exists(collection_name):
        raise HTTPException(
            status_code=404, detail=f"Collection {collection_name} not found."
        )
    return await search_via_frontend(request, collection_name, query)


@app.get("/", include_in_schema=False)
async def search_via_frontend(
    request: Request, collection_name: Optional[str] = None, query: Optional[str] = None
):
    assert vector_db
    result_urls = []
    result_titles = []
    result_snippets = []
    result_dates = []
    result_metadata = []
    if query:
        try:
            search_results: list = await vector_db.search(
                collection_name,
                SearchQuery(query=[query], num_blocks=100),
            )
        except Exception as e:
            logger.exception(e)
            raise HTTPException(
                status_code=500, detail="Internal Server Error. Please try again later."
            )
        try:
            search_results = await batch_llm_rerank([query], search_results)
        except Exception as e:
            # likely due to the content filter of the LLM API
            logger.exception(e)

        search_results: list = search_results[0].results  # one query at a time

        for result in search_results:
            result_urls.append(result.url)
            result_titles.append(result.full_title)
            result_snippets.append(markdown_to_html(result.content))
            date = result.last_edit_date
            if date is not None:
                date = date.strftime("%B %d, %Y")
            result_dates.append(date)
            metadata = result.block_metadata
            if metadata and "language" in metadata:
                try:
                    # TODO why is language sometimes not a language code?
                    # convert language code to human-readable format
                    metadata["language"] = langcodes.Language.get(
                        metadata["language"]
                    ).display_name()
                except Exception as e:
                    logger.warning(
                        f"Failed to convert language code {metadata['language']}: {e}"
                    )
            result_metadata.append(metadata)
        logger.info(f"Returning {len(search_results)} results.")

    public_collections, public_collection_names, human_description_markdown = (
        get_public_indices()
    )

    return templates.TemplateResponse(
        "templates/search.jinja2",
        {
            "request": request,
            "public_collections": public_collections,
            "public_collection_names": public_collection_names,
            "human_description_markdown": human_description_markdown,
            "collection_name": collection_name,
            "title": f"{query} - Search Results" if query else "Search",
            "empty_result": query and len(result_urls) == 0,
            "query": query,
            # Search results:
            "result_urls": result_urls,
            "result_titles": result_titles,
            "result_snippets": result_snippets,
            "result_dates": result_dates,
            "result_metadata": result_metadata,
        },
    )


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
    # No need to call init() here, FastAPI will handle it via the lifespan event
    return app
