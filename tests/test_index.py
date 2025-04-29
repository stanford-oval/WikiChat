import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, "./")
from fastapi.testclient import TestClient
from rich import print

from retrieval.create_index import create_index
from retrieval.qdrant_index import AsyncQdrantVectorDB
from retrieval.retriever_server import app
from retrieval.search_query import SearchFilter, SearchQuery
from tasks.defaults import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_EMBEDDING_MODEL_PORT,
    DEFAULT_EMBEDDING_MODEL_URL,
    DEFAULT_VECTORDB_TYPE,
)

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def setup_session():
    # Code to run before all tests start
    if not os.path.exists("workdir/uploads"):
        os.makedirs("workdir/uploads")
    if os.path.exists("workdir/uploads/test_collection.jsonl"):
        os.remove("workdir/uploads/test_collection.jsonl")
    if os.path.exists("workdir/uploads/test_collection_v2.jsonl"):
        os.remove("workdir/uploads/test_collection_v2.jsonl")
    yield


@pytest.fixture(scope="session", autouse=True)
def teardown_session():
    yield
    # Code to run after all tests finish
    if os.path.exists("workdir/uploads/test_collection.jsonl"):
        os.remove("workdir/uploads/test_collection.jsonl")
    if os.path.exists("workdir/uploads/test_collection_v2.jsonl"):
        os.remove("workdir/uploads/test_collection_v2.jsonl")


def test_upload_fail():
    with open("tests/test_collection_malformed.jsonl", "rb") as file:
        response = client.post(
            "/upload_collection",
            files={
                "file": ("test_collection_malformed.jsonl", file, "application/jsonl")
            },
        )
    assert response.status_code == 400


@pytest.mark.order(1)  # This test will run first
def test_upload_collection_success():
    # upload once
    with open("tests/test_collection.jsonl", "rb") as file:
        response = client.post(
            "/upload_collection",
            files={"file": ("test_collection.jsonl", file, "application/jsonl")},
        )

    assert response.status_code == 200
    assert os.path.exists("workdir/uploads/test_collection.jsonl")

    # upload again
    with open("tests/test_collection.jsonl", "rb") as file:
        response = client.post(
            "/upload_collection",
            files={"file": ("test_collection.jsonl", file, "application/jsonl")},
        )
    assert response.status_code == 200
    assert os.path.exists("workdir/uploads/test_collection.jsonl")
    assert os.path.exists("workdir/uploads/test_collection_v2.jsonl")


@pytest.mark.order(2)
def test_creating_index():
    create_index(
        collection_file="workdir/uploads/test_collection.jsonl",
        num_embedding_workers=1,  # disables multiprocessing to avoid issues with pytest
        embedding_batch_size=8,
        embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
        embedding_model_url=DEFAULT_EMBEDDING_MODEL_URL,
        embedding_model_port=[DEFAULT_EMBEDDING_MODEL_PORT],
        collection_name="test_collection",
        vector_db_type=DEFAULT_VECTORDB_TYPE,
        high_memory=True,
        num_skip=0,
        index_metadata=True,
    )


@pytest.mark.order(3)
@pytest.mark.asyncio(scope="session")
async def test_search_in_vector_db():
    if DEFAULT_VECTORDB_TYPE == "qdrant":
        vector_db_class = AsyncQdrantVectorDB
    else:
        raise ValueError("Invalid vector_db_type")

    # Initialize the vector database asynchronously
    vector_db = vector_db_class(
        vector_db_url="http://localhost",
        embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
        use_onnx=True,
    )

    assert await vector_db.collection_exists("test_collection")

    # Define the search parameters for each test case
    search_params = [
        {"queries": "programming language", "k": 5, "search_filters": []},
        {
            "queries": "programming language",
            "k": 5,
            "search_filters": [
                SearchFilter(
                    field_name="speaker",
                    filter_type="eq",
                    field_value="John Doe",
                )
            ],
        },
        {
            "queries": "programming language",
            "k": 5,
            "search_filters": [
                SearchFilter(
                    field_name="start_time",
                    filter_type="gt",
                    field_value=30000,
                ),
                SearchFilter(
                    field_name="start_time",
                    filter_type="lt",
                    field_value=150000,
                ),
            ],
        },
        {
            "queries": "programming language",
            "k": 5,
            "search_filters": [
                SearchFilter(
                    field_name="last_edit_date",
                    filter_type="gt",
                    field_value=datetime(2024, 10, 15),
                ),
            ],
        },
        {
            "queries": "programming language",
            "k": 5,
            "search_filters": [
                SearchFilter(
                    field_name="week",
                    filter_type="lte",
                    field_value=4,
                ),
            ],
        },
    ]

    for params in search_params:
        results = await vector_db.search(
            collection_name="test_collection",
            search_query=SearchQuery(
                query=params["queries"],
                num_blocks=params["k"],
                search_filters=params["search_filters"],
            ),
        )

        assert len(results) == 1
        assert results[0]
        assert results[0].results

        # check metadata of the results
        print("Search filters used:", params["search_filters"])
        print([r.block_metadata for r in results[0].results])
        print([r.last_edit_date for r in results[0].results])
        print("-" * 50)

    # Delete the test collection
    await vector_db.delete_collection("test_collection")
    assert not (await vector_db.collection_exists("test_collection"))
