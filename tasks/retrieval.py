import sys
from typing import Optional
import os
import signal
from invoke.tasks import task

from tasks.docker_tasks import (
    start_embedding_docker_container,
    start_qdrant_docker_container,
)
from tasks.main import load_api_keys, start_redis
from tasks.preprocessing import (
    get_latest_wikipedia_dump_date,
    get_wikipedia_collection_path,
    preprocess_wikipedia_dump,
)

sys.path.insert(0, "./")
import json

from utils.logging import logger
from tasks.defaults import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_EMBEDDING_MODEL_PORT,
    DEFAULT_EMBEDDING_MODEL_URL,
    DEFAULT_EMBEDDING_USE_ONNX,
    DEFAULT_NUM_GPUS,
    DEFAULT_RETRIEVER_PORT,
    DEFAULT_RETRIEVER_RERANKER_ENGINE,
    DEFAULT_VECTORDB_COLLECTION_NAME,
    DEFAULT_VECTORDB_TYPE,
    DEFAULT_WORKDIR,
)


@task(
    pre=[start_redis, load_api_keys]
)  # Run redis for rate limiting, load API keys for LLM reranker
def start_retriever(
    c,
    vector_db_type: str = DEFAULT_VECTORDB_TYPE,
    workdir: str = DEFAULT_WORKDIR,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    use_onnx: bool = DEFAULT_EMBEDDING_USE_ONNX,
    retriever_port: int = DEFAULT_RETRIEVER_PORT,
    reranker_engine: str = DEFAULT_RETRIEVER_RERANKER_ENGINE,
):
    """
    Starts the retriever server.

    This task runs the retriever server, which is responsible for handling
    retrieval requests. It uses Gunicorn to manage Uvicorn workers for asynchronous processing.
    The retriever server sends search requests to the vector DB Docker container.

    Args:
    - c: Context, automatically passed by invoke.
    - vector_db_type (str): The type of vector database to use for indexing.
    - workdir (str): The working directory where the retriever server will run. Defaults to DEFAULT_WORKDIR.
    - embedding_model_name (str): The HuggingFace ID of the embedding model to use for retrieval. Defaults to DEFAULT_EMBEDDING_MODEL_NAME.
    - use_onnx (bool): Flag indicating whether to use the ONNX version of the embedding model. Defaults to DEFAULT_EMBEDDING_USE_ONNX.
    - retriever_port (int): The port on which the retriever server will listen. Defaults to DEFAULT_RETRIEVER_PORT.
    - reranker_engine (str): The LLM to use for reranking. Defaults to DEFAULT_RETRIEVER_RERANKER_ENGINE.
    """

    if vector_db_type == "qdrant":
        start_qdrant_docker_container(c, workdir=workdir)
    else:
        raise ValueError(f"Invalid vector_db_type {vector_db_type}")

    command = (
        f"gunicorn -k uvicorn.workers.UvicornWorker 'retrieval.retriever_server:gunicorn_app("
        f'embedding_model_name="{embedding_model_name}", '
        f'use_onnx="{use_onnx}", '
        f'reranker_engine="{reranker_engine}", '
        f'vector_db_type="{vector_db_type}")\' '
        f"--access-logfile=- "
        f"--bind 0.0.0.0:{retriever_port} "
        f"--workers 2 --threads 8 "
        f"--pid /tmp/gunicorn.pid "  # Save the PID of the Gunicorn master process to file
        f"--timeout 0"
    )
    c.run(command, pty=False)


def get_gunicorn_pid(pid_file="/tmp/gunicorn.pid"):
    """
    Get the Gunicorn master process ID.
    """
    try:
        with open(pid_file, "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        print(f"PID file {pid_file} not found.")
        return None
    except Exception as e:
        print(f"Error reading PID file: {e}")
        return None


@task
def reload_retriever(c):
    """
    Gracefully reload the Gunicorn retriever server by sending a SIGHUP signal to the master process.
    """
    pid = get_gunicorn_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGHUP)
            print(f"Sent SIGHUP to Gunicorn master process (PID: {pid})")
        except ProcessLookupError:
            print(f"Process with PID {pid} not found.")
        except Exception as e:
            print(f"Error sending SIGHUP: {e}")
    else:
        print("Could not find Gunicorn master PID. Reload aborted.")


@task
def test_index(
    c,
    workdir=DEFAULT_WORKDIR,
    embedding_model_url=DEFAULT_EMBEDDING_MODEL_URL,
    embedding_model_port=DEFAULT_EMBEDDING_MODEL_PORT,
    embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
    num_gpus=DEFAULT_NUM_GPUS,
):
    start_qdrant_docker_container(c, workdir=workdir)
    if (
        "localhost" in embedding_model_url
        or "0.0.0.0" in embedding_model_url
        or "127.0.0.1" in embedding_model_url
    ):
        start_embedding_docker_container(
            c,
            port=embedding_model_port,
            embedding_model=embedding_model_name,
            num_gpus=num_gpus,
        )
    c.run(
        "pytest -rP --color=yes --disable-warnings ./tests/test_index.py ",
        pty=True,
    )


@task
def test_retriever(c, retriever_port=DEFAULT_RETRIEVER_PORT):
    logger.info("Testing the retriever server without LLM reranking...")
    result = c.run(
        f"curl -X POST 0.0.0.0:{retriever_port}/wikipedia "
        f'-H "Content-Type: application/json" '
        f"""-d '{{
            "query": ["What is GPT-4?", "What is LLaMA-3?"],
            "num_blocks": 3
        }}'""",
        pty=True,
        hide=True,
    )
    logger.info(json.dumps(json.loads(result.stdout), indent=2, ensure_ascii=False))

    logger.info("Testing the retriever server with LLM reranking...")
    result = c.run(
        f"curl -X POST 0.0.0.0:{retriever_port}/wikipedia "
        f'-H "Content-Type: application/json" '
        f'-d \'{{"query": ["What is GPT-4?", "What is LLaMA-3?"], "num_blocks": 3, "rerank": true}}\'',
        pty=True,
        hide=True,
    )
    logger.info(json.dumps(json.loads(result.stdout), indent=2, ensure_ascii=False))

    logger.info(
        "Testing the retriever server with LLM reranking and `num_blocks_to_rerank` ..."
    )
    result = c.run(
        f"curl -X POST 0.0.0.0:{retriever_port}/wikipedia "
        f'-H "Content-Type: application/json" '
        f'-d \'{{"query": ["What is GPT-4?", "What is LLaMA-3?"], '
        f'"num_blocks": 3, "rerank": true, "num_blocks_to_rerank": 20}}\'',
        pty=True,
        hide=True,
    )
    logger.info(json.dumps(json.loads(result.stdout), indent=2, ensure_ascii=False))


@task(
    pre=[
        load_api_keys,
    ],
)
def index_collection(
    c,
    collection_path,
    vector_db_type: str = DEFAULT_VECTORDB_TYPE,
    workdir: str = DEFAULT_WORKDIR,
    collection_name: str = DEFAULT_VECTORDB_COLLECTION_NAME,
    embedding_model_url: str = DEFAULT_EMBEDDING_MODEL_URL,
    embedding_model_port: int = DEFAULT_EMBEDDING_MODEL_PORT,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    num_gpus: int = DEFAULT_NUM_GPUS,
    num_skip: int = 0,
    high_memory: bool = False,
    index_metadata: bool = False,
):
    """
    Creates a vector index from a collection file using a specified embedding model.

    This task starts required Docker containers for the text-embedding service and the vector database,
    then indexes the collection file using the specified
    embedding model, and finally stops the Docker containers.

    For reference, indexing the entire English Wikipedia (~50M vectors) takes around 9.5 hours when using an A100 GPU
    Meaning the throughput is about 1.4K vectors per second.
    The actual duration can vary based on the size of your collection file,
    the computational resources available, and the efficiency of the embedding model.

    Args:
        - c (Context): Invoke context, automatically passed by the invoke framework, used for executing shell commands.
        - collection_path (str): Path to a JSONL file (.jsonl or .jsonl.gz file extension) with the appropriate format
        that contains the chunked text of documents you want to index.
        - vector_db_type (str): The type of vector database to use for indexing.
        - workdir (str): The working directory where the retriever server will run. Defaults to DEFAULT_WORKDIR.
        - collection_name (str): The name of the vector DB collection where the indexed data will be stored.
        This parameter allows you to specify a custom name for the collection, which can be useful for organizing multiple indexes or distinguishing between different versions of the same dataset.
        - embedding_model_url (str): The URL at which the embedding model is accessible will be embedding_model_url:embedding_model_port/embed.
        - embedding_model_port (int): The port on which the embedding model server is running.
        - embedding_model_name (str): The HuggingFace ID of the embedding model to use for indexing.
        - num_gpus (int): The number of GPUs to use for indexing. Defaults to DEFAULT_NUM_GPUS.
        - num_skip (int): Number of blocks to skip in the collection file. Useful for resuming indexing.
        - high_memory (bool): Whether to use a high-RAM configuration for the vector database. Defaults to False.
        - index_metadata (bool): Whether to index metadata fields. Defaults to False.
    """
    if (
        "localhost" in embedding_model_url
        or "0.0.0.0" in embedding_model_url
        or "127.0.0.1" in embedding_model_url
    ):
        start_embedding_docker_container(
            c,
            port=embedding_model_port,
            embedding_model=embedding_model_name,
            num_gpus=num_gpus,
        )
    if vector_db_type == "qdrant":
        start_qdrant_docker_container(c, workdir=workdir)
    else:
        raise ValueError(f"Invalid vector_db_type {vector_db_type}")

    if high_memory:
        high_memory_flag = "--high_memory"
    else:
        high_memory_flag = ""

    if index_metadata:
        index_metadata_flag = "--index_metadata"
    else:
        index_metadata_flag = ""

    c.run(
        f"python retrieval/create_index.py "
        f"--collection_file {collection_path} "
        f"--collection_name {collection_name} "
        f"--embedding_model_name {embedding_model_name} "
        f"--embedding_model_port {' '.join([str(embedding_model_port + i) for i in range(num_gpus)])} "
        f"--embedding_model_url {embedding_model_url} "
        f"--vector_db_type {vector_db_type} "
        f"--num_skip {num_skip} "
        f"{high_memory_flag} "
        f"{index_metadata_flag} ",
        pty=True,
    )


@task(
    pre=[load_api_keys],
    iterable=["language"],
)
def index_wikipedia_dump(
    c,
    language,
    collection_name: str = DEFAULT_VECTORDB_COLLECTION_NAME,
    vector_db_type: str = DEFAULT_VECTORDB_TYPE,
    workdir: str = DEFAULT_WORKDIR,
    wikipedia_date: Optional[str] = None,
    embedding_model_url: str = DEFAULT_EMBEDDING_MODEL_URL,
    embedding_model_port: int = DEFAULT_EMBEDDING_MODEL_PORT,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    num_gpus: int = DEFAULT_NUM_GPUS,
    num_skip: int = 0,
    high_memory: bool = False,
):
    """
    Orchestrates the indexing of one or more Wikipedia collections using a specified embedding model.

    This task starts required Docker containers for the text-embedding service and the vector DB vector database,
    processes a Wikipedia dump to create a collection, indexes the collection using the specified
    embedding model, and finally stops the Docker containers.

    The indexing step of the task takes around 9.5 hours when using an A100 GPU
    for indexing the English Wikipedia. The actual duration can vary based on the size of the Wikipedia dump,
    the computational resources available, and the efficiency of the embedding model.

    Args:
        - c (Context): Invoke context, automatically passed by the invoke framework, used for executing shell commands.
        - language (str): The language edition of Wikipedia to index. Can provide multiple languages to index sequentially, e.g., `--language en --language de`.
        - vector_db_type (str): The type of vector database to use for indexing.
        - collection_name (str): The name of the vector DB collection where the indexed data will be stored.
        This parameter allows you to specify a custom name for the collection, which can be useful for organizing multiple indexes or distinguishing between different versions of the same dataset.
        - embedding_model_port (int): The port on which the embedding model server is running.
        - embedding_model_name (str): The HuggingFace ID of the embedding model to use for indexing.
        - workdir (str): The working directory where intermediate and final files are stored.
        - wikipedia_date (str, optional): The date of the Wikipedia dump to use. If not provided, the latest available dump is used.
        - num_gpus (int): The number of GPUs to use for indexing. Defaults to DEFAULT_NUM_GPUS.
        - num_skip (int): Number of blocks to skip in the collection file. Useful for resuming indexing.
        - high_memory (bool): Whether to use a high-RAM configuration for the vector database. Defaults to False.
    """
    assert isinstance(language, list), "language should be a list of languages"

    if vector_db_type == "qdrant":
        start_qdrant_docker_container(c, workdir=workdir)
    else:
        raise ValueError(f"Invalid vector_db_type {vector_db_type}")
    if (
        "localhost" in embedding_model_url
        or "0.0.0.0" in embedding_model_url
        or "127.0.0.1" in embedding_model_url
    ):
        start_embedding_docker_container(
            c,
            workdir=workdir,
            port=embedding_model_port,
            embedding_model=embedding_model_name,
        )

    if not wikipedia_date:
        wikipedia_date = get_latest_wikipedia_dump_date()

    for lang in language:
        logger.info(f"Started indexing for language {lang}")
        preprocess_wikipedia_dump(
            c, workdir=workdir, language=lang, wikipedia_date=wikipedia_date
        )
        collection_path = get_wikipedia_collection_path(
            workdir=workdir, language=lang, wikipedia_date=wikipedia_date
        )

        index_collection(
            c,
            collection_path=collection_path,
            collection_name=collection_name,
            vector_db_type=vector_db_type,
            workdir=workdir,
            embedding_model_url=embedding_model_url,
            embedding_model_port=embedding_model_port,
            embedding_model_name=embedding_model_name,
            num_gpus=num_gpus,
            num_skip=num_skip,
            high_memory=high_memory,
        )
