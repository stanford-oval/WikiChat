import sys

sys.path.insert(0, "./")

# Import specific functions/classes from benchmark
from tasks.benchmark import simulate_users, benchmark_articles, db_to_file

# Import specific functions/classes from docker
from tasks.docker_tasks import (
    stop_docker_container,
    start_qdrant_docker_container,
    start_embedding_docker_container,
    stop_all_docker_containers,
)

# Import specific functions/classes from main
from tasks.main import (
    load_api_keys,
    start_redis,
    start_backend,
    tests,
    demo,
    format_code,
)

# Import specific functions/classes from preprocessing
from tasks.preprocessing import (
    download_wikipedia_dump,
    download_semantic_scholar_dump,
    preprocess_wikipedia_dump,
)

# Import specific functions/classes from retrieval
from tasks.retrieval import start_retriever, index_collection, index_wikipedia_dump

# Import specific functions/classes from setup
from tasks.setup import install_docker, download_azcopy, install

__all__ = [
    "simulate_users",
    "benchmark_articles",
    "db_to_file",
    "stop_docker_container",
    "start_qdrant_docker_container",
    "start_embedding_docker_container",
    "stop_all_docker_containers",
    "load_api_keys",
    "start_redis",
    "start_backend",
    "tests",
    "demo",
    "format_code",
    "download_wikipedia_dump",
    "download_semantic_scholar_dump",
    "preprocess_wikipedia_dump",
    "start_retriever",
    "index_collection",
    "index_wikipedia_dump",
    "install_docker",
    "download_azcopy",
    "install",
]
