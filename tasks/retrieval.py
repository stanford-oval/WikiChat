import glob
import os
import pathlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup
from huggingface_hub import snapshot_download
from invoke import task
from tqdm import tqdm

from tasks.docker_utils import (
    start_embedding_docker_container,
    start_qdrant_docker_container,
    stop_all_docker_containers,
)
from tasks.main import start_redis

sys.path.insert(0, "./")
from pipelines.utils import get_logger
from tasks.defaults import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_EMBEDDING_MODEL_PORT,
    DEFAULT_EMBEDDING_USE_ONNX,
    DEFAULT_NUM_GPUS,
    DEFAULT_QDRANT_COLLECTION_NAME,
    DEFAULT_RETRIEVER_PORT,
    DEFAULT_WIKIPEDIA_DUMP_LANGUAGE,
    DEFAULT_WORKDIR,
)

logger = get_logger(__name__)


def get_latest_wikipedia_dump_date() -> str:
    """
    Fetches the latest Wikipedia HTML dump date from the Wikimedia dumps page.

    The function makes an HTTP request to the Wikimedia dumps page, parses the returned HTML to extract available
    dump dates, and determines the latest date based on the available dates listed.

    Returns:
        str: The latest Wikipedia dump date in YYYYMMDD format.

    Notes:
        The function assumes the availability of dump dates in a specific format (YYYYMMDD/) and relies on the
        structure of the dumps page remaining consistent over time.
    """
    response = requests.get("https://dumps.wikimedia.org/other/enterprise_html/runs/")
    soup = BeautifulSoup(response.text, "html.parser")
    available_html_dump_dates = []
    for tag in soup.find_all("a"):
        if len(tag.text) == 9:
            # tag.text looks like 20240201/
            available_html_dump_dates.append(tag.text[:-1])
    dates = sorted(available_html_dump_dates)
    if not dates:
        raise ValueError(
            "Could not fetch the latest date from wikimedia.org. Please specify the date manually, e.g. by adding --wiki-date=20240201"
        )
    latest_date = dates[-1]
    logger.info(
        "Available dates: %s. Choosing the latest one %s", ", ".join(dates), latest_date
    )
    return latest_date


def get_wikipedia_collection_dir(workdir, language, wikipedia_date) -> str:
    return os.path.join(workdir, language, f"wikipedia_{wikipedia_date}")


def get_wikipedia_collection_path(workdir, language, wikipedia_date) -> str:
    collection_dir = get_wikipedia_collection_dir(workdir, language, wikipedia_date)
    return os.path.join(collection_dir, "collection.jsonl.gz")


def download_chunk_from_url(
    url, start, end, output_path, pbar, file_lock, num_retries=3
):
    """
    Download a chunk of data from a specific URL, within a given byte range, and write it to a part of a file.

    This function attempts to download a specified range of bytes from a given URL and write this data into a part
    of a file denoted by the start byte. If the download fails due to a ChunkedEncodingError, it will retry up to
    a specified number of retries before raising the last encountered error.

    Args:
        url (str): The URL from where to download the chunk.
        start (int): The starting byte in the range request.
        end (int): The ending byte in the range request.
        output_path (Path-like object): The file path where the chunk will be written.
        pbar (tqdm.tqdm): A tqdm progress bar instance to update the download progress.
        file_lock (threading.Lock): A lock to ensure thread-safe writes to the file and updates to the progress bar.
        num_retries (int, optional): The number of times to retry the download in case of failure. Defaults to 3.

    Raises:
        requests.exceptions.ChunkedEncodingError: If downloading the chunk fails after the specified number of retries.
    """
    headers = {"Range": f"bytes={start}-{end}"}
    for retries in range(num_retries):
        try:
            response = requests.get(url, headers=headers, stream=True)
            break
        except requests.exceptions.ChunkedEncodingError:
            logger.warning("Downloading a chunk cause an error. Retrying...")
            if retries == num_retries:
                raise
    with open(output_path, "r+b") as f:
        f.seek(start)  # Move to the correct position in file.
        for chunk in response.iter_content(1024):
            with file_lock:
                f.write(chunk)  # Write the chunk data.
                pbar.update(len(chunk))


def multithreaded_download(url: str, output_path: str, num_parts: int = 3) -> None:
    """
    Download a file in parts concurrently using multiple threads to optimize the download process.

    This function breaks the download into several parts and downloads each part in parallel, thus potentially
    improving the download speed. It is especially useful when dealing with large files and/or rate-limited servers.
    The function ensures the integrity of the downloaded file by preallocating the total expected size and having
    each thread write to its corresponding part of the file.

    Args:
        url (str): The URL of the file to be downloaded.
        output_path (str): The path to which the file will be downloaded.
        num_parts (int, optional): The number of parts into which the download will be split. Defaults to 3, based
        on the typical rate limiting encountered from servers.

    Note:
        The server from which files are being downloaded must support range requests, as the function relies on
        making byte range requests to download different parts of the file concurrently.
    """
    response = requests.head(url)
    total_size = int(response.headers.get("content-length", 0))
    part_size = total_size // num_parts

    # Preallocate file with total size.
    with open(output_path, "wb") as f:
        f.truncate(total_size)

    # Use a threading.Lock to synchronize file access
    file_lock = threading.Lock()

    futures = []
    with ThreadPoolExecutor(max_workers=num_parts) as executor:
        # Initialize progress bar
        pbar = tqdm(
            total=total_size,
            unit="iB",
            unit_scale=True,
            desc="Downloading the HTML dump",
            smoothing=0,
        )

        for i in range(num_parts):
            start = i * part_size
            # Ensure the last part gets any remainder
            end = (start + part_size - 1) if i < num_parts - 1 else ""

            # Submit download tasks
            futures.append(
                executor.submit(
                    download_chunk_from_url,
                    url,
                    start,
                    end,
                    output_path,
                    pbar,
                    file_lock,
                )
            )

        # Wait for all futures to complete
        for future in futures:
            future.result()

        pbar.close()


@task
def download_wikipedia_index(
    c,
    repo_id: str = "stanford-oval/wikipedia_20240401_10-languages_bge-m3_qdrant_index",
    workdir: str = DEFAULT_WORKDIR,
    num_threads: int = 8,
):
    """
    Download and extract a pre-built Qdrant index for Wikipedia from a 🤗 Hub.

    Args:
    - c: Context, automatically passed by invoke.
    - repo_id (str): The 🤗 Hub repository ID from which to download the index files.
    - workdir (str): The working directory where the files will be downloaded and extracted. Defaults to DEFAULT_WORKDIR.
    - num_threads (int): The number of threads to use for downloading and decompressing the files. Defaults to 8.

    Raises:
    - FileNotFoundError: If no part files are found in the specified directory.
    """
    # Download the files
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=workdir,
        allow_patterns="*.tar.*",
        max_workers=num_threads,
    )

    # Find the part files
    part_files = " ".join(sorted(glob.glob(os.path.join(workdir, "*.tar.*"))))

    # Ensure part_files is not empty
    if not part_files:
        raise FileNotFoundError("No part files found in the specified directory.")

    # Decompress and extract the files
    c.run(
        f"cat {part_files} | pigz -d -p {num_threads} | tar --strip-components=2 -xv -C {os.path.join(workdir, 'qdrant_index')}"
    )  # strip-components gets rid of the extra workdir/


@task
def print_wikipedia_dump_date(c, date: Optional[str] = None):
    """
    Prints the specified date in a human-readable format.

    This function takes an optional date string in the format "%Y%m%d" and prints it out in a more
    human-friendly format, "%B %d, %Y". If no date is provided, it defaults to using the date of the latest
    available Wikipedia HTML dump.

    Args:
        c: The context parameter for Invoke tasks, automatically passed by Invoke.
        date (str, optional): The date string in the format "%Y%m%d". If not provided, the latest wikipedia
                              date is used.
    """
    if not date:
        date = get_latest_wikipedia_dump_date()
    print(datetime.strptime(date, "%Y%m%d").strftime("%B %d, %Y"))


@task(pre=[start_qdrant_docker_container, start_redis])  # Run redis for rate limiting
def start_retriever(
    c,
    collection_name=DEFAULT_QDRANT_COLLECTION_NAME,
    embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
    use_onnx=DEFAULT_EMBEDDING_USE_ONNX,
    retriever_port=DEFAULT_RETRIEVER_PORT,
):
    """
    Starts the retriever server.

    This task runs the retriever server, which is responsible for handling
    retrieval requests. It uses Gunicorn to manage Uvicorn workers for asynchronous processing.
    The retriever server sends search requests to the Qdrant docker container.

    Args:
    - c: Context, automatically passed by invoke.
    - collection_name (str): The name of the Qdrant collection to query. Defaults to DEFAULT_QDRANT_COLLECTION_NAME.
    - embedding_model_name (str): The HuggingFace ID of the embedding model to use for retrieval. Defaults to DEFAULT_EMBEDDING_MODEL_NAME.
    - use_onnx (bool): Flag indicating whether to use the ONNX version of the embedding model. Defaults to DEFAULT_EMBEDDING_USE_ONNX.
    - retriever_port (int): The port on which the retriever server will listen. Defaults to DEFAULT_RETRIEVER_PORT.
    """
    command = (
        f"gunicorn -k uvicorn.workers.UvicornWorker 'retrieval.retriever_server:gunicorn_app("
        f'embedding_model_name="{embedding_model_name}", '
        f'use_onnx="{use_onnx}", '
        f'collection_name="{collection_name}")\' '
        f"--access-logfile=- "
        f"--bind 0.0.0.0:{retriever_port} "
        f"--workers 1 --threads 4 "
        f"--timeout 0"
    )
    c.run(command, pty=False)


@task(pre=[start_qdrant_docker_container])
def test_index(
    c,
    collection_name: str = DEFAULT_QDRANT_COLLECTION_NAME,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
):
    """
    Test a Qdrant index.

    This task starts a Qdrant Docker container and then runs a test query to ensure that the index is working correctly.
    Note that this task does not perform the actual indexing; it only tests an index that already exists.

    Args:
    - c: Context, automatically passed by invoke.
    - collection_name (str): Name of the Qdrant collection to test. Defaults to DEFAULT_QDRANT_COLLECTION_NAME.
    - embedding_model_name (str): Name of the embedding model to use for testing. Defaults to DEFAULT_EMBEDDING_MODEL_NAME.
    """
    cmd = (
        f"python retrieval/create_index.py "
        f"--collection_name {collection_name}"
        f"--embedding_model_name {embedding_model_name} "
        f"--test"  # Just test, don't index
    )
    c.run(cmd, pty=True)


@task(
    pre=[
        stop_all_docker_containers,
        start_embedding_docker_container,
        start_qdrant_docker_container,
    ],
    post=[stop_all_docker_containers],
)
def index_collection(
    c,
    collection_path,
    collection_name: str = DEFAULT_QDRANT_COLLECTION_NAME,
    embedding_model_port: int = DEFAULT_EMBEDDING_MODEL_PORT,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
):
    """
    Creates a Qdrant index from a collection file using a specified embedding model.

    This task starts required Docker containers for the text-embedding service and the Qdrant vector database,
    then indexes the collection file using the specified
    embedding model, and finally stops the Docker containers.

    For reference, indexing the entire English Wikipedia (~52M vectors) takes around 9.5 hours when using an A100 GPU
    Meaning the throughput is about 1.4K vectors per second.
    The actual duration can vary based on the size of your collection file,
    the computational resources available, and the efficiency of the embedding model.

    Args:
        - c (Context): Invoke context, automatically passed by the invoke framework, used for executing shell commands.
        - collection_path (str): Path to a JSONL file (.jsonl or .jsonl.gz file extension) with the appropriate format
        that contains the chunked text of documents you want to index.
        - collection_name (str): The name of the Qdrant collection where the indexed data will be stored.
        This parameter allows you to specify a custom name for the collection, which can be useful for organizing multiple indexes or distinguishing between different versions of the same dataset.
        - embedding_model_port (int): The port on which the embedding model server is running.
        - embedding_model_name (str): The HuggingFace ID of the embedding model to use for indexing.
    """
    c.run(
        f"python retrieval/create_index.py "
        f"--collection_file {collection_path} "
        f"--collection_name {collection_name} "
        f"--embedding_model_name {embedding_model_name} "
        f"--model_port {embedding_model_port} "
        f"--index",  # But don't test, because it takes time for Qdrant to optimize the index after we have inserted vectors in bulk.
        pty=True,
    )


@task
def download_wikipedia_dump(
    c,
    workdir: str = DEFAULT_WORKDIR,
    language: str = DEFAULT_WIKIPEDIA_DUMP_LANGUAGE,
    wikipedia_date: Optional[str] = None,
):
    """
    Download a Wikipedia HTML dump from https://dumps.wikimedia.org/other/enterprise_html/runs/

    Args:
    - c: Context, automatically passed by invoke.
    - wikipedia_date (str, optional): The date of the Wikipedia dump to use. If not provided, the latest available dump is used.
    - language: Language edition of Wikipedia.
    """
    if not wikipedia_date:
        wikipedia_date = get_latest_wikipedia_dump_date()
    index_dir = get_wikipedia_collection_dir(workdir, language, wikipedia_date)
    output_path = os.path.join(index_dir, "articles-html.json.tar.gz")
    if os.path.exists(output_path):
        logger.info("Wikipedia dump already exists at %s", output_path)
        return
    download_url = f"https://dumps.wikimedia.org/other/enterprise_html/runs/{wikipedia_date}/{language}wiki-NS0-{wikipedia_date}-ENTERPRISE-HTML.json.tar.gz"

    pathlib.Path(index_dir).mkdir(parents=True, exist_ok=True)
    multithreaded_download(download_url, output_path)


@task
def preprocess_wikipedia_dump(
    c,
    workdir: str = DEFAULT_WORKDIR,
    language: str = DEFAULT_WIKIPEDIA_DUMP_LANGUAGE,
    wikipedia_date: Optional[str] = None,
    pack_to_tokens: int = 200,
    num_exclude_frequent_words_from_translation: int = 1000,
):
    """
    Process Wikipedia HTML dump into a JSONL collection file.
    This takes ~4 hours for English on a 24-core CPU VM. Processing is fully parallelizable so the time is proportional to number of cores available.
    It might take more for other languages, if we need to also get entity translations from Wikidata. This is because of Wikidata's rate limit.


    Args:
    - workdir (str): Working directory for processing
    - language (str): Language of the dump to process
    - wikipedia_date (str, optional): The date of the Wikipedia dump to use. If not provided, the latest available dump is used.
    - pack_to_tokens(int): We try to pack smaller text chunks to get to this number of tokens.
    - num_exclude_frequent_words_from_translation (int): For non-English Wikipedia dumps, we try to find English translations of all article names
    in Wikidata. We will exclude the `num_exclude_frequent_words_from_translation` most common words because neural models are already familiar with these.
    """
    output_path = get_wikipedia_collection_path(workdir, language, wikipedia_date)
    if os.path.exists(output_path):
        logger.info("Collection already exists at %s", output_path)
        return
    download_wikipedia_dump(c, workdir, language, wikipedia_date)

    index_dir = get_wikipedia_collection_dir(workdir, language, wikipedia_date)
    input_path = os.path.join(index_dir, "articles-html.json.tar.gz")
    wikidata_translation_map = os.path.join(
        workdir, "wikidata_translation_map.jsonl.gz"
    )

    # Constructing the command with parameters
    command = (
        f"python wikipedia_preprocessing/preprocess_html_dump.py "
        f"--input_path {input_path} "
        f"--output_path {output_path} "
        f"--wikidata_translation_map {wikidata_translation_map} "
        f"--language {language} "
        f"--should_translate "
        f"--pack_to_tokens {pack_to_tokens} "
        f"--num_exclude_frequent_words_from_translation {num_exclude_frequent_words_from_translation}"
    )

    # Running the command
    c.run(command)


@task(
    pre=[
        stop_all_docker_containers,
        start_embedding_docker_container,
        start_qdrant_docker_container,
    ],
    post=[stop_all_docker_containers],
)
def index_wikipedia_dump(
    c,
    collection_name: str = DEFAULT_QDRANT_COLLECTION_NAME,
    embedding_model_port: int = DEFAULT_EMBEDDING_MODEL_PORT,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    workdir: str = DEFAULT_WORKDIR,
    language: str = DEFAULT_WIKIPEDIA_DUMP_LANGUAGE,
    wikipedia_date: Optional[str] = None,
):
    """
    Orchestrates the indexing of a Wikipedia collection using a specified embedding model.

    This task starts required Docker containers for the text-embedding service and the Qdrant vector database,
    processes a Wikipedia dump to create a collection, indexes the collection using the specified
    embedding model, and finally stops the Docker containers.

    The indexing step of the task takes around 9.5 hours when using an A100 GPU
    for indexing the English Wikipedia. The actual duration can vary based on the size of the Wikipedia dump,
    the computational resources available, and the efficiency of the embedding model.

    Args:
        - c (Context): Invoke context, automatically passed by the invoke framework, used for executing shell commands.
        - collection_name (str): The name of the Qdrant collection where the indexed data will be stored.
        This parameter allows you to specify a custom name for the collection, which can be useful for organizing multiple indexes or distinguishing between different versions of the same dataset.
        - embedding_model_port (int): The port on which the embedding model server is running.
        - embedding_model_name (str): The HuggingFace ID of the embedding model to use for indexing.
        - workdir (str): The working directory where intermediate and final files are stored.
        - language (str): The language edition of Wikipedia to index (e.g., "en" for English).
        - wikipedia_date (str, optional): The date of the Wikipedia dump to use. If not provided, the latest available dump is used.
    """
    if not wikipedia_date:
        wikipedia_date = get_latest_wikipedia_dump_date()
    preprocess_wikipedia_dump(
        c, workdir=workdir, language=language, wikipedia_date=wikipedia_date
    )
    collection_path = get_wikipedia_collection_path(
        workdir=workdir, language=language, wikipedia_date=wikipedia_date
    )
    embedding_model_port_list = []
    for gpu_id in range(DEFAULT_NUM_GPUS):
        embedding_model_port_list.append(str(embedding_model_port + gpu_id))
    embedding_model_port = " ".join(embedding_model_port_list)

    index_collection(
        c,
        collection_path=collection_path,
        collection_name=collection_name,
        embedding_model_port=embedding_model_port,
        embedding_model_name=embedding_model_name,
    )


@task(
    pre=[
        stop_all_docker_containers,
        start_embedding_docker_container,
        start_qdrant_docker_container,
    ],
    post=[stop_all_docker_containers],
    iterable=["language"],
)
def index_multiple_wikipedia_dumps(
    c, language, workdir: str = DEFAULT_WORKDIR, wikipedia_date: Optional[str] = None
):
    """
    Index multiple Wikipedia dumps from different languages in a for loop.
    """
    if not wikipedia_date:
        wikipedia_date = get_latest_wikipedia_dump_date()
    for l in language:
        logger.info("Started indexing for language %s", l)
        index_wikipedia_dump(
            c, workdir=workdir, language=l, wikipedia_date=wikipedia_date
        )
