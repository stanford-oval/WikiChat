import concurrent
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
from huggingface_hub import hf_hub_download, snapshot_download
from invoke import task
from tqdm import tqdm

from tasks.defaults import DEFAULT_WORKDIR
from tasks.main import load_api_keys

sys.path.insert(0, "./")

from utils.logging import logger


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
    if response.status_code != 200:
        raise ValueError(
            f"Failed to fetch the latest date from wikimedia.org. Status code: {response.status_code}"
        )
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
        f"Available dates: {', '.join(dates)}. Choosing the latest one {latest_date}"
    )
    return latest_date


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


def multithreaded_download(
    url: str, output_path: str, num_parts: int = 3, disable_tqdm=False
) -> None:
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
        disable_tqdm (bool, optional): Whether to disable showing a progress bar

    Note:
        The server from which files are being downloaded must support range requests, as the function relies on
        making byte range requests to download different parts of the file concurrently.
    """
    response = requests.head(url)
    total_size = int(response.headers.get("content-length", 0))
    part_size = total_size // num_parts

    # Cretae the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
            desc="Downloading file",
            smoothing=0,
            disable=disable_tqdm,
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


@task(iterable=["language"])
def download_wikipedia_dump(
    c,
    language: str,
    workdir: str = DEFAULT_WORKDIR,
    wikipedia_date: Optional[str] = None,
):
    """
    Download a Wikipedia HTML dump from https://dumps.wikimedia.org/other/enterprise_html/runs/

    Args:
    - c: Context, automatically passed by invoke.
    - language: Language edition of Wikipedia.
    - wikipedia_date (str, optional): The date of the Wikipedia dump to use. If not provided, the latest available dump is used.
    """
    if not wikipedia_date:
        wikipedia_date = get_latest_wikipedia_dump_date()

    if isinstance(language, str):
        language = [language]

    for lang in language:
        index_dir = get_wikipedia_collection_dir(
            workdir=workdir, language=lang, wikipedia_date=wikipedia_date
        )
        output_path = os.path.join(index_dir, "articles-html.json.tar.gz")
        if os.path.exists(output_path):
            logger.info(f"Wikipedia dump already exists at {output_path}")
            continue
        logger.info(f"Downloading Wikipedia dump for {lang} into {output_path}")
        download_url = f"https://dumps.wikimedia.org/other/enterprise_html/runs/{wikipedia_date}/{lang}wiki-NS0-{wikipedia_date}-ENTERPRISE-HTML.json.tar.gz"

        pathlib.Path(index_dir).mkdir(parents=True, exist_ok=True)
        multithreaded_download(download_url, output_path)


@task(pre=[load_api_keys])
def download_semantic_scholar_dump(
    c, workdir: str = DEFAULT_WORKDIR, num_threads: int = 8
):
    """
    This takes about 1.5 hours with 16 threads
    """
    # Read the API key from environment variables
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        raise ValueError("SEMANTIC_SCHOLAR_API_KEY environment variable is not set")

    # Set up headers with the API key
    headers = {"x-api-key": api_key}

    # Call the Semantic Scholar API to get the latest release information
    response = requests.get("https://api.semanticscholar.org/datasets/v1/release/")

    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch release information. Status code: {response.status_code}"
        )

    available_releases = response.json()  # sorted from old to new
    # it takes several days for each S2 release to become fully available. So here we are looking for the latest release that is complete (i.e. has at least 270 files)
    logger.info(f"Found {len(available_releases)} releases")
    for release in reversed(available_releases):
        response = requests.get(
            f"https://api.semanticscholar.org/datasets/v1/release/{release}/dataset/s2orc",
            headers=headers,
        )
        if response.status_code != 200:
            raise Exception(
                f"Failed to fetch release details. Status code: {response.status_code}. Error: {response.text}"
            )
        release_details = response.json()
        file_urls = release_details["files"]
        if len(file_urls) >= 270:
            break

    latest_release = release
    logger.info(f"Latest full release: {latest_release}")
    logger.info(f"Found {len(file_urls)} files to download")

    # Download all files
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for file_idx, url in enumerate(file_urls):
            file_path = os.path.join(
                workdir, "s2", f"semantic_scholar_dump_{file_idx:03d}.jsonl.gz"
            )
            if os.path.exists(file_path):
                logger.info(f"File already exists at {file_path}")
                continue
            future = executor.submit(
                multithreaded_download, url, file_path, num_parts=1
            )
            futures.append(future)

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Files",
            smoothing=0,
        ):
            future.result()  # This will raise any exceptions that occurred during download

    logger.info("All files have been downloaded.")


def get_wikipedia_collection_dir(workdir, language, wikipedia_date) -> str:
    return os.path.join(workdir, "wikipedia", wikipedia_date, language)


def get_wikipedia_collection_path(workdir, language, wikipedia_date) -> str:
    collection_dir = get_wikipedia_collection_dir(
        workdir=workdir, language=language, wikipedia_date=wikipedia_date
    )
    return os.path.join(collection_dir, "collection.parquet")


def get_wikidata_translation_map_path(workdir) -> str:
    return os.path.join(workdir, "wikipedia", "wikidata_translation_map")


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


@task(iterable=["language"], aliases=["preprocess_wikipedia"])
def preprocess_wikipedia_dump(
    c,
    language: str | list[str],
    workdir: str = DEFAULT_WORKDIR,
    wikipedia_date: Optional[str] = None,
    num_exclude_frequent_words_from_translation: int = 5000,
    num_workers=None,
    previous_date: Optional[str] = None,
):
    """
    Process Wikipedia HTML dump into a Parquet collection file.
    This takes ~4 hours for English on a 24-core CPU VM. Processing is fully parallelized so the time is proportional to number of cores available.
    It might take more for other languages, if we need to also get entity translations from Wikidata. This is because of Wikidata's rate limit.


    Args:
    - language (str): Language of the dump to process
    - workdir (str): Working directory for processing
    - wikipedia_date (str, optional): The date of the Wikipedia dump to use. If not provided, the latest available dump is used.
    - num_exclude_frequent_words_from_translation (int): For non-English Wikipedia dumps, we try to find English translations of all article names
    in Wikidata. We will exclude the `num_exclude_frequent_words_from_translation` most common words because neural models are already familiar with these.
    - previous_date (str, optional): Format YYYYMMDD. If provided, we will use the translation map from this date to translate article names.
    """
    if wikipedia_date is None:
        wikipedia_date = get_latest_wikipedia_dump_date()

    if isinstance(language, str):
        language = [language]
    for lang in language:
        output_path = get_wikipedia_collection_path(workdir, lang, wikipedia_date)
        if os.path.exists(output_path):
            logger.info(f"Collection already exists at '{output_path}'")
            continue

        # Download from the Hugging Face Hub, if exists
        try:
            hf_hub_download(
                repo_id="stanford-oval/wikipedia",
                repo_type="dataset",
                filename=f"{wikipedia_date}/{lang}/collection.parquet",
                local_dir=os.path.join(workdir, "wikipedia"),
            )

        except Exception as e:
            if e.__class__.__name__ not in [
                "RepositoryNotFoundError",
                "RevisionNotFoundError",
                "EntryNotFoundError",
                "LocalEntryNotFoundError",
                "EnvironmentError",
            ]:
                logger.warning(
                    f"Error while downloading from Hugging Face Hub: {str(e)}"
                )
        downloaded_file_path = get_wikipedia_collection_path(
            workdir, lang, wikipedia_date
        )
        if os.path.exists(downloaded_file_path):
            logger.info(
                "Found a preprocessed Wikipedia collection on Hugging Face Hub. Skipping preprocessing.",
            )
            continue

        logger.info(
            "No preprocessed Wikipedia collection found on Hugging Face Hub. Proceeding to preprocess the Wikipedia dump."
        )

        if (
            not os.path.exists(get_wikidata_translation_map_path(workdir))
            and lang != "en"
        ):
            logger.info(
                "Downloading Wikidata translation map for non-English Wikipedia dumps..."
            )
            snapshot_download(
                repo_id="stanford-oval/wikipedia",
                repo_type="dataset",
                allow_patterns="wikidata_translation_map*",  # only the translation folder
                local_dir=os.path.join(workdir, "wikipedia"),
            )

        index_dir = get_wikipedia_collection_dir(
            workdir=workdir, language=lang, wikipedia_date=wikipedia_date
        )
        input_path = os.path.join(index_dir, "articles-html.json.tar.gz")
        download_wikipedia_dump(
            c, language=lang, workdir=workdir, wikipedia_date=wikipedia_date
        )
        wikidata_translation_map = get_wikidata_translation_map_path(workdir)
        if num_workers:
            num_workers_flag = f"--num_workers {num_workers}"
        else:
            num_workers_flag = ""

        if previous_date:
            previous_date_flag = f"--previous_date {previous_date}"
        else:
            previous_date_flag = ""

        # Constructing the command with parameters
        command = (
            f"python -m preprocessing.preprocess_wikipedia_html_dump "
            f"--input_path {input_path} "
            f"--output_path {output_path} "
            f"--wikidata_translation_map {wikidata_translation_map} "
            f"--language {lang} "
            f"--should_translate "
            f"--num_exclude_frequent_words_from_translation {num_exclude_frequent_words_from_translation} "
            f"{previous_date_flag} "
            f"{num_workers_flag}"
        )

        # Running the command
        c.run(command)
