import argparse
import asyncio
import io
import os
import pathlib
import re
import sys
import tarfile
from datetime import datetime, timedelta
from multiprocessing import Process, cpu_count
from multiprocessing import SimpleQueue

from preprocessing.entity_translation import url_to_entity_name

import orjson
from tqdm import tqdm

from preprocessing.custom_docling import convert_html_to_blocks

sys.path.insert(0, "./")
import pandas as pd
import transformers

from preprocessing.block import Block, BlockLanguage
from preprocessing.utils import (
    batch_get_wikidata_english_name,
    draw_and_save_histogram_log_bins,
    find_forest_roots_and_members,
    get_from_translation_map,
    get_num_tokens,
    load_translation_map,
)
from preprocessing.wikipedia_disambiguation import is_disambiguation
from utils.logging import logger

transformers.utils.logging.set_verbosity_error()

inverse_redirection_map = (
    {}
)  # used to expand translation search. This map includes a map of each root to itself, for simplicity
frequent_words_to_exclude = set()


def get_wikipedia_block_url(article_url: str, section_title: str, language: str) -> str:
    if language != BlockLanguage.ENGLISH:
        section_title = re.sub(r" \(in English:[^)]*\)", "", section_title)
    # extract the last section titles
    section_title_parts = [p.strip() for p in section_title.split(">")]
    last_section_title = section_title_parts[-1]
    if last_section_title:
        url = f"{article_url}#{last_section_title.replace(' ', '_')}"  # section title should not be URL encoded, but spaces should be replaced with underscores
    else:
        url = article_url
    return url


def extract_article_blocks(
    article: dict,
    language,
    should_translate,
    global_translation_map,
    frequent_words_to_exclude,
    inverse_redirection_map,
) -> list[Block]:
    article_blocks = []
    if "article_body" not in article or "html" not in article["article_body"]:
        return article_blocks

    html = article["article_body"]["html"]
    document_title = article["name"]

    article_blocks = convert_html_to_blocks(
        html=html,
        document_title=document_title,
        should_translate=should_translate,
        global_translation_map=global_translation_map,
        frequent_words_to_exclude=frequent_words_to_exclude,
        inverse_redirection_map=inverse_redirection_map,
    )

    for block in article_blocks:
        if block.num_tokens == 0:
            block.num_tokens = get_num_tokens(block.combined_text)
        if should_translate:
            block.deduplicate_translations()

        block.last_edit_date = article["date_modified"]
        block.language = language

        block.url = get_wikipedia_block_url(
            article_url=article["url"],
            section_title=block.section_title,
            language=language,
        )

    return article_blocks


def article_processor_worker(
    input_queue: SimpleQueue,
    output_queue: SimpleQueue,
    language: str,
    should_translate: bool,
    global_translation_map: dict,
    frequent_words_to_exclude: set,
    inverse_redirection_map: dict,
):
    while True:
        try:
            article = input_queue.get()
        except EOFError:  # faster_fifo raises EOFError when closed
            logger.info("Input queue closed, worker exiting.")
            break

        if article is None:
            break  # termination signal

        try:
            article_blocks = extract_article_blocks(
                article=article,
                language=language,
                should_translate=should_translate,
                global_translation_map=global_translation_map,
                frequent_words_to_exclude=frequent_words_to_exclude,
                inverse_redirection_map=inverse_redirection_map,
            )
            output_queue.put(article_blocks)
        except Exception as e:
            # So that the worker doesn't die and continues processing other articles
            logger.warning(f"Error processing article {article['name']}: {str(e)}")

    output_queue.put(None)  # signal the end


def build_redirection_map(file_path: str) -> dict:
    redirection_incoming_edges: dict[str, set[str]] = (
        {}
    )  # maps an article url with all urls that redirect to it, via one or multiple hops

    for article in tqdm(
        tarfile_loader(file_path),
        desc="Building the Wikipedia redirection graph",
        miniters=1e-6,
        mininterval=0.5,
        unit_scale=1,
        unit=" Articles",
        smoothing=0,
    ):
        if is_disambiguation(article):
            continue

        url = url_to_entity_name(article["url"])
        if url not in redirection_incoming_edges:
            redirection_incoming_edges[url] = set()
        if "redirects" in article:
            # Add redirects even if we have already seen it. This way, multi-hop redirects will be handled correctly.
            for redirect in article["redirects"]:
                redirected_url = url_to_entity_name(redirect["url"])
                redirection_incoming_edges[url].add(redirected_url)

    # the structure of this dictionary describes a forest (i.e. collection of trees), with each item describing the incoming edges of a node
    # we want to find the root of all trees
    redirect_map = find_forest_roots_and_members(redirection_incoming_edges)
    return redirect_map


def tarfile_loader(file_path: str):
    """
    Generator that sequentially loads articles from a tar.gz file containing NDJSON formatted articles,
    using a buffered reader to reduce POSIX read calls.
    """
    buffer_size = 2**20  # 1MB buffer
    try:
        with tarfile.open(file_path, mode="r|gz") as tar_file_:
            for member in tar_file_:
                if member.isfile():
                    try:
                        file_obj = tar_file_.extractfile(member)
                        if file_obj is None:
                            logger.warning(
                                f"Could not extract file object for member: {member.name}"
                            )
                            continue
                        # Use a buffered reader for efficient reading
                        buffered_obj = io.BufferedReader(
                            file_obj, buffer_size=buffer_size
                        )
                        for line in buffered_obj:
                            try:
                                article = orjson.loads(line)
                                yield article
                            except orjson.JSONDecodeError as json_err:
                                logger.warning(
                                    f"Skipping malformed JSON line in {member.name}: {json_err}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Error processing line in {member.name}: {e}"
                                )
                    except KeyError as ke:
                        # tarfile.extractfile can raise KeyError if the member is not found or is not a file/link
                        logger.warning(f"Could not extract member {member.name}: {ke}")
                    except Exception as e:
                        logger.error(f"Error processing member {member.name}: {e}")
    except tarfile.ReadError as tar_err:
        logger.error(f"Error reading tar file {file_path}: {tar_err}")
    except FileNotFoundError:
        logger.error(f"Tar file not found: {file_path}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while opening or reading {file_path}: {e}"
        )


def get_articles_without_disambiguation_or_redirections(
    file_path: str,
    num_workers: int,
    queue: SimpleQueue,
    redirect_map: dict,
    max_articles: int,
    previous_date: datetime | None,
):
    """
    Reads the tar file again, filters articles, and puts them onto the queue for processing.
    Filters out disambiguation pages, redirected pages, duplicates, and optionally old articles.
    """
    counter = 0
    processed_count = 0
    seen_article_names = set()

    # Calculate the date threshold if previous_date is provided
    date_threshold = previous_date - timedelta(days=30) if previous_date else None

    article_iterator = tarfile_loader(file_path)

    while True:
        try:
            article = next(article_iterator)
            processed_count += 1

            # 1. Check if it's a disambiguation page
            if is_disambiguation(article):
                continue

            # 2. Get article name and check if it's a root article we care about
            try:
                article_name = url_to_entity_name(article["url"])
            except KeyError:
                logger.warning("Article missing 'url' key, skipping.")
                continue

            # Check if it's a target of a redirect (present in redirect_map keys)
            # If redirect_map is provided, only process articles that are keys in it.
            if redirect_map and article_name not in redirect_map:
                continue

            # 3. Check if we've already seen this article name in this pass
            if article_name in seen_article_names:
                # This can happen if the dump contains duplicates or near-duplicates
                logger.debug(f"Skipping duplicate article name: {article_name}")
                continue
            seen_article_names.add(article_name)

            # 4. Check modification date if previous_date is set
            if date_threshold:
                try:
                    # Optimize date parsing slightly by checking format implicitly
                    article_date_str = article.get("date_modified")
                    if (
                        not article_date_str
                        or len(article_date_str) != 20
                        or article_date_str[-1] != "Z"
                    ):
                        raise ValueError("Unexpected date format")
                    article_date = datetime.fromisoformat(
                        article_date_str.replace("Z", "+00:00")
                    )

                    if article_date < date_threshold:
                        continue
                except (KeyError, ValueError, TypeError) as date_err:
                    logger.warning(
                        f"Could not parse date_modified ('{article.get('date_modified')}') for article {article_name}: {date_err}. Processing anyway."
                    )
                    # Decide whether to process or skip if date is invalid/missing
                    # Current behavior: process it. Change 'continue' above if skipping is preferred.

            # If all checks pass, put the article on the queue
            queue.put(article)
            counter += 1

            # 5. Check if max_articles limit is reached
            if max_articles > 0 and counter >= max_articles:
                logger.info(f"Reached max_articles limit ({max_articles}).")
                break

        except StopIteration:
            # End of the tar file
            logger.info(
                f"Finished reading articles from {file_path}. Processed {processed_count} raw entries, queued {counter} articles."
            )
            break
        except Exception as e:
            # Catch potential errors during iteration/processing within the loop
            logger.error(f"Error processing article stream: {e}", exc_info=True)
            # Depending on the error, you might want to break or continue
            continue  # Attempt to continue with the next article

    # Signal end to all worker processes
    logger.info("Signaling end to worker processes.")
    for _ in range(num_workers):
        queue.put(None)
    logger.info("All 'None' signals sent to workers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="A Wikipedia HTML dump, which is a tar.gz file containing multiple .ndjson files",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Parquet output file"
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=[lang.value for lang in BlockLanguage],
    )
    parser.add_argument(
        "--should_translate",
        action="store_true",
        help="If we should translate named entities to English using Wikidata. Has no effect if `--language` is English",
    )
    parser.add_argument(
        "--wikidata_translation_map",
        type=str,
        help="Where to read/write the translation mapping we obtain from Wikidata. Should be a Parquet dataset directory.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(4, int(cpu_count() * 0.75)),
        help="Number of worker processes to spawn",
    )
    parser.add_argument(
        "--max_articles",
        type=int,
        default=-1,
        help="Will stop after processing this many articles. -1 means no limit. Used for testing.",
    )
    parser.add_argument(
        "--num_exclude_frequent_words_from_translation",
        type=int,
        default=0,
        help="Will exclude translations for the top N most frequent words used in the English Wikipedia. Numbers will always ne excluded.",
    )
    parser.add_argument(
        "--previous_date",
        type=str,
        default=None,
        help="The date of the previous dump. If provided, we will only process articles that have been modified since then.",
    )

    args = parser.parse_args()
    if args.language == "en":
        args.should_translate = False

    # convert date from YYYYMMDD to datetime
    if args.previous_date:
        args.previous_date = datetime.strptime(args.previous_date, "%Y%m%d")

    redirection_map = build_redirection_map(args.input_path)

    for node, redirectors in redirection_map.items():
        for node2 in redirectors:
            inverse_redirection_map[node2] = node

    logger.info(
        f"Number of articles, excluding disambiguation and redirection pages: {len(redirection_map):,}"
    )
    if args.should_translate:
        if args.num_exclude_frequent_words_from_translation > 0:
            with open("preprocessing/word_list.txt") as f:
                for line in f:
                    frequent_words_to_exclude.add(line.strip().lower())
                    if (
                        len(frequent_words_to_exclude)
                        >= args.num_exclude_frequent_words_from_translation
                    ):
                        break

        global_translation_map = load_translation_map(
            args.wikidata_translation_map, language=args.language
        )

        non_cached_titles = []
        for url in redirection_map:
            if (
                get_from_translation_map(
                    url, inverse_redirection_map, global_translation_map
                )
                is None
            ):
                non_cached_titles.append(url)

        if len(non_cached_titles) > 0:
            logger.info(
                f"Did not find {len(non_cached_titles):,} articles in the translation map, will call the Wikidata API for them",
            )
            asyncio.run(
                batch_get_wikidata_english_name(
                    non_cached_titles,
                    args.language,
                    args.wikidata_translation_map,
                    global_translation_map,
                )
            )
    else:
        global_translation_map = {}

    input_queue = SimpleQueue()
    output_queue = SimpleQueue()
    all_worker_processes = []

    logger.info(f"Using {args.num_workers} workers")

    for worker_id in range(args.num_workers):
        all_worker_processes.append(
            Process(
                target=article_processor_worker,
                args=(
                    input_queue,
                    output_queue,
                    args.language,
                    args.should_translate,
                    global_translation_map,
                    frequent_words_to_exclude,
                    inverse_redirection_map,
                ),
            )
        )

    # The process that feeds the articles to workers
    reader_process = Process(
        target=get_articles_without_disambiguation_or_redirections,
        args=(
            args.input_path,
            args.num_workers,
            input_queue,
            redirection_map,
            args.max_articles,
            args.previous_date,
        ),
    )

    for p in all_worker_processes + [reader_process]:
        p.start()

    workers_finished = 0
    all_blocks = []

    # make parent directories
    pathlib.Path(os.path.dirname(args.output_path)).mkdir(parents=True, exist_ok=True)

    counter = 0
    num_tokens = []

    pbar = tqdm(
        desc="Extracting blocks",
        miniters=1e-6,
        mininterval=0.5,
        unit_scale=1,
        unit=" Articles",
        smoothing=0,
        total=(
            len(redirection_map) if args.max_articles <= 0 else args.max_articles
        ),  # Adjust total for pbar if max_articles is set
    )

    while workers_finished < len(all_worker_processes):
        article_blocks = output_queue.get()
        if article_blocks is None:
            workers_finished += 1
            continue  # Got None signal from a worker

        # Process the received blocks
        for block in article_blocks:
            all_blocks.append(block.model_dump())
            counter += 1
            num_tokens.append(block.num_tokens)
        pbar.update(1)  # Update pbar per article processed

    # Wait for processes to complete cleanly
    logger.info("Waiting for reader process to join...")
    reader_process.join(timeout=10)  # Add timeout
    if reader_process.is_alive():
        logger.warning("Reader process did not join cleanly, terminating.")
        reader_process.terminate()

    logger.info("Waiting for worker processes to join...")
    for p in all_worker_processes:
        p.join(timeout=10)  # Add timeout
        if p.is_alive():
            logger.warning(f"Worker process {p.pid} did not join cleanly, terminating.")
            p.terminate()

    logger.info(f"Saving the collection to '{args.output_path}'")
    df = pd.DataFrame(all_blocks)
    df.to_parquet(args.output_path, index=False)

    logger.info(f"Total number of blocks: {len(all_blocks):,d}")

    if num_tokens:  # Ensure list is not empty before drawing histogram
        draw_and_save_histogram_log_bins(
            num_tokens,
            args.output_path,
        )
    else:
        logger.warning("No tokens were processed, skipping histogram generation.")
