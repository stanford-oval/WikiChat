import asyncio
import os
import sys
from time import time
from urllib.parse import quote

import aiohttp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from retrieval.embedding_model_info import embedding_model_to_parameters

sys.path.insert(0, "./")
from tasks.defaults import DEFAULT_EMBEDDING_MODEL_NAME
from utils.logging import logger


tokenizer = None  # load when needed
translation_prefix = "(in English: "


def extract_english_translations(text):
    """
    Extracts all instances of substrings in the form "(in English: english_translation)"
    from the given text, supporting nested parentheses.

    Parameters:
    - text (str): The text from which to extract the substrings.

    Returns:
    - list of str: A list of extracted substrings in the format "(in English: english_translation)".
    """

    translations = []
    i = 0
    try:
        while True:
            # Detect the start of a potential translation
            i = text.find(translation_prefix, i)
            if i < 0:
                break
            start_index = i
            stack = []
            stack.append("(")
            i += len(translation_prefix)
            while True:
                if text[i] == "(":
                    stack.append(text[i])
                elif text[i] == ")" and stack:
                    stack.pop()
                    # If the stack is empty, we've found a complete translation
                    if not stack:
                        end_index = i + 1  # Include the closing parenthesis
                        break
                i += 1
            translations.append(text[start_index:end_index])
            i = end_index
    except IndexError:
        # this can happen if we truncate section_title too short
        logger.warning(f"Error while extracting English translations from text {text}")

    return translations


def replace_except_first(s, old, new):
    # Find position of the first occurrence
    pos = s.find(old)

    # If the substring is not found, return the original string
    if pos == -1:
        return s

    # Split the string into two parts
    # The first part is up to and including the first occurrence of the substring
    # The second part is the rest of the string
    first_part = s[: pos + len(old)]
    rest = s[pos + len(old) :]

    # Replace the substring in the rest of the string
    rest_replaced = rest.replace(old, new)

    # Concatenate the first part back with the modified rest of the string
    return first_part + rest_replaced


def get_from_translation_map(
    entity: str, inverse_redirection_map: dict, global_translation_map: dict
):
    if entity not in global_translation_map and (
        entity not in inverse_redirection_map
        or inverse_redirection_map[entity] not in global_translation_map
    ):
        return None
    if entity in global_translation_map and global_translation_map[entity] is not None:
        return global_translation_map[entity]
    else:
        return global_translation_map[inverse_redirection_map[entity]]


def load_translation_map(parquet_dataset_dir: str, language: str) -> dict:
    try:
        # Read the Parquet dataset, applying a filter for the 'language' partition
        table = pq.read_table(
            parquet_dataset_dir,
            filters=[("language", "=", language)],
        )

        # Extract the 'translation_key' and 'translation_value' columns directly from the PyArrow table
        translation_keys = table.column("translation_key").to_pylist()
        translation_values = table.column("translation_value").to_pylist()

        # Build the dictionary directly from the PyArrow columns
        translation_map = dict(zip(translation_keys, translation_values))

        # Log the number of translations loaded
        logger.info(
            f"Loaded {len(translation_map):,} Wikidata entity translations for language {language}"
        )
    except FileNotFoundError:
        logger.warning(
            f"Could not find the Wikidata translation map file at '{parquet_dataset_dir}'. Initializing the translation map as an empty dict."
        )
        translation_map = {}
    return translation_map


def append_to_translation_map(parquet_dataset_dir: str, new_rows: list[dict]):
    """
    Appends new translation rows to the existing translation map in a Parquet dataset.
    It doesn't deduplicate the new rows against the old rows due to efficiency reasons.
    However, when we load the translation map, we use the newest row for each translation_key.
    """
    # Convert the list of dictionaries to a PyArrow Table
    logger.info(
        f"Saving {len(new_rows)} new translation rows to '{parquet_dataset_dir}'"
    )

    # Convert the new rows to a pandas DataFrame and remove duplicates based on language and translation_key columns
    new_df = (
        pd.DataFrame(new_rows)
        .drop_duplicates(subset=["language", "translation_key"])
        .reset_index(drop=True)
    )

    # Convert the DataFrame to a PyArrow Table
    new_table = pa.Table.from_pandas(new_df)

    try:
        # Check if the dataset directory exists
        if os.path.exists(parquet_dataset_dir):
            logger.info(
                f"Appending new rows to the existing Parquet dataset at '{parquet_dataset_dir}'"
            )
        else:
            logger.info(f"Creating a new Parquet dataset at '{parquet_dataset_dir}'")

        # Write the new rows to the Parquet dataset, partitioned by the 'language' column
        pq.write_to_dataset(
            new_table,
            root_path=parquet_dataset_dir,
            partition_cols=["language"],
            use_dictionary=True,  # Use dictionary encoding for efficiency
            compression="snappy",  # Use snappy compression for efficient storage
        )

        logger.info("Successfully appended new translation rows to the Parquet dataset")

    except Exception as e:
        logger.warning(f"Failed to save translation map to '{parquet_dataset_dir}'")
        logger.exception(e)


async def get_wikidata_english_name(
    document_title: str, session, language: str, global_translation_map: dict
):
    """
    Returns
        (english_name: str, new_translation_dict: dict)
    """
    if (
        get_from_translation_map(
            document_title, {}, global_translation_map=global_translation_map
        )
        is not None
    ):
        return (
            get_from_translation_map(
                document_title, {}, global_translation_map=global_translation_map
            ),
            {},
        )
    try:
        # the API expects a user agent
        # labels cover more entity-languages, but are sometimes ambiguous. Therefore, we give priority to sitelinks and fallback to labels if needed.
        url = (
            f"https://www.wikidata.org/w/api.php?"
            f"action=wbgetentities&"
            f"normalize=0&"
            f"sites={language}wiki&"
            f"titles={quote(document_title, safe='')}&"
            f"format=json&"
            f"props=sitelinks|labels"  # sitelinks are Wikipedia pages in other languages, lables are Entity translations
        )
        async with session.get(
            url=url,
            headers={"User-Agent": "wikichat/1.0"},
        ) as response:
            a = await response.json()
            wikidata_entity = a["entities"]
            assert len(wikidata_entity) == 1, "found 0 or >1 Wikidata entities"

            wikidata_entity = list(wikidata_entity.items())[0][1]
            sitelinks = (
                wikidata_entity["sitelinks"] if "sitelinks" in wikidata_entity else {}
            )
            sitelink_dict = {}
            for site, v in sitelinks.items():
                if not site.endswith("wiki") or site in [
                    "Wikifunctionswiki",
                    "species",
                    "foundation",
                    "outreach",
                    "mediawiki",
                    "wikimania",
                    "wikifunctions",
                    "sources",
                    "wikidata",
                ]:
                    # These are sites that end with voyage, quote, news
                    continue
                lang = site[: -len("wiki")]
                sitelink_dict[lang] = v["title"]

            english_locale = None
            if "labels" not in wikidata_entity:
                logger.debug(
                    f"Did not find any labels in the Wikidata entry {wikidata_entity}"
                )
                return None, {language: {document_title: ""}}

            if "en" in sitelink_dict:
                english_name = sitelink_dict["en"]
            else:
                if "en" in wikidata_entity["labels"]:
                    english_locale = "en"
                elif "en-gb" in wikidata_entity["labels"]:
                    english_locale = "en-gb"
                elif "en-ca" in wikidata_entity["labels"]:
                    english_locale = "en-ca"
                else:
                    logger.debug(
                        f"Did not find any English labels in Wikidata for {document_title}"
                    )
                    return None, {language: {document_title: ""}}
                english_name = wikidata_entity["labels"][english_locale]["value"]

            new_translation_dict = {}
            set_of_available_languages = set(
                list(sitelink_dict.keys()) + list(wikidata_entity["labels"].keys())
            )

            # No need to include these in the translation map
            for lang in ["en", "en-gb", "en-ca", "commons", "simple"]:
                set_of_available_languages.discard(lang)

            for lang in set_of_available_languages:
                if lang in sitelink_dict:
                    new_translation_dict[lang] = {sitelink_dict[lang]: english_name}
                else:
                    new_translation_dict[lang] = {
                        wikidata_entity["labels"][lang]["value"]: english_name
                    }
            # Add the actual article title as well. Sometimes it is a different version of what we find in Wikidata.
            # E.g. the article title is "Legazpi (Madrid)" but the Wikidata returns "Legazpi"
            if language not in new_translation_dict:
                # this sometimes happens due to the Wikidata entry being incomplete
                new_translation_dict[language] = {}
            new_translation_dict[language][document_title] = english_name

            return english_name, new_translation_dict
    except Exception as e:
        logger.warning(
            f"Unable to get entry for article '{document_title}' due to error"
        )
        logger.exception(e)
        return -1, {language: {document_title: ""}}


async def batch_get_wikidata_english_name(
    document_titles: list[str],
    language: str,
    wikidata_translation_map_path: str,
    global_translation_map: dict,
):
    new_translation_rows = []
    async with aiohttp.ClientSession() as session:
        with logging_redirect_tqdm():
            minibatch_size = 100  # The wikipedia API only allows 100 requests per second, so we batch the requests.
            for i in trange(
                0,
                len(document_titles),
                minibatch_size,
                desc="Getting Wikidata entries",
                smoothing=0,
                unit="Batch",
            ):
                start_time = time()
                batch = document_titles[
                    i : min(len(document_titles), i + minibatch_size)
                ]
                ret = await asyncio.gather(
                    *[
                        get_wikidata_english_name(
                            document_title, session, language, global_translation_map
                        )
                        for document_title in batch
                    ]
                )  # ret is a list of tuples

                # convert to two lists
                batch_english_names, batch_new_translation_dicts = tuple(zip(*ret))
                batch_english_names, batch_new_translation_dicts = (
                    list(batch_english_names),
                    list(batch_new_translation_dicts),
                )

                if -1 in batch_english_names:
                    logger.debug("Reached the Wikidata rate limit. Will wait longer.")
                    # -1 is used to signal a rate limit. wait for longer if we get a rate limit error
                    await asyncio.sleep(2)

                # Add to the global translation dictionary
                for translation_dict in batch_new_translation_dicts:
                    for lang in translation_dict.keys():
                        for k, v in translation_dict[lang].items():
                            new_translation_rows.append(
                                {
                                    "language": lang,
                                    "translation_key": k,
                                    "translation_value": v,
                                },
                            )
                time_passed = time() - start_time
                time_to_wait = 1.1 - time_passed
                if time_to_wait > 0:
                    await asyncio.sleep(time_to_wait)

    append_to_translation_map(
        parquet_dataset_dir=wikidata_translation_map_path, new_rows=new_translation_rows
    )


def print_histogram(data):
    percentiles = np.percentile(data, np.arange(0, 101, 5))
    hist, bin_edges = np.histogram(data, bins=percentiles)

    logger.info("Histogram of number of characters in blocks, every 5th percentile:")
    for i in range(len(hist)):
        # Calculate start (s) and end (e) of the current bin
        s = bin_edges[i]
        e = bin_edges[i + 1]

        # Normalize count to the maximum bar length
        count = hist[i]

        # Create a text bar for each bin with normalized length
        bin_ = f"[{s:0.0f}, {e:0.0f})"
        logger.info(f"{bin_:<20}: ({count})")


def remove_all_extensions(file_path):
    base_name = file_path
    while True:
        base_name, ext = os.path.splitext(base_name)
        if not ext:
            break
    return base_name


def draw_and_save_histogram_log_bins(data, filename: str) -> None:
    import matplotlib.pyplot as plt

    # Create logarithmically spaced bins
    bins = np.logspace(np.log10(min(data)), np.log10(max(data)), num=5000)

    weights = 100 / sum(data)
    plt.hist(
        data, weights=[weights] * len(data), bins=bins, color="blue", edgecolor="blue"
    )
    plt.xscale("log")  # Set x-axis to logarithmic scale
    plt.title("Histogram of Passage Lengths (Logarithmic Bins)")
    plt.xlabel("Passage Length (tokens)")
    plt.ylabel("Frequency (%)")

    context_length = embedding_model_to_parameters[DEFAULT_EMBEDDING_MODEL_NAME][
        "max_sequence_length"
    ]  # Draw a vertical line at x=context_length
    plt.axvline(x=context_length, color="red", linestyle="--", linewidth=1, alpha=0.5)

    # Label the line without rotating the label
    plt.text(
        context_length,
        max(plt.ylim()) * 0.85,
        f"{DEFAULT_EMBEDDING_MODEL_NAME}\ncontext length",
        color="red",
        ha="center",
    )
    median = np.median(data)
    plt.axvline(x=median, color="orange", linestyle="--", linewidth=1, alpha=0.5)
    plt.text(
        median,
        max(plt.ylim()) * 0.75,
        f"Median: {int(median)}",
        color="orange",
        ha="center",
    )
    percentile_99 = np.percentile(data, 99)
    plt.axvline(x=percentile_99, color="green", linestyle="--", linewidth=1, alpha=0.5)
    plt.text(
        percentile_99,
        max(plt.ylim()) * 0.65,
        f"99th Percentile: {int(percentile_99)}",
        color="green",
        ha="center",
    )
    filename = remove_all_extensions(filename) + "_histogram.png"
    plt.savefig(filename)


def find_roots(incoming_edges):
    all_nodes = set(incoming_edges.keys())
    non_root_nodes = set()
    for edges in incoming_edges.values():
        for node in edges:
            non_root_nodes.add(node)

    return set(all_nodes) - set(non_root_nodes)


def find_tree_members(root, incoming_edges):
    visited = set()
    queue = [root]
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            if node in incoming_edges:
                queue.extend(incoming_edges[node])
            else:
                pass  # some redirected pages are never encountered and therefore do not have a separate node in `incoming_edges`

    return visited


def find_forest_roots_and_members(incoming_edges):
    roots = find_roots(incoming_edges)
    trees = {}
    for root in roots:
        trees[root] = find_tree_members(root, incoming_edges)
    return trees


def get_num_tokens(text: str) -> int:
    global tokenizer
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            DEFAULT_EMBEDDING_MODEL_NAME, fast=True
        )
    return len(tokenizer(text)["input_ids"])


def batch_get_num_tokens(texts: list[str]) -> list[int]:
    global tokenizer
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            DEFAULT_EMBEDDING_MODEL_NAME, fast=True
        )
    tokenized_texts = tokenizer(
        texts, padding=False, truncation=False, return_length=True
    )
    return tokenized_texts["length"]


def pack_blocks(blocks: list, pack_to_tokens: int) -> list:
    """
    This function concatenates consecutive Blocks from the same document with the same `section_title`
    as long as their combined length does not exceed `pack_to_tokens` tokens.
    """
    if not blocks:
        return []

    assert all(
        block.document_title == blocks[0].document_title for block in blocks
    ), "All blocks must have the same document_title"

    packed_blocks = []
    current_block = blocks[0]

    current_block.num_tokens = get_num_tokens(current_block.combined_text)
    for next_block in blocks[1:]:
        # Check if the next block has the exact same section title and does not exceed the character limit
        num_tokens_after_merge = current_block.num_tokens + get_num_tokens(
            "\n" + next_block.content
        )
        if (
            next_block.section_title == current_block.section_title
            and num_tokens_after_merge < pack_to_tokens
        ):
            current_block.content += (
                " " + next_block.content
            )  # Concatenate blocks with a space in between
            current_block.num_tokens = num_tokens_after_merge
        else:
            # Once a block reaches the limit or a new title is found, append the current state and move on
            packed_blocks.append(current_block)
            current_block = next_block
            current_block.num_tokens = get_num_tokens(current_block.combined_text)

    # Adding the last accumulated paragraph
    packed_blocks.append(current_block)

    return packed_blocks
