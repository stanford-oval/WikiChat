import asyncio
import sys
from time import time
from urllib.parse import quote

import aiohttp
import numpy as np
import orjsonl
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import AutoTokenizer

sys.path.insert(0, "./")
from pipelines.utils import get_logger

logger = get_logger(__name__)

# Mapping for all languages to English. E.g. global_translation_map["fa"] is a dictionary of Farsi -> English translations
# values can be the empty string "", which means we have already looked up the translations in Wikidata, but did not find the English translation
# this is different from the case that the key is absent, which means we have never looked up that translation in Wikidata
global_translation_map = {}

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3-unsupervised", fast=True)
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
    source_language: str, entity: str, inverse_redirection_map: dict = {}
):
    global global_translation_map
    if source_language not in global_translation_map:
        return None
    if entity not in global_translation_map[source_language] and (
        entity not in inverse_redirection_map
        or inverse_redirection_map[entity]
        not in global_translation_map[source_language]
    ):
        return None
    if (
        entity in global_translation_map[source_language]
        and global_translation_map[source_language][entity] is not None
    ):
        return global_translation_map[source_language][entity]
    else:
        return global_translation_map[source_language][
            inverse_redirection_map[entity]
        ]


def load_translation_map(file_name: str):
    global global_translation_map
    try:
        for language in tqdm(
            orjsonl.stream(file_name), desc="Loading translation map", smoothing=0
        ):
            global_translation_map[language["language"]] = language["translations"]
    except FileNotFoundError as e:
        logger.warning(
            "Could not find the Wikidata translation map file at %s. Initializing the translation map as an empty dictionary.",
            file_name,
        )
        global_translation_map = {}


def save_translation_map(file_name: str):
    global global_translation_map
    orjsonl.save(
        file_name,
        tqdm(
            [
                {
                    "language": language,
                    "translations": global_translation_map[language],
                }
                for language in global_translation_map
            ],
            desc="Saving translation map",
            smoothing=0,
        ),
        compression_format="gz",
    )


async def get_wikidata_english_name(article_title: str, session, language: str):
    """
    Returns
        (english_name: str, new_translation_dict: dict)
    """
    global global_translation_map
    if get_from_translation_map(language, article_title) is not None:
        return get_from_translation_map(language, article_title), {}
    try:
        # the API expects a user agent
        # labels cover more entity-languages, but are sometimes ambiguous. Therefore, we give priority to sitelinks and fallback to labels if needed.
        url = (
            f"https://www.wikidata.org/w/api.php?"
            f"action=wbgetentities&"
            f"normalize=0&"
            f"sites={language}wiki&"
            f"titles={quote(article_title, safe='')}&"
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
                    "Did not find any labels in the Wikidata entry %s",
                    str(wikidata_entity),
                )
                return None, {language: {article_title: ""}}

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
                        "Did not find any English labels in Wikidata for %s",
                        article_title,
                    )
                    return None, {language: {article_title: ""}}
                english_name = wikidata_entity["labels"][english_locale]["value"]

            new_translation_dict = {}
            set_of_available_languages = set(
                list(sitelink_dict.keys()) + list(wikidata_entity["labels"].keys())
            )

            # No need to include these in the translation map
            for l in ["en", "en-gb", "en-ca", "commons", "simple"]:
                set_of_available_languages.discard(l)

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
            new_translation_dict[language][article_title] = english_name

            return english_name, new_translation_dict
    except Exception as e:
        logger.warning(
            "Unable to get entry for article '%s' due to error",
            article_title,
        )
        logger.exception(e)
        return -1, {language: {article_title: ""}}


async def batch_get_wikidata_english_name(article_titles: list[str], language: str):
    global global_translation_map
    async with aiohttp.ClientSession() as session:
        with logging_redirect_tqdm():
            minibatch_size = 100  # The wikipedia API only allows 100 requests per second, so we batch the requests.
            for i in trange(
                0,
                len(article_titles),
                minibatch_size,
                desc="Getting Wikidata entries",
                smoothing=0,
                unit="Batch",
            ):
                start_time = time()
                batch = article_titles[i : min(len(article_titles), i + minibatch_size)]
                ret = await asyncio.gather(
                    *[
                        get_wikidata_english_name(article_title, session, language)
                        for article_title in batch
                    ]
                )  # ret is a list of tuples

                # convert to two lists
                batch_english_names, batch_new_translation_dicts = tuple(zip(*ret))
                batch_english_names, batch_new_translation_dicts = list(
                    batch_english_names
                ), list(batch_new_translation_dicts)

                if -1 in batch_english_names:
                    logger.debug("Reached the Wikidata rate limit. Will wait longer.")
                    # -1 is used to signal a rate limit. wait for longer if we get an rate limit error
                    await asyncio.sleep(2)

                # Add to the global translation dictionary
                for translation_dict in batch_new_translation_dicts:
                    for lang in translation_dict.keys():
                        if lang not in global_translation_map:
                            global_translation_map[lang] = {}
                        for k, v in translation_dict[lang].items():
                            global_translation_map[lang][k] = v
                time_passed = time() - start_time
                time_to_wait = 1.1 - time_passed
                if time_to_wait > 0:
                    await asyncio.sleep(time_to_wait)


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


def draw_and_save_histogram_log_bins(data, filename: str) -> None:
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

    context_length = 8192  # for BAAI/bge-m3-unsupervised
    # Draw a vertical line at x=context_length
    plt.axvline(x=context_length, color="red", linestyle="--", linewidth=1, alpha=0.5)

    # Label the line without rotating the label
    plt.text(
        context_length,
        max(plt.ylim()) * 0.85,
        "BAAI/bge-m3-unsupervised\ncontext length",
        color="red",
        ha="center",
    )
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

def num_tokens(text: str) -> int:
    return len(tokenizer(text)["input_ids"])
