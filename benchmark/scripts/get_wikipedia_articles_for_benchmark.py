import argparse
import asyncio
import json
import sys
from typing import List
from urllib.parse import quote

import aiohttp
import orjsonl
import requests
from datasets import load_dataset
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm

sys.path.insert(0, "./")
from pipelines.utils import get_logger

logger = get_logger(__name__)


def load_collection(collection_path: str, max_articles: int = None):
    all_wiki_titles_intros = {}
    for line in tqdm(orjsonl.stream(collection_path), desc="Loading collection"):
        if (
            line["block_type"] != "text"
            or line["article_title"] in all_wiki_titles_intros
        ):
            continue
        title = line["article_title"]
        all_wiki_titles_intros[title] = line["content_string"]
        if len(all_wiki_titles_intros) == max_articles:
            break
    logger.info("Loaded %d articles from collection", len(all_wiki_titles_intros))

    return all_wiki_titles_intros


def get_most_edited_wikipedia_titles(
    language: str,
    year: str,
    month: str,
    day: str = "all-days",
):
    a = requests.get(
        f"https://wikimedia.org/api/rest_v1/metrics/edited-pages/top-by-edits/{language}.wikipedia/all-editor-types/content/{year}/{month}/{day}"
    )
    results = a.json()["items"][0]["results"][0]["top"]
    titles = [result["page_title"].replace("_", " ") for result in results]
    return titles


def get_most_edited_wikipedia_articles(language: str, all_wiki_titles_intros):
    most_edited_titles = []
    most_edited_titles.extend(get_most_edited_wikipedia_titles(language, "2023", "11"))
    most_edited_titles.extend(get_most_edited_wikipedia_titles(language, "2023", "12"))
    most_edited_titles.extend(get_most_edited_wikipedia_titles(language, "2024", "01"))
    most_edited_titles.extend(get_most_edited_wikipedia_titles(language, "2024", "02"))
    most_edited_titles.extend(get_most_edited_wikipedia_titles(language, "2024", "03"))
    most_edited_titles.extend(get_most_edited_wikipedia_titles(language, "2024", "04"))

    logger.info("most_edited_titles = %s", str(most_edited_titles))

    most_edited_titles_intros = {}
    for title in most_edited_titles:
        if title in most_edited_titles_intros:
            continue
        if title in all_wiki_titles_intros:
            most_edited_titles_intros[title] = all_wiki_titles_intros[title]
        else:
            logger.info("Missing article in collection: %s", title)

    logger.info(
        "Found %d articles out of %d",
        len(most_edited_titles_intros),
        len(most_edited_titles),
    )
    return most_edited_titles_intros


async def get_total_views(article: str, session, end_date: str, language: str):
    try:
        # we input year 2000 to year 3000, and the API will return from the beginning of statistics (2015) to the current unfinished month
        # the API expects a user agent
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{language}.wikipedia.org/all-access/user/{quote(article, safe='')}/monthly/20000101/{end_date}"
        async with session.get(
            url=url,
            headers={"User-Agent": "wikichat/1.0"},
        ) as response:
            a = await response.json()
            if "items" not in a or ("title" in a and a["title"] == "Not found."):
                # logger.info(
                #     "Number of page views for article '%s' not found, probably due to it being too new.", article,
                # )
                return 0
            results = a["items"]
            views = [item["views"] for item in results]
            return sum(views)
    except Exception as e:
        logger.error(
            "Unable to get views for request '%s' due to error %s.", url, str(e)
        )
        return -1


async def batch_get_total_views(articles: List[str], end_date: str, language: str):
    async with aiohttp.ClientSession() as session:
        with logging_redirect_tqdm():
            minibatch_size = 100  # The wikipedia API only allows 100 requests per second, so we batch the requests.
            all_outputs = []
            for i in trange(
                0, len(articles), minibatch_size, desc="getting view stats"
            ):
                batch = articles[i : i + minibatch_size]
                ret = await asyncio.gather(
                    *[
                        get_total_views(article, session, end_date, language)
                        for article in batch
                    ]
                )
                if -1 in ret:
                    # wait for longer if we get an rate limit error
                    await asyncio.sleep(5)
                all_outputs.extend(ret)
                await asyncio.sleep(1)

    return all_outputs


def get_most_and_least_viewed_articles(
    all_wiki_titles_intros,
    num_articles_to_search: int,
    num_articles_to_return: int,
    end_date: str,
    language: str,
):
    """
    num_articles: the number of articles to read from the collection. Due to API limitations, we cannot get all article views in a reasonable time.
    """
    assert num_articles_to_return * 2 <= num_articles_to_search

    all_wiki_titles = list(all_wiki_titles_intros.keys())[:num_articles_to_search]
    all_total_views = {}
    total_views = asyncio.run(
        batch_get_total_views(all_wiki_titles, end_date=end_date, language=language)
    )

    for i, title in enumerate(all_wiki_titles):
        all_total_views[title] = total_views[i]

    # ignore the articles we couldn't find
    all_total_views = [item for item in all_total_views.items() if item[1] > 0]
    # sort by total views
    all_total_views = sorted(all_total_views, key=lambda item: item[1], reverse=True)
    most_viewed_articles = {}
    for title, total_views in all_total_views[:num_articles_to_return]:
        most_viewed_articles[title] = all_wiki_titles_intros[title]

    least_viewed_articles = {}
    for title, total_views in all_total_views[-num_articles_to_return:]:
        least_viewed_articles[title] = all_wiki_titles_intros[title]

    return most_viewed_articles, least_viewed_articles


def get_hotpotqa_articles(all_wiki_titles_intros, num_articles_to_return: int):
    dataset = load_dataset("hotpot_qa", "fullwiki")["validation"]
    count = 0
    ret = []
    for i in range(len(dataset)):
        if count == num_articles_to_return:
            break
        example = dataset[i]
        if example["type"] == "comparison":
            question = example["question"]
            wiki_articles = list(set(example["supporting_facts"]["title"]))
            assert len(wiki_articles) == 2
            if (
                wiki_articles[0] in all_wiki_titles_intros
                and wiki_articles[1] in all_wiki_titles_intros
            ):
                ret.append(
                    {
                        "title_1": wiki_articles[0],
                        "paragraph_1": all_wiki_titles_intros[wiki_articles[0]],
                        "title_2": wiki_articles[1],
                        "paragraph_2": all_wiki_titles_intros[wiki_articles[1]],
                        "hotpot_question": question,
                    }
                )
                count += 1
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recent_output_file",
        type=str,
        required=True,
        help="Where to write the recent articles",
    )
    parser.add_argument(
        "--head_output_file",
        type=str,
        required=True,
        help="Where to write the most viewed articles",
    )
    parser.add_argument(
        "--tail_output_file",
        type=str,
        required=True,
        help="Where to write the least viewed articles",
    )
    parser.add_argument(
        "--multihop_output_file",
        type=str,
        required=True,
        help="Where to write the multihop articles",
    )
    parser.add_argument(
        "--collection_path",
        type=str,
        required=True,
        help="The .jsonl files containing wikipedia chunks.",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="The .jsonl file containing wikipedia chunks.",
    )

    args = parser.parse_args()

    all_wiki_titles_intros = load_collection(args.collection_path)
    # hotpotqa_articles = get_hotpotqa_articles(
    # all_wiki_titles_intros, num_articles_to_return=1000
    # )
    # with open(args.multihop_output_file, "w") as f:
    # json.dump(hotpotqa_articles, f, indent=2, ensure_ascii=False)

    recent_articles = get_most_edited_wikipedia_articles(
        args.language, all_wiki_titles_intros
    )
    with open(args.recent_output_file, "w") as f:
        json.dump(recent_articles, f, indent=2, ensure_ascii=False)

    (
        most_viewed_articles,
        least_viewed_articles,
    ) = get_most_and_least_viewed_articles(
        all_wiki_titles_intros,
        num_articles_to_search=1000000,  # ~6.7M articles in Wikipedia
        num_articles_to_return=5000,
        end_date="20221231",  # end of 2022
        language=args.language,
    )

    with open(args.head_output_file, "w") as f:
        json.dump(most_viewed_articles, f, indent=2, ensure_ascii=False)
    with open(args.tail_output_file, "w") as f:
        json.dump(least_viewed_articles, f, indent=2, ensure_ascii=False)
