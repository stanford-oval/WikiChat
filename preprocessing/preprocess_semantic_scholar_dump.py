from collections import defaultdict
import itertools
import json
import pathlib

from langchain_text_splitters import RecursiveCharacterTextSplitter
import orjsonl
import sys
import argparse
import glob
import os
from multiprocessing import cpu_count, Pool

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


sys.path.insert(0, "./")
from preprocessing.utils import draw_and_save_histogram_log_bins, num_tokens
from preprocessing.block import Block
from pipelines.utils import get_logger

logger = get_logger(__name__)


text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
    chunk_size=500,
    chunk_overlap=0,
    length_function=num_tokens,
    is_separator_regex=False,
)


def is_in_subset(article: dict, subset: list[str]) -> bool:
    if not article["externalids"]:
        return False

    for s in subset:
        if s in article["externalids"] and article["externalids"][s]:
            return True

    return False


def get_section_boundaries(section_headers, article_length):
    section_boundaries = []

    for i in range(len(section_headers) - 1):
        section_boundaries.append(
            (int(section_headers[i]["end"]), int(section_headers[i + 1]["start"]))
        )
    section_boundaries.append((int(section_headers[-1]["end"]), article_length))
    return section_boundaries


def which_section(start, end, section_boundaries):
    for idx, boundary in enumerate(section_boundaries):
        if start >= boundary[0] and end <= boundary[1]:
            return idx
    # print("start, end: ", start, end)
    # print("section_boundaries = ", section_boundaries)
    return None


def extract_article_blocks(article: dict):
    # TODO extract publication date as well
    section_title_texts = defaultdict(list)

    # assert (
    #     article["content"]["source"]["pdfurls"] is None
    #     or len(article["content"]["source"]["pdfurls"]) >= 1
    # )
    article_text = article["content"]["text"]
    titles = set()
    if not article["content"]["annotations"]["title"]:
        logger.warning("Skipping article because it has no title")
        return []
    if not article["content"]["annotations"]["sectionheader"]:
        logger.warning("Skipping article because it has no section headers")
        return []
    if not article["content"]["annotations"]["paragraph"]:
        logger.warning("Skipping article because it has no paragraphs")
        return []
    title = json.loads(article["content"]["annotations"]["title"])
    for t in title:
        start, end = int(t["start"]), int(
            t["end"]
        )  # occasionally, some of them are string instead of int. So we convert them here
        titles.add(article_text[start:end])
    if len(titles) != 1:
        logger.warning("Found %d titles: %s", len(titles), str(titles))
        # when we have multiple titles, the second one is often just author affiliation misclassified as title
        # so we should still select the first title
    article_title = list(titles)[0]

    if article["content"]["annotations"]["abstract"]:
        # Some articles do not have abstracts, for example because they are just the supplementary material to another article
        abstract = json.loads(article["content"]["annotations"]["abstract"])
        abstract_boundaries = set()
        for a in abstract:
            start, end = int(a["start"]), int(a["end"])  # TODO extract method
            abstract_boundaries.add((start, end))

        # assert len(abstract_boundaries) == 1, abstract
        start, end = list(abstract_boundaries)[0]
        abstract_text = article_text[start:end]
        section_title_texts["Abstract"].append(abstract_text)
        # print(abstract_text)

    section_headers = json.loads(article["content"]["annotations"]["sectionheader"])
    section_titles = []
    for header in section_headers:
        # print(section)
        start, end = int(header["start"]), int(header["end"])
        section_title = article_text[start:end]
        if "attributes" in header and "n" in header["attributes"]:
            section_number = header["attributes"]["n"]
            section_title = f"{section_number}. " + section_title
        section_titles.append(section_title)

    # print("section_headers = ", section_headers)
    section_boundaries = get_section_boundaries(section_headers, len(article_text))
    # print("section_boundaries = ", section_boundaries)
    paragraphs = json.loads(article["content"]["annotations"]["paragraph"])
    paragraphs_without_section = []
    for paragraph in paragraphs:
        start, end = int(paragraph["start"]), int(paragraph["end"])
        paragraph_text = article_text[start:end]
        section_idx = which_section(start, end, section_boundaries)
        if not section_idx:
            paragraphs_without_section.append((start, end, paragraph_text))
        else:
            section_title_texts[section_titles[section_idx]].append(paragraph_text)

    # Merge paragraphs without section that have consecutive start and end
    merged_paragraphs_without_section = []
    for i, (start, end, text) in enumerate(paragraphs_without_section):
        if (
            i == 0 or start != merged_paragraphs_without_section[-1][1]
        ):  # Not consecutive
            merged_paragraphs_without_section.append([start, end, text])
        else:  # Consecutive, merge with previous
            merged_paragraphs_without_section[-1][1] = end
            merged_paragraphs_without_section[-1][2] += "\n\n" + text

    _id = article["corpusid"]
    article_blocks = []
    assert isinstance(_id, int)
    for section_title, section_text in section_title_texts.items():
        section_title_texts[section_title] = "\n\n".join(section_text)
        for s in text_splitter.split_text(section_title_texts[section_title]):

            block = Block(
                content_string=s,
                article_title=article_title.title(),
                full_section_title=f"{article_title.title()} > {section_title.title()}",
                block_type="text",
                language="en",
            )
            block.num_tokens = num_tokens(
                block.full_section_title + " " + block.content_string
            )
            article_blocks.append(block)
    for p in merged_paragraphs_without_section:
        for s in text_splitter.split_text(p[2]):
            block = Block(
                content_string=s,
                article_title=article_title.title(),
                full_section_title=f"{article_title.title()} > -",
                block_type="text",
                language="en",
            )
            block.num_tokens = num_tokens(
                block.full_section_title + " " + block.content_string
            )
            article_blocks.append(block)

    return article_blocks


def articles(files: str, max_articles: int, subset: list[str]):

    counter = 0
    all_articles = itertools.chain.from_iterable(orjsonl.stream(f) for f in files)
    for article in all_articles:
        if not is_in_subset(article, subset):
            continue
        yield article
        counter += 1
        if counter == max_articles:
            break


def process_file(_input):
    file, max_articles, subset, counter_start = _input
    all_blocks = []
    counter = counter_start
    for article in articles([file], max_articles, subset):
        article_blocks = extract_article_blocks(article)
        for block in article_blocks:
            all_blocks.append(block.to_json(counter))
            counter += 1
    return all_blocks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input .jsonl.gz files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The output jsonl or json.gz file to write the blocks to.",
    )
    parser.add_argument("--num_workers", type=int, default=max(1, cpu_count() - 4))
    parser.add_argument(
        "--subset",
        type=str,
        nargs="+",
        choices=[
            "arxiv",
            "acl",
            "pubmed",
            "pubmedcentral",
            "dblp",
            "mag",  # MAG is Microsoft Academic Graph, which was discontinued in 2021
        ],
        required=False,
        default=None,
        help="Subset(s) of data to process. Can be one or more of the listed choices. If not provided, will process all available data.",
    )
    parser.add_argument(
        "--max_articles",
        type=int,
        default=-1,
        help="Will stop after processing this many articles. -1 means no limit. Used for testing.",
    )

    args = parser.parse_args()

    input_files = sorted(
        glob.glob(os.path.join(args.input_dir, "semantic_scholar_dump_*.jsonl.gz"))
    )
    if not input_files:
        logger.error(f"No matching files found in {args.input_dir}")
        sys.exit(1)

    num_workers = min(args.num_workers, len(input_files))
    tasks = [
        (file, args.max_articles, args.subset, i * 2000000)
        for i, file in enumerate(input_files)
    ]

    if os.path.exists(args.output_path):
        raise FileExistsError(
            f"Output file '{args.output_path}' already exists. Please choose a different output path or remove the existing file."
        )
    logger.info("Saving the collection to %s", args.output_path)
    # make parent directories
    pathlib.Path(os.path.dirname(args.output_path)).mkdir(parents=True, exist_ok=True)

    with logging_redirect_tqdm():
        with Pool(processes=num_workers) as pool:
            for blocks in tqdm(
                pool.imap_unordered(process_file, tasks),
                total=len(tasks),
                desc="Files",
                position=0,
                leave=True,
            ):
                orjsonl.extend(args.output_path, blocks)

    # save the collection size
    count = 0
    num_tokens_list = []
    for block in tqdm(orjsonl.stream(args.output_path), desc="Calculating output size"):
        count += 1
        num_tokens_list.append(block["num_tokens"])
    with open(
        os.path.join(os.path.dirname(args.output_path), "collection_size.txt"), "w"
    ) as f:
        f.write(str(count))

    logger.info("Total number of blocks: {:,d}".format(count))

    # print_histogram([len(block.content_string) for block in all_blocks])
    draw_and_save_histogram_log_bins(
        num_tokens_list,
        args.output_path.rsplit(".", 1)[0] + "_histogram.png",
    )
