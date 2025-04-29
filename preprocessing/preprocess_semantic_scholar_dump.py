import argparse
import functools
import glob
import itertools
import json
import os
import pathlib
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import orjsonl
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

sys.path.insert(0, "./")
from utils.logging import logger
from preprocessing.block import Block
from preprocessing.utils import (
    draw_and_save_histogram_log_bins,
    get_num_tokens,
    pack_blocks,
)


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
    chunk_size=600,
    chunk_overlap=0,
    length_function=get_num_tokens,
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


def normalize_title(title: str) -> str:
    title = title.replace("\n", " ").strip()
    title = " ".join(title.split())
    title = title.title()
    for phrase in ["citation ; :", "citation:"]:
        if title.lower().startswith(phrase.lower()):
            title = title[len(phrase) :].strip()

    return title


def extract_article_blocks(article: dict) -> list[Block]:
    try:
        # TODO extract publication date as well
        section_title_texts = defaultdict(list)

        # assert (
        #     article["content"]["source"]["pdfurls"] is None
        #     or len(article["content"]["source"]["pdfurls"]) >= 1
        # )
        metadata = {}
        subsets = []
        doi = None
        if article["externalids"]:
            for k in ["arxiv", "acl", "pubmed", "pubmedcentral", "dblp", "mag"]:
                if k in article["externalids"] and article["externalids"][k]:
                    subsets.append(k)

        if (
            article["externalids"]
            and "doi" in article["externalids"]
            and article["externalids"]["doi"]
        ):
            doi = article["externalids"]["doi"]

        metadata["subsets"] = subsets
        metadata["doi"] = doi

        article_text = article["content"]["text"]

        if not article["content"]["annotations"]["title"]:
            logger.debug("Skipping article because it has no title")
            return []
        if not article["content"]["annotations"]["sectionheader"]:
            logger.debug("Skipping article because it has no section headers")
            return []
        if not article["content"]["annotations"]["paragraph"]:
            logger.debug("Skipping article because it has no paragraphs")
            return []
        title = json.loads(article["content"]["annotations"]["title"])
        # print(json.dumps(article, indent=2))
        url = ""
        if not article["content"]["source"]["pdfurls"]:
            if "arxiv" in article["externalids"] and article["externalids"]["arxiv"]:
                arxiv_id = article["externalids"]["arxiv"]
                url = f"https://arxiv.org/pdf/{arxiv_id}"
            elif (
                article["content"]["source"]["oainfo"]
                and "openaccessurl" in article["content"]["source"]["oainfo"]
                and article["content"]["source"]["oainfo"]["openaccessurl"]
            ):
                url = article["content"]["source"]["oainfo"]["openaccessurl"]
            elif (
                "pubmedcentral" in article["externalids"]
                and article["externalids"]["pubmedcentral"]
            ):
                pmc_id = article["externalids"]["pubmedcentral"]
                url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/pdf/"
            elif (
                "pubmed" in article["externalids"] and article["externalids"]["pubmed"]
            ):
                pm_id = article["externalids"]["pubmed"]
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pm_id}/"
            elif "acl" in article["externalids"] and article["externalids"]["acl"]:
                acl_id = article["externalids"]["acl"]
                url = f"https://aclanthology.org/{acl_id}.pdf"
        else:
            url = article["content"]["source"]["pdfurls"][0]

        if not url:
            logger.warning(
                f"Article has no URL: {json.dumps(article, indent=2, ensure_ascii=False)}"
            )
        else:
            url = url.replace(
                "export.arxiv.org", "arxiv.org"
            )  # export arxiv is a mirror, normalize everything to arxiv.org

        titles = set()
        for t in title:
            start, end = (
                int(t["start"]),
                int(t["end"]),
            )  # occasionally, some of them are string instead of int. So we convert them here
            titles.add(article_text[start:end])

        titles = [normalize_title(t) for t in titles]
        titles = [
            t for t in titles if t or "http:" in t.lower()
        ]  # remove empty titles, or titles that have URLs
        seen_titles = set()
        unique_titles = []
        for t in titles:
            if t not in seen_titles:
                seen_titles.add(t)
                unique_titles.append(t)
        titles = unique_titles
        if len(titles) == 2:
            if titles[1] in titles[0]:
                # sometimes the first title has extra stuff like the year, so we should just keep the second title
                titles = [titles[1]]

        if not titles:
            logger.debug("Skipping article because it has no title")
            return []
        if len(titles) != 1:
            logger.warning(f"Found {len(titles)} titles: {titles}")
            # when we have multiple titles, the second one is often just author affiliation misclassified as title
            # so we should still select the first title

        document_title = list(titles)[0]
        document_title = document_title.replace("\n", " ").strip()
        document_title = " ".join(document_title.split())
        document_title = document_title.title()

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
                    document_title=document_title,
                    section_title=section_title.title(),
                    content=s,
                    url=url,
                    block_metadata=metadata,
                )
                block.num_tokens = get_num_tokens(block.combined_text)
                article_blocks.append(block)
        for p in merged_paragraphs_without_section:
            for s in text_splitter.split_text(p[2]):
                block = Block(
                    document_title=document_title,
                    section_title="",
                    content=s,
                    url=url,
                    block_metadata=metadata,
                )
                block.num_tokens = get_num_tokens(block.combined_text)
                article_blocks.append(block)

        # pack smaller blocks into larger ones
        if args.pack_to_tokens > 0:
            article_blocks = pack_blocks(article_blocks, args.pack_to_tokens)

        return article_blocks
    except Exception as e:
        logger.exception(f"Error processing article: {e}")
        return []


def articles(files: str, max_articles: int, subset: list[str]):
    counter = 0
    all_articles = itertools.chain.from_iterable(orjsonl.stream(f) for f in files)
    for article in all_articles:
        if subset and not is_in_subset(article, subset):
            continue
        yield article
        counter += 1
        if counter == max_articles:
            break


def process_file(file, max_articles, subset: list[str]) -> tuple[Block, int, int]:
    """
    Process a file and extract article blocks.

    Returns:
        tuple: A tuple containing the list of all articles' blocks, the number of skipped articles,
               and the total number of articles processed.
    """
    all_blocks = []
    num_skipped_articles = 0
    num_articles = 0
    for article in articles([file], max_articles, subset):
        num_articles += 1
        article_blocks = extract_article_blocks(article)
        if not article_blocks:
            num_skipped_articles += 1
        for block in article_blocks:
            all_blocks.append(block.model_dump())
    return all_blocks, num_skipped_articles, num_articles


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
        help="The output .parquet file to write the blocks to.",
    )
    parser.add_argument("--num_workers", type=int, default=max(1, cpu_count() - 8))
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
        help="Will stop after processing this many articles per dump file. -1 means no limit. Used for testing.",
    )
    parser.add_argument(
        "--pack_to_tokens",
        type=int,
        default=0,
        help="If consecutive paragraphs in the same subsection are small, we greedily concatenate them together, while keeping the result shorter than this many tokens."
        " This helps reduce the number of vector embeddings when indexing. DEFAULT_EMBEDDING_MODEL_NAME tokenizer is used to determine token boundaries.",
    )

    args = parser.parse_args()
    if not args.output_path.endswith(".parquet"):
        raise ValueError("Output path must be a .parquet file.")

    input_files = sorted(
        glob.glob(os.path.join(args.input_dir, "semantic_scholar_dump_*.jsonl.gz"))
    )
    if not input_files:
        logger.error(f"No matching files found in {args.input_dir}")
        sys.exit(1)

    num_workers = min(args.num_workers, len(input_files))

    logger.info(f"Saving the collection to '{args.output_path}'")
    # make parent directories
    pathlib.Path(os.path.dirname(args.output_path)).mkdir(parents=True, exist_ok=True)

    num_total_skipped_articles = 0
    num_articles = 0
    num_duplicates = 0
    seen_titles_in_previous_files = set()
    num_tokens_list = []

    parquet_writer = None

    with logging_redirect_tqdm(loggers=[logger]):
        with Pool(processes=num_workers) as pool:
            for file_blocks, skipped_articles, total_articles in tqdm(
                pool.imap_unordered(
                    functools.partial(
                        process_file, max_articles=args.max_articles, subset=args.subset
                    ),
                    input_files,
                ),
                total=len(input_files),
                desc="Files",
                position=0,
                leave=True,
            ):
                seen_titles_in_this_files = set()
                num_total_skipped_articles += skipped_articles
                num_articles += total_articles
                logger.info(
                    f"So far, we have skipped {num_total_skipped_articles:,} articles, which is {num_total_skipped_articles / num_articles * 100:.1f}% of all articles ({num_articles:,})."
                )

                # Collect blocks for this iteration
                current_blocks = []

                # Deduplicate articles based on their titles
                for block in file_blocks:
                    seen_titles_in_this_files.add(block["document_title"])
                    if block["document_title"] in seen_titles_in_previous_files:
                        num_duplicates += 1
                        continue
                    # Append the block to the current list for this iteration
                    current_blocks.append(block)
                    num_tokens_list.append(block["num_tokens"])

                seen_titles_in_previous_files.update(seen_titles_in_this_files)

                # Convert the current blocks to a DataFrame
                if current_blocks:
                    df = pd.DataFrame(current_blocks)

                    # Convert the DataFrame to an Arrow Table
                    table = pa.Table.from_pandas(df)

                    # Write the table to the Parquet file in chunks
                    if parquet_writer is None:
                        # Open the Parquet file for the first time and write the schema
                        parquet_writer = pq.ParquetWriter(
                            args.output_path, table.schema
                        )

                    # Append the current chunk to the Parquet file
                    parquet_writer.write_table(table)

    # Close the Parquet writer after all chunks are written
    if parquet_writer:
        parquet_writer.close()

    logger.info(f"Removed {num_duplicates:,d} blocks with duplicate document titles.")
    # Read the Parquet file's metadata and print its size and number of rows
    parquet_file = pq.ParquetFile(args.output_path)
    metadata = parquet_file.metadata
    num_rows = metadata.num_rows

    logger.info(f"Parquet file number of rows: {num_rows:,d}")

    # print_histogram([len(block.content) for block in all_blocks])
    draw_and_save_histogram_log_bins(
        num_tokens_list,
        args.output_path,
    )
