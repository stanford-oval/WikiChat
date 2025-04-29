import argparse
import os
import sys

from huggingface_hub import upload_file, upload_folder
from tqdm import tqdm

sys.path.insert(0, "./")
from utils.logging import logger


repo_id = "stanford-oval/wikipedia"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload preprocessed Wikipedia collection files to HuggingFace Hub"
    )
    parser.add_argument(
        "--dates",
        nargs="*",
        default=[],
        help="List of Wikipedia dump dates (e.g., 20240401). If not provided, the list will be empty.",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=[],
        help="List of language codes (e.g., de en fr). If not provided, the list will be empty. Use `all` to upload all languages.",
    )
    parser.add_argument("--workdir", default="workdir", help="Working directory")

    args = parser.parse_args()

    if args.languages == ["all"]:
        args.languages = [
            lang
            for lang in os.listdir(f"{args.workdir}/wikipedia/{args.dates[0]}")
            if os.path.isdir(f"{args.workdir}/wikipedia/{args.dates[0]}/{lang}")
        ]

    if args.dates and args.languages:
        for date, language in tqdm(
            [(date, lang) for date in args.dates for lang in args.languages],
            desc="Uploading",
        ):
            logger.info(f"Uploading {date}/{language} to {repo_id}")
            for file in [
                "collection.parquet",
                "collection_histogram.png",
            ]:
                upload_file(
                    path_or_fileobj=f"{args.workdir}/wikipedia/{date}/{language}/{file}",
                    path_in_repo=f"{date}/{language}/{file}",
                    repo_id=repo_id,
                    repo_type="dataset",
                )

    # Upload the Wikidata translation map
    wikidata_translation_path = os.path.join(
        args.workdir, "wikipedia", "wikidata_translation_map"
    )
    if os.path.exists(wikidata_translation_path):
        upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=wikidata_translation_path,
            path_in_repo="wikidata_translation_map",
        )
