import argparse
from huggingface_hub import HfApi
import gzip
import shutil
import os

from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload preprocessed Wikipedia collection files to HuggingFace Hub"
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        required=True,
        help="List of Wikipedia dump dates (e.g., 20240401)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        help="List of language codes (e.g., de en fr)",
    )

    args = parser.parse_args()

    api = HfApi()
    for date, language in tqdm(
        [(date, lang) for date in args.dates for lang in args.languages],
        desc="Uploading",
    ):
        # Extract the .gz file
        gz_file = f"workdir/{language}/wikipedia_{date}/collection.jsonl.gz"
        extracted_file = f"workdir/{language}/wikipedia_{date}/collection.jsonl"

        with gzip.open(gz_file, "rb") as f_in:
            with open(extracted_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Now the extracted file is ready for upload
        for file in ["collection.jsonl", "collection_histogram.png"]:
            api.upload_file(
                path_or_fileobj=f"workdir/{language}/wikipedia_{date}/{file}",
                path_in_repo=f"{date}/{language}/{file}",
                repo_id="stanford-oval/wikipedia",
                repo_type="dataset",
                run_as_future=True,
            )

    # Remove the extracted files
    for date, language in [
        (date, lang) for date in args.dates for lang in args.languages
    ]:
        extracted_file = f"workdir/{language}/wikipedia_{date}/collection.jsonl"
        os.remove(extracted_file)
