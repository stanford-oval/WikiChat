import argparse
import sys

import orjsonl
from tqdm import tqdm

sys.path.insert(0, ".")
from preprocessing.block import Block

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert old JSONL collection to new format"
    )
    parser.add_argument(
        "--collection_path", type=str, required=True, help="Path to the collection file"
    )
    args = parser.parse_args()

    if not args.collection_path.endswith(".jsonl"):
        raise ValueError("Collection file should be in JSONL format")

    new_rows = []
    for row in tqdm(orjsonl.load(args.collection_path)):
        new_row = {}
        if "id" in row:
            del row["id"]
        if (
            "last_edit_date" in row
            and row["last_edit_date"]
            and ":24" in row["last_edit_date"]
        ):
            # the format is MM:DD:YY like 10:4:24. Convert it to YYYY-MM-DD
            month, day, year = row["last_edit_date"].split(":")
            year = "20" + year
            row["last_edit_date"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        # Convert extra fields to block_metadata
        extra_keys = []
        for key in row.keys():
            if key not in [
                "last_edit_date",
                "content",
                "title",
                "document_title",
                "full_section_title",
                "section_title",
                "url",
            ]:
                extra_keys.append(key)

        print("Found extra keys", extra_keys)
        if "block_metadata" not in row:
            row["block_metadata"] = {}
        for key in extra_keys:
            row["block_metadata"][key] = row[key]
            del row[key]

        block = Block(**row)
        new_rows.append(block.model_dump())

    orjsonl.save(args.collection_path.replace(".jsonl", "_new.jsonl"), new_rows)
