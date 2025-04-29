import argparse
import os

import orjsonl
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rich import print
from tqdm import tqdm


def convert_jsonl_to_parquet(input_file, chunk_size=10000):
    if input_file.endswith(".jsonl.gz"):
        output_file = input_file.replace(".jsonl.gz", ".parquet")
    else:
        output_file = input_file.replace(".jsonl", ".parquet")

    # Initialize an empty list to store the data
    data = []
    parquet_writer = None

    # Process the JSONL file in chunks
    for i, row in enumerate(
        tqdm(
            orjsonl.stream(input_file),
            desc="Processing JSONL",
            miniters=1e-6,
            unit_scale=1,
            unit=" Rows",
            smoothing=0,
        )
    ):
        data.append(row)

        # Once we have enough rows, convert to DataFrame and write to Parquet
        if (i + 1) % chunk_size == 0:
            df = pd.DataFrame(data)
            table = pa.Table.from_pandas(df)

            if parquet_writer is None:
                # Create a new Parquet file and write the first chunk
                parquet_writer = pq.ParquetWriter(output_file, table.schema)
            parquet_writer.write_table(table)

            # Clear the data list to free up memory
            data = []

    # Write any remaining data that didn't fill up a full chunk
    if data:
        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df)
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(output_file, table.schema)
        parquet_writer.write_table(table)

    # Close the Parquet writer
    if parquet_writer is not None:
        parquet_writer.close()

    print(f"Conversion complete. Parquet file saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert JSONL files to Parquet format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing JSONL files in its subdirectories",
    )
    args = parser.parse_args()

    # Get all JSONL files in the input directory
    jsonl_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".jsonl") or file.endswith(".jsonl.gz"):
                jsonl_files.append(os.path.join(root, file))
    print(f"Found {len(jsonl_files)} JSONL files in {args.input_dir}:")
    print(jsonl_files)

    for input_file in tqdm(jsonl_files, desc="Converting JSONL to Parquet"):
        convert_jsonl_to_parquet(input_file)
