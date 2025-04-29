import argparse
import os
import sys

import pyarrow.parquet as pq
import rich
from rich.panel import Panel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.block import Block


def stream_parquet_file(file_path, N):
    parquet_file = pq.ParquetFile(file_path)
    count = 0

    # Iterate over row groups
    for row_group in range(parquet_file.num_row_groups):
        # Read the row group as a table
        table = parquet_file.read_row_group(row_group)

        # Convert the table to batches and iterate over rows
        for batch in table.to_batches():
            # Convert the batch to a dictionary of columns
            batch_dict = batch.to_pydict()
            num_rows = len(
                next(iter(batch_dict.values()))
            )  # Get the number of rows in the batch

            # Iterate over each row by index
            for i in range(num_rows):
                if count < N:
                    # Construct a row dictionary where each key is a column name and value is the row's value
                    row = {col: batch_dict[col][i] for col in batch_dict}
                    yield row
                    count += 1
                else:
                    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection_path",
        type=str,
        required=True,
        help="Path to the collection Parquet file",
    )
    args = parser.parse_args()

    collection_path = args.collection_path

    for row in stream_parquet_file(collection_path, 10):
        block = Block(**row)
        panel = Panel(f"[link={block.url}]{block.full_title}[/link]\n\n{block.content}")
        rich.print(panel)
