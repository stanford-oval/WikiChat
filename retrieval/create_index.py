import argparse
import asyncio
import logging
import os
import random
import sys
from datetime import timedelta
from multiprocessing import Process, SimpleQueue
from time import time
from typing import Callable

import orjsonl
import requests
from tqdm import tqdm


sys.path.insert(0, "./")
import pyarrow.parquet as pq
from qdrant_client.models import PayloadSchemaType

from utils.logging import logger
from preprocessing.block import Block
from retrieval.embedding_model_info import (
    get_embedding_model_parameters,
    get_supported_embedding_models,
)
from retrieval.qdrant_index import AsyncQdrantVectorDB
import math
import numpy as np


embedding_request_headers = {"Content-Type": "application/json"}
embedding_api_key = os.environ.get("EMBEDDING_API_KEY", None)
if embedding_api_key:
    embedding_request_headers["Authorization"] = f"Bearer {embedding_api_key}"
    logger.info("Loaded API key for embedding requests")


def embed_tei(
    text: str | list[str],
    is_query: bool,
    query_template: Callable[[str], str],
    embedding_model_url: str,
    embedding_model_port: list[int],
    matryoshka: bool,
) -> list[list[float]] | None:
    if not isinstance(text, list):
        text = [text]
    if is_query:
        text = [query_template(t) for t in text]
    selected_port = random.choice(embedding_model_port)  # load balancing
    try:
        resp = requests.post(
            f"{embedding_model_url}:{selected_port}/embed",
            json={"inputs": text, "normalize": True, "truncate": True},
            timeout=300,  # seconds
            headers=embedding_request_headers,
        )
    except requests.exceptions.RequestException as e:
        logging.warning(
            f"Skipping batch because the embedding server returned error {str(e)}"
        )
        return None

    if resp.status_code == 413:
        # the batch has been too long for HTTP
        logging.warning(
            "Skipping batch because it was too large for HTTP. Consider decreasing --embedding_batch_size"
        )
        return None
    if resp.status_code != 200:
        logging.warning(
            f"Skipping batch because the embedding server returned error code {resp.status_code}: {resp.text}"
        )
        return None
    embedding = resp.json()

    if matryoshka:
        # truncate each embedding to 256 dimensions
        embedding = [e[:256] for e in embedding]
        # renormalize the embeddings
        embedding = np.array(embedding)
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        # Avoid division by zero by setting zero norms to one
        norms[norms == 0] = 1
        normalized_embeddings = (embedding / norms).tolist()
        for emb in embedding:
            norm = math.sqrt(sum(x * x for x in emb))
            if norm > 0:
                normalized_embeddings.append([x / norm for x in emb])
            else:
                normalized_embeddings.append(emb)
        embedding = normalized_embeddings
    return embedding


def embed_batch(
    input_queue: SimpleQueue,
    output_queue: SimpleQueue,
    query_template: Callable[[str], str],
    embedding_model_url: str,
    embedding_model_port: list[int],
    matryoshka: bool,
):
    while True:
        item = input_queue.get()
        if item is None:
            break
        batch_blocks = item

        batch_text = [
            f"Title: {block.full_title}\n{block.content}" for block in batch_blocks
        ]
        batch_embeddings = embed_tei(
            list(batch_text),
            is_query=False,
            query_template=query_template,
            embedding_model_url=embedding_model_url,
            embedding_model_port=embedding_model_port,
            matryoshka=matryoshka,
        )  # Ensure list is passed for batch processing

        if batch_embeddings is not None:
            output_queue.put((batch_blocks, batch_embeddings))

    output_queue.put(None)


vector_db = None


def write_to_vector_db(
    num_workers: int,
    input_queue: SimpleQueue,
    collection_size: int,
    num_skip: int,
    embedding_model_name: str,
    collection_name: str,
    high_memory: bool,
    index_metadata: bool,
):
    asyncio.run(
        async_write_to_vector_db(
            num_workers,
            input_queue,
            collection_size,
            num_skip,
            embedding_model_name,
            collection_name,
            high_memory,
            index_metadata,
        )
    )


async def async_write_to_vector_db(
    num_workers: int,
    input_queue: SimpleQueue,
    collection_size: int,
    num_skip: int,
    embedding_model_name: str,
    collection_name: str,
    high_memory: bool,
    index_metadata: bool,
):
    pbar = tqdm(
        desc="Indexing collection",
        miniters=1e-6,
        mininterval=0.5,
        unit_scale=1,
        unit=" Blocks",
        dynamic_ncols=True,
        smoothing=0,
        total=collection_size - num_skip,
    )
    vector_db_class = AsyncQdrantVectorDB
    vector_db = vector_db_class(
        vector_db_url="http://localhost",
        embedding_model_name=embedding_model_name,
        skip_loading_embedding_model=True,
    )
    await vector_db.create_collection_if_not_exists(
        collection_name,
        high_memory=high_memory,
    )

    finished_workers = 0
    metadata_fields = []
    metadata_types = []
    is_first_batch = True

    while True:
        item = input_queue.get()
        if item is None:
            finished_workers += 1
            if finished_workers == num_workers:
                break
            continue
        batch_blocks, batch_embeddings = item

        await vector_db.add_blocks(
            collection_name=collection_name,
            vectors=batch_embeddings,
            blocks=batch_blocks,
        )

        if is_first_batch:
            metadata_fields, metadata_types = batch_blocks[
                0
            ].get_metadata_fields_and_types()
            is_first_batch = False

        pbar.update(len(batch_blocks))

    if index_metadata:
        field_types = [PayloadSchemaType.KEYWORD, PayloadSchemaType.DATETIME]
        field_names = ["document_title", "last_edit_date"]

        for f_name, f_type in zip(metadata_fields, metadata_types):
            if f_type is str:
                field_types.append(PayloadSchemaType.KEYWORD)
            elif f_type is int:
                field_types.append(PayloadSchemaType.INTEGER)
            elif f_type is float:
                field_types.append(PayloadSchemaType.FLOAT)
            elif f_type is bool:
                field_types.append(PayloadSchemaType.BOOL)
            else:
                raise ValueError(
                    f"Unsupported type {f_type} for metadata field {f_name}"
                )

            field_names.append(f_name)

        logger.info(
            f"Creating metadata index for fields {field_names} with types {field_types}"
        )
        await vector_db.create_metadata_index(
            collection_name=collection_name,
            field_names=field_names,
            field_types=field_types,
        )

    await vector_db.close_connections()


logger = logging.getLogger(__name__)


def batch_generator(
    collection_file,
    embedding_batch_size,
    num_skip: int,
):
    """
    Generator function to yield batches of data from a JSONL file using `orjsonl` or a parquet file using `pyarrow`.
    """
    batch = []
    count = 0
    if collection_file.endswith(".jsonl") or collection_file.endswith(".jsonl.gz"):
        for row in orjsonl.stream(collection_file):
            try:
                block = Block(**row)
            except Exception as e:
                logger.error(f"Error creating Block: {e}. Skipping.")
                continue

            if count < num_skip:
                count += 1
                continue
            batch.append(block)

            if len(batch) == embedding_batch_size:
                yield batch
                batch = []

    elif collection_file.endswith(".parquet"):
        parquet_file = pq.ParquetFile(collection_file)

        for row_group in range(parquet_file.num_row_groups):
            # Read the entire row group as a PyArrow Table
            table = parquet_file.read_row_group(row_group)

            # Skip rows if necessary
            if count + table.num_rows <= num_skip:
                count += table.num_rows
                logger.info(f"Skipped {count:,} blocks")
                continue

            # Calculate the starting row index after skipping
            start_idx = max(0, num_skip - count)

            # Process rows in bulk
            for i in range(start_idx, table.num_rows):
                row = {col: table[col][i].as_py() for col in table.column_names}

                block = Block(**row)
                batch.append(block)

                if len(batch) == embedding_batch_size:
                    yield batch
                    batch = []

    # Yield the last partial batch
    if len(batch) > 0:
        yield batch


def create_index(
    collection_file,
    num_embedding_workers,
    embedding_batch_size,
    embedding_model_name,
    embedding_model_url,
    embedding_model_port: list[int],
    collection_name,
    vector_db_type,
    high_memory,
    num_skip,
    index_metadata: bool,
):
    embedding_model_params = get_embedding_model_parameters(embedding_model_name)
    query_template = embedding_model_params["query_template"]
    matryoshka = embedding_model_params.get("matryoshka", False)

    if collection_file.endswith(".parquet"):
        parquet_file = pq.ParquetFile(collection_file)
        collection_size = parquet_file.metadata.num_rows
    else:
        assert collection_file.endswith(".jsonl") or collection_file.endswith(
            ".jsonl.gz"
        )
        logger.warning(
            "For better performance, consider converting the file to Parquet before running this script."
        )
        collection_size = 0

    input_queue = SimpleQueue()
    vector_queue = SimpleQueue()
    start_time = time()

    batches = batch_generator(collection_file, embedding_batch_size, num_skip)

    if num_embedding_workers == 1:
        # Single worker mode, no multiprocessing, but keep queue logic intact for testing purposes

        # Simulate the producer-consumer pattern in the main process
        for batch in batches:
            logger.info("adding batch to queue")
            input_queue.put(batch)

        # Signal the end of input
        input_queue.put(None)

        logger.info("processing batch in main process")
        embed_batch(
            input_queue,
            vector_queue,
            query_template,
            embedding_model_url,
            embedding_model_port,
            matryoshka=matryoshka,
        )

        write_to_vector_db(
            num_embedding_workers,
            vector_queue,
            collection_size,
            num_skip,
            embedding_model_name,
            collection_name,
            high_memory,
            index_metadata,
        )

    else:
        # Multiprocessing mode
        vector_db_worker = Process(
            target=write_to_vector_db,
            args=(
                num_embedding_workers,
                vector_queue,
                collection_size,
                num_skip,
                embedding_model_name,
                collection_name,
                high_memory,
                index_metadata,
            ),
        )
        all_workers = [vector_db_worker]

        for _ in range(num_embedding_workers):
            p = Process(
                target=embed_batch,
                args=(
                    input_queue,
                    vector_queue,
                    query_template,
                    embedding_model_url,
                    embedding_model_port,
                    matryoshka,
                ),
            )
            all_workers.append(p)

        for p in all_workers:
            p.start()

        # main process reads and feeds the collection to workers
        for batch in batches:
            input_queue.put(batch)
        for _ in range(num_embedding_workers):
            input_queue.put(None)

        for p in all_workers:
            p.join()

    end_time = time()
    logger.info(f"Indexing took {str(timedelta(seconds=int(end_time - start_time)))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection_file", type=str, default=None, help="Parquet file to read from."
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        required=True,
        choices=get_supported_embedding_models(),
    )
    parser.add_argument(
        "--embedding_model_port",
        type=int,
        nargs="+",
        default=None,
        help="The port(s) to which the embedding model is accessible. In multi-GPU settings, you can run the embedding server on different GPUs and ports.",
    )
    parser.add_argument(
        "--embedding_model_url",
        type=str,
        default=None,
        help="The URL at which the embedding model is accessible will be embedding_model_url:embedding_model_port/embed",
    )
    parser.add_argument(
        "--num_embedding_workers",
        type=int,
        default=32,
        help="The number of processes that send embedding requests to GPU. Using too few will underutilize the GPU, and using too many will add overhead.",
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=32,
        help="The size of each request sent to GPU. The actual batch size is `embedding_batch_size * num_embedding_workers`",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        required=True,
        help="The name of the collection, e.g. wikipedia.",
    )
    parser.add_argument(
        "--vector_db_type",
        type=str,
        choices=["qdrant"],
        required=True,
        help="The type of vector database to use for indexing.",
    )
    parser.add_argument(
        "--high_memory",
        action="store_true",
        help="If set, will keep the index and vectors in RAM for faster search. Will use a lot more memory.",
    )
    parser.add_argument(
        "--num_skip",
        type=int,
        default=0,
        help="Number of blocks to skip in the collection file. Useful for resuming indexing.",
    )
    parser.add_argument(
        "--index_metadata",
        action="store_true",
        help="Index the metadata of the blocks.",
    )

    args = parser.parse_args()

    create_index(
        collection_file=args.collection_file,
        num_embedding_workers=args.num_embedding_workers,
        embedding_batch_size=args.embedding_batch_size,
        embedding_model_name=args.embedding_model_name,
        embedding_model_url=args.embedding_model_url,
        embedding_model_port=args.embedding_model_port,
        collection_name=args.collection_name,
        vector_db_type=args.vector_db_type,
        high_memory=args.high_memory,
        num_skip=args.num_skip,
        index_metadata=args.index_metadata,
    )
