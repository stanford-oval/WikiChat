import argparse
import asyncio
import json
import logging
import os
import random
import sys
from datetime import timedelta
from multiprocessing import Process, SimpleQueue
from time import time

import orjsonl
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    Datatype,
    Distance,
    HnswConfigDiff,
    OptimizersConfig,
    PayloadSchemaType,
    VectorParams,
)
from tqdm import tqdm


sys.path.insert(0, "./")
from tasks.defaults import DEFAULT_QDRANT_COLLECTION_NAME
from pipelines.utils import get_logger
from retrieval.qdrant_index import QdrantIndex

logger = get_logger(__name__)

model_port = []


def embed_tei(text: str | list, is_query: bool, query_prefix: str = "") -> list:
    if not isinstance(text, list):
        text = [text]
    if is_query:
        text = [query_prefix + t for t in text]
    selected_port = random.choices(model_port)[0]  # load balancing
    resp = requests.post(
        f"http://0.0.0.0:{selected_port}/embed",
        json={"inputs": text, "normalize": True, "truncate": True},
        timeout=300,  # seconds
    )
    if resp.status_code == 413:
        # the batch has been too long for HTTP
        logging.warning(
            "Skipping batch because it was too large for HTTP. Consider decreasing --embedding_batch_size"
        )
        return None
    if resp.status_code != 200:
        logging.warning(
            "Skipping batch because the embedding server returned error code %s",
            str(resp.status_code),
        )
        return None
    embedding = resp.json()
    return embedding


def index_batch(input_queue: SimpleQueue, output_queue: SimpleQueue):
    while True:
        item = input_queue.get()
        if item is None:
            break
        batch_blocks = item

        batch_text = [
            "Title: "
            + block["full_section_title"]
            + ". "
            + block["content_string"].strip()
            for block in batch_blocks
        ]
        batch_embeddings = embed_tei(
            list(batch_text), is_query=False
        )  # Ensure list is passed for batch processing

        if batch_embeddings is not None:
            output_queue.put((batch_blocks, batch_embeddings))

    output_queue.put(None)


def commit_to_index(
    num_workers: int,
    input_queue: SimpleQueue,
    collection_name: str,
    collection_size: int,
):
    pbar = tqdm(
        desc="Indexing collection",
        miniters=1e-6,
        unit_scale=1,
        unit=" Block",
        dynamic_ncols=True,
        smoothing=0,
        total=collection_size,
    )

    qdrant_client = QdrantClient(url="http://localhost", timeout=60, prefer_grpc=True)
    if not qdrant_client.collection_exists(collection_name=collection_name):
        logger.info(
            "Did not find collection %s in Qdrant, creating it...",
            collection_name,
        )
        result = qdrant_client.create_collection(
            collection_name=collection_name,
            shard_number=2,
            on_disk_payload=True,
            vectors_config=VectorParams(
                size=embedding_size,
                distance=Distance.DOT,
                on_disk=True,
                datatype=Datatype.FLOAT16,
            ),
            # optimizers_config=OptimizersConfig(
            #     indexing_threshold=10
            # ),
            hnsw_config=HnswConfigDiff(
                m=64, ef_construct=100, full_scan_threshold=10, on_disk=True
            ),
            quantization_config=BinaryQuantization(
                binary=BinaryQuantizationConfig(
                    always_ram=True,
                ),
            ),
        )
        if result:
            logger.info("Collection creation was successful")
        else:
            raise RuntimeError("Could not create the collection in Qdrant.")

    finished_workers = 0
    while True:
        item = input_queue.get()
        if item is None:
            finished_workers += 1
            if finished_workers == num_workers:
                break
            continue
        batch_blocks, batch_embeddings = item
        qdrant_client.upload_collection(
            collection_name=collection_name,
            vectors=batch_embeddings,
            payload=[
                {
                    "text": block["content_string"],
                    "title": block["article_title"],
                    "full_section_title": block["full_section_title"],
                    "language": block["language"],
                    "block_type": block["block_type"],
                    "last_edit_date": block["last_edit_date"],
                }
                for block in batch_blocks
            ],
            ids=[
                abs(
                    hash(
                        block["content_string"]
                        + " "
                        + block["full_section_title"]
                        + block["language"],
                    )
                )
                for block in batch_blocks
            ],
            max_retries=3,
            wait=True,
        )

        pbar.update(len(batch_blocks))

    # index these payload fields, so that we can filter searched based on them
    # logger.info("Indexing payload fields.")
    # qdrant_client.create_payload_index(
    #     collection_name=args.collection_name,
    #     field_name="language",
    #     field_schema=PayloadSchemaType.KEYWORD,
    #     wait=False,
    # )
    # qdrant_client.create_payload_index(
    #     collection_name=args.collection_name,
    #     field_name="block_type",
    #     field_schema=PayloadSchemaType.KEYWORD,
    #     wait=False,
    # )
    # qdrant_client.create_payload_index(
    #     collection_name=args.collection_name,
    #     field_name="title",
    #     field_schema=PayloadSchemaType.KEYWORD,
    #     wait=False,
    # )

    qdrant_client.close()


def batch_generator(collection_file, embedding_batch_size):
    """Generator function to yield batches of data from the queue."""
    batch = []
    for block in orjsonl.stream(collection_file, compression_format="gz"):
        batch.append(block)
        if len(batch) == embedding_batch_size:
            yield batch
            batch = []

    # yield the last partial batch
    if len(batch) > 0:
        yield batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection_file", type=str, default=None, help=".jsonl file to read from."
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        choices=QdrantIndex.get_supported_embedding_models(),
        default="BAAI/bge-m3",
    )
    parser.add_argument(
        "--model_port",
        type=int,
        nargs="+",
        default=None,
        help="The port(s) to which the embedding model is accessible. In multi-GPU settings, you can run the embedding server on different GPUs and ports.",
    )
    parser.add_argument(
        "--num_embedding_workers",
        type=int,
        default=10,
        help="The number of processes that send embedding requests to GPU. Using too few will underutilize the GPU, and using too many will add overhead.",
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=48,
        help="The size of each request sent to GPU. The actual batch size is `embedding_batch_size * num_embedding_workers`",
    )
    parser.add_argument("--collection_name", default=DEFAULT_QDRANT_COLLECTION_NAME, type=str)
    parser.add_argument(
        "--index",
        action="store_true",
        help="If set, will index the provided `--collection_file`.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="If set, will run a test query on the provided index.",
    )

    args = parser.parse_args()

    model_port = args.model_port
    embedding_size = QdrantIndex.get_embedding_model_parameters(args.embedding_model)[
        "embedding_dimension"
    ]
    query_prefix = QdrantIndex.get_embedding_model_parameters(args.embedding_model)[
        "query_prefix"
    ]

    if args.index:
        collection_size = 0
        size_file = os.path.join(
            os.path.dirname(args.collection_file), "collection_size.txt"
        )
        try:
            with open(size_file) as f:
                collection_size = int(f.read().strip())
        except Exception as e:
            logger.warning(
                "Could not read the collection size from %s, defaulting to zero.",
                size_file,
            )

        input_queue = SimpleQueue()
        vector_queue = SimpleQueue()
        start_time = time()

        qdrant_worker = Process(
            target=commit_to_index,
            args=(
                args.num_embedding_workers,
                vector_queue,
                args.collection_name,
                collection_size,
            ),
        )
        all_workers = [qdrant_worker]

        for _ in range(args.num_embedding_workers):
            p = Process(target=index_batch, args=(input_queue, vector_queue))
            all_workers.append(p)

        for p in all_workers:
            p.start()

        # main process reads and feeds the collection to workers
        batches = batch_generator(args.collection_file, args.embedding_batch_size)
        for batch in batches:
            input_queue.put(batch)
        for _ in range(args.num_embedding_workers):
            input_queue.put(None)

        for p in all_workers:
            p.join()

        end_time = time()
        logger.info(
            "Indexing took %s", str(timedelta(seconds=int(end_time - start_time)))
        )

    if args.test:
        # Retrieve a test query
        logger.info("Testing the index")
        queries = [
            "Tell me about Haruki Murakami",
            "Who is the current monarch of the UK?",
        ]

        with QdrantIndex(
            args.embedding_model, args.collection_name, use_onnx=True
        ) as index:
            results = asyncio.run(index.search(queries, 5))
            logger.info(json.dumps(results, indent=2, ensure_ascii=False))
