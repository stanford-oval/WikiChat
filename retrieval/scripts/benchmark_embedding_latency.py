import sys
from time import time

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from retrieval_commons import embedding_model_to_parameters
from tqdm import trange
from transformers import AutoTokenizer

sys.path.insert(0, "./")
import argparse

from utils.logging import logger


def embed_queries(queries: list[str]):
    queries = [query_template(t) for t in queries]
    onnx_config = embedding_model_to_parameters[embedding_model_name]["onnx_config"]

    start_time = time()
    inputs = embedding_tokenizer(
        queries, padding=True, truncation=True, return_tensors="np"
    )
    inputs_onnx = {k: ort.OrtValue.ortvalue_from_numpy(v) for k, v in inputs.items()}
    embeddings = ort_session.run(None, inputs_onnx)[
        onnx_config["embedding_index_in_output"]
    ]
    latency = time() - start_time
    embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )  # do this calculation because the actual retriever has to do this as well

    return latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark embedding latency.")
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        required=True,
        help="The name of the embedding model to use.",
    )
    args = parser.parse_args()

    embedding_model_name = args.embedding_model_name

    query_template = embedding_model_to_parameters[embedding_model_name][
        "query_template"
    ]
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    if "onnx_config" not in embedding_model_to_parameters[embedding_model_name]:
        raise ValueError(f"ONNX has not been tested for '{embedding_model_name}'.")
    onnx_config = embedding_model_to_parameters[embedding_model_name]["onnx_config"]

    # Download the onnx model files
    onnx_repo_id = onnx_config["onnx_model_repo"]
    onnx_model_path = hf_hub_download(
        repo_id=onnx_repo_id, filename=onnx_config["onnx_model_file"]
    )

    logger.info(f"Using ONNX model: '{onnx_config['onnx_model_file']}'")
    ort_session = ort.InferenceSession(onnx_model_path)

    total_latency = 0
    num_iterations = 100
    test_queries = [
        "Hello, how are you?",
        "Tell me about Haruki Murakami's childhood in Japan.",
        "The capital of France is Paris, not Berlin, and not NYC.",
    ]
    for _ in trange(num_iterations):
        total_latency += embed_queries(test_queries)

    average_latency = total_latency / num_iterations
    logger.info(
        f"Latency when embedding {len(test_queries)} queries, averaged over {num_iterations} iterations: {average_latency:.2f} seconds"
    )
