"""Default values"""

DEFAULT_NUM_GPUS = 1
DEFAULT_EMBEDDING_MODEL_PORT = (
    6338  # and 6339, etc. if there is more than one GPU available
)
DEFAULT_EMBEDDING_MODEL_URL = "http://localhost"
QDRANT_VERSION = "1.13.6"
TEI_VERSION = "1.7.0"
# See https://github.com/huggingface/text-embeddings-inference?tab=readme-ov-file#docker-images
TEI_DOCKER_CONTAINER_HARDWARE_ARCHITECTURE = ""  # e.g. "hopper" for H100 GPU, "turing" for T4 GPU, "cpu" for CPU (not recommended). Default is "" which means Ampere 80 architecture (A100, A30 etc. GPUs).
TGI_VERSION = "2.2.0"
DEFAULT_RETRIEVER_PORT = 5100
DEFAULT_RETRIEVER_RERANKER_ENGINE = "gpt-4o-mini"
DEFAULT_EMBEDDING_USE_ONNX = False
DEFAULT_EMBEDDING_MODEL_NAME = "Snowflake/snowflake-arctic-embed-l-v2.0"
DEFAULT_WORKDIR = "workdir"
DEFAULT_VECTORDB_COLLECTION_NAME = "wikipedia_20250320"
DEFAULT_VECTORDB_PORT = 6333
DEFAULT_VECTORDB_TYPE = "qdrant"

DEFAULT_BACKEND_PORT = 5001
DEFAULT_REDIS_PORT = 6379
DEFAULT_DISTILLED_MODEL_PORT = 5002  # 5002 is reserved for local models

CHATBOT_DEFAULT_CONFIG = {
    "engine": "gpt-4o",
    "do_refine": False,
    "corpus_id": DEFAULT_VECTORDB_COLLECTION_NAME,
    "retriever_endpoint": "https://search.genie.stanford.edu/" + DEFAULT_VECTORDB_COLLECTION_NAME,
    "do_reranking": True,
    "query_pre_reranking_num": 20,
    "query_post_reranking_num": 3,
    "claim_pre_reranking_num": 10,
    "claim_post_reranking_num": 2,
}
