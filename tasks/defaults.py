"""Default values"""

DEFAULT_NUM_GPUS = 1
DEFAULT_EMBEDDING_MODEL_PORT = (
    6338  # and 6339, etc. if there is more than one GPU available
)
QDRANT_VERSION = "1.11.0"
TEI_VERSION = "1.5.0"
TGI_VERSION = "2.2.0"
DEFAULT_RETRIEVER_PORT = 5100
DEFAULT_EMBEDDING_USE_ONNX = True
DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_WIKIPEDIA_DUMP_LANGUAGE = "en"
DEFAULT_WORKDIR = "workdir"
DEFAULT_QDRANT_COLLECTION_NAME="wikipedia"

DEFAULT_BACKEND_PORT = 5001
DEFAULT_REDIS_PORT = 6379
DEFAULT_DISTILLED_MODEL_PORT = 5002  # 5002 is reserved for local models

CHATBOT_DEFAULT_CONFIG = {
    "engine": "gpt-4o",
    "pipeline": "early_combine",
    "temperature": 0.0,
    "top_p": 0.9,
    "retriever_endpoint": "https://wikichat.genie.stanford.edu/search",
    "skip_verification": False,
    "skip_query": False,
    "fuse_claim_splitting": True,
    "generation_prompt": "generate_split_claims.prompt",
    "retrieval_num": 4,
    "do_refine": True,
    "refinement_prompt": "refine.prompt",
    "draft_prompt": "draft.prompt",
    "retrieval_reranking_method": "llm",
    "retrieval_reranking_num": 20,
    "evi_num": 2,
    "evidence_reranking_method": "none",
    "evidence_reranking_num": 2,
    "generate_engine": None,
    "draft_engine": None,
    "refine_engine": None,
}
