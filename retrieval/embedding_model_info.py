# Find more models at https://huggingface.co/spaces/mteb/leaderboard
# model_size and sequence_length are not used in the code, but they are included in the parameters for reference
embedding_model_to_parameters = {
    # BAAI BGE models
    "BAAI/bge-m3": {
        "embedding_dimension": 1024,
        "query_template": lambda query: f"{query}",  # no prefix needed for this model
        "model_size": "567M",
        "max_sequence_length": 8192,
        "onnx_config": {
            "onnx_model_repo": "BAAI/bge-m3",  # The same repo has the ONNX model as well
            "onnx_model_file": "onnx/model.onnx",
            "onnx_data_file": "onnx/model.onnx_data",
            "embedding_index_in_output": 1,
        },
    },  # Supports more than 100 languages.
    "BAAI/bge-large-en-v1.5": {
        "embedding_dimension": 1024,
        "query_template": lambda query: f"Represent this sentence for searching relevant passages: {query}",
        "model_size": "335M",
        "max_sequence_length": 512,
    },
    "BAAI/bge-base-en-v1.5": {
        "embedding_dimension": 768,
        "query_template": lambda query: f"Represent this sentence for searching relevant passages: {query}",
        "model_size": "109M",
        "max_sequence_length": 512,
    },
    "BAAI/bge-small-en-v1.5": {
        "embedding_dimension": 384,
        "query_template": lambda query: f"Represent this sentence for searching relevant passages: {query}",
        "model_size": "33M",
        "max_sequence_length": 512,
        "onnx_config": {
            "onnx_model_repo": "BAAI/bge-small-en-v1.5",  # The same repo has the ONNX model as well
            "onnx_model_file": "onnx/model.onnx",
            "onnx_data_file": "onnx/model.onnx",
            "embedding_index_in_output": 0,
        },
    },
    "BAAI/bge-multilingual-gemma2": {
        "embedding_dimension": 3584,
        "query_template": lambda query: f"<instruct>Given a web search query, retrieve relevant passages that answer the query.\n<query>{query}'",
        "model_size": "9242M",
        "max_sequence_length": 8192,
    },
    # Alibaba GTE models
    "Alibaba-NLP/gte-base-en-v1.5": {
        "embedding_dimension": 768,
        "query_template": lambda query: f"{query}",
        "model_size": "137M",
        "max_sequence_length": 8192,
    },
    "Alibaba-NLP/gte-large-en-v1.5": {
        "embedding_dimension": 1024,
        "query_template": lambda query: f"{query}",
        "model_size": "434M",
        "max_sequence_length": 8192,
    },
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": {
        "embedding_dimension": 1536,
        "query_template": lambda query: f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {query}",
        "model_size": "1776M",
        "max_sequence_length": 32768,
    },
    "Alibaba-NLP/gte-Qwen2-7B-instruct": {
        "embedding_dimension": 3584,
        "query_template": lambda query: f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {query}",
        "model_size": "7000M",
        "max_sequence_length": 32768,
    },
    "Alibaba-NLP/gte-multilingual-base": {
        "embedding_dimension": 768,
        "query_template": lambda query: f"{query}",
        "model_size": "305M",
        "max_sequence_length": 8192,
        "onnx_config": {
            "onnx_model_repo": "onnx-community/gte-multilingual-base",
            "onnx_model_file": "onnx/model.onnx",  # depending on the CPU, other types might be faster
            "onnx_data_file": "onnx/model.onnx",
            "embedding_index_in_output": 1,
        },
    },  # Supports over 70 languages.
    "Snowflake/snowflake-arctic-embed-l-v2.0": {
        "embedding_dimension": 256,  # 1024, with matryoshka support for 256
        "matryoshka": True,
        "query_template": lambda query: f"query: {query}",
        "model_size": "303M",
        "max_sequence_length": 8192,
        "onnx_config": {
            "onnx_model_repo": "Snowflake/snowflake-arctic-embed-l-v2.0",
            "onnx_model_file": "onnx/model.onnx",  # depending on the CPU, other types might be faster
            "onnx_data_file": "onnx/model.onnx_data",
            "embedding_index_in_output": 1,
        },
    },
}


def get_embedding_model_parameters(embedding_model: str):
    return embedding_model_to_parameters[embedding_model]


def get_supported_embedding_models():
    return embedding_model_to_parameters.keys()
