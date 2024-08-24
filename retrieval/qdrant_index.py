import asyncio
import math
from time import time
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    QuantizationSearchParams,
    ScoredPoint,
    SearchParams,
    SearchRequest,
)
from transformers import AutoModel, AutoTokenizer

from pipelines.utils import get_logger

logger = get_logger(__name__)

embedding_model_to_parameters = {
    "BAAI/bge-m3": {
        "embedding_dimension": 1024,
        "query_prefix": "",
    },  # Supports more than 100 languages. no prefix needed for this model
    "BAAI/bge-large-en-v1.5": {
        "embedding_dimension": 1024,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "BAAI/bge-base-en-v1.5": {
        "embedding_dimension": 768,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "BAAI/bge-small-en-v1.5": {
        "embedding_dimension": 384,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": {
        "embedding_dimension": 1536,
        "query_prefix": "Given a web search query, retrieve relevant passages that answer the query",
    },
    "Alibaba-NLP/gte-multilingual-base": {
        "embedding_dimension": 768,
        "query_prefix": "",
    },  # Supports over 70 languages. Model Size: 305M
}


class SearchResult(BaseModel):
    score: list[float] = Field(
        default_factory=list,
        json_schema_extra={
            "example": [0.6902604699134827, 0.6895850896835327, 0.6812092661857605]
        },
    )
    text: list[str] = Field(
        default_factory=list,
        json_schema_extra={
            "example": [
                "GPT-4（ジーピーティーフォー、Generative Pre-trained Transformer 4）とは、OpenAI (in English: OpenAI)によって開発されたマルチモーダル (in English: Multimodal learning)（英語版）大規模言語モデル (in English: Large language model)である。2023年3月14日に公開された。自然言語処理 (in English: Natural language processing)にTransformer (機械学習モデル) (in English: Transformer)を採用しており、教師なし学習 (in English: Unsupervised learning)によって大規模なニューラルネットワーク (in English: Neural network)を学習させ、その後、人間のフィードバックからの強化学習 (in English: Reinforcement learning from human feedback)（RLHF）を行っている。",
                'Generative Pre-trained Transformer 4 (GPT-4) é um Modelo de linguagem grande (in English: Large language model) multimodal criado pela OpenAI (in English: OpenAI) e o quarto modelo da série GPT.  Foi lançado em 14 de março de 2023, e se tornou publicamente aberto de forma limitada por meio do ChatGPT (in English: ChatGPT) Plus, com o seu acesso à API comercial sendo provido por uma lista de espera. Sendo um Transformador (in English: Transformer), foi pré-treinado para prever o próximo Token (informática) (usando dados públicos e "licenciados de provedores terceirizados"), e então foi aperfeiçoado através de uma técnica de aprendizagem por reforço com humanos.',
                "GPT-4 es un modelo de Inteligencia artificial multimodal que puede generar texto a partir de diferentes tipos de entradas, como texto o imágenes. GPT-4 es un modelo multimodal porque puede Procesador de texto (in English: Word processor) y combinar diferentes modalidades de información, como el Lengua natural (in English: Natural language) y la Visión artificial (in English: Computer vision) Esto le da una ventaja sobre los modelos que solo pueden manejar una modalidad, ya que puede aprovechar el contexto y el conocimiento de múltiples fuentes.GPT-4 utiliza una técnica llamada fusión cruzada, que le permite integrar Información (in English: Information) de diferentes modalidades en una sola representación, lo que mejora su capacidad de Entendimiento (in English: Understanding) y generación",
            ]
        },
    )
    title: list[str] = Field(
        default_factory=list,
        json_schema_extra={
            "example": [
                "GPT-4",
                "GPT-4",
                "Inteligencia artificial multimodal (in English: Multimodal artificial intelligence)",
            ]
        },
    )
    full_section_title: list[str] = Field(
        default_factory=list,
        json_schema_extra={
            "example": [
                "GPT-4",
                "GPT-4",
                "Inteligencia artificial multimodal (in English: Multimodal artificial intelligence) > GPT-4",
            ]
        },
    )
    block_type: list[str] = Field(
        default_factory=list, json_schema_extra={"example": ["text", "text", "text"]}
    )
    language: list[str] = Field(
        default_factory=list, json_schema_extra={"example": ["ja", "pt", "es"]}
    )
    last_edit_date: list[str] = Field(
        default_factory=list,
        json_schema_extra={
            "example": [
                "2024-03-01T14:03:10Z",
                "2024-01-10T18:38:38Z",
                "2024-03-28T22:44:02Z",
            ]
        },
    )
    prob: list[float] = Field(
        default_factory=list,
        json_schema_extra={
            "example": [0.33441298757005206, 0.3341872079017531, 0.3313998045281948]
        },
    )


class QdrantIndex:
    query_prefix: str
    embedding_model: Any
    embedding_tokenizer: Any
    qdrant_client: AsyncQdrantClient
    collection_name: str
    use_onnx: bool

    def __init__(
        self,
        embedding_model_name: str,
        collection_name: str,
        qdrant_url: str = None,
        use_onnx: bool = False,
    ):
        self.qdrant_client = AsyncQdrantClient(
            url=qdrant_url, timeout=10, prefer_grpc=True
        )  # Lower timeout so that we don't get stuck on long retrieval jobs
        self.collection_name = collection_name
        self.use_onnx = use_onnx
        self.query_prefix = embedding_model_to_parameters[embedding_model_name][
            "query_prefix"
        ]
        self.search_params = SearchParams(
            indexed_only=True,
            hnsw_ef=200,
            exact=False,
            quantization=QuantizationSearchParams(
                ignore=False,
                rescore=True,
                oversampling=3.0,
            ),
        )

        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        if self.use_onnx:
            if embedding_model_name != "BAAI/bge-m3":
                raise ValueError("ONNX has only been tested for bge-m3.")

            # Download the .onnx model files
            onnx_repo_id = "aapot/bge-m3-onnx"
            onnx_model_path = hf_hub_download(
                repo_id=onnx_repo_id, filename="model.onnx"
            )
            hf_hub_download(repo_id=onnx_repo_id, filename="model.onnx.data")
            hf_hub_download(repo_id=onnx_repo_id, filename="ort_config.json")

            self.ort_session = ort.InferenceSession(onnx_model_path)
        else:
            self.embedding_model = AutoModel.from_pretrained(embedding_model_name)

        # self.embedding_model.eval()
        logger.info("Successfully loaded the embedding model " + embedding_model_name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup code here, for example closing the connection
        asyncio.run(self.qdrant_client.close())
        return False  # Return True if you handled an exception, False to propagate it

    async def search(
        self, queries: list[str] | str, k: int, filters: dict[str, list[str]] = None
    ) -> list[SearchResult]:
        """
        Returns a list of SearchResult, one SearchResult per query
        """
        was_list = True
        if isinstance(queries, str):
            queries = [queries]
            was_list = False
        query_embeddings = self.embed_queries(queries)

        start_time = time()

        batch_results = await self.qdrant_client.search_batch(
            collection_name=self.collection_name,
            requests=[
                SearchRequest(
                    vector=vector,
                    with_vector=False,
                    with_payload=True,
                    limit=k,
                    params=self.search_params,
                    filter=(
                        Filter(
                            must=[  # 'must' acts like AND, 'should' acts like OR
                                FieldCondition(
                                    key=key,
                                    match=MatchAny(any=list(value)),
                                )
                                for key, value in filters.items()
                            ]
                        )
                        if filters
                        else None
                    ),
                )
                for vector in query_embeddings
            ],
        )
        logger.info("Nearest neighbor search took %.2f seconds", (time() - start_time))

        ret = []
        for result in batch_results:
            # each iteration is for one query in the batch
            result_dict = QdrantIndex.search_result_to_pydantic(result)
            ret.append(result_dict)

        if was_list:
            assert len(ret) == len(queries)
        else:
            assert len(ret) == 1
        return ret

    def search_result_to_pydantic(search_result: list[ScoredPoint]) -> SearchResult:
        """
        Results for one query
        """
        ret = {
            "score": [],
            "text": [],
            "title": [],
            "full_section_title": [],
            "block_type": [],
            "language": [],
            "last_edit_date": [],
            # "vector":[]
        }
        for scored_point in search_result:
            payload = scored_point.payload
            if not payload:
                # TODO sometimes the payload is None, but not sure if this is a qdrant bug or due to a crash during indexing
                continue
            ret["score"].append(
                scored_point.score
            )  # score is a similarity score, meaning that the higher the score, the more relevant the result
            ret["text"].append(payload["text"])
            ret["title"].append(payload["title"])
            ret["full_section_title"].append(payload["full_section_title"])
            ret["block_type"].append(payload["block_type"])
            ret["language"].append(payload["language"])
            ret["last_edit_date"].append(payload["last_edit_date"])
            # ret["vector"].append(scored_point.vector)

        # take the softmax
        passage_probs = [math.exp(score) for score in ret["score"]]
        ret["prob"] = [prob / sum(passage_probs) for prob in passage_probs]

        return SearchResult(**ret)

    def embed_queries(self, queries: list[str]):
        start_time = time()

        queries = [self.query_prefix + t for t in queries]

        if self.use_onnx:
            inputs = self.embedding_tokenizer(
                queries, padding=True, truncation=True, return_tensors="np"
            )
            inputs_onnx = {
                k: ort.OrtValue.ortvalue_from_numpy(v) for k, v in inputs.items()
            }
            embeddings = self.ort_session.run(None, inputs_onnx)[0]
            # normalize embeddings
            normalized_embeddings = embeddings / np.linalg.norm(
                embeddings, axis=1, keepdims=True
            )
        else:
            encoded_input = self.embedding_tokenizer(
                queries, padding=True, truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                model_output = self.embedding_model(**encoded_input)
                # Perform pooling. In this case, cls pooling.
                embeddings = model_output[0][:, 0]
                # normalize embeddings
                normalized_embeddings = torch.nn.functional.normalize(
                    embeddings, p=2, dim=1
                )

        logger.info(
            "Embedding the query into a vector took %.2f seconds", (time() - start_time)
        )

        return normalized_embeddings.tolist()

    @staticmethod
    def get_embedding_model_parameters(embedding_model: str):
        return embedding_model_to_parameters[embedding_model]

    @staticmethod
    def get_supported_embedding_models():
        return embedding_model_to_parameters.keys()
