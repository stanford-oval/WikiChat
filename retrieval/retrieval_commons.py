import abc
import asyncio
import json
import sys
from time import time
from typing import Optional, Union

import numpy as np
import torch
from async_lru import alru_cache
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModel, AutoTokenizer

from retrieval.search_query import SearchFilter, SearchQuery
from retrieval.search_result_block import SearchResultBlock

sys.path.insert(0, "./")
from utils.logging import logger
from preprocessing.block import Block
from retrieval.embedding_model_info import get_embedding_model_parameters


class QueryResult(BaseModel):
    results: list[SearchResultBlock] = Field(
        default_factory=list, description="The list of search results"
    )

    model_config = ConfigDict(extra="forbid")  # Disallow extra fields

    def to_string(self, n: Optional[int] = None, id_prefix: str = "") -> str:
        if n is None:
            n = len(self.results)
        if len(self.results) == 0:
            return "No relevant results found."
        return "\n\n".join(
            [
                f"[{id_prefix}{i + 1}] Title: {r.full_title}\n{r.content}"
                for i, r in enumerate(self.results[:n])
            ]
        )

    def to_string_for_review(
        self,
        n: Optional[int] = None,
        id_prefix: str = "",
        action_index: int = 1,
    ) -> tuple[str, int]:
        if n is None:
            n = len(self.results)
        if len(self.results) == 0:
            return "No relevant results found.", 0
        return "\n\n".join(
            [
                f"<strong>[{id_prefix}{i + action_index}] Title: {r.full_title}</strong>\n\n{r.content}"
                for i, r in enumerate(self.results[:n])
            ]
        ), len(self.results[:n])

    @staticmethod
    def to_markdown(obj: Union["QueryResult", list[SearchResultBlock]]) -> str:
        results = obj.results if isinstance(obj, QueryResult) else obj
        ret = []
        for r in results:
            ret.append(f"#### [{r.full_title}]({r.url})\n{r.content}")

        if ret:
            return "\n\n".join(ret)
        return "No relevant results found."


class AsyncVectorDB(abc.ABC):
    def __init__(
        self,
        embedding_model_name: str,
        vector_db_url: str = None,
        use_onnx: bool = False,
        skip_loading_embedding_model: bool = False,
    ):
        self.embedding_model_name = embedding_model_name
        self.vector_db_url = vector_db_url
        self.use_onnx = use_onnx

        self.embedding_dimension = get_embedding_model_parameters(embedding_model_name)[
            "embedding_dimension"
        ]
        self.query_template = get_embedding_model_parameters(embedding_model_name)[
            "query_template"
        ]
        if not skip_loading_embedding_model:
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(
                embedding_model_name
            )
            if self.use_onnx:
                import onnxruntime as ort

                self.ort = ort  # Store ort as an attribute of the class
                if "onnx_config" not in get_embedding_model_parameters(
                    embedding_model_name
                ):
                    raise ValueError(
                        f"ONNX has not been tested for '{embedding_model_name}'."
                    )
                logger.info("Using ONNX for the embedding model.")
                onnx_config = get_embedding_model_parameters(embedding_model_name)[
                    "onnx_config"
                ]

                # Download the onnx model files
                onnx_repo_id = onnx_config["onnx_model_repo"]
                onnx_model_path = hf_hub_download(
                    repo_id=onnx_repo_id, filename=onnx_config["onnx_model_file"]
                )
                hf_hub_download(
                    repo_id=onnx_repo_id, filename=onnx_config["onnx_data_file"]
                )
                # hf_hub_download(repo_id=onnx_repo_id, filename="ort_config.json")

                self.ort_session = self.ort.InferenceSession(onnx_model_path)
            else:
                logger.info(
                    "ONNX is not being used for the embedding model. Use `--use-onnx` flag to use ONNX."
                )
                self.embedding_model = AutoModel.from_pretrained(
                    embedding_model_name, trust_remote_code=True
                )

            # self.embedding_model.eval()
            logger.info(
                f"Successfully loaded the embedding model '{embedding_model_name}'"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup code here, for example closing the connection
        asyncio.run(self.close_connections())
        return False  # Return False to propagate exceptions

    @abc.abstractmethod
    async def _batch_vector_search(
        self,
        collection_name: str,
        vectors: list[list[float]],
        search_filters: list[SearchFilter],
        k: int,
    ) -> list[QueryResult]:
        """
        Perform a batch search on the vector database with the given query vectors and search filters.

        Args:
            vectors (list[list[float]]): A list of lists where each inner list represents a vector to search for.
            search_filters (list[SearchFilter]): A list of search filters to apply to the search results.
            k (int): The maximum number of search results to return for each query vector.

        Returns:
            list[SearchResult]: A list of search results, where each result corresponds to a query vector.
        """
        raise NotImplementedError

    @alru_cache(maxsize=10000)
    async def search(
        self,
        collection_name: str,
        search_query: SearchQuery,
    ) -> list[QueryResult]:
        """
        Returns a list of SearchResult, one SearchResult per query
        """
        if not await self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        queries = search_query.query
        was_list = True
        if isinstance(queries, str):
            queries = [queries]
            was_list = False
        logger.info(
            f"Searching for {len(queries)} {'query' if len(queries) == 1 else 'queries'}: {json.dumps(queries, indent=2, ensure_ascii=False)}"
        )
        query_embeddings = self._embed_queries(queries)

        start_time = time()
        batch_results = await self._batch_vector_search(
            collection_name,
            query_embeddings,
            search_query.search_filters,
            search_query.num_blocks,
        )
        logger.info(f"Nearest neighbor search took {time() - start_time:.2f} seconds")

        if was_list:
            assert len(batch_results) == len(queries)
        else:
            assert len(batch_results) == 1
        return batch_results

    def _embed_queries(self, queries: list[str]):
        start_time = time()

        queries = [self.query_template(t) for t in queries]
        embedding_dimension = get_embedding_model_parameters(self.embedding_model_name)[
            "embedding_dimension"
        ]
        matryoshka = get_embedding_model_parameters(self.embedding_model_name).get(
            "matryoshka", False
        )

        if self.use_onnx:
            onnx_config = get_embedding_model_parameters(self.embedding_model_name)[
                "onnx_config"
            ]
            inputs = self.embedding_tokenizer(
                queries, padding=True, truncation=True, return_tensors="np"
            )
            inputs_onnx = {
                k: self.ort.OrtValue.ortvalue_from_numpy(v) for k, v in inputs.items()
            }
            embeddings = self.ort_session.run(None, inputs_onnx)[
                onnx_config["embedding_index_in_output"]
            ]

            # first truncate, then normalize
            if matryoshka:
                embeddings = embeddings[:, :embedding_dimension]
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

            # first truncate, then normalize
            if matryoshka:
                embeddings = embeddings[:, :embedding_dimension]
            # normalize embeddings
            normalized_embeddings = torch.nn.functional.normalize(
                embeddings, p=2, dim=1
            )

        logger.info(
            f"Embedding {'query' if len(queries) == 1 else 'queries'} took {time() - start_time:.2f} seconds"
        )
        assert normalized_embeddings.shape[0] == len(queries)

        assert normalized_embeddings.shape[1] == embedding_dimension, (
            f"Expected {embedding_dimension} dimensions, got {normalized_embeddings.shape[1]} dimensions"
        )

        return normalized_embeddings.tolist()

    async def add_blocks(
        self, collection_name: str, vectors: list[list[float]], blocks: list[Block]
    ):
        """
        Uploads the vectors and payloads to the vector database.

        Args:
            vectors (list[list[float]]): A list of vectors to upload.
            payloads: A list of payloads to upload.
            ids: A list of ids to upload.
        """
        raise NotImplementedError

    async def create_collection_if_not_exists(
        self, collection_name: str, high_memory: bool = False
    ):
        if not await self.collection_exists(collection_name):
            logger.info(
                f"Did not find collection '{collection_name}' in VectorDB, creating it..."
            )
            result = await self._create_collection(collection_name, high_memory)
            if result:
                logger.info("Collection creation was successful")
            else:
                # print(result)
                raise RuntimeError("Could not create the collection in VectorDB.")

    async def collection_exists(self, collection_name: str) -> bool:
        return collection_name in await self.list_collections()

    @abc.abstractmethod
    async def list_collections(self) -> list[str]:
        raise NotImplementedError

    @abc.abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    async def _create_collection(self, collection_name: str, high_memory: bool) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    async def close_connections(self):
        raise NotImplementedError
