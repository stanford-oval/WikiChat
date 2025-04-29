import math
from enum import Enum

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    Datatype,
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    QuantizationSearchParams,
    ScoredPoint,
    SearchParams,
    SearchRequest,
    VectorParams,
)

from preprocessing.block import Block
from retrieval.retrieval_commons import AsyncVectorDB, QueryResult, SearchResultBlock
from retrieval.search_query import SearchFilter
from tasks.defaults import DEFAULT_VECTORDB_PORT


class AsyncQdrantVectorDB(AsyncVectorDB):
    def __init__(
        self,
        embedding_model_name: str,
        vector_db_url: str = None,
        use_onnx: bool = False,
        skip_loading_embedding_model: bool = False,
    ):
        super().__init__(
            embedding_model_name,
            vector_db_url,
            use_onnx,
            skip_loading_embedding_model,
        )
        self.vector_db_client = AsyncQdrantClient(
            url=vector_db_url,
            timeout=20,
            prefer_grpc=True,
            port=DEFAULT_VECTORDB_PORT,
            grpc_port=DEFAULT_VECTORDB_PORT + 1,
        )  # Lower timeout so that we don't get stuck on long retrieval jobs

        # These parameters result in a 99.1% precision@10 compared to exact search, using Alibaba-NLP/gte-multilingual-base embedding model on the English Wikipedia index
        # This means that out of the top 10 results we get using approximate search with binary quantization using these parameters, 99.1% of them are also in the top 10 results of exact search
        # Oversampling is important here, without it, the precision@10 is only ~66%
        self.search_params = SearchParams(
            indexed_only=True,
            hnsw_ef=64,
            exact=False,
            quantization=QuantizationSearchParams(
                ignore=False,
                rescore=True,
                oversampling=3.0,
            ),
        )

    async def close_connections(self):
        await self.vector_db_client.close()

    async def _batch_vector_search(
        self,
        collection_name: str,
        vectors: list[list[float]],
        search_filters: list[SearchFilter],
        k: int,
    ) -> list[QueryResult]:
        batch_results = await self.vector_db_client.search_batch(
            collection_name=collection_name,
            requests=[
                SearchRequest(
                    vector=v,
                    with_vector=False,
                    with_payload=True,
                    limit=k,
                    params=self.search_params,
                    filter=SearchFilter.to_qdrant_filter(search_filters),
                )
                for v in vectors
            ],
        )

        ret = []
        for result in batch_results:
            # each iteration is for one query in the batch
            pydantic_result = AsyncQdrantVectorDB._search_result_to_pydantic(result)
            ret.append(pydantic_result)
        return ret

    async def list_collections(self) -> list[str]:
        collections = await self.vector_db_client.get_collections()
        return [c.name for c in collections.collections]

    async def delete_collection(self, collection_name: str) -> None:
        await self.vector_db_client.delete_collection(collection_name)

    async def add_blocks(
        self, collection_name: str, vectors: list[list[float]], blocks: list[Block]
    ):
        payload = []
        for b in blocks:
            p = {
                "document_title": b.document_title,
                "section_title": b.section_title,
                "content": b.content,
                "url": b.url,
                "last_edit_date": b.last_edit_date,
            }
            if b.block_metadata:
                for key, value in b.block_metadata.items():
                    if isinstance(value, Enum):
                        value = value.value
                    p[key] = value
            payload.append(p)

        self.vector_db_client.upload_collection(
            collection_name=collection_name,
            vectors=vectors,
            payload=payload,
            ids=[b.id for b in blocks],
            max_retries=3,
            # wait=True,
        )

    async def create_metadata_index(
        self,
        collection_name: str,
        field_names: list[str],
        field_types: list[PayloadSchemaType],
    ) -> None:
        for f_name, f_type in zip(field_names, field_types):
            await self.vector_db_client.create_payload_index(
                collection_name=collection_name,
                field_name=f_name,
                field_schema=f_type,
                wait=False,
            )

    async def _create_collection(self, collection_name: str, high_memory: bool) -> bool:
        if high_memory:
            index_on_disk = False
            vectors_on_disk = False
        else:
            index_on_disk = True
            vectors_on_disk = True

        quantization_config = BinaryQuantization(
            binary=BinaryQuantizationConfig(
                always_ram=True,
            ),
        )

        ret = await self.vector_db_client.create_collection(
            collection_name=collection_name,
            shard_number=None,  # Qdrant's default is one shard per node
            on_disk_payload=True,
            vectors_config=VectorParams(
                size=self.embedding_dimension,
                distance=Distance.DOT,
                datatype=Datatype.FLOAT16,
                on_disk=vectors_on_disk,
            ),
            hnsw_config=HnswConfigDiff(
                m=64,
                ef_construct=100,
                full_scan_threshold=10,
                on_disk=index_on_disk,
            ),
            quantization_config=quantization_config,
        )
        return ret

    def _search_result_to_pydantic(search_result: list[ScoredPoint]) -> QueryResult:
        """
        Results for one query
        """
        results = []
        probabilities = [scored_point.score for scored_point in search_result]
        # take the softmax
        probabilities = [math.exp(score) for score in probabilities]
        probabilities = [prob / sum(probabilities) for prob in probabilities]

        for scored_point, prob in zip(search_result, probabilities):
            payload = scored_point.payload
            if not payload:
                # TODO sometimes the payload is None, but not sure if this is a qdrant bug or due to a crash during indexing
                continue
            block_metadata = {
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "document_title",
                    "section_title",
                    "content",
                    "url",
                    "last_edit_date",
                }
            }
            results.append(
                SearchResultBlock(
                    document_title=payload["document_title"],
                    section_title=payload["section_title"],
                    content=payload["content"],
                    url=payload.get("url", None),
                    last_edit_date=payload["last_edit_date"],
                    block_metadata=block_metadata,
                    similarity_score=scored_point.score,
                    probability_score=prob,
                )
            )

        return QueryResult(results=results)
