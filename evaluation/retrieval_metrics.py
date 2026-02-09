"""
Retrieval quality metrics for evaluating search results.

Provides standard IR metrics including Precision@K, Recall@K, MRR, NDCG,
as well as semantic similarity and diversity metrics.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    """Result of a single metric computation."""

    metric_name: str = Field(..., description="Name of the metric")
    value: float = Field(..., description="Computed metric value")
    k: Optional[int] = Field(None, description="K value if applicable")
    details: Optional[dict] = Field(None, description="Additional metric details")


class QueryMetrics(BaseModel):
    """Aggregated metrics for a single query."""

    query: str = Field(..., description="The query string")
    precision_at_k: dict[int, float] = Field(
        default_factory=dict, description="Precision@K for various K values"
    )
    recall_at_k: dict[int, float] = Field(
        default_factory=dict, description="Recall@K for various K values"
    )
    mrr: float = Field(0.0, description="Mean Reciprocal Rank")
    ndcg_at_k: dict[int, float] = Field(
        default_factory=dict, description="NDCG@K for various K values"
    )
    semantic_similarity: float = Field(
        0.0, description="Average semantic similarity of results"
    )
    diversity: float = Field(0.0, description="Diversity score of results")


@dataclass
class RetrievalMetrics:
    """
    Comprehensive retrieval metrics calculator.

    Computes standard IR metrics for evaluating retrieval quality:
    - Precision@K: Fraction of retrieved documents that are relevant
    - Recall@K: Fraction of relevant documents that are retrieved
    - MRR: Mean Reciprocal Rank of first relevant result
    - NDCG@K: Normalized Discounted Cumulative Gain
    - Semantic Similarity: Cosine similarity between query and results
    - Diversity: Pairwise dissimilarity among retrieved results
    """

    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])

    def compute_all_metrics(
        self,
        query: str,
        retrieved_ids: list[str],
        relevant_ids: list[str],
        relevance_scores: Optional[dict[str, float]] = None,
        query_embedding: Optional[list[float]] = None,
        result_embeddings: Optional[list[list[float]]] = None,
    ) -> QueryMetrics:
        """
        Compute all retrieval metrics for a single query.

        Args:
            query: The query string
            retrieved_ids: List of retrieved document IDs in rank order
            relevant_ids: List of relevant document IDs (ground truth)
            relevance_scores: Optional graded relevance scores per document
            query_embedding: Optional query embedding for semantic similarity
            result_embeddings: Optional result embeddings for similarity/diversity

        Returns:
            QueryMetrics with all computed metrics
        """
        metrics = QueryMetrics(query=query)

        for k in self.k_values:
            if k <= len(retrieved_ids):
                metrics.precision_at_k[k] = precision_at_k(
                    retrieved_ids, relevant_ids, k
                )
                metrics.recall_at_k[k] = recall_at_k(retrieved_ids, relevant_ids, k)
                metrics.ndcg_at_k[k] = ndcg_at_k(
                    retrieved_ids, relevant_ids, k, relevance_scores
                )

        metrics.mrr = mean_reciprocal_rank(retrieved_ids, relevant_ids)

        if query_embedding is not None and result_embeddings:
            metrics.semantic_similarity = semantic_similarity(
                query_embedding, result_embeddings
            )

        if result_embeddings and len(result_embeddings) > 1:
            metrics.diversity = diversity_score(result_embeddings)

        return metrics


def precision_at_k(
    retrieved_ids: list[str], relevant_ids: list[str], k: int
) -> float:
    """
    Compute Precision@K.

    Args:
        retrieved_ids: List of retrieved document IDs in rank order
        relevant_ids: List of relevant document IDs
        k: Number of top results to consider

    Returns:
        Precision@K value between 0 and 1
    """
    if k <= 0:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    relevant_retrieved = retrieved_at_k.intersection(relevant_set)

    return len(relevant_retrieved) / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Compute Recall@K.

    Args:
        retrieved_ids: List of retrieved document IDs in rank order
        relevant_ids: List of relevant document IDs
        k: Number of top results to consider

    Returns:
        Recall@K value between 0 and 1
    """
    if not relevant_ids or k <= 0:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    relevant_retrieved = retrieved_at_k.intersection(relevant_set)

    return len(relevant_retrieved) / len(relevant_set)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Returns the reciprocal of the rank of the first relevant result.

    Args:
        retrieved_ids: List of retrieved document IDs in rank order
        relevant_ids: List of relevant document IDs

    Returns:
        MRR value between 0 and 1
    """
    relevant_set = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def dcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
    relevance_scores: Optional[dict[str, float]] = None,
) -> float:
    """
    Compute Discounted Cumulative Gain at K.

    Args:
        retrieved_ids: List of retrieved document IDs in rank order
        relevant_ids: List of relevant document IDs
        k: Number of top results to consider
        relevance_scores: Optional graded relevance scores (default: binary)

    Returns:
        DCG@K value
    """
    if k <= 0:
        return 0.0

    relevant_set = set(relevant_ids)
    dcg = 0.0

    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_set:
            if relevance_scores and doc_id in relevance_scores:
                rel = relevance_scores[doc_id]
            else:
                rel = 1.0
            dcg += (2**rel - 1) / math.log2(rank + 1)

    return dcg


def ideal_dcg_at_k(
    relevant_ids: list[str],
    k: int,
    relevance_scores: Optional[dict[str, float]] = None,
) -> float:
    """
    Compute Ideal DCG at K (best possible DCG).

    Args:
        relevant_ids: List of relevant document IDs
        k: Number of top results to consider
        relevance_scores: Optional graded relevance scores

    Returns:
        Ideal DCG@K value
    """
    if not relevant_ids or k <= 0:
        return 0.0

    if relevance_scores:
        sorted_scores = sorted(
            [relevance_scores.get(doc_id, 1.0) for doc_id in relevant_ids],
            reverse=True,
        )
    else:
        sorted_scores = [1.0] * len(relevant_ids)

    idcg = 0.0
    for rank, rel in enumerate(sorted_scores[:k], start=1):
        idcg += (2**rel - 1) / math.log2(rank + 1)

    return idcg


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
    relevance_scores: Optional[dict[str, float]] = None,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K.

    Args:
        retrieved_ids: List of retrieved document IDs in rank order
        relevant_ids: List of relevant document IDs
        k: Number of top results to consider
        relevance_scores: Optional graded relevance scores

    Returns:
        NDCG@K value between 0 and 1
    """
    dcg = dcg_at_k(retrieved_ids, relevant_ids, k, relevance_scores)
    idcg = ideal_dcg_at_k(relevant_ids, k, relevance_scores)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity value between -1 and 1
    """
    arr1 = np.array(vec1)
    arr2 = np.array(vec2)

    dot_product = np.dot(arr1, arr2)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def semantic_similarity(
    query_embedding: list[float], result_embeddings: list[list[float]]
) -> float:
    """
    Compute average semantic similarity between query and results.

    Args:
        query_embedding: Query embedding vector
        result_embeddings: List of result embedding vectors

    Returns:
        Average cosine similarity between query and all results
    """
    if not result_embeddings:
        return 0.0

    similarities = [
        cosine_similarity(query_embedding, result_emb)
        for result_emb in result_embeddings
    ]

    return float(np.mean(similarities))


def diversity_score(result_embeddings: list[list[float]]) -> float:
    """
    Compute diversity score as average pairwise dissimilarity.

    Higher scores indicate more diverse results.

    Args:
        result_embeddings: List of result embedding vectors

    Returns:
        Diversity score between 0 and 1
    """
    n = len(result_embeddings)
    if n < 2:
        return 0.0

    total_dissimilarity = 0.0
    pair_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            similarity = cosine_similarity(result_embeddings[i], result_embeddings[j])
            total_dissimilarity += 1.0 - similarity
            pair_count += 1

    return total_dissimilarity / pair_count if pair_count > 0 else 0.0


def aggregate_metrics(query_metrics_list: list[QueryMetrics]) -> dict:
    """
    Aggregate metrics across multiple queries.

    Args:
        query_metrics_list: List of QueryMetrics for individual queries

    Returns:
        Dictionary with mean and std for each metric
    """
    if not query_metrics_list:
        return {}

    all_k_values = set()
    for qm in query_metrics_list:
        all_k_values.update(qm.precision_at_k.keys())
        all_k_values.update(qm.recall_at_k.keys())
        all_k_values.update(qm.ndcg_at_k.keys())

    aggregated = {}

    for k in sorted(all_k_values):
        precision_values = [
            qm.precision_at_k.get(k, 0.0) for qm in query_metrics_list
        ]
        recall_values = [qm.recall_at_k.get(k, 0.0) for qm in query_metrics_list]
        ndcg_values = [qm.ndcg_at_k.get(k, 0.0) for qm in query_metrics_list]

        aggregated[f"precision@{k}"] = {
            "mean": float(np.mean(precision_values)),
            "std": float(np.std(precision_values)),
        }
        aggregated[f"recall@{k}"] = {
            "mean": float(np.mean(recall_values)),
            "std": float(np.std(recall_values)),
        }
        aggregated[f"ndcg@{k}"] = {
            "mean": float(np.mean(ndcg_values)),
            "std": float(np.std(ndcg_values)),
        }

    mrr_values = [qm.mrr for qm in query_metrics_list]
    aggregated["mrr"] = {
        "mean": float(np.mean(mrr_values)),
        "std": float(np.std(mrr_values)),
    }

    similarity_values = [qm.semantic_similarity for qm in query_metrics_list]
    if any(v > 0 for v in similarity_values):
        aggregated["semantic_similarity"] = {
            "mean": float(np.mean(similarity_values)),
            "std": float(np.std(similarity_values)),
        }

    diversity_values = [qm.diversity for qm in query_metrics_list]
    if any(v > 0 for v in diversity_values):
        aggregated["diversity"] = {
            "mean": float(np.mean(diversity_values)),
            "std": float(np.std(diversity_values)),
        }

    return aggregated

