"""
Retrieval Quality Evaluation Framework for WikiChat.

This module provides tools for evaluating retrieval quality including:
- Precision@K, Recall@K, MRR, NDCG metrics
- Semantic similarity scoring
- Diversity metrics
- Ground truth dataset building
- Batch evaluation and comparison reporting
"""

from evaluation.retrieval_metrics import (
    RetrievalMetrics,
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    semantic_similarity,
    diversity_score,
)
from evaluation.ground_truth_builder import GroundTruthBuilder, RelevanceLabel
from evaluation.eval_runner import EvaluationRunner, EvaluationConfig, EvaluationReport

__all__ = [
    "RetrievalMetrics",
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "semantic_similarity",
    "diversity_score",
    "GroundTruthBuilder",
    "RelevanceLabel",
    "EvaluationRunner",
    "EvaluationConfig",
    "EvaluationReport",
]

