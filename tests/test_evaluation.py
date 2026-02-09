"""
Tests for the retrieval quality evaluation framework.

Tests cover:
- Retrieval metrics computation
- Ground truth builder functionality
- Evaluation runner operations
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, "./")

from evaluation.retrieval_metrics import (
    RetrievalMetrics,
    QueryMetrics,
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    dcg_at_k,
    cosine_similarity,
    semantic_similarity,
    diversity_score,
    aggregate_metrics,
)
from evaluation.ground_truth_builder import (
    GroundTruthBuilder,
    GroundTruthDataset,
    RelevanceLabel,
    RelevanceAnnotation,
)
from evaluation.eval_runner import (
    EvaluationRunner,
    EvaluationConfig,
    EvaluationReport,
    RetrievalResult,
    compare_evaluations,
    generate_comparison_report,
)


class TestPrecisionAtK:
    """Tests for precision_at_k metric."""

    def test_perfect_precision(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "b", "c", "d", "e"]
        assert precision_at_k(retrieved, relevant, 5) == 1.0

    def test_zero_precision(self):
        retrieved = ["a", "b", "c"]
        relevant = ["x", "y", "z"]
        assert precision_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_precision(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = ["a", "c"]
        assert precision_at_k(retrieved, relevant, 4) == 0.5

    def test_precision_at_1(self):
        retrieved = ["a", "b", "c"]
        relevant = ["a"]
        assert precision_at_k(retrieved, relevant, 1) == 1.0

    def test_precision_with_k_larger_than_retrieved(self):
        retrieved = ["a", "b"]
        relevant = ["a", "b", "c"]
        assert precision_at_k(retrieved, relevant, 5) == 0.4

    def test_precision_k_zero(self):
        retrieved = ["a", "b"]
        relevant = ["a"]
        assert precision_at_k(retrieved, relevant, 0) == 0.0


class TestRecallAtK:
    """Tests for recall_at_k metric."""

    def test_perfect_recall(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "b", "c"]
        assert recall_at_k(retrieved, relevant, 5) == 1.0

    def test_zero_recall(self):
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b", "c"]
        assert recall_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_recall(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = ["a", "c", "e", "f"]
        assert recall_at_k(retrieved, relevant, 4) == 0.5

    def test_recall_with_empty_relevant(self):
        retrieved = ["a", "b"]
        relevant = []
        assert recall_at_k(retrieved, relevant, 2) == 0.0

    def test_recall_k_zero(self):
        retrieved = ["a", "b"]
        relevant = ["a"]
        assert recall_at_k(retrieved, relevant, 0) == 0.0


class TestMRR:
    """Tests for mean_reciprocal_rank metric."""

    def test_first_position(self):
        retrieved = ["a", "b", "c"]
        relevant = ["a"]
        assert mean_reciprocal_rank(retrieved, relevant) == 1.0

    def test_second_position(self):
        retrieved = ["x", "a", "b"]
        relevant = ["a"]
        assert mean_reciprocal_rank(retrieved, relevant) == 0.5

    def test_third_position(self):
        retrieved = ["x", "y", "a"]
        relevant = ["a"]
        assert abs(mean_reciprocal_rank(retrieved, relevant) - 1 / 3) < 1e-6

    def test_no_relevant(self):
        retrieved = ["x", "y", "z"]
        relevant = ["a"]
        assert mean_reciprocal_rank(retrieved, relevant) == 0.0

    def test_multiple_relevant(self):
        retrieved = ["x", "a", "b"]
        relevant = ["a", "b"]
        assert mean_reciprocal_rank(retrieved, relevant) == 0.5


class TestNDCG:
    """Tests for NDCG metric."""

    def test_perfect_ndcg(self):
        retrieved = ["a", "b", "c"]
        relevant = ["a", "b", "c"]
        assert ndcg_at_k(retrieved, relevant, 3) == 1.0

    def test_zero_ndcg(self):
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b", "c"]
        assert ndcg_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_ndcg(self):
        retrieved = ["a", "x", "b", "y"]
        relevant = ["a", "b", "c"]
        ndcg = ndcg_at_k(retrieved, relevant, 4)
        assert 0 < ndcg < 1

    def test_ndcg_with_graded_relevance(self):
        retrieved = ["a", "b", "c"]
        relevant = ["a", "b", "c"]
        relevance_scores = {"a": 3.0, "b": 2.0, "c": 1.0}
        ndcg = ndcg_at_k(retrieved, relevant, 3, relevance_scores)
        assert ndcg == 1.0

    def test_dcg_calculation(self):
        retrieved = ["a", "b"]
        relevant = ["a", "b"]
        dcg = dcg_at_k(retrieved, relevant, 2)
        assert dcg > 0


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self):
        vec = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        assert abs(cosine_similarity(vec1, vec2)) < 1e-6

    def test_opposite_vectors(self):
        vec1 = [1.0, 1.0]
        vec2 = [-1.0, -1.0]
        assert abs(cosine_similarity(vec1, vec2) + 1.0) < 1e-6

    def test_zero_vector(self):
        vec1 = [0.0, 0.0]
        vec2 = [1.0, 1.0]
        assert cosine_similarity(vec1, vec2) == 0.0


class TestSemanticSimilarity:
    """Tests for semantic_similarity function."""

    def test_high_similarity(self):
        query = [1.0, 0.0, 0.0]
        results = [[0.9, 0.1, 0.0], [0.8, 0.2, 0.0]]
        sim = semantic_similarity(query, results)
        assert sim > 0.9

    def test_low_similarity(self):
        query = [1.0, 0.0, 0.0]
        results = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        sim = semantic_similarity(query, results)
        assert sim < 0.1

    def test_empty_results(self):
        query = [1.0, 0.0, 0.0]
        results = []
        assert semantic_similarity(query, results) == 0.0


class TestDiversityScore:
    """Tests for diversity_score function."""

    def test_identical_results(self):
        embeddings = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        assert diversity_score(embeddings) < 0.1

    def test_diverse_results(self):
        embeddings = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        div = diversity_score(embeddings)
        assert div > 0.5

    def test_single_result(self):
        embeddings = [[1.0, 0.0]]
        assert diversity_score(embeddings) == 0.0


class TestRetrievalMetrics:
    """Tests for RetrievalMetrics class."""

    def test_compute_all_metrics(self):
        metrics = RetrievalMetrics(k_values=[1, 3, 5])
        result = metrics.compute_all_metrics(
            query="test query",
            retrieved_ids=["a", "b", "c", "d", "e"],
            relevant_ids=["a", "c", "e"],
        )
        assert isinstance(result, QueryMetrics)
        assert result.query == "test query"
        assert 1 in result.precision_at_k
        assert result.mrr > 0

    def test_with_embeddings(self):
        metrics = RetrievalMetrics()
        query_emb = [1.0, 0.0, 0.0]
        result_embs = [[0.9, 0.1, 0.0], [0.8, 0.2, 0.0]]
        result = metrics.compute_all_metrics(
            query="test",
            retrieved_ids=["a", "b"],
            relevant_ids=["a"],
            query_embedding=query_emb,
            result_embeddings=result_embs,
        )
        assert result.semantic_similarity > 0
        assert result.diversity >= 0


class TestAggregateMetrics:
    """Tests for aggregate_metrics function."""

    def test_aggregate_multiple_queries(self):
        metrics_list = [
            QueryMetrics(
                query="q1",
                precision_at_k={1: 1.0, 3: 0.67},
                recall_at_k={1: 0.5, 3: 1.0},
                mrr=1.0,
                ndcg_at_k={1: 1.0, 3: 0.9},
            ),
            QueryMetrics(
                query="q2",
                precision_at_k={1: 0.0, 3: 0.33},
                recall_at_k={1: 0.0, 3: 0.5},
                mrr=0.5,
                ndcg_at_k={1: 0.0, 3: 0.5},
            ),
        ]
        aggregated = aggregate_metrics(metrics_list)
        assert "precision@1" in aggregated
        assert "mrr" in aggregated
        assert "mean" in aggregated["precision@1"]
        assert "std" in aggregated["precision@1"]

    def test_aggregate_empty_list(self):
        assert aggregate_metrics([]) == {}


class TestGroundTruthBuilder:
    """Tests for GroundTruthBuilder class."""

    def test_add_annotation(self):
        builder = GroundTruthBuilder("test_dataset")
        builder.add_annotation(
            query="test query",
            document_id="doc1",
            document_title="Test Doc",
            document_content="Test content",
            label=RelevanceLabel.RELEVANT,
        )
        assert len(builder.dataset.annotations) == 1
        assert builder.dataset.annotations[0].label == RelevanceLabel.RELEVANT

    def test_get_relevant_ids(self):
        builder = GroundTruthBuilder("test")
        builder.add_annotation(
            query="q1",
            document_id="d1",
            document_title="D1",
            document_content="C1",
            label=RelevanceLabel.HIGHLY_RELEVANT,
        )
        builder.add_annotation(
            query="q1",
            document_id="d2",
            document_title="D2",
            document_content="C2",
            label=RelevanceLabel.NOT_RELEVANT,
        )
        relevant = builder.dataset.get_relevant_ids("q1")
        assert "d1" in relevant
        assert "d2" not in relevant

    def test_get_relevance_scores(self):
        builder = GroundTruthBuilder("test")
        builder.add_annotation(
            query="q1",
            document_id="d1",
            document_title="D1",
            document_content="C1",
            label=RelevanceLabel.HIGHLY_RELEVANT,
        )
        scores = builder.dataset.get_relevance_scores("q1")
        assert scores["d1"] == 3.0

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_gt.json")

            builder = GroundTruthBuilder("test_dataset", "Test description")
            builder.add_annotation(
                query="q1",
                document_id="d1",
                document_title="D1",
                document_content="C1",
                label=RelevanceLabel.RELEVANT,
            )
            builder.save(filepath)

            loaded = GroundTruthBuilder.load(filepath)
            assert loaded.dataset.name == "test_dataset"
            assert len(loaded.dataset.annotations) == 1

    def test_export_and_import(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "export.json")

            builder = GroundTruthBuilder("test")
            builder.add_annotation(
                query="q1",
                document_id="d1",
                document_title="D1",
                document_content="C1",
                label=RelevanceLabel.RELEVANT,
            )
            builder.export_for_sharing(filepath)

            imported = GroundTruthBuilder.import_from_sharing_format(filepath)
            assert len(imported.dataset.annotations) == 1

    def test_get_statistics(self):
        builder = GroundTruthBuilder("test")
        builder.add_annotation(
            query="q1",
            document_id="d1",
            document_title="D1",
            document_content="C1",
            label=RelevanceLabel.RELEVANT,
        )
        builder.add_annotation(
            query="q1",
            document_id="d2",
            document_title="D2",
            document_content="C2",
            label=RelevanceLabel.NOT_RELEVANT,
        )
        stats = builder.get_statistics()
        assert stats["total_annotations"] == 2
        assert stats["unique_queries"] == 1


class TestEvaluationRunner:
    """Tests for EvaluationRunner class."""

    @pytest.fixture
    def sample_ground_truth(self):
        builder = GroundTruthBuilder("test_gt")
        builder.add_annotation(
            query="What is Python?",
            document_id="doc1",
            document_title="Python Programming",
            document_content="Python is a programming language",
            label=RelevanceLabel.HIGHLY_RELEVANT,
        )
        builder.add_annotation(
            query="What is Python?",
            document_id="doc2",
            document_title="Snake Info",
            document_content="Python is also a snake",
            label=RelevanceLabel.RELEVANT,
        )
        return builder.dataset

    @pytest.fixture
    def sample_config(self):
        return EvaluationConfig(
            name="test_eval",
            k_values=[1, 3],
            compute_semantic_similarity=False,
            compute_diversity=False,
        )

    def test_evaluate_single(self, sample_ground_truth, sample_config):
        runner = EvaluationRunner(
            ground_truth=sample_ground_truth, config=sample_config
        )
        result = RetrievalResult(
            query="What is Python?",
            retrieved_ids=["doc1", "doc3", "doc2"],
        )
        metrics = runner.evaluate_single(result)
        assert metrics.precision_at_k[1] == 1.0
        assert metrics.mrr == 1.0

    def test_evaluate_batch(self, sample_ground_truth, sample_config):
        runner = EvaluationRunner(
            ground_truth=sample_ground_truth, config=sample_config
        )
        results = [
            RetrievalResult(
                query="What is Python?",
                retrieved_ids=["doc1", "doc2"],
            ),
        ]
        report = runner.evaluate_batch(results)
        assert isinstance(report, EvaluationReport)
        assert report.num_queries == 1

    def test_save_report(self, sample_ground_truth, sample_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EvaluationConfig(
                name="test_eval",
                k_values=[1, 3],
                output_dir=tmpdir,
            )
            runner = EvaluationRunner(
                ground_truth=sample_ground_truth, config=config
            )
            results = [
                RetrievalResult(
                    query="What is Python?",
                    retrieved_ids=["doc1"],
                ),
            ]
            report = runner.evaluate_batch(results)
            filepath = runner.save_report(report)
            assert os.path.exists(filepath)


class TestCompareEvaluations:
    """Tests for evaluation comparison functions."""

    def test_compare_evaluations(self):
        reports = [
            EvaluationReport(
                config=EvaluationConfig(name="config_a"),
                ground_truth_name="gt",
                num_queries=10,
                aggregated_metrics={
                    "precision@1": {"mean": 0.8, "std": 0.1},
                    "mrr": {"mean": 0.9, "std": 0.05},
                },
            ),
            EvaluationReport(
                config=EvaluationConfig(name="config_b"),
                ground_truth_name="gt",
                num_queries=10,
                aggregated_metrics={
                    "precision@1": {"mean": 0.7, "std": 0.15},
                    "mrr": {"mean": 0.85, "std": 0.08},
                },
            ),
        ]
        comparison = compare_evaluations(reports)
        assert "config_a" in comparison["reports"]
        assert "config_b" in comparison["reports"]
        assert comparison["best_per_metric"]["precision@1"] == "config_a"

    def test_generate_comparison_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reports = [
                EvaluationReport(
                    config=EvaluationConfig(name="config_a"),
                    ground_truth_name="gt",
                    num_queries=10,
                    aggregated_metrics={
                        "precision@1": {"mean": 0.8, "std": 0.1},
                    },
                ),
            ]
            output_path = os.path.join(tmpdir, "comparison.json")
            generate_comparison_report(reports, output_path)
            assert os.path.exists(output_path)


class TestRelevanceLabel:
    """Tests for RelevanceLabel enum."""

    def test_label_scores(self):
        assert RelevanceLabel.HIGHLY_RELEVANT.score == 3.0
        assert RelevanceLabel.RELEVANT.score == 2.0
        assert RelevanceLabel.PARTIALLY_RELEVANT.score == 1.0
        assert RelevanceLabel.NOT_RELEVANT.score == 0.0

