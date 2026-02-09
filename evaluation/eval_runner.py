"""
Batch evaluation runner for retrieval quality assessment.

Provides tools for running evaluations across multiple queries and
generating comparison reports.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field

from evaluation.ground_truth_builder import GroundTruthBuilder, GroundTruthDataset
from evaluation.retrieval_metrics import (
    QueryMetrics,
    RetrievalMetrics,
    aggregate_metrics,
)


class EvaluationConfig(BaseModel):
    """Configuration for an evaluation run."""

    name: str = Field(..., description="Evaluation run name")
    k_values: list[int] = Field(
        default_factory=lambda: [1, 3, 5, 10],
        description="K values for metrics computation",
    )
    compute_semantic_similarity: bool = Field(
        True, description="Whether to compute semantic similarity"
    )
    compute_diversity: bool = Field(True, description="Whether to compute diversity")
    output_dir: str = Field("evaluation_results", description="Output directory")


class RetrievalResult(BaseModel):
    """Result from a retrieval system for a single query."""

    query: str = Field(..., description="The query string")
    retrieved_ids: list[str] = Field(..., description="Retrieved document IDs")
    retrieved_titles: list[str] = Field(
        default_factory=list, description="Retrieved document titles"
    )
    scores: list[float] = Field(
        default_factory=list, description="Retrieval scores"
    )
    embeddings: Optional[list[list[float]]] = Field(
        None, description="Document embeddings if available"
    )
    query_embedding: Optional[list[float]] = Field(
        None, description="Query embedding if available"
    )


class EvaluationReport(BaseModel):
    """Complete evaluation report."""

    config: EvaluationConfig = Field(..., description="Evaluation configuration")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Evaluation timestamp"
    )
    ground_truth_name: str = Field(..., description="Ground truth dataset name")
    num_queries: int = Field(..., description="Number of queries evaluated")
    per_query_metrics: list[QueryMetrics] = Field(
        default_factory=list, description="Metrics for each query"
    )
    aggregated_metrics: dict = Field(
        default_factory=dict, description="Aggregated metrics across queries"
    )
    metadata: dict = Field(default_factory=dict, description="Additional metadata")

    def to_summary_string(self) -> str:
        """Generate a human-readable summary of the evaluation."""
        lines = [
            f"Evaluation Report: {self.config.name}",
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Ground Truth: {self.ground_truth_name}",
            f"Queries Evaluated: {self.num_queries}",
            "",
            "Aggregated Metrics:",
            "-" * 40,
        ]

        for metric_name, values in sorted(self.aggregated_metrics.items()):
            if isinstance(values, dict):
                mean = values.get("mean", 0)
                std = values.get("std", 0)
                lines.append(f"  {metric_name}: {mean:.4f} (+/- {std:.4f})")
            else:
                lines.append(f"  {metric_name}: {values:.4f}")

        return "\n".join(lines)


@dataclass
class EvaluationRunner:
    """
    Runner for batch retrieval evaluation.

    Evaluates retrieval results against ground truth datasets and
    generates comprehensive reports.
    """

    ground_truth: GroundTruthDataset
    config: EvaluationConfig
    metrics_calculator: RetrievalMetrics = field(init=False)

    def __post_init__(self):
        self.metrics_calculator = RetrievalMetrics(k_values=self.config.k_values)

    def evaluate_single(self, result: RetrievalResult) -> QueryMetrics:
        """
        Evaluate a single retrieval result.

        Args:
            result: Retrieval result for a query

        Returns:
            QueryMetrics for the result
        """
        relevant_ids = self.ground_truth.get_relevant_ids(result.query)
        relevance_scores = self.ground_truth.get_relevance_scores(result.query)

        return self.metrics_calculator.compute_all_metrics(
            query=result.query,
            retrieved_ids=result.retrieved_ids,
            relevant_ids=relevant_ids,
            relevance_scores=relevance_scores if relevance_scores else None,
            query_embedding=(
                result.query_embedding
                if self.config.compute_semantic_similarity
                else None
            ),
            result_embeddings=(
                result.embeddings if self.config.compute_diversity else None
            ),
        )

    def evaluate_batch(self, results: list[RetrievalResult]) -> EvaluationReport:
        """
        Evaluate a batch of retrieval results.

        Args:
            results: List of retrieval results

        Returns:
            Complete evaluation report
        """
        per_query_metrics = []

        for result in results:
            if result.query in self.ground_truth.get_queries():
                metrics = self.evaluate_single(result)
                per_query_metrics.append(metrics)

        aggregated = aggregate_metrics(per_query_metrics)

        return EvaluationReport(
            config=self.config,
            ground_truth_name=self.ground_truth.name,
            num_queries=len(per_query_metrics),
            per_query_metrics=per_query_metrics,
            aggregated_metrics=aggregated,
        )

    def save_report(self, report: EvaluationReport) -> str:
        """
        Save evaluation report to file.

        Args:
            report: Evaluation report to save

        Returns:
            Path to saved report
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.name}_{timestamp}.json"
        filepath = output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2, default=str)

        summary_path = output_dir / f"{self.config.name}_{timestamp}_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(report.to_summary_string())

        return str(filepath)

    @classmethod
    def from_ground_truth_file(
        cls, ground_truth_path: str, config: EvaluationConfig
    ) -> "EvaluationRunner":
        """
        Create runner from a ground truth file.

        Args:
            ground_truth_path: Path to ground truth JSON file
            config: Evaluation configuration

        Returns:
            Configured EvaluationRunner
        """
        builder = GroundTruthBuilder.load(ground_truth_path)
        return cls(ground_truth=builder.dataset, config=config)


def compare_evaluations(reports: list[EvaluationReport]) -> dict:
    """
    Compare multiple evaluation reports.

    Args:
        reports: List of evaluation reports to compare

    Returns:
        Comparison dictionary with metrics for each report
    """
    if not reports:
        return {}

    comparison = {
        "reports": [r.config.name for r in reports],
        "metrics_comparison": {},
    }

    all_metrics = set()
    for report in reports:
        all_metrics.update(report.aggregated_metrics.keys())

    for metric in sorted(all_metrics):
        comparison["metrics_comparison"][metric] = {}
        for report in reports:
            if metric in report.aggregated_metrics:
                values = report.aggregated_metrics[metric]
                if isinstance(values, dict):
                    comparison["metrics_comparison"][metric][report.config.name] = {
                        "mean": values.get("mean", 0),
                        "std": values.get("std", 0),
                    }
                else:
                    comparison["metrics_comparison"][metric][report.config.name] = {
                        "mean": values,
                        "std": 0,
                    }

    best_per_metric = {}
    for metric, results in comparison["metrics_comparison"].items():
        best_name = None
        best_value = -float("inf")
        for name, values in results.items():
            if values["mean"] > best_value:
                best_value = values["mean"]
                best_name = name
        best_per_metric[metric] = best_name

    comparison["best_per_metric"] = best_per_metric

    return comparison


def generate_comparison_report(
    reports: list[EvaluationReport], output_path: str
) -> None:
    """
    Generate a comparison report for multiple evaluations.

    Args:
        reports: List of evaluation reports
        output_path: Path to save the comparison report
    """
    comparison = compare_evaluations(reports)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    summary_lines = [
        "Evaluation Comparison Report",
        "=" * 50,
        "",
        f"Comparing: {', '.join(comparison['reports'])}",
        "",
        "Metrics Comparison:",
        "-" * 50,
    ]

    for metric, results in sorted(comparison["metrics_comparison"].items()):
        summary_lines.append(f"\n{metric}:")
        for name, values in results.items():
            is_best = comparison["best_per_metric"].get(metric) == name
            marker = " *" if is_best else ""
            summary_lines.append(
                f"  {name}: {values['mean']:.4f} (+/- {values['std']:.4f}){marker}"
            )

    summary_lines.extend(
        [
            "",
            "-" * 50,
            "* indicates best performing configuration",
        ]
    )

    summary_path = path.with_suffix(".txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))


async def run_retrieval_evaluation(
    retriever_fn: Callable[[str], RetrievalResult],
    ground_truth_path: str,
    config: EvaluationConfig,
    queries: Optional[list[str]] = None,
) -> EvaluationReport:
    """
    Run a complete retrieval evaluation.

    Args:
        retriever_fn: Function that takes a query and returns RetrievalResult
        ground_truth_path: Path to ground truth dataset
        config: Evaluation configuration
        queries: Optional list of queries to evaluate (uses all if not provided)

    Returns:
        Complete evaluation report
    """
    runner = EvaluationRunner.from_ground_truth_file(ground_truth_path, config)

    if queries is None:
        queries = runner.ground_truth.get_queries()

    results = []
    for query in queries:
        result = retriever_fn(query)
        results.append(result)

    report = runner.evaluate_batch(results)
    runner.save_report(report)

    return report

