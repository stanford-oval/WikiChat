# Retrieval Quality Evaluation Framework

A comprehensive evaluation framework for assessing retrieval quality in WikiChat.

## Overview

This module provides tools for:
- Computing standard IR metrics (Precision@K, Recall@K, MRR, NDCG)
- Semantic similarity and diversity scoring
- Building ground truth datasets with LLM-assisted labeling
- Batch evaluation and comparison reporting

## Installation

The evaluation module uses dependencies already included in WikiChat:
- `numpy` for numerical computations
- `pydantic` for data validation

Optional dependencies for LLM-assisted labeling:
- `chainlite` (included in WikiChat)

## Usage

### Computing Retrieval Metrics

```python
from evaluation.retrieval_metrics import (
    RetrievalMetrics,
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
)

# Simple metric computation
retrieved_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
relevant_ids = ["doc1", "doc3", "doc7"]

p_at_5 = precision_at_k(retrieved_ids, relevant_ids, k=5)
r_at_5 = recall_at_k(retrieved_ids, relevant_ids, k=5)
mrr = mean_reciprocal_rank(retrieved_ids, relevant_ids)

# Comprehensive metrics with RetrievalMetrics class
metrics = RetrievalMetrics(k_values=[1, 3, 5, 10])
query_metrics = metrics.compute_all_metrics(
    query="What is Python?",
    retrieved_ids=retrieved_ids,
    relevant_ids=relevant_ids,
)
print(f"Precision@5: {query_metrics.precision_at_k[5]}")
print(f"MRR: {query_metrics.mrr}")
```

### Building Ground Truth Datasets

```python
from evaluation.ground_truth_builder import GroundTruthBuilder, RelevanceLabel

# Create a new dataset
builder = GroundTruthBuilder(
    dataset_name="wikipedia_retrieval_eval",
    description="Evaluation dataset for Wikipedia retrieval",
)

# Add manual annotations
builder.add_annotation(
    query="What is Python programming?",
    document_id="doc_python_intro",
    document_title="Python (programming language)",
    document_content="Python is a high-level programming language...",
    label=RelevanceLabel.HIGHLY_RELEVANT,
)

# Save the dataset
builder.save("ground_truth/wikipedia_eval.json")

# Load an existing dataset
loaded_builder = GroundTruthBuilder.load("ground_truth/wikipedia_eval.json")
```

### LLM-Assisted Labeling

```python
import asyncio
from evaluation.ground_truth_builder import GroundTruthBuilder

builder = GroundTruthBuilder(
    dataset_name="auto_labeled",
    llm_engine="gpt-4o",  # or any configured engine
)

# Label a single document
annotation = asyncio.run(builder.label_with_llm(
    query="What is machine learning?",
    document_id="doc1",
    document_title="Machine Learning Overview",
    document_content="Machine learning is a subset of artificial intelligence...",
))
```

### Running Batch Evaluations

```python
from evaluation.eval_runner import (
    EvaluationRunner,
    EvaluationConfig,
    RetrievalResult,
)
from evaluation.ground_truth_builder import GroundTruthBuilder

# Load ground truth
builder = GroundTruthBuilder.load("ground_truth/wikipedia_eval.json")

# Configure evaluation
config = EvaluationConfig(
    name="baseline_evaluation",
    k_values=[1, 3, 5, 10],
    output_dir="evaluation_results",
)

# Create runner
runner = EvaluationRunner(
    ground_truth=builder.dataset,
    config=config,
)

# Prepare results (from your retrieval system)
results = [
    RetrievalResult(
        query="What is Python?",
        retrieved_ids=["doc1", "doc2", "doc3"],
    ),
]

# Run evaluation
report = runner.evaluate_batch(results)
print(report.to_summary_string())

# Save report
runner.save_report(report)
```

### Comparing Configurations

```python
from evaluation.eval_runner import compare_evaluations, generate_comparison_report

reports = [baseline_report, improved_report]  # EvaluationReport objects
comparison = compare_evaluations(reports)
generate_comparison_report(reports, "comparison_results/report.json")
```

## Metrics Reference

| Metric | Description | Range |
|--------|-------------|-------|
| Precision@K | Fraction of top-K results that are relevant | [0, 1] |
| Recall@K | Fraction of relevant documents in top-K | [0, 1] |
| MRR | Reciprocal rank of first relevant result | [0, 1] |
| NDCG@K | Normalized DCG accounting for rank positions | [0, 1] |
| Semantic Similarity | Cosine similarity between query and results | [-1, 1] |
| Diversity | Average pairwise dissimilarity of results | [0, 1] |

## File Structure

```
evaluation/
    __init__.py           # Module exports
    retrieval_metrics.py  # Core metric implementations
    ground_truth_builder.py  # Dataset building tools
    eval_runner.py        # Batch evaluation runner
    README.md             # This file
```

## Testing

```bash
pytest tests/test_evaluation.py -v
```

