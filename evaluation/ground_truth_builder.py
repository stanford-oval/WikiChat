"""
Ground truth dataset builder for retrieval evaluation.

Provides tools for building and managing ground truth relevance labels,
including LLM-assisted relevance labeling.
"""

import asyncio
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class RelevanceLabel(str, Enum):
    """Relevance labels for ground truth annotation."""

    HIGHLY_RELEVANT = "highly_relevant"
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    NOT_RELEVANT = "not_relevant"

    @property
    def score(self) -> float:
        """Convert label to numeric score for graded metrics."""
        scores = {
            RelevanceLabel.HIGHLY_RELEVANT: 3.0,
            RelevanceLabel.RELEVANT: 2.0,
            RelevanceLabel.PARTIALLY_RELEVANT: 1.0,
            RelevanceLabel.NOT_RELEVANT: 0.0,
        }
        return scores[self]


class RelevanceAnnotation(BaseModel):
    """A single relevance annotation for a query-document pair."""

    query: str = Field(..., description="The query string")
    document_id: str = Field(..., description="The document identifier")
    document_title: str = Field(..., description="Document title for reference")
    document_content: str = Field(..., description="Document content snippet")
    label: RelevanceLabel = Field(..., description="Relevance label")
    confidence: float = Field(
        1.0, ge=0.0, le=1.0, description="Annotation confidence score"
    )
    annotator: str = Field("human", description="Annotator identifier")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Annotation timestamp"
    )
    notes: Optional[str] = Field(None, description="Optional annotation notes")


class GroundTruthDataset(BaseModel):
    """A collection of relevance annotations for evaluation."""

    name: str = Field(..., description="Dataset name")
    description: str = Field("", description="Dataset description")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    annotations: list[RelevanceAnnotation] = Field(
        default_factory=list, description="List of annotations"
    )
    metadata: dict = Field(default_factory=dict, description="Additional metadata")

    def get_relevant_ids(self, query: str) -> list[str]:
        """Get list of relevant document IDs for a query."""
        return [
            ann.document_id
            for ann in self.annotations
            if ann.query == query and ann.label != RelevanceLabel.NOT_RELEVANT
        ]

    def get_relevance_scores(self, query: str) -> dict[str, float]:
        """Get graded relevance scores for a query."""
        return {
            ann.document_id: ann.label.score
            for ann in self.annotations
            if ann.query == query
        }

    def get_queries(self) -> list[str]:
        """Get list of unique queries in the dataset."""
        return list(set(ann.query for ann in self.annotations))


class GroundTruthBuilder:
    """
    Builder for creating ground truth relevance datasets.

    Supports both manual annotation and LLM-assisted labeling.
    """

    def __init__(
        self,
        dataset_name: str,
        description: str = "",
        llm_engine: Optional[str] = None,
    ):
        """
        Initialize the ground truth builder.

        Args:
            dataset_name: Name for the dataset
            description: Dataset description
            llm_engine: Optional LLM engine for assisted labeling
        """
        self.dataset = GroundTruthDataset(
            name=dataset_name,
            description=description,
        )
        self.llm_engine = llm_engine

    def add_annotation(
        self,
        query: str,
        document_id: str,
        document_title: str,
        document_content: str,
        label: RelevanceLabel,
        confidence: float = 1.0,
        annotator: str = "human",
        notes: Optional[str] = None,
    ) -> None:
        """
        Add a manual relevance annotation.

        Args:
            query: The query string
            document_id: Document identifier
            document_title: Document title
            document_content: Document content snippet
            label: Relevance label
            confidence: Annotation confidence
            annotator: Annotator identifier
            notes: Optional notes
        """
        annotation = RelevanceAnnotation(
            query=query,
            document_id=document_id,
            document_title=document_title,
            document_content=document_content,
            label=label,
            confidence=confidence,
            annotator=annotator,
            notes=notes,
        )
        self.dataset.annotations.append(annotation)

    async def label_with_llm(
        self,
        query: str,
        document_id: str,
        document_title: str,
        document_content: str,
    ) -> RelevanceAnnotation:
        """
        Use LLM to generate a relevance label.

        Args:
            query: The query string
            document_id: Document identifier
            document_title: Document title
            document_content: Document content

        Returns:
            RelevanceAnnotation with LLM-generated label
        """
        if self.llm_engine is None:
            raise ValueError("LLM engine not configured for assisted labeling")

        try:
            from chainlite import llm_generation_chain
        except ImportError:
            raise ImportError(
                "chainlite is required for LLM-assisted labeling. "
                "Install it with: pip install chainlite"
            )

        prompt = f"""Evaluate the relevance of the following document to the query.

Query: {query}

Document Title: {document_title}
Document Content: {document_content}

Rate the relevance using one of these labels:
- highly_relevant: The document directly and completely answers the query
- relevant: The document provides useful information related to the query
- partially_relevant: The document has some connection to the query but is not very helpful
- not_relevant: The document is not related to the query

Respond with ONLY the label (e.g., "highly_relevant")."""

        chain = llm_generation_chain(
            template_file=None,
            engine=self.llm_engine,
            max_tokens=20,
        )

        response = await chain.ainvoke({"text": prompt})
        response = response.strip().lower().replace(" ", "_")

        label_map = {
            "highly_relevant": RelevanceLabel.HIGHLY_RELEVANT,
            "relevant": RelevanceLabel.RELEVANT,
            "partially_relevant": RelevanceLabel.PARTIALLY_RELEVANT,
            "not_relevant": RelevanceLabel.NOT_RELEVANT,
        }

        label = label_map.get(response, RelevanceLabel.NOT_RELEVANT)

        annotation = RelevanceAnnotation(
            query=query,
            document_id=document_id,
            document_title=document_title,
            document_content=document_content,
            label=label,
            confidence=0.8,
            annotator=f"llm:{self.llm_engine}",
        )

        self.dataset.annotations.append(annotation)
        return annotation

    async def batch_label_with_llm(
        self,
        items: list[dict],
    ) -> list[RelevanceAnnotation]:
        """
        Batch label multiple query-document pairs with LLM.

        Args:
            items: List of dicts with keys: query, document_id, document_title, document_content

        Returns:
            List of RelevanceAnnotations
        """
        tasks = [
            self.label_with_llm(
                query=item["query"],
                document_id=item["document_id"],
                document_title=item["document_title"],
                document_content=item["document_content"],
            )
            for item in items
        ]
        return await asyncio.gather(*tasks)

    def save(self, file_path: str) -> None:
        """
        Save the dataset to a JSON file.

        Args:
            file_path: Path to save the dataset
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.dataset.model_dump(mode="json"), f, indent=2, default=str)

    @classmethod
    def load(cls, file_path: str) -> "GroundTruthBuilder":
        """
        Load a dataset from a JSON file.

        Args:
            file_path: Path to the dataset file

        Returns:
            GroundTruthBuilder with loaded dataset
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = GroundTruthDataset.model_validate(data)
        builder = cls(dataset_name=dataset.name, description=dataset.description)
        builder.dataset = dataset
        return builder

    def export_for_sharing(self, file_path: str, include_content: bool = True) -> None:
        """
        Export dataset in a format suitable for sharing.

        Args:
            file_path: Path to save the export
            include_content: Whether to include document content
        """
        export_data = {
            "name": self.dataset.name,
            "description": self.dataset.description,
            "version": "1.0",
            "queries": {},
        }

        for ann in self.dataset.annotations:
            if ann.query not in export_data["queries"]:
                export_data["queries"][ann.query] = []

            doc_data = {
                "document_id": ann.document_id,
                "document_title": ann.document_title,
                "label": ann.label.value,
                "score": ann.label.score,
            }

            if include_content:
                doc_data["document_content"] = ann.document_content

            export_data["queries"][ann.query].append(doc_data)

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def import_from_sharing_format(cls, file_path: str) -> "GroundTruthBuilder":
        """
        Import dataset from sharing format.

        Args:
            file_path: Path to the import file

        Returns:
            GroundTruthBuilder with imported dataset
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        builder = cls(
            dataset_name=data.get("name", "imported_dataset"),
            description=data.get("description", ""),
        )

        label_map = {
            "highly_relevant": RelevanceLabel.HIGHLY_RELEVANT,
            "relevant": RelevanceLabel.RELEVANT,
            "partially_relevant": RelevanceLabel.PARTIALLY_RELEVANT,
            "not_relevant": RelevanceLabel.NOT_RELEVANT,
        }

        for query, documents in data.get("queries", {}).items():
            for doc in documents:
                builder.add_annotation(
                    query=query,
                    document_id=doc["document_id"],
                    document_title=doc["document_title"],
                    document_content=doc.get("document_content", ""),
                    label=label_map.get(doc["label"], RelevanceLabel.NOT_RELEVANT),
                )

        return builder

    def get_statistics(self) -> dict:
        """
        Get statistics about the dataset.

        Returns:
            Dictionary with dataset statistics
        """
        queries = self.dataset.get_queries()
        label_counts = {label: 0 for label in RelevanceLabel}

        for ann in self.dataset.annotations:
            label_counts[ann.label] += 1

        return {
            "total_annotations": len(self.dataset.annotations),
            "unique_queries": len(queries),
            "label_distribution": {
                label.value: count for label, count in label_counts.items()
            },
            "avg_annotations_per_query": (
                len(self.dataset.annotations) / len(queries) if queries else 0
            ),
        }

