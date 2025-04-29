from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from qdrant_client.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    MatchValue,
    Range,
)

from preprocessing.block import BlockLanguage, BlockMetadataType


class SearchFilterType(str, Enum):
    """
    SearchFilterType is an enumeration that defines different types of search filters.

    Attributes:
        eq (str): Represents the "equal to" filter.
        ne (str): Represents the "not equal to" filter.
        lt (str): Represents the "less than" filter.
        lte (str): Represents the "less than or equal to" filter.
        gt (str): Represents the "greater than" filter.
        gte (str): Represents the "greater than or equal to" filter.
    """

    EQUAL_TO = "eq"
    NOT_EQUAL_TO = "ne"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL_TO = "lte"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL_TO = "gte"


class SearchFilter(BaseModel):
    field_name: str = Field(
        ...,
        description="The name of the field to filter on",
        json_schema_extra={"example": "language"},
    )
    filter_type: SearchFilterType = Field(
        ...,
        description="The type of filter to apply, can be one of 'eq', 'ne', 'lt', 'lte', 'gt', 'gte'",
        json_schema_extra={"example": "eq"},
    )
    field_value: BlockMetadataType = Field(
        ...,
        description="The value to filter on",
        json_schema_extra={"example": BlockLanguage.ENGLISH},
    )

    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def to_qdrant_filter(metadata_filters: list["SearchFilter"]) -> Filter:
        if not metadata_filters:
            return None

        must = []
        must_not = []
        for metadata_filter in metadata_filters:
            if (
                isinstance(metadata_filter.field_value, datetime)
                or metadata_filter.field_name == "last_edit_date"
            ):
                range_class = DatetimeRange
            else:
                range_class = Range
            if metadata_filter.filter_type == "eq":
                must.append(
                    FieldCondition(
                        key=metadata_filter.field_name,
                        match=MatchValue(value=metadata_filter.field_value),
                    )
                )
            elif metadata_filter.filter_type == "ne":
                must_not.append(
                    FieldCondition(
                        key=metadata_filter.field_name,
                        match=MatchValue(value=metadata_filter.field_value),
                    )
                )
            elif metadata_filter.filter_type == "lt":
                must.append(
                    FieldCondition(
                        key=metadata_filter.field_name,
                        range=range_class(lt=metadata_filter.field_value),
                    )
                )
            elif metadata_filter.filter_type == "lte":
                must.append(
                    FieldCondition(
                        key=metadata_filter.field_name,
                        range=range_class(lte=metadata_filter.field_value),
                    )
                )
            elif metadata_filter.filter_type == "gt":
                must.append(
                    FieldCondition(
                        key=metadata_filter.field_name,
                        range=range_class(gt=metadata_filter.field_value),
                    )
                )
            elif metadata_filter.filter_type == "gte":
                must.append(
                    FieldCondition(
                        key=metadata_filter.field_name,
                        range=range_class(gte=metadata_filter.field_value),
                    )
                )

        return Filter(must=must, must_not=must_not)


class SearchQuery(BaseModel):
    query: list[str] = Field(
        default=...,  # This means the field is required
        description="A list of queries in natural language. Can be a list of one query as well.",
        json_schema_extra={"example": ["What is GPT-4?", "What is LLaMA-3?"]},
    )
    num_blocks: int = Field(
        default=...,
        description="Number of results to return",
        gt=0,
        le=100,
        json_schema_extra={"example": 3},
    )
    search_filters: list[SearchFilter] = Field(
        default_factory=list,
        description="List of filters to apply to search results. Multiple filters will be combined using AND logic.",
    )

    rerank: bool = Field(
        default=False,
        description="Whether to rerank the results using a language model",
        json_schema_extra={"example": False},
    )
    num_blocks_to_rerank: int = Field(
        default=0,
        description="Number of results to retriever for reranking. Must be greater than or equal to num_blocks",
        ge=0,
        le=100,
        json_schema_extra={"example": 10},
    )

    @field_validator("query", mode="before")
    def validate_query(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            if len(value) < 1:
                raise ValueError("String query must be non-empty")
            if len(value) > 512:
                raise ValueError("String query must be at most 512 characters long")
            value = [value]

        elif isinstance(value, list):
            if len(value) < 1:
                raise ValueError("List query must contain at least one item")
            if len(value) > 100:
                raise ValueError("List query must contain at most 100 items")
            for item in value:
                if not isinstance(item, str) or len(item) < 1:
                    raise ValueError("Each item in the list must be a non-empty string")
                if len(item) > 512:
                    raise ValueError(
                        "Each item in the list must be at most 512 characters long"
                    )
        else:
            raise TypeError("Query must be either a string or a list of strings")
        return value

    @model_validator(mode="after")
    def validate_num_blocks_to_rerank(
        cls, search_query: "SearchQuery"
    ) -> "SearchQuery":
        num_blocks = search_query.num_blocks
        if not search_query.rerank or search_query.num_blocks_to_rerank < num_blocks:
            search_query.num_blocks_to_rerank = num_blocks
        return search_query

    def __hash__(self):
        # Return the hash of a tuple containing all fields
        return hash(
            (
                tuple(self.query),
                self.num_blocks,
                self.rerank,
                self.num_blocks_to_rerank,
            )
        )

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,  # This allows the API to accept string values for enums
    )
