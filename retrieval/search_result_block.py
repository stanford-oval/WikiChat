from pydantic import ConfigDict, Field, field_validator

from preprocessing.block import Block


class SearchResultBlock(Block):
    """
    A block of search result with additional similarity and probability scores.
    """

    similarity_score: float = Field(
        default=...,
        json_schema_extra={"example": 0.681},
        description="The similarity score between the search result and the query.",
    )
    probability_score: float = Field(
        default=...,
        json_schema_extra={"example": 0.331},
        description="Normalized similarity score. Can be viewed as the estimated probability that this SearchResultBlock is the answer to the query.",
    )
    summary: list[str] = Field(
        default_factory=list,
        json_schema_extra={"example": ["bullet point 1", "bullet point 2"]},
        description="Bullet points from the search result that are relevant to the query.",
    )

    model_config = ConfigDict(extra="forbid")  # Disallow extra fields

    @staticmethod
    def _round_to_three_digits(value: float) -> float:
        return round(value, 3)

    @field_validator("similarity_score", "probability_score", mode="before")
    def validate_scores(cls, value):
        return cls._round_to_three_digits(value)

    def __hash__(self) -> int:
        return hash(self.combined_text)

    def __eq__(self, value):
        return self.combined_text == value.combined_text
