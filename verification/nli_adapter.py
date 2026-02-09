"""
Natural Language Inference (NLI) adapter for entailment checking.

Provides a unified interface for various NLI models including
HuggingFace transformers-based models.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field


class NLIPrediction(str, Enum):
    """NLI prediction labels."""

    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


class NLIResult(BaseModel):
    """Result from an NLI model prediction."""

    premise: str = Field(..., description="The premise text")
    hypothesis: str = Field(..., description="The hypothesis text")
    prediction: NLIPrediction = Field(..., description="Predicted label")
    entailment_prob: float = Field(
        0.0, ge=0.0, le=1.0, description="Probability of entailment"
    )
    contradiction_prob: float = Field(
        0.0, ge=0.0, le=1.0, description="Probability of contradiction"
    )
    neutral_prob: float = Field(
        0.0, ge=0.0, le=1.0, description="Probability of neutral"
    )

    @property
    def confidence(self) -> float:
        """Get confidence of the predicted label."""
        if self.prediction == NLIPrediction.ENTAILMENT:
            return self.entailment_prob
        elif self.prediction == NLIPrediction.CONTRADICTION:
            return self.contradiction_prob
        else:
            return self.neutral_prob


@dataclass
class NLIAdapter:
    """
    Adapter for NLI models supporting entailment checking.

    Provides a unified interface for various NLI backends including
    HuggingFace transformers models.
    """

    model_name: str = "microsoft/deberta-v3-base-mnli-fever-anli"
    device: str = "cpu"
    batch_size: int = 8
    _pipeline: Optional[object] = field(default=None, init=False)
    _initialized: bool = field(default=False, init=False)

    def _initialize(self) -> None:
        """Initialize the NLI model."""
        if self._initialized:
            return

        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "transformers is required for NLI. "
                "Install it with: pip install transformers"
            )

        self._pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            device=self.device if self.device != "cpu" else -1,
            top_k=None,
        )
        self._initialized = True

    def _parse_result(
        self, premise: str, hypothesis: str, result: list[dict]
    ) -> NLIResult:
        """Parse pipeline result into NLIResult."""
        probs = {
            NLIPrediction.ENTAILMENT: 0.0,
            NLIPrediction.CONTRADICTION: 0.0,
            NLIPrediction.NEUTRAL: 0.0,
        }

        label_map = {
            "ENTAILMENT": NLIPrediction.ENTAILMENT,
            "CONTRADICTION": NLIPrediction.CONTRADICTION,
            "NEUTRAL": NLIPrediction.NEUTRAL,
            "entailment": NLIPrediction.ENTAILMENT,
            "contradiction": NLIPrediction.CONTRADICTION,
            "neutral": NLIPrediction.NEUTRAL,
            "LABEL_0": NLIPrediction.CONTRADICTION,
            "LABEL_1": NLIPrediction.NEUTRAL,
            "LABEL_2": NLIPrediction.ENTAILMENT,
        }

        for item in result:
            label = item.get("label", "")
            score = item.get("score", 0.0)
            if label in label_map:
                probs[label_map[label]] = score

        prediction = max(probs, key=probs.get)

        return NLIResult(
            premise=premise,
            hypothesis=hypothesis,
            prediction=prediction,
            entailment_prob=probs[NLIPrediction.ENTAILMENT],
            contradiction_prob=probs[NLIPrediction.CONTRADICTION],
            neutral_prob=probs[NLIPrediction.NEUTRAL],
        )

    async def predict(self, premise: str, hypothesis: str) -> NLIResult:
        """
        Predict NLI relationship between premise and hypothesis.

        Args:
            premise: The premise (evidence) text
            hypothesis: The hypothesis (claim) text

        Returns:
            NLIResult with prediction and probabilities
        """
        self._initialize()

        input_text = f"{premise} [SEP] {hypothesis}"

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self._pipeline(input_text)
        )

        return self._parse_result(premise, hypothesis, result)

    async def predict_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[NLIResult]:
        """
        Batch predict NLI relationships.

        Args:
            pairs: List of (premise, hypothesis) tuples

        Returns:
            List of NLIResult objects
        """
        self._initialize()

        inputs = [f"{p} [SEP] {h}" for p, h in pairs]

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, lambda: self._pipeline(inputs, batch_size=self.batch_size)
        )

        return [
            self._parse_result(p, h, r) for (p, h), r in zip(pairs, results)
        ]

    def predict_sync(self, premise: str, hypothesis: str) -> NLIResult:
        """
        Synchronous prediction for non-async contexts.

        Args:
            premise: The premise text
            hypothesis: The hypothesis text

        Returns:
            NLIResult with prediction
        """
        self._initialize()

        input_text = f"{premise} [SEP] {hypothesis}"
        result = self._pipeline(input_text)

        return self._parse_result(premise, hypothesis, result)


class MockNLIAdapter:
    """
    Mock NLI adapter for testing without model loading.

    Returns configurable predictions for testing purposes.
    """

    def __init__(
        self,
        default_prediction: NLIPrediction = NLIPrediction.NEUTRAL,
        default_confidence: float = 0.8,
    ):
        """
        Initialize mock adapter.

        Args:
            default_prediction: Default prediction to return
            default_confidence: Default confidence score
        """
        self.default_prediction = default_prediction
        self.default_confidence = default_confidence
        self._overrides: dict[str, NLIResult] = {}

    def set_response(
        self, hypothesis: str, prediction: NLIPrediction, confidence: float = 0.9
    ) -> None:
        """Set a specific response for a hypothesis."""
        probs = {
            NLIPrediction.ENTAILMENT: 0.1,
            NLIPrediction.CONTRADICTION: 0.1,
            NLIPrediction.NEUTRAL: 0.1,
        }
        probs[prediction] = confidence
        remaining = 1.0 - confidence
        for p in probs:
            if p != prediction:
                probs[p] = remaining / 2

        self._overrides[hypothesis] = NLIResult(
            premise="",
            hypothesis=hypothesis,
            prediction=prediction,
            entailment_prob=probs[NLIPrediction.ENTAILMENT],
            contradiction_prob=probs[NLIPrediction.CONTRADICTION],
            neutral_prob=probs[NLIPrediction.NEUTRAL],
        )

    async def predict(self, premise: str, hypothesis: str) -> NLIResult:
        """Return configured or default prediction."""
        if hypothesis in self._overrides:
            result = self._overrides[hypothesis]
            return NLIResult(
                premise=premise,
                hypothesis=hypothesis,
                prediction=result.prediction,
                entailment_prob=result.entailment_prob,
                contradiction_prob=result.contradiction_prob,
                neutral_prob=result.neutral_prob,
            )

        probs = {
            NLIPrediction.ENTAILMENT: 0.1,
            NLIPrediction.CONTRADICTION: 0.1,
            NLIPrediction.NEUTRAL: 0.1,
        }
        probs[self.default_prediction] = self.default_confidence

        return NLIResult(
            premise=premise,
            hypothesis=hypothesis,
            prediction=self.default_prediction,
            entailment_prob=probs[NLIPrediction.ENTAILMENT],
            contradiction_prob=probs[NLIPrediction.CONTRADICTION],
            neutral_prob=probs[NLIPrediction.NEUTRAL],
        )

    async def predict_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[NLIResult]:
        """Batch predict using mock responses."""
        return [await self.predict(p, h) for p, h in pairs]


def create_nli_adapter(
    model_name: Optional[str] = None,
    use_mock: bool = False,
    device: str = "cpu",
) -> Union[NLIAdapter, MockNLIAdapter]:
    """
    Factory function to create an NLI adapter.

    Args:
        model_name: HuggingFace model name (None for default)
        use_mock: Whether to use mock adapter (for testing)
        device: Device to run model on

    Returns:
        NLIAdapter or MockNLIAdapter instance
    """
    if use_mock:
        return MockNLIAdapter()

    if model_name is None:
        model_name = "microsoft/deberta-v3-base-mnli-fever-anli"

    return NLIAdapter(model_name=model_name, device=device)

