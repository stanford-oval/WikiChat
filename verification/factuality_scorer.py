"""
Factuality scoring and tracking for dialogue responses.

Provides aggregate factuality metrics and historical tracking across
dialogue turns.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from verification.claim_verifier import (
    ClaimVerification,
    VerificationResult,
    VerificationVerdict,
)


class FactualityScore(BaseModel):
    """Factuality score for a single response."""

    response_id: str = Field(..., description="Unique identifier for the response")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Score timestamp"
    )
    total_claims: int = Field(0, description="Total number of claims")
    supported_claims: int = Field(0, description="Number of supported claims")
    refuted_claims: int = Field(0, description="Number of refuted claims")
    uncertain_claims: int = Field(0, description="Number of uncertain claims")
    factuality_ratio: float = Field(
        0.0, ge=0.0, le=1.0, description="Ratio of supported claims"
    )
    weighted_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Weighted factuality score"
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Average confidence in verifications"
    )

    @classmethod
    def from_verification_result(
        cls, result: VerificationResult, response_id: str
    ) -> "FactualityScore":
        """
        Create a factuality score from a verification result.

        Args:
            result: The verification result
            response_id: Unique response identifier

        Returns:
            FactualityScore instance
        """
        supported = sum(
            1
            for v in result.verifications
            if v.verdict == VerificationVerdict.SUPPORTED
        )
        refuted = sum(
            1
            for v in result.verifications
            if v.verdict == VerificationVerdict.REFUTED
        )
        uncertain = sum(
            1
            for v in result.verifications
            if v.verdict == VerificationVerdict.NOT_ENOUGH_INFO
        )

        total = len(result.verifications)
        factuality_ratio = supported / total if total > 0 else 0.0

        if total > 0:
            weighted_sum = 0.0
            for v in result.verifications:
                if v.verdict == VerificationVerdict.SUPPORTED:
                    weighted_sum += v.confidence
                elif v.verdict == VerificationVerdict.REFUTED:
                    weighted_sum -= v.confidence * 0.5
            weighted_score = max(0.0, min(1.0, (weighted_sum / total + 1) / 2))
        else:
            weighted_score = 0.5

        avg_confidence = (
            sum(v.confidence for v in result.verifications) / total
            if total > 0
            else 0.0
        )

        return cls(
            response_id=response_id,
            total_claims=total,
            supported_claims=supported,
            refuted_claims=refuted,
            uncertain_claims=uncertain,
            factuality_ratio=factuality_ratio,
            weighted_score=weighted_score,
            confidence=avg_confidence,
        )


class TurnFactuality(BaseModel):
    """Factuality information for a dialogue turn."""

    turn_id: int = Field(..., description="Turn number in the dialogue")
    user_utterance: str = Field(..., description="User's message")
    agent_utterance: str = Field(..., description="Agent's response")
    score: FactualityScore = Field(..., description="Factuality score")
    verifications: list[ClaimVerification] = Field(
        default_factory=list, description="Individual claim verifications"
    )


class DialogueFactuality(BaseModel):
    """Factuality tracking for an entire dialogue."""

    dialogue_id: str = Field(..., description="Unique dialogue identifier")
    start_time: datetime = Field(
        default_factory=datetime.now, description="Dialogue start time"
    )
    turns: list[TurnFactuality] = Field(
        default_factory=list, description="Per-turn factuality"
    )
    overall_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Overall dialogue factuality"
    )
    total_claims: int = Field(0, description="Total claims across dialogue")
    supported_claims: int = Field(0, description="Total supported claims")
    refuted_claims: int = Field(0, description="Total refuted claims")

    def add_turn(self, turn: TurnFactuality) -> None:
        """Add a turn and update aggregate metrics."""
        self.turns.append(turn)
        self._update_aggregates()

    def _update_aggregates(self) -> None:
        """Update aggregate metrics from all turns."""
        self.total_claims = sum(t.score.total_claims for t in self.turns)
        self.supported_claims = sum(t.score.supported_claims for t in self.turns)
        self.refuted_claims = sum(t.score.refuted_claims for t in self.turns)

        if self.total_claims > 0:
            self.overall_score = self.supported_claims / self.total_claims
        else:
            self.overall_score = 0.0


@dataclass
class FactualityScorer:
    """
    Scorer for tracking and aggregating factuality across responses.

    Provides methods for scoring individual responses and tracking
    factuality across dialogue sessions.
    """

    dialogue_id: str
    dialogue: DialogueFactuality = field(init=False)
    _turn_counter: int = field(default=0, init=False)

    def __post_init__(self):
        self.dialogue = DialogueFactuality(dialogue_id=self.dialogue_id)

    def score_response(
        self,
        verification_result: VerificationResult,
        user_utterance: str,
    ) -> FactualityScore:
        """
        Score a single response and add to dialogue history.

        Args:
            verification_result: Verification result for the response
            user_utterance: The user's input

        Returns:
            FactualityScore for the response
        """
        response_id = f"{self.dialogue_id}_turn_{self._turn_counter}"
        score = FactualityScore.from_verification_result(
            verification_result, response_id
        )

        turn = TurnFactuality(
            turn_id=self._turn_counter,
            user_utterance=user_utterance,
            agent_utterance=verification_result.utterance,
            score=score,
            verifications=verification_result.verifications,
        )

        self.dialogue.add_turn(turn)
        self._turn_counter += 1

        return score

    def get_current_score(self) -> float:
        """Get the current overall factuality score."""
        return self.dialogue.overall_score

    def get_statistics(self) -> dict:
        """
        Get comprehensive statistics about dialogue factuality.

        Returns:
            Dictionary with factuality statistics
        """
        if not self.dialogue.turns:
            return {
                "dialogue_id": self.dialogue_id,
                "num_turns": 0,
                "overall_score": 0.0,
                "total_claims": 0,
                "claim_breakdown": {
                    "supported": 0,
                    "refuted": 0,
                    "uncertain": 0,
                },
            }

        scores = [t.score for t in self.dialogue.turns]

        return {
            "dialogue_id": self.dialogue_id,
            "num_turns": len(self.dialogue.turns),
            "overall_score": self.dialogue.overall_score,
            "total_claims": self.dialogue.total_claims,
            "claim_breakdown": {
                "supported": self.dialogue.supported_claims,
                "refuted": self.dialogue.refuted_claims,
                "uncertain": sum(s.uncertain_claims for s in scores),
            },
            "per_turn_scores": [
                {
                    "turn": t.turn_id,
                    "factuality_ratio": t.score.factuality_ratio,
                    "weighted_score": t.score.weighted_score,
                    "claims": t.score.total_claims,
                }
                for t in self.dialogue.turns
            ],
            "average_claims_per_turn": (
                self.dialogue.total_claims / len(self.dialogue.turns)
            ),
            "average_confidence": sum(s.confidence for s in scores) / len(scores),
        }

    def get_problematic_claims(self) -> list[dict]:
        """
        Get all refuted or uncertain claims for review.

        Returns:
            List of problematic claims with context
        """
        problematic = []

        for turn in self.dialogue.turns:
            for v in turn.verifications:
                if v.verdict != VerificationVerdict.SUPPORTED:
                    problematic.append(
                        {
                            "turn_id": turn.turn_id,
                            "claim": v.claim,
                            "verdict": v.verdict.value,
                            "confidence": v.confidence,
                            "evidence_count": len(v.evidence),
                            "context": turn.user_utterance,
                        }
                    )

        return problematic

    def save(self, output_path: str) -> None:
        """
        Save dialogue factuality data to file.

        Args:
            output_path: Path to save the data
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self.dialogue.model_dump(mode="json"),
                f,
                indent=2,
                default=str,
            )

    @classmethod
    def load(cls, input_path: str) -> "FactualityScorer":
        """
        Load dialogue factuality data from file.

        Args:
            input_path: Path to the saved data

        Returns:
            FactualityScorer with loaded data
        """
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dialogue = DialogueFactuality.model_validate(data)
        scorer = cls(dialogue_id=dialogue.dialogue_id)
        scorer.dialogue = dialogue
        scorer._turn_counter = len(dialogue.turns)

        return scorer

    def generate_report(self) -> str:
        """
        Generate a human-readable factuality report.

        Returns:
            Formatted report string
        """
        stats = self.get_statistics()

        lines = [
            f"Factuality Report: {self.dialogue_id}",
            "=" * 50,
            "",
            f"Number of Turns: {stats['num_turns']}",
            f"Overall Factuality Score: {stats['overall_score']:.2%}",
            "",
            "Claim Breakdown:",
            f"  Total Claims: {stats['total_claims']}",
            f"  Supported: {stats['claim_breakdown']['supported']}",
            f"  Refuted: {stats['claim_breakdown']['refuted']}",
            f"  Uncertain: {stats['claim_breakdown']['uncertain']}",
            "",
        ]

        if stats.get("per_turn_scores"):
            lines.append("Per-Turn Scores:")
            for turn_stat in stats["per_turn_scores"]:
                lines.append(
                    f"  Turn {turn_stat['turn']}: "
                    f"{turn_stat['factuality_ratio']:.2%} "
                    f"({turn_stat['claims']} claims)"
                )

        problematic = self.get_problematic_claims()
        if problematic:
            lines.extend(
                [
                    "",
                    "Problematic Claims:",
                    "-" * 40,
                ]
            )
            for claim_info in problematic[:10]:
                lines.append(
                    f"  [{claim_info['verdict']}] {claim_info['claim'][:80]}..."
                    if len(claim_info["claim"]) > 80
                    else f"  [{claim_info['verdict']}] {claim_info['claim']}"
                )

        return "\n".join(lines)


def aggregate_factuality_scores(scorers: list[FactualityScorer]) -> dict:
    """
    Aggregate factuality statistics across multiple dialogues.

    Args:
        scorers: List of FactualityScorer instances

    Returns:
        Aggregated statistics dictionary
    """
    if not scorers:
        return {}

    all_stats = [s.get_statistics() for s in scorers]

    total_turns = sum(s["num_turns"] for s in all_stats)
    total_claims = sum(s["total_claims"] for s in all_stats)
    total_supported = sum(s["claim_breakdown"]["supported"] for s in all_stats)
    total_refuted = sum(s["claim_breakdown"]["refuted"] for s in all_stats)
    total_uncertain = sum(s["claim_breakdown"]["uncertain"] for s in all_stats)

    overall_scores = [s["overall_score"] for s in all_stats if s["num_turns"] > 0]

    return {
        "num_dialogues": len(scorers),
        "total_turns": total_turns,
        "total_claims": total_claims,
        "overall_factuality": total_supported / total_claims if total_claims > 0 else 0,
        "claim_breakdown": {
            "supported": total_supported,
            "refuted": total_refuted,
            "uncertain": total_uncertain,
        },
        "average_score_per_dialogue": (
            sum(overall_scores) / len(overall_scores) if overall_scores else 0
        ),
        "min_score": min(overall_scores) if overall_scores else 0,
        "max_score": max(overall_scores) if overall_scores else 0,
    }

