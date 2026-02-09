"""
Tests for the hallucination detection module.

Tests cover:
- Claim extraction and verification
- NLI adapter functionality
- Factuality scoring
"""

import asyncio
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, "./")

from verification.claim_verifier import (
    ClaimVerifier,
    ClaimVerification,
    VerificationResult,
    VerificationVerdict,
    EvidenceMatch,
)
from verification.factuality_scorer import (
    FactualityScorer,
    FactualityScore,
    DialogueFactuality,
    TurnFactuality,
    aggregate_factuality_scores,
)
from verification.nli_adapter import (
    NLIAdapter,
    NLIResult,
    NLIPrediction,
    MockNLIAdapter,
    create_nli_adapter,
)


class TestVerificationVerdict:
    """Tests for VerificationVerdict enum."""

    def test_is_factual(self):
        assert VerificationVerdict.SUPPORTED.is_factual is True
        assert VerificationVerdict.REFUTED.is_factual is False
        assert VerificationVerdict.NOT_ENOUGH_INFO.is_factual is False


class TestEvidenceMatch:
    """Tests for EvidenceMatch model."""

    def test_create_evidence_match(self):
        match = EvidenceMatch(
            document_title="Test Title",
            document_content="Test content",
            relevance_score=0.9,
            entailment_score=0.8,
            url="https://example.com",
        )
        assert match.document_title == "Test Title"
        assert match.relevance_score == 0.9


class TestClaimVerification:
    """Tests for ClaimVerification model."""

    def test_create_verification(self):
        verification = ClaimVerification(
            claim="The sky is blue.",
            evidence=[],
            verdict=VerificationVerdict.SUPPORTED,
            confidence=0.95,
        )
        assert verification.claim == "The sky is blue."
        assert verification.verdict == VerificationVerdict.SUPPORTED


class TestVerificationResult:
    """Tests for VerificationResult model."""

    def test_compute_overall_all_supported(self):
        result = VerificationResult(
            utterance="Test utterance",
            claims=["claim1", "claim2"],
            verifications=[
                ClaimVerification(
                    claim="claim1",
                    evidence=[],
                    verdict=VerificationVerdict.SUPPORTED,
                    confidence=0.9,
                ),
                ClaimVerification(
                    claim="claim2",
                    evidence=[],
                    verdict=VerificationVerdict.SUPPORTED,
                    confidence=0.85,
                ),
            ],
        )
        result.compute_overall()
        assert result.overall_verdict == VerificationVerdict.SUPPORTED
        assert result.factuality_ratio == 1.0

    def test_compute_overall_with_refuted(self):
        result = VerificationResult(
            utterance="Test utterance",
            claims=["claim1", "claim2"],
            verifications=[
                ClaimVerification(
                    claim="claim1",
                    evidence=[],
                    verdict=VerificationVerdict.SUPPORTED,
                    confidence=0.9,
                ),
                ClaimVerification(
                    claim="claim2",
                    evidence=[],
                    verdict=VerificationVerdict.REFUTED,
                    confidence=0.85,
                ),
            ],
        )
        result.compute_overall()
        assert result.overall_verdict == VerificationVerdict.REFUTED
        assert result.factuality_ratio == 0.5

    def test_compute_overall_empty(self):
        result = VerificationResult(
            utterance="Test",
            claims=[],
            verifications=[],
        )
        result.compute_overall()
        assert result.overall_verdict == VerificationVerdict.NOT_ENOUGH_INFO
        assert result.factuality_ratio == 0.0


class TestClaimVerifier:
    """Tests for ClaimVerifier class."""

    def test_simple_claim_extraction(self):
        verifier = ClaimVerifier()
        claims = verifier._simple_claim_extraction(
            "Python is a programming language. It was created by Guido van Rossum."
        )
        assert len(claims) == 2
        assert "Python is a programming language" in claims[0]

    def test_simple_claim_extraction_removes_questions(self):
        verifier = ClaimVerifier()
        claims = verifier._simple_claim_extraction(
            "Python is great. What do you think? It is popular."
        )
        assert not any("?" in claim for claim in claims)

    def test_keyword_overlap_score(self):
        verifier = ClaimVerifier()
        score = verifier._keyword_overlap_score(
            "Python is a programming language",
            "Python is a popular programming language used worldwide",
        )
        assert score > 0.5

    def test_keyword_overlap_score_no_overlap(self):
        verifier = ClaimVerifier()
        score = verifier._keyword_overlap_score(
            "Python is a programming language",
            "Java coffee beans are delicious",
        )
        assert score < 0.3

    @pytest.mark.asyncio
    async def test_verify_claim_with_mock_nli(self):
        mock_nli = MockNLIAdapter(
            default_prediction=NLIPrediction.ENTAILMENT, default_confidence=0.9
        )
        verifier = ClaimVerifier(nli_adapter=mock_nli, entailment_threshold=0.7)

        evidence = [
            {
                "title": "Python",
                "content": "Python is a high-level programming language.",
                "url": "https://python.org",
            }
        ]

        result = await verifier.verify_claim(
            "Python is a programming language.", evidence
        )

        assert result.verdict == VerificationVerdict.SUPPORTED
        assert result.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_verify_utterance(self):
        verifier = ClaimVerifier()

        evidence = [
            {
                "title": "Python Programming",
                "content": "Python is a popular programming language created by Guido van Rossum.",
            }
        ]

        result = await verifier.verify_utterance(
            utterance="Python is a programming language. It was created by Guido.",
            evidence_list=evidence,
        )

        assert isinstance(result, VerificationResult)
        assert len(result.claims) > 0
        assert len(result.verifications) == len(result.claims)

    @pytest.mark.asyncio
    async def test_verify_with_existing_claims(self):
        verifier = ClaimVerifier()

        existing_claims = ["Python is a programming language."]
        evidence = [
            {
                "title": "Python",
                "content": "Python is a programming language.",
            }
        ]

        result = await verifier.verify_utterance(
            utterance="Python is great!",
            evidence_list=evidence,
            existing_claims=existing_claims,
        )

        assert result.claims == existing_claims


class TestNLIResult:
    """Tests for NLIResult model."""

    def test_confidence_entailment(self):
        result = NLIResult(
            premise="test",
            hypothesis="test",
            prediction=NLIPrediction.ENTAILMENT,
            entailment_prob=0.9,
            contradiction_prob=0.05,
            neutral_prob=0.05,
        )
        assert result.confidence == 0.9

    def test_confidence_contradiction(self):
        result = NLIResult(
            premise="test",
            hypothesis="test",
            prediction=NLIPrediction.CONTRADICTION,
            entailment_prob=0.05,
            contradiction_prob=0.9,
            neutral_prob=0.05,
        )
        assert result.confidence == 0.9


class TestMockNLIAdapter:
    """Tests for MockNLIAdapter class."""

    @pytest.mark.asyncio
    async def test_default_prediction(self):
        adapter = MockNLIAdapter(
            default_prediction=NLIPrediction.NEUTRAL, default_confidence=0.8
        )
        result = await adapter.predict("premise", "hypothesis")
        assert result.prediction == NLIPrediction.NEUTRAL

    @pytest.mark.asyncio
    async def test_set_response(self):
        adapter = MockNLIAdapter()
        adapter.set_response(
            "specific hypothesis", NLIPrediction.ENTAILMENT, confidence=0.95
        )
        result = await adapter.predict("any premise", "specific hypothesis")
        assert result.prediction == NLIPrediction.ENTAILMENT
        assert result.entailment_prob == 0.95

    @pytest.mark.asyncio
    async def test_batch_predict(self):
        adapter = MockNLIAdapter(default_prediction=NLIPrediction.ENTAILMENT)
        pairs = [("p1", "h1"), ("p2", "h2"), ("p3", "h3")]
        results = await adapter.predict_batch(pairs)
        assert len(results) == 3
        assert all(r.prediction == NLIPrediction.ENTAILMENT for r in results)


class TestCreateNLIAdapter:
    """Tests for create_nli_adapter factory."""

    def test_create_mock_adapter(self):
        adapter = create_nli_adapter(use_mock=True)
        assert isinstance(adapter, MockNLIAdapter)

    def test_create_real_adapter(self):
        adapter = create_nli_adapter(use_mock=False)
        assert isinstance(adapter, NLIAdapter)


class TestFactualityScore:
    """Tests for FactualityScore model."""

    def test_from_verification_result(self):
        result = VerificationResult(
            utterance="Test",
            claims=["c1", "c2", "c3"],
            verifications=[
                ClaimVerification(
                    claim="c1",
                    evidence=[],
                    verdict=VerificationVerdict.SUPPORTED,
                    confidence=0.9,
                ),
                ClaimVerification(
                    claim="c2",
                    evidence=[],
                    verdict=VerificationVerdict.REFUTED,
                    confidence=0.8,
                ),
                ClaimVerification(
                    claim="c3",
                    evidence=[],
                    verdict=VerificationVerdict.NOT_ENOUGH_INFO,
                    confidence=0.7,
                ),
            ],
        )
        result.compute_overall()

        score = FactualityScore.from_verification_result(result, "test_id")
        assert score.total_claims == 3
        assert score.supported_claims == 1
        assert score.refuted_claims == 1
        assert score.uncertain_claims == 1
        assert abs(score.factuality_ratio - 1 / 3) < 0.01


class TestDialogueFactuality:
    """Tests for DialogueFactuality model."""

    def test_add_turn(self):
        dialogue = DialogueFactuality(dialogue_id="test_dialogue")

        turn = TurnFactuality(
            turn_id=0,
            user_utterance="Hello",
            agent_utterance="Hi there!",
            score=FactualityScore(
                response_id="t0",
                total_claims=2,
                supported_claims=2,
                refuted_claims=0,
                uncertain_claims=0,
                factuality_ratio=1.0,
                weighted_score=0.9,
                confidence=0.85,
            ),
            verifications=[],
        )

        dialogue.add_turn(turn)
        assert len(dialogue.turns) == 1
        assert dialogue.total_claims == 2
        assert dialogue.overall_score == 1.0


class TestFactualityScorer:
    """Tests for FactualityScorer class."""

    def test_score_response(self):
        scorer = FactualityScorer(dialogue_id="test")

        result = VerificationResult(
            utterance="Test response",
            claims=["claim1"],
            verifications=[
                ClaimVerification(
                    claim="claim1",
                    evidence=[],
                    verdict=VerificationVerdict.SUPPORTED,
                    confidence=0.9,
                )
            ],
        )
        result.compute_overall()

        score = scorer.score_response(result, "Test input")
        assert score.total_claims == 1
        assert score.factuality_ratio == 1.0
        assert scorer.get_current_score() == 1.0

    def test_get_statistics(self):
        scorer = FactualityScorer(dialogue_id="test")

        for i in range(3):
            result = VerificationResult(
                utterance=f"Response {i}",
                claims=[f"claim{i}"],
                verifications=[
                    ClaimVerification(
                        claim=f"claim{i}",
                        evidence=[],
                        verdict=VerificationVerdict.SUPPORTED,
                        confidence=0.8,
                    )
                ],
            )
            result.compute_overall()
            scorer.score_response(result, f"Input {i}")

        stats = scorer.get_statistics()
        assert stats["num_turns"] == 3
        assert stats["total_claims"] == 3
        assert stats["overall_score"] == 1.0

    def test_get_problematic_claims(self):
        scorer = FactualityScorer(dialogue_id="test")

        result = VerificationResult(
            utterance="Test",
            claims=["good claim", "bad claim"],
            verifications=[
                ClaimVerification(
                    claim="good claim",
                    evidence=[],
                    verdict=VerificationVerdict.SUPPORTED,
                    confidence=0.9,
                ),
                ClaimVerification(
                    claim="bad claim",
                    evidence=[],
                    verdict=VerificationVerdict.REFUTED,
                    confidence=0.85,
                ),
            ],
        )
        result.compute_overall()
        scorer.score_response(result, "Input")

        problematic = scorer.get_problematic_claims()
        assert len(problematic) == 1
        assert problematic[0]["claim"] == "bad claim"

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "factuality.json")

            scorer = FactualityScorer(dialogue_id="test")
            result = VerificationResult(
                utterance="Test",
                claims=["claim1"],
                verifications=[
                    ClaimVerification(
                        claim="claim1",
                        evidence=[],
                        verdict=VerificationVerdict.SUPPORTED,
                        confidence=0.9,
                    )
                ],
            )
            result.compute_overall()
            scorer.score_response(result, "Input")
            scorer.save(filepath)

            loaded = FactualityScorer.load(filepath)
            assert loaded.dialogue_id == "test"
            assert len(loaded.dialogue.turns) == 1

    def test_generate_report(self):
        scorer = FactualityScorer(dialogue_id="test")

        result = VerificationResult(
            utterance="Test response",
            claims=["claim1"],
            verifications=[
                ClaimVerification(
                    claim="claim1",
                    evidence=[],
                    verdict=VerificationVerdict.SUPPORTED,
                    confidence=0.9,
                )
            ],
        )
        result.compute_overall()
        scorer.score_response(result, "Input")

        report = scorer.generate_report()
        assert "Factuality Report" in report
        assert "test" in report


class TestAggregateFactualityScores:
    """Tests for aggregate_factuality_scores function."""

    def test_aggregate_multiple_scorers(self):
        scorers = []
        for i in range(3):
            scorer = FactualityScorer(dialogue_id=f"dialogue_{i}")
            result = VerificationResult(
                utterance=f"Response {i}",
                claims=[f"claim{i}"],
                verifications=[
                    ClaimVerification(
                        claim=f"claim{i}",
                        evidence=[],
                        verdict=VerificationVerdict.SUPPORTED,
                        confidence=0.8,
                    )
                ],
            )
            result.compute_overall()
            scorer.score_response(result, f"Input {i}")
            scorers.append(scorer)

        aggregated = aggregate_factuality_scores(scorers)
        assert aggregated["num_dialogues"] == 3
        assert aggregated["total_claims"] == 3
        assert aggregated["overall_factuality"] == 1.0

    def test_aggregate_empty_list(self):
        assert aggregate_factuality_scores([]) == {}


class TestIntegration:
    """Integration tests for verification pipeline."""

    @pytest.mark.asyncio
    async def test_full_verification_pipeline(self):
        mock_nli = MockNLIAdapter(
            default_prediction=NLIPrediction.ENTAILMENT, default_confidence=0.85
        )
        verifier = ClaimVerifier(nli_adapter=mock_nli)
        scorer = FactualityScorer(dialogue_id="integration_test")

        evidence = [
            {
                "title": "Python",
                "content": "Python is a high-level programming language created by Guido van Rossum in 1991.",
            }
        ]

        result = await verifier.verify_utterance(
            utterance="Python is a programming language. It was created by Guido van Rossum.",
            evidence_list=evidence,
        )

        score = scorer.score_response(result, "Tell me about Python")

        assert score.total_claims > 0
        assert scorer.get_current_score() > 0
        stats = scorer.get_statistics()
        assert stats["num_turns"] == 1

