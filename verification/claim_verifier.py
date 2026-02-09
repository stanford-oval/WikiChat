"""
Claim extraction and verification against retrieved evidence.

Provides tools for extracting claims from agent utterances and verifying
them against search results.
"""

import asyncio
import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class VerificationVerdict(str, Enum):
    """Verdict for a claim verification."""

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"

    @property
    def is_factual(self) -> bool:
        """Check if the verdict indicates factuality."""
        return self == VerificationVerdict.SUPPORTED


class EvidenceMatch(BaseModel):
    """A piece of evidence matched to a claim."""

    document_title: str = Field(..., description="Title of the source document")
    document_content: str = Field(..., description="Content of the evidence")
    relevance_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Relevance score"
    )
    entailment_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Entailment score from NLI"
    )
    url: Optional[str] = Field(None, description="URL of the source")


class ClaimVerification(BaseModel):
    """Result of verifying a single claim."""

    claim: str = Field(..., description="The claim being verified")
    evidence: list[EvidenceMatch] = Field(
        default_factory=list, description="Evidence used for verification"
    )
    verdict: VerificationVerdict = Field(
        ..., description="Verification verdict"
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Confidence in the verdict"
    )
    explanation: Optional[str] = Field(
        None, description="Explanation for the verdict"
    )


class VerificationResult(BaseModel):
    """Complete verification result for an agent utterance."""

    utterance: str = Field(..., description="The original agent utterance")
    claims: list[str] = Field(default_factory=list, description="Extracted claims")
    verifications: list[ClaimVerification] = Field(
        default_factory=list, description="Verification results per claim"
    )
    overall_verdict: VerificationVerdict = Field(
        VerificationVerdict.NOT_ENOUGH_INFO,
        description="Overall verification verdict",
    )
    factuality_ratio: float = Field(
        0.0, ge=0.0, le=1.0, description="Ratio of supported claims"
    )

    def compute_overall(self) -> None:
        """Compute overall verdict and factuality ratio."""
        if not self.verifications:
            self.overall_verdict = VerificationVerdict.NOT_ENOUGH_INFO
            self.factuality_ratio = 0.0
            return

        supported = sum(
            1
            for v in self.verifications
            if v.verdict == VerificationVerdict.SUPPORTED
        )
        refuted = sum(
            1
            for v in self.verifications
            if v.verdict == VerificationVerdict.REFUTED
        )

        self.factuality_ratio = supported / len(self.verifications)

        if refuted > 0:
            self.overall_verdict = VerificationVerdict.REFUTED
        elif supported == len(self.verifications):
            self.overall_verdict = VerificationVerdict.SUPPORTED
        else:
            self.overall_verdict = VerificationVerdict.NOT_ENOUGH_INFO


class ClaimVerifier:
    """
    Verifier for extracting and checking claims against evidence.

    Uses NLI models to verify entailment between claims and retrieved evidence.
    """

    def __init__(
        self,
        nli_adapter: Optional["NLIAdapter"] = None,
        llm_engine: Optional[str] = None,
        entailment_threshold: float = 0.7,
        contradiction_threshold: float = 0.7,
    ):
        """
        Initialize the claim verifier.

        Args:
            nli_adapter: NLI adapter for entailment checking
            llm_engine: LLM engine for claim extraction (if not using existing claims)
            entailment_threshold: Threshold for SUPPORTED verdict
            contradiction_threshold: Threshold for REFUTED verdict
        """
        self.nli_adapter = nli_adapter
        self.llm_engine = llm_engine
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold

    async def extract_claims(self, utterance: str) -> list[str]:
        """
        Extract verifiable claims from an utterance.

        Args:
            utterance: The agent utterance

        Returns:
            List of extracted claims
        """
        if self.llm_engine is None:
            return self._simple_claim_extraction(utterance)

        try:
            from chainlite import llm_generation_chain
        except ImportError:
            return self._simple_claim_extraction(utterance)

        prompt = f"""Extract factual claims from the following text. Each claim should be:
1. A complete, self-contained statement
2. Verifiable (not an opinion or subjective statement)
3. Atomic (one fact per claim)

Text: {utterance}

List each claim on a separate line, starting with "- ". 
If there are no verifiable claims, respond with "None"."""

        chain = llm_generation_chain(
            template_file=None,
            engine=self.llm_engine,
            max_tokens=500,
        )

        response = await chain.ainvoke({"text": prompt})

        claims = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("- "):
                claim = line[2:].strip()
                if claim and claim.lower() != "none":
                    claims.append(claim)

        return claims if claims else self._simple_claim_extraction(utterance)

    def _simple_claim_extraction(self, utterance: str) -> list[str]:
        """
        Simple sentence-based claim extraction fallback.

        Args:
            utterance: The agent utterance

        Returns:
            List of sentences as claims
        """
        utterance = re.sub(r"\[\d+\]", "", utterance)

        sentences = re.split(r"(?<=[.!?])\s+", utterance.strip())

        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and not sentence.endswith("?"):
                claims.append(sentence)

        return claims

    async def verify_claim(
        self,
        claim: str,
        evidence_list: list[dict],
    ) -> ClaimVerification:
        """
        Verify a single claim against evidence.

        Args:
            claim: The claim to verify
            evidence_list: List of evidence dicts with title, content, url

        Returns:
            ClaimVerification result
        """
        evidence_matches = []
        max_entailment = 0.0
        max_contradiction = 0.0

        for evidence in evidence_list:
            content = evidence.get("content", "")
            if not content:
                continue

            if self.nli_adapter is not None:
                nli_result = await self.nli_adapter.predict(
                    premise=content, hypothesis=claim
                )

                entailment_score = nli_result.entailment_prob
                contradiction_score = nli_result.contradiction_prob
            else:
                entailment_score = self._keyword_overlap_score(claim, content)
                contradiction_score = 0.0

            max_entailment = max(max_entailment, entailment_score)
            max_contradiction = max(max_contradiction, contradiction_score)

            if entailment_score > 0.3:
                evidence_matches.append(
                    EvidenceMatch(
                        document_title=evidence.get("title", "Unknown"),
                        document_content=content[:500],
                        relevance_score=evidence.get("score", 0.0),
                        entailment_score=entailment_score,
                        url=evidence.get("url"),
                    )
                )

        if max_entailment >= self.entailment_threshold:
            verdict = VerificationVerdict.SUPPORTED
            confidence = max_entailment
        elif max_contradiction >= self.contradiction_threshold:
            verdict = VerificationVerdict.REFUTED
            confidence = max_contradiction
        else:
            verdict = VerificationVerdict.NOT_ENOUGH_INFO
            confidence = 1.0 - max(max_entailment, max_contradiction)

        return ClaimVerification(
            claim=claim,
            evidence=evidence_matches,
            verdict=verdict,
            confidence=confidence,
        )

    def _keyword_overlap_score(self, claim: str, evidence: str) -> float:
        """
        Simple keyword overlap score for fallback verification.

        Args:
            claim: The claim text
            evidence: The evidence text

        Returns:
            Overlap score between 0 and 1
        """
        claim_words = set(claim.lower().split())
        evidence_words = set(evidence.lower().split())

        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "but", "and", "or", "if",
            "because", "until", "while", "this", "that", "these", "those",
        }

        claim_words = claim_words - stopwords
        evidence_words = evidence_words - stopwords

        if not claim_words:
            return 0.0

        overlap = len(claim_words.intersection(evidence_words))
        return overlap / len(claim_words)

    async def verify_utterance(
        self,
        utterance: str,
        evidence_list: list[dict],
        existing_claims: Optional[list[str]] = None,
    ) -> VerificationResult:
        """
        Verify all claims in an utterance.

        Args:
            utterance: The agent utterance
            evidence_list: List of evidence dicts
            existing_claims: Pre-extracted claims (optional)

        Returns:
            Complete VerificationResult
        """
        if existing_claims is not None:
            claims = existing_claims
        else:
            claims = await self.extract_claims(utterance)

        verifications = []
        for claim in claims:
            verification = await self.verify_claim(claim, evidence_list)
            verifications.append(verification)

        result = VerificationResult(
            utterance=utterance,
            claims=claims,
            verifications=verifications,
        )
        result.compute_overall()

        return result

    async def verify_dialogue_turn(
        self,
        agent_utterance: str,
        filtered_search_results: list,
        llm_claims: Optional[list[str]] = None,
    ) -> VerificationResult:
        """
        Verify claims in a dialogue turn using WikiChat's data structures.

        Args:
            agent_utterance: The agent's response
            filtered_search_results: List of SearchResultBlock objects
            llm_claims: Pre-extracted claims from WikiChat pipeline

        Returns:
            VerificationResult for the turn
        """
        evidence_list = []
        for result in filtered_search_results:
            evidence_list.append(
                {
                    "title": getattr(result, "full_title", ""),
                    "content": getattr(result, "content", ""),
                    "url": getattr(result, "url", None),
                    "score": getattr(result, "probability_score", 0.0),
                }
            )

        return await self.verify_utterance(
            utterance=agent_utterance,
            evidence_list=evidence_list,
            existing_claims=llm_claims,
        )

