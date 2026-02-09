"""
Hallucination Detection Module for WikiChat.

This module provides tools for post-hoc verification of generated claims
against retrieved evidence, including:
- Claim extraction and verification
- NLI-based entailment checking
- Factuality scoring and tracking
"""

from verification.claim_verifier import (
    ClaimVerifier,
    ClaimVerification,
    VerificationVerdict,
)
from verification.factuality_scorer import (
    FactualityScorer,
    FactualityScore,
    DialogueFactuality,
)
from verification.nli_adapter import (
    NLIAdapter,
    NLIResult,
    NLIPrediction,
)

__all__ = [
    "ClaimVerifier",
    "ClaimVerification",
    "VerificationVerdict",
    "FactualityScorer",
    "FactualityScore",
    "DialogueFactuality",
    "NLIAdapter",
    "NLIResult",
    "NLIPrediction",
]

