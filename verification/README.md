# Hallucination Detection Module

A post-hoc verification module for detecting and scoring hallucinations in WikiChat responses.

## Overview

This module provides tools for:
- Extracting verifiable claims from agent utterances
- Verifying claims against retrieved evidence using NLI models
- Tracking factuality scores across dialogue sessions
- Generating factuality reports

## Installation

The verification module uses dependencies already included in WikiChat:
- `pydantic` for data validation

Optional dependencies for NLI-based verification:
- `transformers` for HuggingFace NLI models

## Usage

### Basic Claim Verification

```python
import asyncio
from verification.claim_verifier import ClaimVerifier
from verification.nli_adapter import create_nli_adapter

# Create verifier with NLI support
nli_adapter = create_nli_adapter(
    model_name="microsoft/deberta-v3-base-mnli-fever-anli",
    device="cpu",  # or "cuda" for GPU
)
verifier = ClaimVerifier(nli_adapter=nli_adapter)

# Prepare evidence from retrieval
evidence = [
    {
        "title": "Python (programming language)",
        "content": "Python is a high-level programming language created by Guido van Rossum.",
        "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
    }
]

# Verify an utterance
result = asyncio.run(verifier.verify_utterance(
    utterance="Python is a programming language created by Guido van Rossum in 1991.",
    evidence_list=evidence,
))

print(f"Overall verdict: {result.overall_verdict}")
print(f"Factuality ratio: {result.factuality_ratio:.2%}")
for v in result.verifications:
    print(f"  - {v.claim}: {v.verdict} ({v.confidence:.2%})")
```

### Integration with WikiChat Pipeline

```python
from verification.claim_verifier import ClaimVerifier
from verification.nli_adapter import MockNLIAdapter  # or NLIAdapter for real NLI

# Use with dialogue turn data
async def verify_turn(dialogue_turn):
    verifier = ClaimVerifier(nli_adapter=MockNLIAdapter())
    
    result = await verifier.verify_dialogue_turn(
        agent_utterance=dialogue_turn.agent_utterance,
        filtered_search_results=dialogue_turn.filtered_search_results,
        llm_claims=dialogue_turn.llm_claims,  # Reuse existing claims
    )
    return result
```

### Factuality Scoring and Tracking

```python
from verification.factuality_scorer import FactualityScorer
from verification.claim_verifier import ClaimVerifier

# Create scorer for a dialogue session
scorer = FactualityScorer(dialogue_id="session_123")

# For each turn, score the response
for turn in dialogue_turns:
    verification_result = await verifier.verify_utterance(
        utterance=turn.agent_utterance,
        evidence_list=evidence,
    )
    
    score = scorer.score_response(
        verification_result=verification_result,
        user_utterance=turn.user_utterance,
    )
    print(f"Turn factuality: {score.factuality_ratio:.2%}")

# Get overall statistics
stats = scorer.get_statistics()
print(f"Overall dialogue factuality: {stats['overall_score']:.2%}")
print(f"Total claims: {stats['total_claims']}")

# Get problematic claims for review
problematic = scorer.get_problematic_claims()
for claim in problematic:
    print(f"[{claim['verdict']}] {claim['claim']}")

# Save factuality data
scorer.save("factuality_logs/session_123.json")

# Generate human-readable report
print(scorer.generate_report())
```

### Using Different NLI Models

```python
from verification.nli_adapter import NLIAdapter, MockNLIAdapter

# Use a real NLI model
real_adapter = NLIAdapter(
    model_name="microsoft/deberta-v3-base-mnli-fever-anli",
    device="cuda",  # or "cpu"
    batch_size=16,
)

# For testing without model loading
mock_adapter = MockNLIAdapter(
    default_prediction=NLIPrediction.NEUTRAL,
    default_confidence=0.8,
)

# Configure specific responses for testing
mock_adapter.set_response(
    "Python was created by Guido",
    NLIPrediction.ENTAILMENT,
    confidence=0.95,
)
```

### Aggregating Across Multiple Dialogues

```python
from verification.factuality_scorer import aggregate_factuality_scores

scorers = [scorer1, scorer2, scorer3]  # FactualityScorer instances
aggregated = aggregate_factuality_scores(scorers)

print(f"Total dialogues: {aggregated['num_dialogues']}")
print(f"Overall factuality: {aggregated['overall_factuality']:.2%}")
print(f"Average per dialogue: {aggregated['average_score_per_dialogue']:.2%}")
```

## Verification Verdicts

| Verdict | Description | Interpretation |
|---------|-------------|----------------|
| SUPPORTED | Claim is entailed by evidence | Factual |
| REFUTED | Claim contradicts evidence | Hallucination |
| NOT_ENOUGH_INFO | Insufficient evidence | Uncertain |

## File Structure

```
verification/
    __init__.py           # Module exports
    claim_verifier.py     # Claim extraction and verification
    factuality_scorer.py  # Scoring and tracking
    nli_adapter.py        # NLI model interface
    README.md             # This file
```

## NLI Models

Recommended models for verification:
- `microsoft/deberta-v3-base-mnli-fever-anli` (default, best accuracy)
- `facebook/bart-large-mnli` (good balance of speed/accuracy)
- `cross-encoder/nli-deberta-v3-small` (faster, smaller)

## Testing

```bash
pytest tests/test_verification.py -v
```

## Research Applications

This module enables:
- **Factuality Metrics**: Quantifiable hallucination rates per response
- **Self-Verification**: Flag uncertain statements for user review
- **Ablation Studies**: Measure impact of different pipeline configurations
- **Benchmark Evaluation**: Compare factuality across models/settings

