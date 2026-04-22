"""Integration-ish tests that simulate how Stage 5 judges will consume
the provider abstraction: build a request, ask for a strict structured
output, and get a validated pydantic instance back.

Uses the mock client so the test is fully deterministic and offline.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from src.core.exceptions import ProviderError, ProviderTimeoutError
from src.llm import LLMRequest, MockLLMClient, RetryPolicy


class JudgeOutput(BaseModel):
    """A representative judge output schema.

    Real judge schemas live in src/judges/ in Stage 5; this mimics the
    shape so we are sure the provider layer is capable of delivering it.
    """

    pillar: str
    score: int = Field(..., ge=1, le=5)
    confidence: float = Field(..., ge=0.0, le=1.0)
    decision_summary: str
    failure_tags: list[str] = Field(default_factory=list)
    why_not_higher: str
    why_not_lower: str


def _request() -> LLMRequest:
    return LLMRequest(
        system_prompt="You are an evaluation judge.",
        user_prompt="Score the response.",
        retry=RetryPolicy(max_attempts=3, initial_backoff_s=0.0, jitter_s=0.0),
    )


def test_judge_style_structured_call_returns_validated_model() -> None:
    script = [
        JudgeOutput(
            pillar="factual_accuracy",
            score=4,
            confidence=0.85,
            decision_summary="Mostly correct.",
            failure_tags=["minor_inaccuracy"],
            why_not_higher="One minor unsupported claim.",
            why_not_lower="Core answer remains accurate.",
        )
    ]
    client = MockLLMClient(structured_script=script)

    resp = client.generate_structured(_request(), JudgeOutput)

    assert isinstance(resp.parsed, JudgeOutput)
    assert resp.parsed.score == 4
    assert "minor_inaccuracy" in resp.parsed.failure_tags
    assert resp.attempts == 1


def test_judge_style_call_handles_dict_payload() -> None:
    client = MockLLMClient(
        structured_script=[
            {
                "pillar": "relevance",
                "score": 5,
                "confidence": 0.99,
                "decision_summary": "Perfect.",
                "failure_tags": [],
                "why_not_higher": "Capped at 5.",
                "why_not_lower": "Nothing missing.",
            }
        ]
    )
    resp = client.generate_structured(_request(), JudgeOutput)
    assert resp.parsed.pillar == "relevance"
    assert resp.parsed.score == 5


def test_judge_style_call_surfaces_validation_error() -> None:
    client = MockLLMClient(
        structured_script=[
            {
                "pillar": "relevance",
                "score": 7,  # out of range
                "confidence": 0.5,
                "decision_summary": "Bad",
                "failure_tags": [],
                "why_not_higher": "",
                "why_not_lower": "",
            }
        ]
    )
    with pytest.raises(ProviderError):
        client.generate_structured(_request(), JudgeOutput)


def test_judge_style_call_retries_transient_failures() -> None:
    good = JudgeOutput(
        pillar="toxicity",
        score=5,
        confidence=0.95,
        decision_summary="Not toxic.",
        failure_tags=[],
        why_not_higher="Capped at 5.",
        why_not_lower="No toxic content detected.",
    )
    client = MockLLMClient(
        structured_script=[
            ProviderTimeoutError("slow"),
            ProviderTimeoutError("slow again"),
            good,
        ]
    )
    resp = client.generate_structured(_request(), JudgeOutput)
    assert resp.parsed.score == 5
    assert resp.attempts == 3


def test_judge_style_call_bails_after_max_attempts() -> None:
    client = MockLLMClient(structured_script=[ProviderTimeoutError("slow") for _ in range(3)])
    with pytest.raises(ProviderTimeoutError):
        client.generate_structured(_request(), JudgeOutput)
    assert client.structured_script_remaining == 0
