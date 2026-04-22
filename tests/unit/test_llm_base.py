"""Tests for :mod:`src.llm.base`.

Exercises:
- Request / retry policy validation.
- The base class retry engine with a counting subclass that records
  attempts and raises scripted exceptions.
- Latency timing populates a real number.
- Non-retryable errors surface immediately.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from src.core.exceptions import (
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from src.llm import base as base_module
from src.llm.base import (
    LLMClient,
    LLMRequest,
    LLMUsage,
    RetryPolicy,
)


class _Schema(BaseModel):
    value: int


class _CountingClient(LLMClient):
    """Subclass that replays scripted responses/exceptions per call."""

    def __init__(
        self,
        *,
        text_script: list[Any] | None = None,
        structured_script: list[Any] | None = None,
    ) -> None:
        super().__init__(model_name="counting-v0")
        self.text_script = list(text_script or [])
        self.structured_script = list(structured_script or [])
        self.text_calls = 0
        self.structured_calls = 0

    def _invoke_text(self, request: LLMRequest) -> tuple[str, LLMUsage, Any]:
        self.text_calls += 1
        item = self.text_script.pop(0)
        if isinstance(item, Exception):
            raise item
        return (
            str(item),
            LLMUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            {"attempt": self.text_calls},
        )

    def _invoke_structured(
        self,
        request: LLMRequest,
        schema: type[Any],
    ) -> tuple[Any, LLMUsage, Any]:
        self.structured_calls += 1
        item = self.structured_script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, schema):
            parsed: Any = item
        else:
            parsed = schema.model_validate(item)
        return parsed, LLMUsage(), {"attempt": self.structured_calls}


@pytest.fixture(autouse=True)
def _nosleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make retry backoff instant for tests."""
    monkeypatch.setattr(base_module, "_sleep", lambda _seconds: None)


# ---------------------------------------------------------------------------
# RetryPolicy / LLMRequest model validation
# ---------------------------------------------------------------------------


def test_retry_policy_defaults_are_reasonable() -> None:
    p = RetryPolicy()
    assert p.max_attempts == 3
    assert p.initial_backoff_s > 0
    assert p.max_backoff_s >= p.initial_backoff_s


def test_retry_policy_rejects_negative_backoff() -> None:
    with pytest.raises(ValidationError):
        RetryPolicy(initial_backoff_s=-0.1)


def test_retry_policy_rejects_zero_max_attempts() -> None:
    with pytest.raises(ValidationError):
        RetryPolicy(max_attempts=0)


def test_llm_request_requires_user_prompt() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(user_prompt="")


def test_llm_request_temperature_bounds() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(user_prompt="hi", temperature=-0.1)
    with pytest.raises(ValidationError):
        LLMRequest(user_prompt="hi", temperature=2.5)


def test_llm_request_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        LLMRequest.model_validate({"user_prompt": "hi", "rogue_field": "nope"})


# ---------------------------------------------------------------------------
# Retry engine
# ---------------------------------------------------------------------------


def test_generate_text_succeeds_on_first_attempt() -> None:
    client = _CountingClient(text_script=["hello"])
    request = LLMRequest(user_prompt="hi")
    resp = client.generate_text(request)
    assert resp.text == "hello"
    assert resp.attempts == 1
    assert resp.latency_ms >= 0.0
    assert resp.usage.total_tokens == 2
    assert resp.model_name == "counting-v0"


def test_generate_text_retries_on_rate_limit_then_succeeds() -> None:
    client = _CountingClient(
        text_script=[ProviderRateLimitError("429"), "hello"],
    )
    request = LLMRequest(user_prompt="hi")
    resp = client.generate_text(request)
    assert resp.text == "hello"
    assert resp.attempts == 2
    assert client.text_calls == 2


def test_generate_text_retries_on_timeout_then_succeeds() -> None:
    client = _CountingClient(
        text_script=[ProviderTimeoutError("slow"), "hello"],
    )
    resp = client.generate_text(LLMRequest(user_prompt="hi"))
    assert resp.text == "hello"
    assert resp.attempts == 2


def test_generate_text_bails_after_exhausting_attempts() -> None:
    errs = [ProviderRateLimitError("429") for _ in range(3)]
    client = _CountingClient(text_script=errs)
    req = LLMRequest(
        user_prompt="hi",
        retry=RetryPolicy(max_attempts=3, initial_backoff_s=0.0, jitter_s=0.0),
    )
    with pytest.raises(ProviderRateLimitError):
        client.generate_text(req)
    assert client.text_calls == 3


def test_generate_text_does_not_retry_on_non_retryable_provider_error() -> None:
    client = _CountingClient(text_script=[ProviderError("bad request")])
    with pytest.raises(ProviderError):
        client.generate_text(LLMRequest(user_prompt="hi"))
    assert client.text_calls == 1


def test_generate_structured_returns_validated_instance() -> None:
    client = _CountingClient(structured_script=[{"value": 42}])
    resp = client.generate_structured(LLMRequest(user_prompt="hi"), _Schema)
    assert isinstance(resp.parsed, _Schema)
    assert resp.parsed.value == 42
    assert resp.attempts == 1


def test_generate_structured_retries_on_timeout() -> None:
    client = _CountingClient(structured_script=[ProviderTimeoutError("slow"), _Schema(value=7)])
    resp = client.generate_structured(LLMRequest(user_prompt="hi"), _Schema)
    assert resp.parsed.value == 7
    assert resp.attempts == 2


# ---------------------------------------------------------------------------
# Backoff helper
# ---------------------------------------------------------------------------


def test_backoff_seconds_is_bounded_by_cap() -> None:
    for attempt in range(1, 10):
        got = base_module._backoff_seconds(
            attempt=attempt,
            initial=1.0,
            multiplier=2.0,
            cap=3.0,
            jitter=0.0,
        )
        assert got <= 3.0


def test_backoff_seconds_with_no_jitter_is_deterministic() -> None:
    a = base_module._backoff_seconds(attempt=2, initial=1.0, multiplier=2.0, cap=10.0, jitter=0.0)
    b = base_module._backoff_seconds(attempt=2, initial=1.0, multiplier=2.0, cap=10.0, jitter=0.0)
    assert a == b == 2.0


def test_model_name_must_be_nonempty() -> None:
    """Subclasses must provide a non-empty model name via super().__init__."""

    class _Bad(LLMClient):
        def _invoke_text(
            self, request: LLMRequest
        ) -> tuple[str, LLMUsage, Any]:  # pragma: no cover
            raise NotImplementedError

        def _invoke_structured(
            self,
            request: LLMRequest,
            schema: type[Any],
        ) -> tuple[Any, LLMUsage, Any]:  # pragma: no cover
            raise NotImplementedError

    with pytest.raises(ValueError, match="model_name"):
        _Bad(model_name="")
