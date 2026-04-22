"""Tests for :class:`src.llm.MockLLMClient`."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from src.core.exceptions import ProviderError, ProviderRateLimitError
from src.llm import LLMRequest, LLMUsage, MockLLMClient, RetryPolicy


class _Schema(BaseModel):
    ok: bool
    count: int


def _req(prompt: str = "hi") -> LLMRequest:
    return LLMRequest(
        user_prompt=prompt,
        retry=RetryPolicy(max_attempts=3, initial_backoff_s=0.0, jitter_s=0.0),
    )


# ---------------------------------------------------------------------------
# Text scripting
# ---------------------------------------------------------------------------


def test_mock_returns_scripted_text_in_order() -> None:
    client = MockLLMClient(text_script=["first", "second"])
    r1 = client.generate_text(_req("a"))
    r2 = client.generate_text(_req("b"))
    assert (r1.text, r2.text) == ("first", "second")
    assert len(client.calls) == 2
    assert [c.request.user_prompt for c in client.calls] == ["a", "b"]


def test_mock_raises_if_text_script_exhausted() -> None:
    client = MockLLMClient(text_script=["only"])
    client.generate_text(_req())
    with pytest.raises(ProviderError, match="no scripted text responses"):
        client.generate_text(_req())


def test_mock_text_fn_overrides_script() -> None:
    def fn(req: LLMRequest) -> str:
        return req.user_prompt.upper()

    client = MockLLMClient(text_fn=fn)
    resp = client.generate_text(_req("hello"))
    assert resp.text == "HELLO"


def test_mock_interleaves_exceptions_with_text_responses() -> None:
    client = MockLLMClient(
        text_script=[ProviderRateLimitError("429"), "ok"],
    )
    resp = client.generate_text(_req())
    assert resp.text == "ok"
    assert resp.attempts == 2


# ---------------------------------------------------------------------------
# Structured scripting
# ---------------------------------------------------------------------------


def test_mock_accepts_schema_instance() -> None:
    client = MockLLMClient(structured_script=[_Schema(ok=True, count=1)])
    out = client.generate_structured(_req(), _Schema)
    assert out.parsed.ok is True
    assert out.parsed.count == 1


def test_mock_accepts_dict_and_validates_schema() -> None:
    client = MockLLMClient(structured_script=[{"ok": False, "count": 2}])
    out = client.generate_structured(_req(), _Schema)
    assert out.parsed.count == 2


def test_mock_rejects_bad_dict_with_provider_error() -> None:
    client = MockLLMClient(structured_script=[{"ok": "not-a-bool"}])
    with pytest.raises(ProviderError, match="does not validate"):
        client.generate_structured(_req(), _Schema)


def test_mock_structured_fn_overrides_script() -> None:
    def fn(req: LLMRequest, schema: type[BaseModel]) -> Any:
        assert schema is _Schema
        return _Schema(ok=True, count=len(req.user_prompt))

    client = MockLLMClient(structured_fn=fn)
    out = client.generate_structured(_req("foo"), _Schema)
    assert out.parsed.count == 3


def test_mock_records_schema_name_on_structured_calls() -> None:
    client = MockLLMClient(structured_script=[_Schema(ok=True, count=0)])
    client.generate_structured(_req(), _Schema)
    assert client.calls[-1].kind == "structured"
    assert client.calls[-1].schema_name == "_Schema"


def test_queue_helpers_append_without_clearing() -> None:
    client = MockLLMClient(text_script=["a"])
    client.queue_text("b")
    assert client.text_script_remaining == 2


def test_reset_clears_recorded_calls_but_keeps_queue() -> None:
    client = MockLLMClient(text_script=["a"])
    client.generate_text(_req())
    assert client.calls
    client.reset()
    assert client.calls == []


def test_mock_usage_is_returned_on_both_paths() -> None:
    client = MockLLMClient(
        text_script=["x"],
        structured_script=[_Schema(ok=True, count=0)],
        usage=LLMUsage(input_tokens=3, output_tokens=5, total_tokens=8),
    )
    text = client.generate_text(_req())
    structured = client.generate_structured(_req(), _Schema)
    assert text.usage.total_tokens == 8
    assert structured.usage.total_tokens == 8
