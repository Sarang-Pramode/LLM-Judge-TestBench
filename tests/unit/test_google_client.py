"""Tests for :mod:`src.llm.google_client`.

These tests never touch the real Gemini API. Instead we inject a
``chat_model_factory`` that returns a stand-in chat model with the
LangChain ``invoke`` / ``with_structured_output`` shape we care about,
then assert on:

- Message construction (system + user prompts -> LangChain messages).
- Text extraction from ``AIMessage``-like responses.
- Structured output via ``with_structured_output(include_raw=True)``.
- Error translation (rate limit / timeout / generic).
- Usage metadata extraction.

The error-translation helpers are tested directly since they are pure
functions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, SecretStr

from src.core.exceptions import (
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from src.llm import LLMRequest, LLMUsage, RetryPolicy
from src.llm.google_client import (
    GoogleGenAIClient,
    _build_messages,
    _extract_parsed,
    _extract_text,
    _extract_usage,
    _translate_error,
)


class _Out(BaseModel):
    decision: str
    confidence: float


class _FakeAIMessage:
    def __init__(
        self,
        content: Any = "",
        usage_metadata: dict[str, int] | None = None,
    ) -> None:
        self.content = content
        self.usage_metadata = usage_metadata
        self.response_metadata: dict[str, Any] = {}


class _FakeStructuredRunnable:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def invoke(self, _messages: Any, config: Any = None) -> Any:
        return self._payload


class _FakeChat:
    def __init__(
        self,
        *,
        invoke_fn: Callable[[Any], Any] | None = None,
        structured_payload: Any | None = None,
        structured_error: Exception | None = None,
    ) -> None:
        self._invoke_fn = invoke_fn
        self._structured_payload = structured_payload
        self._structured_error = structured_error
        self.invoke_calls: list[Any] = []
        self.structured_requests: list[tuple[type[BaseModel], bool]] = []

    def invoke(self, messages: Any, config: Any = None) -> Any:
        self.invoke_calls.append({"messages": messages, "config": config})
        if self._invoke_fn is None:
            return _FakeAIMessage(content="default")
        return self._invoke_fn(messages)

    def with_structured_output(
        self,
        schema: type[BaseModel],
        *,
        include_raw: bool = False,
    ) -> Any:
        self.structured_requests.append((schema, include_raw))
        if self._structured_error is not None:
            err = self._structured_error
            self._structured_error = None

            class _Raiser:
                def invoke(self, _m: Any, config: Any = None) -> Any:
                    raise err

            return _Raiser()
        return _FakeStructuredRunnable(self._structured_payload)


def _factory_for(chat: _FakeChat) -> Callable[..., _FakeChat]:
    def _factory(**_kwargs: Any) -> _FakeChat:
        return chat

    return _factory


def _req(**overrides: Any) -> LLMRequest:
    defaults: dict[str, Any] = {
        "user_prompt": "hi",
        "retry": RetryPolicy(max_attempts=2, initial_backoff_s=0.0, jitter_s=0.0),
    }
    defaults.update(overrides)
    return LLMRequest(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_requires_api_key() -> None:
    with pytest.raises(ProviderError, match="api_key"):
        GoogleGenAIClient(api_key=None, chat_model_factory=_factory_for(_FakeChat()))


def test_accepts_plain_string_or_secret_str() -> None:
    fake = _FakeChat()
    c1 = GoogleGenAIClient(api_key="plain-key", chat_model_factory=_factory_for(fake))
    c2 = GoogleGenAIClient(api_key=SecretStr("secret-key"), chat_model_factory=_factory_for(fake))
    assert c1.model_name  # smoke
    assert c2.model_name


def test_construction_errors_translate_to_provider_error() -> None:
    def boom(**_kwargs: Any) -> Any:
        raise RuntimeError("simulated chat model init failure")

    with pytest.raises(ProviderError, match="simulated chat model init"):
        GoogleGenAIClient(api_key="k", chat_model_factory=boom)


# ---------------------------------------------------------------------------
# Text path
# ---------------------------------------------------------------------------


def test_generate_text_builds_system_and_user_messages() -> None:
    captured: list[Any] = []

    def invoke(messages: Any) -> Any:
        captured.append(messages)
        return _FakeAIMessage(content="ok")

    chat = _FakeChat(invoke_fn=invoke)
    client = GoogleGenAIClient(api_key="k", chat_model_factory=_factory_for(chat))
    resp = client.generate_text(_req(system_prompt="sys", user_prompt="hello"))

    assert resp.text == "ok"
    sent = captured[0]
    assert isinstance(sent[0], SystemMessage)
    assert isinstance(sent[1], HumanMessage)
    assert sent[0].content == "sys"
    assert sent[1].content == "hello"


def test_generate_text_extracts_usage_metadata() -> None:
    chat = _FakeChat(
        invoke_fn=lambda _m: _FakeAIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 3,
                "output_tokens": 5,
                "total_tokens": 8,
            },
        )
    )
    client = GoogleGenAIClient(api_key="k", chat_model_factory=_factory_for(chat))
    resp = client.generate_text(_req())
    assert resp.usage == LLMUsage(input_tokens=3, output_tokens=5, total_tokens=8)


def test_generate_text_falls_back_to_response_metadata_token_usage() -> None:
    msg = _FakeAIMessage(content="ok")
    msg.response_metadata = {
        "token_usage": {"prompt_tokens": 9, "completion_tokens": 11, "total_tokens": 20}
    }
    chat = _FakeChat(invoke_fn=lambda _m: msg)
    client = GoogleGenAIClient(api_key="k", chat_model_factory=_factory_for(chat))
    resp = client.generate_text(_req())
    assert resp.usage.input_tokens == 9
    assert resp.usage.output_tokens == 11


def test_generate_text_translates_rate_limit() -> None:
    class ResourceExhausted(Exception):
        pass

    chat = _FakeChat(invoke_fn=lambda _m: (_ for _ in ()).throw(ResourceExhausted("429")))
    client = GoogleGenAIClient(api_key="k", chat_model_factory=_factory_for(chat))
    with pytest.raises(ProviderRateLimitError):
        client.generate_text(_req(retry=RetryPolicy(max_attempts=1)))


def test_generate_text_translates_timeout() -> None:
    class DeadlineExceeded(Exception):
        pass

    chat = _FakeChat(invoke_fn=lambda _m: (_ for _ in ()).throw(DeadlineExceeded("slow")))
    client = GoogleGenAIClient(api_key="k", chat_model_factory=_factory_for(chat))
    with pytest.raises(ProviderTimeoutError):
        client.generate_text(_req(retry=RetryPolicy(max_attempts=1)))


def test_generate_text_translates_generic_error() -> None:
    chat = _FakeChat(invoke_fn=lambda _m: (_ for _ in ()).throw(RuntimeError("wat")))
    client = GoogleGenAIClient(api_key="k", chat_model_factory=_factory_for(chat))
    with pytest.raises(ProviderError, match="wat"):
        client.generate_text(_req(retry=RetryPolicy(max_attempts=1)))


# ---------------------------------------------------------------------------
# Structured path
# ---------------------------------------------------------------------------


def test_generate_structured_unwraps_parsed_field() -> None:
    raw_ai = _FakeAIMessage(
        content="",
        usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    )
    payload = {
        "raw": raw_ai,
        "parsed": _Out(decision="yes", confidence=0.9),
        "parsing_error": None,
    }
    chat = _FakeChat(structured_payload=payload)
    client = GoogleGenAIClient(api_key="k", chat_model_factory=_factory_for(chat))

    resp = client.generate_structured(_req(), _Out)
    assert resp.parsed.decision == "yes"
    assert resp.parsed.confidence == pytest.approx(0.9)
    assert resp.usage.total_tokens == 3
    # Client should have asked for include_raw=True:
    assert chat.structured_requests == [(_Out, True)]


def test_generate_structured_accepts_dict_parsed_value() -> None:
    payload = {
        "raw": _FakeAIMessage(content=""),
        "parsed": {"decision": "no", "confidence": 0.1},
        "parsing_error": None,
    }
    chat = _FakeChat(structured_payload=payload)
    client = GoogleGenAIClient(api_key="k", chat_model_factory=_factory_for(chat))
    resp = client.generate_structured(_req(), _Out)
    assert resp.parsed.decision == "no"


def test_generate_structured_surfaces_parsing_error_as_provider_error() -> None:
    payload = {
        "raw": _FakeAIMessage(content=""),
        "parsed": None,
        "parsing_error": "expected object got string",
    }
    chat = _FakeChat(structured_payload=payload)
    client = GoogleGenAIClient(api_key="k", chat_model_factory=_factory_for(chat))
    with pytest.raises(ProviderError, match="structured output failed to parse"):
        client.generate_structured(_req(retry=RetryPolicy(max_attempts=1)), _Out)


def test_generate_structured_translates_invocation_error() -> None:
    chat = _FakeChat(structured_error=RuntimeError("api exploded"))
    client = GoogleGenAIClient(api_key="k", chat_model_factory=_factory_for(chat))
    with pytest.raises(ProviderError, match="api exploded"):
        client.generate_structured(_req(retry=RetryPolicy(max_attempts=1)), _Out)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_build_messages_skips_system_when_absent() -> None:
    msgs = _build_messages(_req(user_prompt="hi"))
    assert len(msgs) == 1
    assert isinstance(msgs[0], HumanMessage)


def test_extract_text_handles_list_of_parts() -> None:
    msg = _FakeAIMessage(
        content=[
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"},
        ]
    )
    assert _extract_text(msg) == "hello world"


def test_extract_usage_returns_empty_when_absent() -> None:
    msg = _FakeAIMessage(content="x")
    assert _extract_usage(msg) == LLMUsage()


def test_extract_parsed_rejects_unexpected_types() -> None:
    with pytest.raises(ProviderError):
        _extract_parsed(object(), _Out)


def test_translate_error_never_double_wraps() -> None:
    original = ProviderRateLimitError("existing")
    translated = _translate_error(original, context="x")
    assert translated is original


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ResourceExhausted", ProviderRateLimitError),
        ("DeadlineExceeded", ProviderTimeoutError),
        ("RuntimeError", ProviderError),
    ],
)
def test_translate_error_by_name(name: str, expected: type[ProviderError]) -> None:
    exc = type(name, (Exception,), {})("boom")
    got = _translate_error(exc, context="t")
    assert isinstance(got, expected)


def test_translate_error_by_message_sniff_when_name_is_generic() -> None:
    exc = RuntimeError("Request timed out after 30s")
    assert isinstance(_translate_error(exc, context="t"), ProviderTimeoutError)
    exc2 = RuntimeError("Quota exceeded for requests")
    assert isinstance(_translate_error(exc2, context="t"), ProviderRateLimitError)
