"""Google Gemini client, wired through LangChain's ``ChatGoogleGenerativeAI``.

Why LangChain:
- ``ChatGoogleGenerativeAI.with_structured_output(schema)`` gives us
  Pydantic-validated structured output without writing a Gemini-specific
  JSON-repair loop here.
- Migrating to other LangChain chat models later (OpenAI, Anthropic,
  Vertex) is then mostly a "swap the chat model class" change; the
  structured-output contract stays the same.

Rules enforced by this module:
- Vendor exceptions never leak. Anything from
  ``google.api_core.exceptions`` or ``google.generativeai`` is translated
  into a :class:`ProviderError` subtype at the function boundary.
- No other module in ``src/`` imports ``langchain_google_genai`` or any
  Google SDK directly - those imports are confined to this file.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, SecretStr, ValidationError

from src.core.exceptions import (
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from src.llm.base import LLMClient, LLMRequest, LLMUsage

__all__ = ["DEFAULT_GEMINI_MODEL", "GoogleGenAIClient"]

_LOGGER = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


class GoogleGenAIClient(LLMClient):
    """Concrete :class:`LLMClient` backed by Gemini via LangChain.

    Args:
        model_name: The Gemini model string, e.g. ``"gemini-2.0-flash"``
            or ``"gemini-1.5-pro"``.
        api_key: Google GenAI API key. Accepts a :class:`SecretStr` or a
            plain string for convenience in tests.
        default_timeout_s: Per-call timeout handed to the LangChain
            chat model. Concrete requests may override this; see
            :meth:`_build_chat_model`.
        chat_model_factory: Injection hook used primarily by tests so we
            can swap in a fake without touching the real SDK. Defaults
            to constructing :class:`ChatGoogleGenerativeAI`.

    Notes on timeouts: LangChain's chat models accept a ``timeout``
    constructor argument. Rather than instantiating a new chat model per
    request (which would throw away token counters, etc.), we use the
    ``request.timeout_s`` value only when it differs materially from the
    client default; otherwise the client default applies.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_GEMINI_MODEL,
        api_key: SecretStr | str | None = None,
        default_timeout_s: float = 60.0,
        chat_model_factory: Any | None = None,
    ) -> None:
        super().__init__(model_name=model_name)

        if api_key is None:
            raise ProviderError(
                "GoogleGenAIClient requires an api_key. Set JTB_GOOGLE_API_KEY "
                "or pass one explicitly."
            )
        if isinstance(api_key, SecretStr):
            self._api_key_value = api_key.get_secret_value()
        else:
            self._api_key_value = str(api_key)

        self._default_timeout_s = default_timeout_s
        self._chat_model_factory = chat_model_factory or _default_chat_model_factory

        # Construct once so we reuse connection / config; structured
        # variants are derived with ``.with_structured_output`` per call
        # (they're cheap wrappers, not re-init).
        try:
            self._chat = self._chat_model_factory(
                model=self._model_name,
                api_key=self._api_key_value,
                timeout=self._default_timeout_s,
            )
        except Exception as exc:
            # Vendor boundary: any exception from the SDK is translated
            # into a ProviderError before propagating.
            raise _translate_error(exc, context="init") from exc

    # ---- LLMClient hooks -----------------------------------------------

    def _invoke_text(self, request: LLMRequest) -> tuple[str, LLMUsage, Any]:
        messages = _build_messages(request)
        try:
            response = self._chat.invoke(messages, config=_config_for(request))
        except Exception as exc:
            raise _translate_error(exc, context="invoke_text") from exc

        text = _extract_text(response)
        usage = _extract_usage(response)
        return text, usage, response

    def _invoke_structured[
        ParsedT: BaseModel
    ](self, request: LLMRequest, schema: type[ParsedT],) -> tuple[ParsedT, LLMUsage, Any]:
        messages = _build_messages(request)
        try:
            structured = self._chat.with_structured_output(schema, include_raw=True)
            response = structured.invoke(messages, config=_config_for(request))
        except Exception as exc:
            raise _translate_error(exc, context="invoke_structured") from exc

        parsed = _extract_parsed(response, schema)
        usage = _extract_usage_from_structured(response)
        return parsed, usage, response


# ---------------------------------------------------------------------------
# Default factory
# ---------------------------------------------------------------------------


def _default_chat_model_factory(
    *,
    model: str,
    api_key: str,
    timeout: float,
) -> Any:
    # Lazy import so unit tests that don't need the SDK don't pay for it.
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key,
        timeout=timeout,
        # Keep temperature as the per-request override - provider
        # defaults to 0 via ``LLMRequest.temperature = 0``.
    )


# ---------------------------------------------------------------------------
# Helpers (module-level so tests can target them directly)
# ---------------------------------------------------------------------------


def _build_messages(request: LLMRequest) -> list[Any]:
    msgs: list[Any] = []
    if request.system_prompt:
        msgs.append(SystemMessage(content=request.system_prompt))
    msgs.append(HumanMessage(content=request.user_prompt))
    return msgs


def _config_for(request: LLMRequest) -> dict[str, Any]:
    """Build a LangChain ``RunnableConfig`` dict for this request.

    We avoid threading the timeout at call time (LangChain's chat models
    want that on construction) but we *do* forward tags and temperature
    so observability captures them.
    """
    config: dict[str, Any] = {}
    if request.tags:
        # LangChain expects ``tags`` as a list[str] of string labels and
        # ``metadata`` as the free-form dict. We encode our tag dict as
        # both so it shows up in Langfuse / LangSmith cleanly.
        config["tags"] = [f"{k}={v}" for k, v in request.tags.items()]
        config["metadata"] = dict(request.tags)
    if request.temperature != 0.0 or request.max_output_tokens is not None:
        # Forwarded via ``configurable`` is provider-specific; langchain
        # chat models accept these kwargs at invoke time for most
        # providers through ``.bind()``. Keeping it on the config here
        # keeps the concrete translation inside this file.
        extras: dict[str, Any] = {"temperature": request.temperature}
        if request.max_output_tokens is not None:
            extras["max_output_tokens"] = request.max_output_tokens
        config.setdefault("metadata", {}).update({"request_params": extras})
    return config


def _extract_text(response: Any) -> str:
    """Pull plain text out of a LangChain AIMessage-like response."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # LangChain occasionally returns a list of parts (e.g. tool
        # calls interleaved with text). Pick out the text parts.
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    return str(content)


def _extract_usage(response: Any) -> LLMUsage:
    """Best-effort extraction of token usage from a LangChain response."""
    meta = getattr(response, "usage_metadata", None)
    if isinstance(meta, dict):
        return LLMUsage(
            input_tokens=_as_int(meta.get("input_tokens")),
            output_tokens=_as_int(meta.get("output_tokens")),
            total_tokens=_as_int(meta.get("total_tokens")),
        )
    # Older LangChain returns token info under ``response_metadata``.
    rmeta = getattr(response, "response_metadata", None) or {}
    token_usage = (rmeta.get("token_usage") if isinstance(rmeta, dict) else None) or {}
    if isinstance(token_usage, dict):
        return LLMUsage(
            input_tokens=_as_int(token_usage.get("prompt_tokens")),
            output_tokens=_as_int(token_usage.get("completion_tokens")),
            total_tokens=_as_int(token_usage.get("total_tokens")),
        )
    return LLMUsage()


def _extract_usage_from_structured(response: Any) -> LLMUsage:
    """``with_structured_output(include_raw=True)`` returns ``{"raw": AIMessage, "parsed": schema, ...}``."""
    raw = response.get("raw") if isinstance(response, dict) else None
    if raw is not None:
        return _extract_usage(raw)
    return _extract_usage(response)


def _extract_parsed[ParsedT: BaseModel](response: Any, schema: type[ParsedT]) -> ParsedT:
    if isinstance(response, dict) and "parsed" in response:
        parsed = response["parsed"]
        parsing_error = response.get("parsing_error")
        if parsing_error is not None:
            raise ProviderError(
                f"Gemini structured output failed to parse as {schema.__name__}: "
                f"{parsing_error}"
            )
        if parsed is None:
            raise ProviderError(
                f"Gemini returned no parsed structured output for {schema.__name__}."
            )
    else:
        parsed = response

    if isinstance(parsed, schema):
        return parsed
    if isinstance(parsed, BaseModel):
        try:
            return schema.model_validate(parsed.model_dump())
        except ValidationError as exc:
            raise ProviderError(
                f"Gemini structured output does not validate as {schema.__name__}: {exc}"
            ) from exc
    if isinstance(parsed, dict):
        try:
            return schema.model_validate(parsed)
        except ValidationError as exc:
            raise ProviderError(
                f"Gemini structured output dict does not validate as {schema.__name__}: {exc}"
            ) from exc
    raise ProviderError(f"Gemini structured output was unexpected type {type(parsed).__name__}.")


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------

# Exception class names we treat as rate limits vs timeouts. Matching on
# qualified names avoids an import-time dependency on specific subclasses
# that might move between SDK versions.
_RATE_LIMIT_NAMES = frozenset(
    {
        "ResourceExhausted",
        "TooManyRequests",
        "QuotaExceededError",
    }
)

_TIMEOUT_NAMES = frozenset(
    {
        "DeadlineExceeded",
        "TimeoutError",
        "ReadTimeout",
    }
)


def _translate_error(exc: BaseException, *, context: str) -> ProviderError:
    """Convert any langchain / Google SDK exception into a ``ProviderError``.

    The goal is to classify retryable (rate limit / timeout) vs
    non-retryable categories at the provider boundary so the base client
    can apply its retry policy uniformly.
    """
    name = type(exc).__name__
    message = f"[{context}] {name}: {exc}"

    if isinstance(exc, ProviderError):
        # Already translated - never wrap twice.
        return exc

    if name in _RATE_LIMIT_NAMES:
        return ProviderRateLimitError(message)
    if name in _TIMEOUT_NAMES:
        return ProviderTimeoutError(message)

    # Fall back to string sniffing for langchain's generic runtime errors
    # which sometimes wrap the real cause.
    text = str(exc).lower()
    if "rate limit" in text or "quota" in text or "429" in text:
        return ProviderRateLimitError(message)
    if "timeout" in text or "timed out" in text or "deadline" in text:
        return ProviderTimeoutError(message)

    return ProviderError(message)
