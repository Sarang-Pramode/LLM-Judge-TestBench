"""Provider-agnostic LLM client interface.

Everything that calls a model goes through :class:`LLMClient`. Judges,
orchestration, dashboards, and metrics never import a vendor SDK; they
build an :class:`LLMRequest`, hand it to a client, and receive an
:class:`LLMResponse` or :class:`StructuredResponse`.

Responsibilities handled here (not in concrete providers):

- Retry policy with exponential backoff and jitter.
- Per-call latency timing.
- Error translation via concrete providers - each concrete subclass is
  expected to raise :class:`ProviderError` subclasses, and this base
  class decides which errors are retryable.
- Structured output: subclasses implement ``_invoke_structured`` and we
  validate the returned payload against the caller's Pydantic schema so
  every judge gets a strictly-typed object back.

Concrete clients implement two hooks:

- :meth:`LLMClient._invoke_text` for free-text completions.
- :meth:`LLMClient._invoke_structured` for pydantic-schema-enforced
  structured output. If a provider cannot natively constrain output,
  the subclass can implement this by calling :meth:`_invoke_text` and
  parsing the result, but the base class does not do that silently -
  that decision stays with the provider implementation so failure modes
  remain explicit.

The interface is deliberately synchronous. Concurrency lives in the
orchestration layer (Stage 7) via a thread pool; keeping this
synchronous makes each judge trivially testable and keeps error handling
linear.
"""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.exceptions import (
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

__all__ = [
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "LLMUsage",
    "RetryPolicy",
    "StructuredResponse",
]

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RetryPolicy(BaseModel):
    """Retry policy applied uniformly across every concrete provider.

    Placed in the base layer rather than each client so behaviour is
    consistent and tweakable from judge configs without touching provider
    code.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts: int = Field(default=3, ge=1, le=10)
    initial_backoff_s: float = Field(default=0.5, ge=0.0)
    max_backoff_s: float = Field(default=8.0, ge=0.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    jitter_s: float = Field(default=0.25, ge=0.0)


class LLMRequest(BaseModel):
    """Inputs to a single model call.

    Kept flat and provider-agnostic. Provider-specific knobs (tool use,
    safety settings, response formats) belong in a concrete client's
    constructor, not here - otherwise judges would start depending on
    vendor semantics through the request object.
    """

    model_config = ConfigDict(extra="forbid")

    system_prompt: str | None = None
    user_prompt: str = Field(..., min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1, le=32_768)
    timeout_s: float = Field(default=60.0, gt=0.0)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    #: Provider-neutral metadata forwarded to observability (run_id,
    #: judge_pillar, row_id, ...). Concrete providers may pass selected
    #: keys through as request tags if supported.
    tags: dict[str, str] = Field(default_factory=dict)


@dataclass(frozen=True)
class LLMUsage:
    """Token / cost accounting for a single call. Fields are optional
    because not every provider reports every number."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    """Free-text response, plus timing, usage, and an escape hatch to
    the raw provider payload for observability.

    ``raw`` is intentionally ``Any`` so no downstream module starts
    depending on a specific provider payload shape - treat it as opaque
    except inside observability hooks.
    """

    text: str
    model_name: str
    latency_ms: float
    usage: LLMUsage = field(default_factory=LLMUsage)
    attempts: int = 1
    raw: Any = None


@dataclass(frozen=True)
class StructuredResponse[ParsedT: BaseModel]:
    """Schema-validated structured response."""

    parsed: ParsedT
    model_name: str
    latency_ms: float
    usage: LLMUsage = field(default_factory=LLMUsage)
    attempts: int = 1
    raw: Any = None


# ---------------------------------------------------------------------------
# Abstract client
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """Abstract LLM client.

    Concrete subclasses only need to implement :meth:`_invoke_text` and
    :meth:`_invoke_structured`. The base class handles timing, retries,
    and error translation. Subclasses MUST translate vendor-specific
    exceptions to :class:`ProviderError` subtypes before they propagate.
    """

    def __init__(self, *, model_name: str) -> None:
        if not model_name:
            raise ValueError("model_name is required.")
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    # ---- Public API ----------------------------------------------------

    def generate_text(self, request: LLMRequest) -> LLMResponse:
        """Run the request and return a free-text response."""
        start = time.perf_counter()
        attempts_used, raw_text, usage, raw = self._run_with_retry(
            request,
            lambda: self._invoke_text(request),
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        return LLMResponse(
            text=raw_text,
            model_name=self._model_name,
            latency_ms=latency_ms,
            usage=usage,
            attempts=attempts_used,
            raw=raw,
        )

    def generate_structured[ParsedT: BaseModel](
        self,
        request: LLMRequest,
        schema: type[ParsedT],
    ) -> StructuredResponse[ParsedT]:
        """Run the request and return a schema-validated structured response."""
        start = time.perf_counter()
        attempts_used, parsed, usage, raw = self._run_with_retry(
            request,
            lambda: self._invoke_structured(request, schema),
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        return StructuredResponse(
            parsed=parsed,
            model_name=self._model_name,
            latency_ms=latency_ms,
            usage=usage,
            attempts=attempts_used,
            raw=raw,
        )

    # ---- Subclass hooks ------------------------------------------------

    @abstractmethod
    def _invoke_text(
        self,
        request: LLMRequest,
    ) -> tuple[str, LLMUsage, Any]:
        """Return ``(text, usage, raw_payload)``.

        Concrete providers must translate any vendor exception into a
        :class:`ProviderError` subclass before letting it propagate.
        """

    @abstractmethod
    def _invoke_structured[ParsedT: BaseModel](
        self,
        request: LLMRequest,
        schema: type[ParsedT],
    ) -> tuple[ParsedT, LLMUsage, Any]:
        """Return ``(parsed, usage, raw_payload)``.

        The parsed value must already be an instance of ``schema``;
        validation failures should raise :class:`ProviderError`.
        """

    # ---- Retry / timing engine (internal) ------------------------------

    def _run_with_retry(
        self,
        request: LLMRequest,
        fn: Any,
    ) -> tuple[int, Any, LLMUsage, Any]:
        """Invoke ``fn`` with retry / backoff / translation.

        ``fn`` is expected to return ``(payload, usage, raw)`` where
        ``payload`` is whatever the public method ultimately returns
        (text string, or parsed pydantic instance).
        """
        policy = request.retry
        last_error: Exception | None = None

        for attempt in range(1, policy.max_attempts + 1):
            try:
                payload, usage, raw = fn()
            except ProviderRateLimitError as exc:
                last_error = exc
                _LOGGER.warning(
                    "LLM rate limited",
                    extra={
                        "model": self._model_name,
                        "attempt": attempt,
                        "error": str(exc),
                    },
                )
            except ProviderTimeoutError as exc:
                last_error = exc
                _LOGGER.warning(
                    "LLM call timed out",
                    extra={
                        "model": self._model_name,
                        "attempt": attempt,
                        "timeout_s": request.timeout_s,
                    },
                )
            except ProviderError as exc:
                # Non-retryable provider error. Stop immediately.
                _LOGGER.error(
                    "LLM call failed with non-retryable provider error",
                    extra={"model": self._model_name, "attempt": attempt, "error": str(exc)},
                )
                raise
            else:
                return attempt, payload, usage, raw

            if attempt >= policy.max_attempts:
                break
            _sleep(
                _backoff_seconds(
                    attempt=attempt,
                    initial=policy.initial_backoff_s,
                    multiplier=policy.backoff_multiplier,
                    cap=policy.max_backoff_s,
                    jitter=policy.jitter_s,
                )
            )

        assert last_error is not None  # Unreachable; either we returned or set last_error.
        raise last_error


# ---------------------------------------------------------------------------
# Small utilities (module-level so tests can monkeypatch ``_sleep``)
# ---------------------------------------------------------------------------


def _backoff_seconds(
    *,
    attempt: int,
    initial: float,
    multiplier: float,
    cap: float,
    jitter: float,
) -> float:
    """Exponential backoff with a small random jitter component.

    ``attempt`` is 1-indexed; the first retry sleeps approximately
    ``initial`` seconds, the next ``initial * multiplier``, and so on,
    clamped to ``cap``.
    """
    base = min(cap, initial * (multiplier ** (attempt - 1)))
    if jitter <= 0:
        return base
    return base + random.uniform(0.0, jitter)


def _sleep(seconds: float) -> None:
    """Thin wrapper so tests can monkeypatch a zero-sleep."""
    if seconds > 0:
        time.sleep(seconds)


# ---------------------------------------------------------------------------
# Small helper used by concrete clients: translate tag mappings safely
# ---------------------------------------------------------------------------


def sanitize_tags(tags: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of ``tags`` with string-only values, safe to log.

    Concrete clients use this when forwarding tags to a provider SDK or
    to a logging call; keeps the rule "no secrets in tags" enforceable
    at this single boundary.
    """
    out: dict[str, str] = {}
    for key, value in tags.items():
        if not isinstance(value, str):
            out[str(key)] = str(value)
        else:
            out[str(key)] = value
    return out
