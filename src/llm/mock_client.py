"""Deterministic in-memory client for tests and offline development.

The mock supports three scripting styles:

1. A fixed sequence of text / structured responses (one per call).
2. A callable the test supplies that receives the :class:`LLMRequest`
   and returns the payload - useful for asserting against prompt shape.
3. A scripted queue of exceptions (rate limits / timeouts) interleaved
   with successful responses to exercise the base class retry engine.

Every call is recorded in :attr:`MockLLMClient.calls` so tests can
assert on what prompts were sent and how many attempts happened.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from src.core.exceptions import ProviderError
from src.llm.base import LLMClient, LLMRequest, LLMUsage

__all__ = [
    "MockCall",
    "MockLLMClient",
    "MockScript",
    "StructuredScript",
    "TextScript",
]


@dataclass(frozen=True)
class MockCall:
    """Record of a single call routed through the mock."""

    kind: str  # "text" | "structured"
    request: LLMRequest
    schema_name: str | None = None


# Script item types: either a payload or an exception to raise.
#
# - For text scripts: ``str`` payloads, or any :class:`Exception`.
# - For structured scripts: a pydantic instance / dict, or any
#   :class:`Exception`.
TextScript = Sequence[str | Exception]
StructuredScript = Sequence[BaseModel | dict[str, Any] | Exception]
MockScript = TextScript | StructuredScript


class MockLLMClient(LLMClient):
    """An in-memory client with deterministic, scripted behavior.

    Typical usage in tests::

        client = MockLLMClient(
            model_name="mock-v1",
            text_script=["answer-1", "answer-2"],
            structured_script=[MyPydantic(...)],
            usage=LLMUsage(input_tokens=10, output_tokens=5),
        )
    """

    def __init__(
        self,
        *,
        model_name: str = "mock-llm-v0",
        text_script: TextScript | None = None,
        structured_script: StructuredScript | None = None,
        text_fn: Callable[[LLMRequest], str] | None = None,
        structured_fn: Callable[[LLMRequest, type[BaseModel]], BaseModel] | None = None,
        usage: LLMUsage | None = None,
    ) -> None:
        super().__init__(model_name=model_name)
        self._text_queue: deque[str | Exception] = deque(text_script or ())
        self._structured_queue: deque[BaseModel | dict[str, Any] | Exception] = deque(
            structured_script or ()
        )
        self._text_fn = text_fn
        self._structured_fn = structured_fn
        self._usage = usage if usage is not None else LLMUsage()
        self.calls: list[MockCall] = []
        # Guards the script queues and the call log. Stage 7's runner
        # drives judges from multiple threads, so two concurrent calls
        # against the same mock must not race on ``popleft()`` or on
        # ``calls.append()``.
        self._lock = threading.Lock()

    # ---- Script management helpers -------------------------------------

    def queue_text(self, *items: str | Exception) -> None:
        """Append items to the text-response queue."""
        with self._lock:
            self._text_queue.extend(items)

    def queue_structured(self, *items: BaseModel | dict[str, Any] | Exception) -> None:
        """Append items to the structured-response queue."""
        with self._lock:
            self._structured_queue.extend(items)

    def reset(self) -> None:
        """Forget recorded calls without changing queued scripts."""
        with self._lock:
            self.calls.clear()

    @property
    def text_script_remaining(self) -> int:
        with self._lock:
            return len(self._text_queue)

    @property
    def structured_script_remaining(self) -> int:
        with self._lock:
            return len(self._structured_queue)

    # ---- LLMClient hooks -----------------------------------------------

    def _invoke_text(self, request: LLMRequest) -> tuple[str, LLMUsage, Any]:
        # Record the call under the lock so concurrent invocations don't
        # interleave entries in ``calls``. ``text_fn`` runs *outside* the
        # lock so a test-supplied function that blocks (e.g. to simulate
        # latency) can't deadlock other workers.
        with self._lock:
            self.calls.append(MockCall(kind="text", request=request))
            if self._text_fn is None:
                if not self._text_queue:
                    raise ProviderError(
                        f"MockLLMClient[{self.model_name}] has no scripted text responses "
                        "left; queue another response or supply text_fn."
                    )
                item: str | Exception = self._text_queue.popleft()
                if isinstance(item, Exception):
                    raise item
                return item, self._usage, {"mock": True, "source": "script"}

        # text_fn branch: dispatched outside the lock.
        assert self._text_fn is not None
        out = self._text_fn(request)
        return out, self._usage, {"mock": True, "source": "fn"}

    def _invoke_structured[
        ParsedT: BaseModel
    ](self, request: LLMRequest, schema: type[ParsedT],) -> tuple[ParsedT, LLMUsage, Any]:
        with self._lock:
            self.calls.append(
                MockCall(kind="structured", request=request, schema_name=schema.__name__)
            )
            if self._structured_fn is None:
                if not self._structured_queue:
                    raise ProviderError(
                        f"MockLLMClient[{self.model_name}] has no scripted structured "
                        "responses left; queue another response or supply structured_fn."
                    )
                queued: BaseModel | dict[str, Any] | Exception = self._structured_queue.popleft()
                if isinstance(queued, Exception):
                    raise queued
                parsed = _coerce_to_schema(queued, schema)
                return parsed, self._usage, {"mock": True, "source": "script"}

        assert self._structured_fn is not None
        out = self._structured_fn(request, schema)
        parsed = _coerce_to_schema(out, schema)
        return parsed, self._usage, {"mock": True, "source": "fn"}


def _coerce_to_schema[ParsedT: BaseModel](item: Any, schema: type[ParsedT]) -> ParsedT:
    """Accept either a schema instance or a dict and return ``schema``."""
    if isinstance(item, schema):
        return item
    if isinstance(item, BaseModel):
        # Support "same shape, different class" - useful when tests reuse
        # an already-built judge output under a stricter subclass.
        try:
            return schema.model_validate(item.model_dump())
        except ValidationError as exc:
            raise ProviderError(
                f"MockLLMClient returned a {type(item).__name__} that does "
                f"not validate as {schema.__name__}: {exc}"
            ) from exc
    if isinstance(item, dict):
        try:
            return schema.model_validate(item)
        except ValidationError as exc:
            raise ProviderError(
                f"MockLLMClient received a dict that does not validate as {schema.__name__}: {exc}"
            ) from exc
    raise ProviderError(
        f"MockLLMClient cannot coerce {type(item).__name__} into {schema.__name__}; "
        "supply a schema instance, a dict, or an Exception."
    )
