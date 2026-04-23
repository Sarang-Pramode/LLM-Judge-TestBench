"""Langfuse tracer for per-row judge calls.

Langfuse captures the fine-grained detail MLflow is not good at: every
``(record_id, pillar)`` becomes an observation with the prompt context,
the structured judge output, usage, latency, and any parse errors.

Design mirrors :mod:`src.observability.mlflow_logger`:

1. **Never break a run.** Every public method is guarded.
2. **Lazy import.** The ``langfuse`` SDK is only imported when the
   tracer is enabled and no backend is injected, so stripped
   deployments don't need the dep.
3. **Injectable backend.** Tests inject a minimal fake that records
   calls; production resolves to a real ``Langfuse`` client.
4. **Thread-safe.** The runner fan-outs via a thread pool; the tracer
   stores its root span in an ``threading.Lock``-protected reference
   and creates children via that root so children attach to the right
   trace regardless of which worker thread records them.

Observation shape
-----------------

- One root **span** per run, carrying :func:`to_langfuse_metadata` output.
- One **generation** per judge call (score, decision_summary, usage),
  attached as a child observation of the root span.
- Successful calls include the parsed judge output; failures record the
  exception message as ``status_message`` with level ``ERROR``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, Protocol

from src.core.types import NormalizedRow
from src.judges.base import JudgeOutcome
from src.observability.run_metadata import RunMetadata, to_langfuse_metadata

__all__ = [
    "LangfuseTracer",
    "build_langfuse_tracer",
]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend protocols
# ---------------------------------------------------------------------------


class _LangfuseObservation(Protocol):
    """The subset of a Langfuse span/generation handle we actually use."""

    def update(self, **kwargs: Any) -> Any: ...

    def end(self, **kwargs: Any) -> Any: ...

    def start_observation(self, **kwargs: Any) -> Any: ...


class _LangfuseBackend(Protocol):
    """The subset of ``Langfuse`` we use."""

    def start_observation(self, **kwargs: Any) -> Any: ...

    def flush(self) -> None: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_warned_errors: set[str] = set()


def _warn_once(tag: str, exc: BaseException) -> None:
    key = f"{tag}:{type(exc).__name__}"
    if key in _warned_errors:
        return
    _warned_errors.add(key)
    logger.warning("LangfuseTracer.%s failed (%s): %s", tag, type(exc).__name__, exc)


def _guarded(tag: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        def inner(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                _warn_once(tag, exc)
                return None

        inner.__name__ = fn.__name__
        inner.__doc__ = fn.__doc__
        return inner

    return wrap


def _resolve_default_backend(
    *,
    host: str | None,
    public_key: str | None,
    secret_key: str | None,
) -> _LangfuseBackend | None:
    """Instantiate a real Langfuse client, or return None on failure.

    We require all three of host/public_key/secret_key to be present; a
    subset indicates a half-configured env and we'd rather no-op than
    send traces to the wrong project.
    """
    if not (host and public_key and secret_key):
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(  # type: ignore[return-value]
            host=host,
            public_key=public_key,
            secret_key=secret_key,
        )
    except ImportError:
        logger.info(
            "langfuse is not installed; LangfuseTracer will run in disabled mode. "
            "Install the ``langfuse`` package to enable per-row tracing.",
        )
        return None
    except Exception as exc:  # broad on purpose - client init can raise various things
        logger.warning("Failed to construct Langfuse client (%s): %s", type(exc).__name__, exc)
        return None


# ---------------------------------------------------------------------------
# Redaction helpers
# ---------------------------------------------------------------------------


# PII redaction lives here rather than in individual call sites so the
# policy is centralized. For now we support a simple opt-in: when
# ``redact_pii`` is True, we send only lengths + hashes for free-text
# fields. Richer redaction (regex-based) can be added without touching
# call sites.
def _redact_text(text: str, *, redact: bool, max_len: int = 2048) -> str:
    if redact:
        return f"<redacted len={len(text)}>"
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... <truncated {len(text) - max_len} chars>"


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


class LangfuseTracer:
    """Adapter that turns evaluation events into Langfuse observations."""

    def __init__(
        self,
        *,
        host: str | None,
        public_key: str | None,
        secret_key: str | None,
        enabled: bool = True,
        redact_pii: bool = False,
        backend: _LangfuseBackend | None = None,
    ) -> None:
        self._host = host
        self._redact_pii = redact_pii

        resolved = (
            backend
            if backend is not None
            else _resolve_default_backend(host=host, public_key=public_key, secret_key=secret_key)
        )
        self._enabled = bool(enabled and resolved is not None)
        self._backend: _LangfuseBackend | None = resolved if self._enabled else None

        self._root_lock = threading.Lock()
        self._root_observation: _LangfuseObservation | None = None
        self._run_id: str | None = None

    # ---- Public properties --------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def run_id(self) -> str | None:
        return self._run_id

    # ---- Lifecycle ----------------------------------------------------

    @_guarded("start_run")
    def start_run(self, meta: RunMetadata) -> None:
        """Open the per-run root span. Idempotent."""
        if not self._enabled or self._backend is None:
            return
        with self._root_lock:
            if self._root_observation is not None:
                return
            self._root_observation = self._backend.start_observation(
                name=f"jtb.run.{meta.run_id}",
                as_type="span",
                input={
                    "pillars": list(meta.pillars),
                    "dataset_row_count": meta.dataset_row_count,
                    "provider": meta.provider,
                },
                metadata=to_langfuse_metadata(meta),
            )
            self._run_id = meta.run_id

    @_guarded("end_run")
    def end_run(self, status: str = "FINISHED", summary: dict[str, Any] | None = None) -> None:
        """Close the root span and flush pending events.

        No-op when no run has been started (idempotent with respect to
        :meth:`start_run`): skipping flush in that case keeps tests
        deterministic and avoids a network round-trip when the caller
        didn't actually do anything traceable.
        """
        if not self._enabled or self._backend is None:
            return
        with self._root_lock:
            root = self._root_observation
            self._root_observation = None
        if root is None:
            return
        output = {"status": status}
        if summary is not None:
            output.update(summary)
        root.update(output=output, level="DEFAULT" if status == "FINISHED" else "ERROR")
        root.end()
        try:
            self._backend.flush()
        except Exception as exc:
            _warn_once("flush", exc)

    # ---- Per-row tracing ----------------------------------------------

    @_guarded("log_outcome")
    def log_outcome(self, outcome: JudgeOutcome, row: NormalizedRow) -> None:
        """Emit one ``generation`` observation for a finished judge call.

        Creates the observation off the root span via ``start_observation``
        and immediately ends it, so the tracer never relies on the Python
        ``with`` context propagating across threads.
        """
        if not self._enabled or self._backend is None:
            return
        with self._root_lock:
            root = self._root_observation
        if root is None:
            # start_run wasn't called (or failed silently). We still want
            # to record something, so we create a top-level observation.
            parent: _LangfuseBackend | _LangfuseObservation = self._backend
        else:
            parent = root

        metadata: dict[str, Any] = {
            "record_id": outcome.record_id,
            "pillar": outcome.pillar,
            "latency_ms": outcome.latency_ms,
            "attempts": outcome.attempts,
            "category": row.category,
        }
        if outcome.prompt_versions is not None:
            metadata["prompt_version"] = outcome.prompt_versions[0]
            metadata["rubric_version"] = outcome.prompt_versions[1]
        if outcome.extras:
            metadata["extras"] = dict(outcome.extras)

        input_payload = {
            "user_input": _redact_text(row.user_input, redact=self._redact_pii),
            "agent_output": _redact_text(row.agent_output, redact=self._redact_pii),
        }

        output_payload: Any
        status_message: str | None = None
        level: str = "DEFAULT"
        if outcome.result is not None:
            output_payload = {
                "score": outcome.result.score,
                "confidence": outcome.result.confidence,
                "decision_summary": outcome.result.decision_summary,
                "failure_tags": list(outcome.result.failure_tags or ()),
            }
        else:
            output_payload = None
            status_message = outcome.error or outcome.error_type or "judge_failed"
            level = "ERROR"

        usage_details: dict[str, int] | None = None
        if outcome.usage is not None:
            usage_details = {
                "input": int(outcome.usage.input_tokens or 0),
                "output": int(outcome.usage.output_tokens or 0),
                "total": int(
                    (outcome.usage.input_tokens or 0) + (outcome.usage.output_tokens or 0)
                ),
            }

        observation = parent.start_observation(
            name=f"{outcome.pillar}@{outcome.record_id}",
            as_type="generation",
            input=input_payload,
            output=output_payload,
            metadata=metadata,
            model=outcome.model_name,
            usage_details=usage_details,
            level=level,
            status_message=status_message,
        )
        observation.end()

    # ---- Low-level helpers --------------------------------------------

    @_guarded("flush")
    def flush(self) -> None:
        """Force pending events to be sent. Safe to call multiple times."""
        if not self._enabled or self._backend is None:
            return
        self._backend.flush()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_langfuse_tracer(
    *,
    host: str | None = None,
    public_key: str | None = None,
    secret_key: str | None = None,
    enabled: bool = True,
    redact_pii: bool | None = None,
    backend: _LangfuseBackend | None = None,
) -> LangfuseTracer:
    """Construct a tracer, defaulting to :class:`AppSettings`.

    Credentials come from the settings layer if not explicitly passed.
    The returned tracer is always safe to call: missing creds produce a
    disabled tracer.
    """
    if host is None or public_key is None or secret_key is None or redact_pii is None:
        from src.core.settings import get_settings

        settings = get_settings()
        if host is None:
            host = settings.langfuse_host
        if public_key is None and settings.langfuse_public_key is not None:
            public_key = settings.langfuse_public_key.get_secret_value()
        if secret_key is None and settings.langfuse_secret_key is not None:
            secret_key = settings.langfuse_secret_key.get_secret_value()
        if redact_pii is None:
            redact_pii = settings.redact_pii
    return LangfuseTracer(
        host=host,
        public_key=public_key,
        secret_key=secret_key,
        enabled=enabled,
        redact_pii=bool(redact_pii),
        backend=backend,
    )
