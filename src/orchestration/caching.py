"""Outcome-level cache for the evaluation runner.

We cache at the :class:`~src.judges.base.JudgeOutcome` layer rather than
at the raw LLM request layer. Two reasons:

1. A cache hit should skip prompt building + LLM call + parsing + rubric
   post-validation in one go. That is the expensive path; caching only
   the provider response still pays for the prompt build and parse on
   every hit.
2. Judges may pull in side-inputs (completeness KB match, task profile,
   rubric anchors) that a naive request-level key would miss. Caching
   the outcome lets us build the key from the inputs we already have:
   ``(pillar, bundle versions, model, row JSON, kb fingerprint)``.

The cache is an abstract interface:

- :class:`NoCache` is the default - zero overhead, always a miss.
- :class:`InMemoryOutcomeCache` is a thread-safe dict, suitable for a
  single runner process. File-backed implementations (useful for
  resuming a crashed run) can plug in later without changing the runner.

All implementations MUST be safe to call from multiple worker threads
simultaneously. Runner's concurrency model is threaded, not async; the
cache sits in every task's hot path.
"""

from __future__ import annotations

import hashlib
import json
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.core.types import NormalizedRow
from src.judges.config import JudgeBundle

if TYPE_CHECKING:
    from src.judges.base import JudgeOutcome
    from src.llm.base import LLMClient

__all__ = [
    "InMemoryOutcomeCache",
    "NoCache",
    "OutcomeCache",
    "compute_cache_key",
]


class OutcomeCache(ABC):
    """Contract for outcome caches. Thread-safe by implementation."""

    @abstractmethod
    def get(self, key: str) -> JudgeOutcome | None:
        """Return a cached outcome or ``None`` if absent."""

    @abstractmethod
    def set(self, key: str, outcome: JudgeOutcome) -> None:
        """Insert an outcome. No-op if the cache rejects this entry."""


class NoCache(OutcomeCache):
    """Null-object cache. Always misses; ``set`` is a no-op.

    Default for :class:`~src.orchestration.runner.RunPlan` so users opt
    into caching explicitly. That makes cache-driven bugs easier to
    spot: if you didn't wire a cache, you can't possibly be hitting one.
    """

    def get(self, key: str) -> JudgeOutcome | None:
        return None

    def set(self, key: str, outcome: JudgeOutcome) -> None:
        return None


class InMemoryOutcomeCache(OutcomeCache):
    """Thread-safe in-memory cache.

    Backed by a plain dict guarded by a :class:`threading.Lock`. We only
    lock around dict mutations; reads are also locked because CPython's
    dict is thread-safe at the operation level, but a ``get`` that
    races against a ``set`` on the same key could otherwise return an
    in-flight value without the memory-barrier the lock provides.
    """

    def __init__(self) -> None:
        self._store: dict[str, JudgeOutcome] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> JudgeOutcome | None:
        with self._lock:
            return self._store.get(key)

    def set(self, key: str, outcome: JudgeOutcome) -> None:
        # Only cache successful outcomes. Caching errors would make
        # retries useless and mask transient provider issues.
        if not outcome.ok:
            return
        with self._lock:
            self._store[key] = outcome

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


def compute_cache_key(
    *,
    pillar: str,
    bundle: JudgeBundle,
    llm: LLMClient,
    row: NormalizedRow,
    kb_fingerprint: str | None,
) -> str:
    """Build a stable SHA256 cache key for one ``(pillar, row)`` task.

    Inputs that must invalidate the cache when they change:

    - ``pillar`` (obviously).
    - ``bundle.config.prompt_version`` / ``bundle.config.rubric_version``
      - changing either means a different prompt shape.
    - ``bundle.config.model_alias`` + ``llm.model_name`` - user-facing
      alias and the actual model the runner resolved it to.
    - ``bundle.config.temperature`` / ``max_output_tokens`` - different
      sampling settings can yield different scores.
    - ``row.model_dump_json(sort_keys=True)`` - canonical row payload.
    - ``kb_fingerprint`` - completeness judge behaviour depends on the
      KB contents even though the KB isn't part of the row.

    What we deliberately DON'T include:

    - ``run_id`` / ``run_config_hash`` - these change per run but do
      not change the answer; keying off them would defeat caching.
    - ``reviewer_name`` - already part of the row payload.
    - Retry policy - behavioural, not semantic.

    Returns a 64-char hex digest.
    """
    payload = {
        "pillar": pillar,
        "prompt_version": bundle.config.prompt_version,
        "rubric_version": bundle.config.rubric_version,
        "model_alias": bundle.config.model_alias,
        "model_name": llm.model_name,
        "temperature": bundle.config.temperature,
        "max_output_tokens": bundle.config.max_output_tokens,
        "kb_fingerprint": kb_fingerprint,
        "row": row.model_dump(mode="json"),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
