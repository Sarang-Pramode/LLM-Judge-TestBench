"""Orchestration layer: parallel evaluation, batching, and caching.

Stage 7 introduces the :class:`EvaluationRunner` that fans (row, pillar)
tasks across a thread pool, routes them through per-provider throttles,
and collects deterministic, failure-isolated results.

Public API is intentionally small; anything not re-exported here should
be considered an internal implementation detail subject to change.
"""

from src.orchestration.batching import TaskSpec, chunk, plan_tasks
from src.orchestration.caching import (
    InMemoryOutcomeCache,
    NoCache,
    OutcomeCache,
    compute_cache_key,
)
from src.orchestration.concurrency import (
    ConcurrencyPolicy,
    ProviderThrottle,
    RunAborted,
    acquire_throttle,
    execute_parallel,
)
from src.orchestration.runner import (
    EvaluationRunner,
    RunPlan,
    RunResult,
    RunSummary,
)

__all__ = [
    "ConcurrencyPolicy",
    "EvaluationRunner",
    "InMemoryOutcomeCache",
    "NoCache",
    "OutcomeCache",
    "ProviderThrottle",
    "RunAborted",
    "RunPlan",
    "RunResult",
    "RunSummary",
    "TaskSpec",
    "acquire_throttle",
    "chunk",
    "compute_cache_key",
    "execute_parallel",
    "plan_tasks",
]
