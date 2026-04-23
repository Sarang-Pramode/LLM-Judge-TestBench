"""Canonical session-state keys and run-config dataclass.

Streamlit's ``st.session_state`` is a flat namespace shared across
pages. Without a single source of truth, pages drift out of sync and
renaming a key becomes a repo-wide search. This module defines every
key we use, prefixed ``jtb.`` so IDE completion / grep filters cleanly.

Pages import the constants from here; *never* spell session keys
as raw strings inside page code.

A companion :class:`RunConfig` dataclass captures the user's choices
from the Configure page so the Run page gets a typed hand-off rather
than a bag of strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from src.core.constants import PILLARS

__all__ = [
    "SS_BASELINE_SNAPSHOT",
    "SS_DATASET_NAME",
    "SS_LAST_RUN_RESULT",
    "SS_LAST_RUN_ROWS",
    "SS_MAPPING",
    "SS_NORMALIZED_ROWS",
    "SS_RUN_CONFIG",
    "SS_SOURCE_COLUMNS",
    "SS_SOURCE_ROWS",
    "RunConfig",
]


# ---------------------------------------------------------------------------
# Session-state key registry
# ---------------------------------------------------------------------------

# Upload page (Stage 2, already writes these)
SS_SOURCE_COLUMNS: Final[str] = "jtb.source_columns"
SS_SOURCE_ROWS: Final[str] = "jtb.source_rows"
SS_MAPPING: Final[str] = "jtb.mapping"
SS_NORMALIZED_ROWS: Final[str] = "jtb.normalized_rows"
SS_DATASET_NAME: Final[str] = "jtb.dataset_name"

# Configure + run pages (Stage 9)
SS_RUN_CONFIG: Final[str] = "jtb.run_config"
SS_LAST_RUN_RESULT: Final[str] = "jtb.last_run_result"
#: Snapshot of the normalized rows used for the last run - pinned so
#: the dashboard keeps working even if the user uploads a new file
#: without re-running.
SS_LAST_RUN_ROWS: Final[str] = "jtb.last_run_rows"
#: Optional pinned baseline for drift (dict from
#: :meth:`BaselineSnapshot.to_serializable`).
SS_BASELINE_SNAPSHOT: Final[str] = "jtb.baseline_snapshot"


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """User-selected parameters for a single evaluation run.

    Kept intentionally small. Per-pillar model/alias overrides belong
    in the YAML configs under ``configs/judges/`` and are already
    wired through :class:`JudgeBundle` - surfacing them in the UI
    would blur the config-driven boundary we established in Stage 4.

    Attributes:
        pillars: Which pillars to evaluate in this run. Order
            preserved for deterministic output.
        provider: ``"mock"`` during development / demos, ``"google"``
            for real Google GenAI calls. The orchestration runner
            does not care which - it talks to :class:`LLMClient`.
        max_workers: Upper bound on parallel judge calls. Capped
            sensibly in the UI so users can't spawn 1k threads by
            accident.
        per_provider_limit: Per-client rate-limit ceiling. ``None``
            disables throttling.
        use_cache: Whether to use the in-memory outcome cache so a
            re-run on the same dataset skips repeated work.
        enable_mlflow: Whether the Run page should try to log aggregate
            metrics to MLflow. Opt-out (defaults to ``True``) because
            the logger gracefully no-ops when ``JTB_MLFLOW_TRACKING_URI``
            isn't configured - disabling it explicitly only matters for
            users who *have* MLflow set up but want to skip this run.
        enable_langfuse: Same semantics as ``enable_mlflow`` but for
            per-row Langfuse tracing.
    """

    pillars: tuple[str, ...] = tuple(PILLARS)
    provider: str = "mock"
    max_workers: int = 8
    per_provider_limit: int | None = None
    use_cache: bool = True
    enable_mlflow: bool = True
    enable_langfuse: bool = True
    #: Arbitrary extra context attached to the next :class:`RunContext`
    #: (dataset fingerprint lives here, model alias, etc.). Kept as a
    #: plain dict so future knobs don't require a schema change.
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.pillars:
            raise ValueError("RunConfig.pillars must be non-empty.")
        if len(set(self.pillars)) != len(self.pillars):
            raise ValueError("RunConfig.pillars must not contain duplicates.")
        unknown = set(self.pillars) - set(PILLARS)
        if unknown:
            raise ValueError(f"RunConfig.pillars has unknown pillar(s): {sorted(unknown)}")
        if self.provider not in {"mock", "google"}:
            raise ValueError(
                f"RunConfig.provider must be 'mock' or 'google'; got {self.provider!r}."
            )
        if self.max_workers < 1:
            raise ValueError("RunConfig.max_workers must be >= 1.")
        if self.per_provider_limit is not None and self.per_provider_limit < 1:
            raise ValueError("RunConfig.per_provider_limit must be None or >= 1.")
