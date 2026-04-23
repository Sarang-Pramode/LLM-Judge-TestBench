"""Run metadata and fingerprinting.

Pure-Python helpers that capture everything needed to reproduce a run:

- A content-based ``dataset_fingerprint`` derived from the normalized rows
  actually sent to the judges (not the source filename), so two runs on
  the same rows produce the same fingerprint even across machines.
- A ``run_config_hash`` over the user-selected pillars / provider /
  concurrency knobs so dashboards can group "identical config" runs.
- A :class:`RunMetadata` aggregate combining all version strings the
  observability layer cares about: prompt/rubric versions per pillar,
  model aliases, KB version, run_id.
- Flattening helpers (:func:`to_mlflow_params`, :func:`to_langfuse_metadata`)
  that backends consume. The backends stay dumb; this module is the only
  place that knows *what* to log.

No external dependencies. Safe to import from every layer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.core.types import NormalizedRow
from src.judges.config import JudgeBundle
from src.llm.base import LLMClient

__all__ = [
    "RunMetadata",
    "build_run_metadata",
    "dataset_fingerprint",
    "run_config_hash",
    "to_langfuse_metadata",
    "to_mlflow_params",
    "to_mlflow_tags",
]


# ---------------------------------------------------------------------------
# Fingerprinting primitives
# ---------------------------------------------------------------------------


_FINGERPRINT_PREFIX = "sha256:"
_FINGERPRINT_SHORT_LEN = 16


def _sha256_short(payload: str) -> str:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_FINGERPRINT_SHORT_LEN]
    return f"{_FINGERPRINT_PREFIX}{digest}"


def dataset_fingerprint(rows: Sequence[NormalizedRow]) -> str:
    """Content-addressed fingerprint of the normalized rows.

    Deterministic across processes / machines because we serialize only
    the four required columns in a canonical order. Empty datasets still
    produce a valid fingerprint (of the empty string) so downstream code
    never has to branch on "no rows".

    We deliberately ignore optional fields (``retrieved_context``,
    ``chat_history``, ``metadata``) because they are noisy and often
    reshaped by upstream ETL; two semantically identical runs would
    otherwise fingerprint differently.
    """
    parts = [
        "\t".join((row.record_id, row.user_input, row.agent_output, row.category)) for row in rows
    ]
    return _sha256_short("\n".join(parts))


def run_config_hash(config: Mapping[str, Any]) -> str:
    """Canonical fingerprint of a run-config mapping.

    Accepts any JSON-serializable mapping (pillars, provider, concurrency,
    etc.). Sorting keys makes the hash stable regardless of insertion
    order, and ``default=str`` is a safety net for odd primitives like
    ``Path`` or ``Decimal`` that a caller might leak in.
    """
    encoded = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    return _sha256_short(encoded)


# ---------------------------------------------------------------------------
# Aggregate metadata model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunMetadata:
    """Everything observability backends need about a single run.

    Purposefully a plain frozen dataclass: MLflow and Langfuse both want
    plain mappings at the edge, and callers frequently want to log
    partial snapshots (e.g. only params up front, then metrics later).
    Keeping this flat avoids forcing backends to traverse Pydantic models.

    ``provider`` is the high-level provider identifier (``"mock"``,
    ``"google"``, etc.). Per-pillar concrete model names live in
    :attr:`model_name_by_pillar` because two pillars can share a
    provider but use different models.
    """

    run_id: str
    started_at: datetime
    dataset_fingerprint: str
    dataset_row_count: int
    pillars: tuple[str, ...]
    provider: str
    model_alias_by_pillar: Mapping[str, str]
    model_name_by_pillar: Mapping[str, str]
    prompt_version_by_pillar: Mapping[str, str]
    rubric_version_by_pillar: Mapping[str, str]
    run_config_hash: str | None = None
    kb_version: str | None = None
    extra_tags: Mapping[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_run_metadata(
    *,
    run_id: str,
    dataset_fingerprint: str,
    dataset_row_count: int,
    pillars: Sequence[str],
    bundles: Mapping[str, JudgeBundle],
    llm_by_pillar: Mapping[str, LLMClient],
    provider: str = "unknown",
    run_config: Mapping[str, Any] | None = None,
    kb_version: str | None = None,
    started_at: datetime | None = None,
    extra_tags: Mapping[str, str] | None = None,
) -> RunMetadata:
    """Assemble a :class:`RunMetadata` from the same inputs the runner uses.

    Raises:
        KeyError: if ``pillars`` contains an entry without a matching
            bundle or LLM client. We want this to fail loudly; a silent
            fallback would break reproducibility.
    """
    missing_bundles = [p for p in pillars if p not in bundles]
    if missing_bundles:
        raise KeyError(f"bundles missing entries for pillars: {sorted(missing_bundles)}")
    missing_clients = [p for p in pillars if p not in llm_by_pillar]
    if missing_clients:
        raise KeyError(f"llm_by_pillar missing entries for pillars: {sorted(missing_clients)}")

    ordered = tuple(pillars)
    return RunMetadata(
        run_id=run_id,
        started_at=started_at or datetime.now(UTC),
        dataset_fingerprint=dataset_fingerprint,
        dataset_row_count=dataset_row_count,
        pillars=ordered,
        provider=provider,
        model_alias_by_pillar={p: bundles[p].config.model_alias for p in ordered},
        model_name_by_pillar={p: llm_by_pillar[p].model_name for p in ordered},
        prompt_version_by_pillar={p: bundles[p].config.prompt_version for p in ordered},
        rubric_version_by_pillar={p: bundles[p].rubric.version for p in ordered},
        run_config_hash=run_config_hash(run_config) if run_config is not None else None,
        kb_version=kb_version,
        extra_tags=dict(extra_tags) if extra_tags else {},
    )


# ---------------------------------------------------------------------------
# Flatteners
# ---------------------------------------------------------------------------


def to_mlflow_params(meta: RunMetadata) -> dict[str, str]:
    """Flatten to string/string pairs suitable for ``mlflow.log_params``.

    MLflow requires param values to be strings and caps them around 500
    chars; our inputs are already short identifiers so no truncation is
    needed, but we coerce to ``str`` defensively.
    """
    params: dict[str, str] = {
        "run_id": meta.run_id,
        "dataset_fingerprint": meta.dataset_fingerprint,
        "dataset_row_count": str(meta.dataset_row_count),
        "pillars": ",".join(meta.pillars),
        "provider": meta.provider,
    }
    if meta.run_config_hash is not None:
        params["run_config_hash"] = meta.run_config_hash
    if meta.kb_version is not None:
        params["kb_version"] = meta.kb_version
    for pillar in meta.pillars:
        params[f"model_alias_{pillar}"] = meta.model_alias_by_pillar[pillar]
        params[f"model_name_{pillar}"] = meta.model_name_by_pillar[pillar]
        params[f"prompt_version_{pillar}"] = meta.prompt_version_by_pillar[pillar]
        params[f"rubric_version_{pillar}"] = meta.rubric_version_by_pillar[pillar]
    return params


def to_mlflow_tags(meta: RunMetadata) -> dict[str, str]:
    """Tags are free-form labels; MLflow filters by these in the UI.

    We send a small curated subset so the MLflow Experiments UI stays
    scannable. Anything richer (per-pillar specifics) goes via params.
    """
    tags: dict[str, str] = {
        "jtb.run_id": meta.run_id,
        "jtb.provider": meta.provider,
        "jtb.dataset_fingerprint": meta.dataset_fingerprint,
    }
    if meta.kb_version is not None:
        tags["jtb.kb_version"] = meta.kb_version
    tags.update({f"jtb.{k}": v for k, v in meta.extra_tags.items()})
    return tags


def to_langfuse_metadata(meta: RunMetadata) -> dict[str, Any]:
    """Structured dict for Langfuse trace ``metadata`` (nested OK here).

    Langfuse accepts arbitrary JSON, so we keep the nesting more natural
    than the flattened MLflow params: per-pillar values group under
    their pillar name rather than being prefixed.
    """
    per_pillar: dict[str, dict[str, str]] = {}
    for pillar in meta.pillars:
        per_pillar[pillar] = {
            "model_alias": meta.model_alias_by_pillar[pillar],
            "model_name": meta.model_name_by_pillar[pillar],
            "prompt_version": meta.prompt_version_by_pillar[pillar],
            "rubric_version": meta.rubric_version_by_pillar[pillar],
        }
    payload: dict[str, Any] = {
        "run_id": meta.run_id,
        "dataset_fingerprint": meta.dataset_fingerprint,
        "dataset_row_count": meta.dataset_row_count,
        "provider": meta.provider,
        "pillars": list(meta.pillars),
        "per_pillar": per_pillar,
    }
    if meta.run_config_hash is not None:
        payload["run_config_hash"] = meta.run_config_hash
    if meta.kb_version is not None:
        payload["kb_version"] = meta.kb_version
    if meta.extra_tags:
        payload["extra_tags"] = dict(meta.extra_tags)
    return payload
