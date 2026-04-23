"""North-star threshold gates for agreement metrics (Risk / Product).

Loads :file:`configs/evaluation_thresholds.yaml` and evaluates each
:class:`PillarAgreement` into a discrete ``pass | warn | fail`` status
with human-readable reasons.

Convention
----------

- **Higher-is-better** metrics (``within_1_rate``, ``weighted_kappa``,
  ``severity_aware_alignment``): *pass* if value >= target; *warn* if
  warn <= value < target; *fail* if value < warn.
- **Lower-is-better** metrics (``mean_absolute_error``,
  ``large_miss_rate``): *pass* if value <= target; *warn* if target <
  value <= warn; *fail* if value > warn.

``large_miss_rate`` is computed as ``off_by_2_rate + off_by_3_plus_rate``
from :class:`PillarAgreement`. ``weighted_kappa`` may be ``None`` on
tiny samples; missing kappa bounds are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.core.exceptions import ConfigLoadError
from src.evaluation.agreement import AgreementReport

__all__ = [
    "GateResult",
    "GateStatus",
    "MetricBounds",
    "PillarGateReport",
    "RunThresholdReport",
    "evaluate_agreement_against_thresholds",
    "load_evaluation_thresholds",
    "resolve_pillar_bounds",
]


class GateStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class MetricBounds(BaseModel):
    """Target (good) and warn (minimum acceptable) for one metric."""

    model_config = ConfigDict(extra="forbid")

    target: float
    warn: float


class ThresholdsFile(BaseModel):
    """On-disk schema for ``evaluation_thresholds.yaml``."""

    model_config = ConfigDict(extra="forbid")

    version: str = "1"
    defaults: dict[str, dict[str, float]] = Field(default_factory=dict)
    pillars: dict[str, dict[str, dict[str, float]]] = Field(default_factory=dict)


def load_evaluation_thresholds(path: Path | str) -> ThresholdsFile:
    """Load and validate the thresholds YAML."""
    p = Path(path)
    if not p.exists():
        raise ConfigLoadError(f"Thresholds file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"Invalid YAML in {p}: {exc}") from exc
    if raw is None or not isinstance(raw, dict):
        raise ConfigLoadError(f"Thresholds file {p} must be a mapping at top level.")
    return ThresholdsFile.model_validate(raw)


def resolve_pillar_bounds(
    cfg: ThresholdsFile,
    pillar: str,
) -> dict[str, MetricBounds]:
    """Merge defaults with optional per-pillar overrides."""
    out: dict[str, MetricBounds] = {}
    for key, spec in cfg.defaults.items():
        out[key] = MetricBounds.model_validate(spec)
    override = cfg.pillars.get(pillar, {})
    for key, spec in override.items():
        out[key] = MetricBounds.model_validate(spec)
    return out


def _rate_triplet(
    value: float,
    *,
    bounds: MetricBounds,
    higher_is_better: bool,
) -> tuple[GateStatus, str]:
    if higher_is_better:
        if value >= bounds.target:
            return GateStatus.PASS, f"value {value:.4f} >= target {bounds.target:.4f}"
        if value >= bounds.warn:
            return (
                GateStatus.WARN,
                f"value {value:.4f} in [warn, target) = [{bounds.warn:.4f}, {bounds.target:.4f})",
            )
        return GateStatus.FAIL, f"value {value:.4f} < warn {bounds.warn:.4f}"
    # lower is better
    if value <= bounds.target:
        return GateStatus.PASS, f"value {value:.4f} <= target {bounds.target:.4f}"
    if value <= bounds.warn:
        return (
            GateStatus.WARN,
            f"value {value:.4f} in (target, warn] = ({bounds.target:.4f}, {bounds.warn:.4f}]",
        )
    return GateStatus.FAIL, f"value {value:.4f} > warn {bounds.warn:.4f}"


@dataclass(frozen=True)
class GateResult:
    """Single-metric gate outcome."""

    metric: str
    status: GateStatus
    value: float | None
    detail: str


@dataclass(frozen=True)
class PillarGateReport:
    """All gates for one pillar."""

    pillar: str
    gates: tuple[GateResult, ...]
    overall: GateStatus

    @staticmethod
    def _rollup(gates: tuple[GateResult, ...]) -> GateStatus:
        if not gates:
            return GateStatus.SKIP
        statuses = {g.status for g in gates if g.status != GateStatus.SKIP}
        if not statuses:
            return GateStatus.SKIP
        if GateStatus.FAIL in statuses:
            return GateStatus.FAIL
        if GateStatus.WARN in statuses:
            return GateStatus.WARN
        return GateStatus.PASS


@dataclass(frozen=True)
class RunThresholdReport:
    """Gates for every pillar in an agreement report."""

    version: str
    per_pillar: dict[str, PillarGateReport] = field(default_factory=dict)

    def worst_status(self) -> GateStatus:
        prims = [r.overall for r in self.per_pillar.values()]
        if not prims:
            return GateStatus.SKIP
        if GateStatus.FAIL in prims:
            return GateStatus.FAIL
        if GateStatus.WARN in prims:
            return GateStatus.WARN
        if GateStatus.PASS in prims:
            return GateStatus.PASS
        return GateStatus.SKIP


def evaluate_agreement_against_thresholds(
    report: AgreementReport,
    cfg: ThresholdsFile,
) -> RunThresholdReport:
    """Evaluate each pillar in ``report`` against merged bounds."""
    per_pillar: dict[str, PillarGateReport] = {}
    for pillar in report.pillars():
        bounds = resolve_pillar_bounds(cfg, pillar)
        pa = report.per_pillar[pillar]
        gates: list[GateResult] = []

        if "within_1_rate" in bounds:
            st, det = _rate_triplet(
                pa.within_1_rate, bounds=bounds["within_1_rate"], higher_is_better=True
            )
            gates.append(GateResult("within_1_rate", st, pa.within_1_rate, det))

        if "weighted_kappa" in bounds and pa.weighted_kappa is not None:
            st, det = _rate_triplet(
                pa.weighted_kappa,
                bounds=bounds["weighted_kappa"],
                higher_is_better=True,
            )
            gates.append(GateResult("weighted_kappa", st, pa.weighted_kappa, det))
        elif "weighted_kappa" in bounds:
            gates.append(
                GateResult(
                    "weighted_kappa",
                    GateStatus.SKIP,
                    None,
                    "weighted_kappa undefined for this sample",
                )
            )

        if "mean_absolute_error" in bounds:
            st, det = _rate_triplet(
                pa.mean_absolute_error,
                bounds=bounds["mean_absolute_error"],
                higher_is_better=False,
            )
            gates.append(GateResult("mean_absolute_error", st, pa.mean_absolute_error, det))

        if "severity_aware_alignment" in bounds:
            st, det = _rate_triplet(
                pa.severity_aware_alignment,
                bounds=bounds["severity_aware_alignment"],
                higher_is_better=True,
            )
            gates.append(
                GateResult("severity_aware_alignment", st, pa.severity_aware_alignment, det)
            )

        if "large_miss_rate" in bounds:
            large_miss = pa.off_by_2_rate + pa.off_by_3_plus_rate
            st, det = _rate_triplet(
                large_miss, bounds=bounds["large_miss_rate"], higher_is_better=False
            )
            gates.append(GateResult("large_miss_rate", st, large_miss, det))

        gtuple = tuple(gates)
        per_pillar[pillar] = PillarGateReport(
            pillar=pillar,
            gates=gtuple,
            overall=PillarGateReport._rollup(gtuple),
        )

    return RunThresholdReport(version=cfg.version, per_pillar=per_pillar)


def threshold_report_to_mlflow_metrics(report: RunThresholdReport) -> dict[str, float]:
    """Encode gate status as floats for MLflow (pass=2, warn=1, fail=0, skip=-1)."""
    encoding = {
        GateStatus.PASS: 2.0,
        GateStatus.WARN: 1.0,
        GateStatus.FAIL: 0.0,
        GateStatus.SKIP: -1.0,
    }
    out: dict[str, float] = {}
    for pillar, pr in report.per_pillar.items():
        safe_p = pillar.replace("/", "_")
        out[f"threshold/{safe_p}/overall"] = float(encoding[pr.overall])
        for g in pr.gates:
            out[f"threshold/{safe_p}/{g.metric}"] = float(encoding[g.status])
            if g.value is not None:
                out[f"threshold/{safe_p}/{g.metric}_value"] = float(g.value)
    return out


def threshold_report_to_serializable(report: RunThresholdReport) -> dict[str, Any]:
    """JSON-friendly dict for MLflow artifacts."""
    return {
        "version": report.version,
        "worst_status": report.worst_status().value,
        "pillars": {
            p: {
                "overall": pr.overall.value,
                "gates": [
                    {
                        "metric": g.metric,
                        "status": g.status.value,
                        "value": g.value,
                        "detail": g.detail,
                    }
                    for g in pr.gates
                ],
            }
            for p, pr in report.per_pillar.items()
        },
    }
