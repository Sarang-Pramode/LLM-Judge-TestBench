"""Row-level and distribution diagnostics for judge vs SME scores.

Stdlib-only. Used by the Risk evidence UI and MLflow artifacts.

Regression convention: **human ~ judge** (human as dependent variable)
so slope/intercept describe how SME ratings relate to judge scores.
Identity agreement is the line human = judge.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.core.constants import SCORE_MAX, SCORE_MIN
from src.evaluation.join import ScoredItem

__all__ = [
    "BaselineSnapshot",
    "OLSResult",
    "PillarDiagnostics",
    "RunDiagnostics",
    "build_baseline_snapshot",
    "compute_pillar_diagnostics",
    "compute_run_diagnostics",
    "histogram_pmf",
    "jensen_shannon_divergence",
    "psi_ordinal",
]


def histogram_pmf(
    scores: Sequence[int], *, lo: int = SCORE_MIN, hi: int = SCORE_MAX
) -> dict[int, float]:
    """Probability mass on each ordinal bin (uniform if empty)."""
    n = len(scores)
    if n == 0:
        width = hi - lo + 1
        return {s: 1.0 / width for s in range(lo, hi + 1)}
    c = Counter(scores)
    return {s: c.get(s, 0) / n for s in range(lo, hi + 1)}


def _kl_divergence(p: dict[int, float], q: dict[int, float], *, eps: float = 1e-12) -> float:
    keys = sorted(set(p) | set(q))
    total = 0.0
    for k in keys:
        pk = max(p.get(k, 0.0), eps)
        qk = max(q.get(k, 0.0), eps)
        total += pk * math.log(pk / qk)
    return total


def jensen_shannon_divergence(
    p: dict[int, float],
    q: dict[int, float],
    *,
    eps: float = 1e-12,
) -> float:
    """JS divergence (natural log) on shared discrete support; bounded [0, ln 2]."""
    keys = sorted(set(p) | set(q))
    m = {k: 0.5 * (max(p.get(k, 0.0), eps) + max(q.get(k, 0.0), eps)) for k in keys}
    pp = {k: max(p.get(k, 0.0), eps) for k in keys}
    qq = {k: max(q.get(k, 0.0), eps) for k in keys}
    # renormalize m
    s = sum(m.values())
    m = {k: v / s for k, v in m.items()}
    return 0.5 * _kl_divergence(pp, m) + 0.5 * _kl_divergence(qq, m)


def psi_ordinal(
    expected_pmf: dict[int, float],
    actual_pmf: dict[int, float],
    *,
    eps: float = 1e-6,
) -> float:
    """Population Stability Index style sum on 1..5 bins."""
    total = 0.0
    for s in range(SCORE_MIN, SCORE_MAX + 1):
        e = max(expected_pmf.get(s, 0.0), eps)
        a = max(actual_pmf.get(s, 0.0), eps)
        total += (a - e) * math.log(a / e)
    return total


def _ols_y_on_x(xs: list[float], ys: list[float]) -> tuple[float, float, float] | None:
    """Return (slope, intercept, r_squared) for y ~ x, or None if degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    if ss_tot <= 0:
        r2 = 1.0 if all(abs(ys[i] - (slope * xs[i] + intercept)) < 1e-9 for i in range(n)) else 0.0
    else:
        ss_res = sum((ys[i] - (slope * xs[i] + intercept)) ** 2 for i in range(n))
        r2 = 1.0 - ss_res / ss_tot
    return slope, intercept, max(0.0, min(1.0, r2))


@dataclass(frozen=True)
class OLSResult:
    slope: float
    intercept: float
    r_squared: float


@dataclass(frozen=True)
class PillarDiagnostics:
    pillar: str
    support: int
    judge_pmf: dict[int, float]
    human_pmf: dict[int, float]
    mean_residual_judge_minus_human: float
    pct_positive_residual: float
    ols_human_on_judge: OLSResult | None
    js_vs_baseline: float | None = None
    psi_vs_baseline: float | None = None


@dataclass(frozen=True)
class RunDiagnostics:
    pillars: dict[str, PillarDiagnostics] = field(default_factory=dict)
    dataset_fingerprint: str | None = None
    baseline_run_id: str | None = None
    baseline_fingerprint: str | None = None
    baseline_compatible: bool = False

    def to_serializable(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "dataset_fingerprint": self.dataset_fingerprint,
            "baseline_run_id": self.baseline_run_id,
            "baseline_fingerprint": self.baseline_fingerprint,
            "baseline_compatible": self.baseline_compatible,
            "pillars": {},
        }
        for name, d in self.pillars.items():
            ols = d.ols_human_on_judge
            out["pillars"][name] = {
                "support": d.support,
                "judge_pmf": d.judge_pmf,
                "human_pmf": d.human_pmf,
                "mean_residual_judge_minus_human": d.mean_residual_judge_minus_human,
                "pct_positive_residual": d.pct_positive_residual,
                "ols_human_on_judge": (
                    None
                    if ols is None
                    else {
                        "slope": ols.slope,
                        "intercept": ols.intercept,
                        "r_squared": ols.r_squared,
                    }
                ),
                "js_vs_baseline": d.js_vs_baseline,
                "psi_vs_baseline": d.psi_vs_baseline,
            }
        return out


@dataclass(frozen=True)
class BaselineSnapshot:
    """Pinned judge PMFs per pillar for drift comparison."""

    dataset_fingerprint: str
    pillars: tuple[str, ...]
    judge_pmfs: dict[str, dict[int, float]]
    run_id: str | None = None

    def to_serializable(self) -> dict[str, Any]:
        return {
            "dataset_fingerprint": self.dataset_fingerprint,
            "pillars": list(self.pillars),
            "judge_pmfs": {
                k: {str(sk): sv for sk, sv in v.items()} for k, v in self.judge_pmfs.items()
            },
            "run_id": self.run_id,
        }

    @staticmethod
    def from_serializable(data: dict[str, Any]) -> BaselineSnapshot:
        pmfs_raw = data.get("judge_pmfs") or {}
        judge_pmfs: dict[str, dict[int, float]] = {}
        for pillar, bins in pmfs_raw.items():
            judge_pmfs[pillar] = {int(k): float(v) for k, v in bins.items()}
        return BaselineSnapshot(
            dataset_fingerprint=str(data["dataset_fingerprint"]),
            pillars=tuple(data.get("pillars") or ()),
            judge_pmfs=judge_pmfs,
            run_id=data.get("run_id"),
        )


def build_baseline_snapshot(
    items: Sequence[ScoredItem],
    *,
    dataset_fingerprint: str,
    pillars: Sequence[str],
    run_id: str | None = None,
) -> BaselineSnapshot:
    by_pillar: dict[str, list[ScoredItem]] = {}
    for it in items:
        by_pillar.setdefault(it.pillar, []).append(it)
    pmfs: dict[str, dict[int, float]] = {}
    for p in pillars:
        scores = [it.judge_score for it in by_pillar.get(p, ())]
        pmfs[p] = histogram_pmf(scores)
    return BaselineSnapshot(
        dataset_fingerprint=dataset_fingerprint,
        pillars=tuple(pillars),
        judge_pmfs=pmfs,
        run_id=run_id,
    )


def compute_pillar_diagnostics(
    items: Sequence[ScoredItem],
    *,
    pillar: str,
    baseline_judge_pmf: dict[int, float] | None = None,
) -> PillarDiagnostics:
    slice_items = [it for it in items if it.pillar == pillar]
    n = len(slice_items)
    if n == 0:
        empty_pmf = histogram_pmf(())
        return PillarDiagnostics(
            pillar=pillar,
            support=0,
            judge_pmf=empty_pmf,
            human_pmf=empty_pmf,
            mean_residual_judge_minus_human=0.0,
            pct_positive_residual=0.0,
            ols_human_on_judge=None,
            js_vs_baseline=None,
            psi_vs_baseline=None,
        )
    j_scores = [it.judge_score for it in slice_items]
    h_scores = [it.human_score for it in slice_items]
    residuals = [j - h for j, h in zip(j_scores, h_scores, strict=True)]
    mean_res = sum(residuals) / n
    pct_pos = sum(1 for r in residuals if r > 0) / n
    jp = histogram_pmf(j_scores)
    hp = histogram_pmf(h_scores)
    xs = [float(x) for x in j_scores]
    ys = [float(y) for y in h_scores]
    ols_t = _ols_y_on_x(xs, ys)
    ols = (
        None if ols_t is None else OLSResult(slope=ols_t[0], intercept=ols_t[1], r_squared=ols_t[2])
    )
    js_b = psi_b = None
    if baseline_judge_pmf is not None:
        js_b = jensen_shannon_divergence(jp, baseline_judge_pmf)
        psi_b = psi_ordinal(baseline_judge_pmf, jp)
    return PillarDiagnostics(
        pillar=pillar,
        support=n,
        judge_pmf=jp,
        human_pmf=hp,
        mean_residual_judge_minus_human=mean_res,
        pct_positive_residual=pct_pos,
        ols_human_on_judge=ols,
        js_vs_baseline=js_b,
        psi_vs_baseline=psi_b,
    )


def compute_run_diagnostics(
    items: Sequence[ScoredItem],
    *,
    pillars: Sequence[str],
    dataset_fingerprint: str | None = None,
    baseline: BaselineSnapshot | None = None,
) -> RunDiagnostics:
    """Compute per-pillar diagnostics; drift vs baseline when fingerprints match."""
    base_fp = baseline.dataset_fingerprint if baseline else None
    compatible = (
        baseline is not None
        and dataset_fingerprint is not None
        and base_fp == dataset_fingerprint
        and set(pillars) <= set(baseline.pillars)
    )
    out: dict[str, PillarDiagnostics] = {}
    for p in pillars:
        bpmf = baseline.judge_pmfs.get(p) if baseline and compatible else None
        out[p] = compute_pillar_diagnostics(items, pillar=p, baseline_judge_pmf=bpmf)
    return RunDiagnostics(
        pillars=out,
        dataset_fingerprint=dataset_fingerprint,
        baseline_run_id=baseline.run_id if baseline else None,
        baseline_fingerprint=base_fp,
        baseline_compatible=compatible,
    )
