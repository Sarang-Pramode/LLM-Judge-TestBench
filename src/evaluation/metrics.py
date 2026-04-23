"""Pure-math metric helpers operating on ``(judge, human)`` score pairs.

Design rules:

- No dependencies on our domain types. Every function takes a sequence
  of ``(judge_score, human_score)`` integer pairs. That keeps golden
  tests tiny and lets us swap in synthetic inputs without building
  NormalizedRow fixtures.
- No plotting. Ever. The dashboard (Stage 9) is the only place that
  knows how to render a confusion matrix or a distribution.
- Graceful degradation. Metrics that mathematically require a
  non-degenerate sample (e.g. Spearman on constant inputs) return
  ``None`` rather than raising. Empty inputs return ``None`` for
  "undefined on 0 samples" metrics and ``0`` / ``0.0`` / ``{}`` for
  counting metrics.
- No numpy / scipy dependency. The project is already-light on
  runtime deps; these implementations are short enough (and well
  covered by the test suite) that we don't need a linalg toolkit.

The weighted kappa implementation uses Cohen's quadratic-weighted
kappa by default - standard for ordinal Likert-style ratings - and
also supports linear weights. The Spearman coefficient is implemented
as Pearson-on-ranks with average-rank tie handling.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from src.core.constants import DISTANCE_WEIGHTS, SCORE_MAX, SCORE_MIN

__all__ = [
    "ScorePair",
    "confusion_matrix",
    "exact_match_rate",
    "mean_absolute_error",
    "off_by_2_rate",
    "off_by_3_plus_rate",
    "score_distribution",
    "severity_aware_alignment",
    "spearman_correlation",
    "weighted_kappa",
    "within_1_rate",
]


#: ``(judge_score, human_score)``. Using a plain tuple keeps the API
#: friendly for quick tests and one-line list-comprehensions. A named
#: dataclass would be over-engineering for a two-tuple.
ScorePair = tuple[int, int]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_list(pairs: Iterable[ScorePair]) -> list[ScorePair]:
    """Materialise pairs once so we can iterate multiple times safely."""
    return list(pairs)


def _counts_by_distance(pairs: Sequence[ScorePair]) -> Counter[int]:
    """Histogram of absolute judge-vs-human distances."""
    return Counter(abs(j - h) for j, h in pairs)


# ---------------------------------------------------------------------------
# Rate metrics
# ---------------------------------------------------------------------------


def exact_match_rate(pairs: Iterable[ScorePair]) -> float:
    """Fraction of pairs where ``judge == human``. Returns 0.0 on empty."""
    items = _as_list(pairs)
    if not items:
        return 0.0
    matches = sum(1 for j, h in items if j == h)
    return matches / len(items)


def within_1_rate(pairs: Iterable[ScorePair]) -> float:
    """Fraction of pairs where ``|judge - human| <= 1``.

    This includes exact matches by definition - don't subtract
    :func:`exact_match_rate` when combining.
    """
    items = _as_list(pairs)
    if not items:
        return 0.0
    return sum(1 for j, h in items if abs(j - h) <= 1) / len(items)


def off_by_2_rate(pairs: Iterable[ScorePair]) -> float:
    """Fraction of pairs where ``|judge - human| == 2``.

    This is an exact-distance bucket (not a cumulative "off-by-2+"
    rate) so that the three distance buckets (``exact``, ``within_1``
    - exact, ``off_by_2``, ``off_by_3_plus``) partition the sample.
    """
    items = _as_list(pairs)
    if not items:
        return 0.0
    return sum(1 for j, h in items if abs(j - h) == 2) / len(items)


def off_by_3_plus_rate(pairs: Iterable[ScorePair]) -> float:
    """Fraction of pairs where ``|judge - human| >= 3``.

    Intentionally a cumulative tail bucket - these are the "large
    miss" cases dashboards highlight as severe disagreements.
    """
    items = _as_list(pairs)
    if not items:
        return 0.0
    return sum(1 for j, h in items if abs(j - h) >= 3) / len(items)


def mean_absolute_error(pairs: Iterable[ScorePair]) -> float:
    """Mean ``|judge - human|`` across pairs. Returns 0.0 on empty."""
    items = _as_list(pairs)
    if not items:
        return 0.0
    return sum(abs(j - h) for j, h in items) / len(items)


# ---------------------------------------------------------------------------
# Severity-aware alignment
# ---------------------------------------------------------------------------


def severity_aware_alignment(
    pairs: Iterable[ScorePair],
    *,
    weights: Mapping[int, float] = DISTANCE_WEIGHTS,
) -> float:
    """Business-friendly "how aligned are we overall" score.

    Each pair contributes the weight associated with its distance bucket
    (see :data:`src.core.constants.DISTANCE_WEIGHTS`); the run-wide
    score is the mean. A distance not present in ``weights`` falls back
    to ``0.0`` so a misconfigured weight map degrades gracefully.

    Returns 0.0 on empty input - if there's nothing to score, by
    definition no alignment evidence exists.
    """
    items = _as_list(pairs)
    if not items:
        return 0.0
    total = 0.0
    for j, h in items:
        distance = abs(j - h)
        total += weights.get(distance, 0.0)
    return total / len(items)


# ---------------------------------------------------------------------------
# Distributions + confusion matrix
# ---------------------------------------------------------------------------


def score_distribution(
    scores: Iterable[int],
    *,
    score_min: int = SCORE_MIN,
    score_max: int = SCORE_MAX,
) -> dict[int, int]:
    """Return ``{score: count}`` covering the full 1..5 range.

    The returned dict always contains every integer in
    ``[score_min, score_max]`` (even zeros) so downstream plots don't
    have to backfill missing bins. Scores outside the range are
    counted under the nearest edge - a signal that something upstream
    broke rather than a silent drop.
    """
    dist = dict.fromkeys(range(score_min, score_max + 1), 0)
    for s in scores:
        clamped = max(score_min, min(score_max, s))
        dist[clamped] += 1
    return dist


def confusion_matrix(
    pairs: Iterable[ScorePair],
    *,
    score_min: int = SCORE_MIN,
    score_max: int = SCORE_MAX,
) -> list[list[int]]:
    """Return a square confusion matrix indexed by ``human`` x ``judge``.

    Row index ``i`` corresponds to ``human_score = score_min + i``;
    column index ``j`` to ``judge_score = score_min + j``. That
    orientation matches the typical seaborn heatmap convention
    ("human on y, judge on x") and keeps plot code trivial.

    Scores outside the range are clamped to the nearest edge (same
    contract as :func:`score_distribution`).
    """
    k = score_max - score_min + 1
    matrix = [[0 for _ in range(k)] for _ in range(k)]
    for j, h in pairs:
        row = max(score_min, min(score_max, h)) - score_min
        col = max(score_min, min(score_max, j)) - score_min
        matrix[row][col] += 1
    return matrix


# ---------------------------------------------------------------------------
# Weighted kappa (Cohen) - ordinal agreement
# ---------------------------------------------------------------------------


def weighted_kappa(
    pairs: Iterable[ScorePair],
    *,
    weights: Literal["linear", "quadratic"] = "quadratic",
    score_min: int = SCORE_MIN,
    score_max: int = SCORE_MAX,
) -> float | None:
    """Cohen's weighted kappa for ordinal agreement.

    ``quadratic`` (default) is the conventional choice for Likert-style
    judge vs SME comparisons: it amplifies large disagreements and
    barely penalises near-misses. ``linear`` is the alternative
    citation-friendly variant.

    Returns ``None`` when the kappa is undefined - specifically when
    the expected-agreement denominator is zero (a degenerate
    distribution where one side is constant), or on fewer than two
    pairs.

    Reference: Cohen (1968). Formula:

        kappa = 1 - sum(w_ij * O_ij) / sum(w_ij * E_ij)

    with weights ``w_ij = (i - j) ** p / (K - 1) ** p`` where ``p=1``
    (linear) or ``p=2`` (quadratic) and ``K`` is the number of
    categories.
    """
    items = _as_list(pairs)
    if len(items) < 2:
        return None

    k = score_max - score_min + 1
    if k < 2:  # pragma: no cover - defensive
        return None

    # Observed joint counts. We index by human (row) / judge (col).
    observed = confusion_matrix(items, score_min=score_min, score_max=score_max)

    # Marginal distributions -> expected joint under independence.
    row_totals = [sum(row) for row in observed]
    col_totals = [sum(observed[r][c] for r in range(k)) for c in range(k)]
    n = sum(row_totals)
    if n == 0:  # pragma: no cover - empty guarded above
        return None

    expected = [[(row_totals[r] * col_totals[c]) / n for c in range(k)] for r in range(k)]

    # Weight matrix.
    denom = (k - 1) ** (2 if weights == "quadratic" else 1)
    w = [
        [((r - c) ** (2 if weights == "quadratic" else 1)) / denom for c in range(k)]
        for r in range(k)
    ]

    num = sum(w[r][c] * observed[r][c] for r in range(k) for c in range(k))
    den = sum(w[r][c] * expected[r][c] for r in range(k) for c in range(k))
    if den == 0:
        return None
    return 1.0 - num / den


# ---------------------------------------------------------------------------
# Spearman rank correlation
# ---------------------------------------------------------------------------


def spearman_correlation(pairs: Iterable[ScorePair]) -> float | None:
    """Spearman's rank correlation with average-rank tie handling.

    Implemented as Pearson correlation applied to the rank transforms
    of the two score series. Returns ``None`` when the correlation is
    undefined - either fewer than two pairs or one series has no
    variance (constant scores).
    """
    items = _as_list(pairs)
    if len(items) < 2:
        return None

    judge_scores = [j for j, _ in items]
    human_scores = [h for _, h in items]

    ranks_j = _average_ranks(judge_scores)
    ranks_h = _average_ranks(human_scores)

    return _pearson(ranks_j, ranks_h)


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return 1-based ranks with ties assigned the average rank.

    Classic example: ``[10, 20, 20, 30]`` -> ``[1, 2.5, 2.5, 4]``.
    """
    sorted_pairs = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_pairs):
        j = i
        while j + 1 < len(sorted_pairs) and sorted_pairs[j + 1][1] == sorted_pairs[i][1]:
            j += 1
        # Average of the inclusive rank span [i+1, j+1].
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            orig_index = sorted_pairs[k][0]
            ranks[orig_index] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson product-moment correlation. Returns ``None`` if
    undefined (constant series)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sx = 0.0
    sy = 0.0
    sxy = 0.0
    for x, y in zip(xs, ys, strict=True):
        dx = x - mean_x
        dy = y - mean_y
        sx += dx * dx
        sy += dy * dy
        sxy += dx * dy
    if sx == 0.0 or sy == 0.0:
        return None
    return sxy / math.sqrt(sx * sy)
