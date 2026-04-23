# Metrics

All metrics are computed in `src/evaluation/` as pure functions of a per-row
results DataFrame + metadata. No plotting, no IO, no Streamlit. Charts live
in `src/dashboard/charts.py` (Altair) and `src/dashboard/plotly_charts.py`
(interactive risk views) and consume the outputs of this module.

## Per-pillar metrics (required)

Computed against human labels when `label_<pillar>` is present; otherwise
returned as `None` / `"insufficient_labels"`.

| Metric                     | Type     | Notes                                                              |
|----------------------------|----------|--------------------------------------------------------------------|
| `exact_match_rate`         | float    | fraction of rows where `judge == human`.                           |
| `within_1_rate`            | float    | fraction where `abs(judge - human) <= 1`.                          |
| `off_by_2_rate`            | float    | fraction where `abs(judge - human) == 2`.                          |
| `off_by_3_plus_rate`       | float    | fraction where `abs(judge - human) >= 3`.                          |
| `mean_absolute_error`      | float    | mean of `abs(judge - human)`.                                      |
| `weighted_kappa`           | float    | Cohen's kappa with **quadratic** weights on a 1–5 scale.            |
| `spearman_correlation`     | float    | Spearman ρ between judge and human scores.                         |
| `score_distribution`       | dict     | `{1: n1, 2: n2, 3: n3, 4: n4, 5: n5}` for judge (and for human).   |
| `confusion_matrix`         | 5×5 array| rows = human, cols = judge.                                        |
| `severity_aware_alignment` | float    | business-friendly score; see mapping below.                        |

### Severity-aware alignment score

Maps absolute score distance to a bounded weight, then averages over all
rows with a human label.

| distance | weight |
|----------|--------|
| 0        | 1.00   |
| 1        | 0.75   |
| 2        | 0.40   |
| 3        | 0.10   |
| 4        | 0.00   |

```python
DISTANCE_WEIGHTS = {0: 1.00, 1: 0.75, 2: 0.40, 3: 0.10, 4: 0.00}

def severity_aware_alignment(judge: list[int], human: list[int]) -> float:
    pairs = [(j, h) for j, h in zip(judge, human) if j is not None and h is not None]
    if not pairs:
        return float("nan")
    return sum(DISTANCE_WEIGHTS[abs(j - h)] for j, h in pairs) / len(pairs)
```

This is a single business-friendly number per pillar; report per-category
and per-reviewer versions too.

## Run-level metrics (required)

Computed from the orchestration log + provider accounting.

| Metric                       | Notes                                       |
|------------------------------|---------------------------------------------|
| `total_rows`                 | rows in the normalized dataset.             |
| `rows_successfully_scored`   | completed judge calls per pillar.           |
| `rows_failed_parsing`        | rows tagged `parse_failed` per pillar.       |
| `avg_latency_ms` per judge   | mean wall-clock per call.                   |
| `tokens_in` per judge        | sum of prompt tokens.                       |
| `tokens_out` per judge       | sum of completion tokens.                   |
| `cost_estimate` per judge    | USD using rates in `configs/models.yaml`.   |

All logged as MLflow metrics with the pillar name as a suffix.

## Slices

All per-pillar metrics MUST be recomputed on each slice.

### `by_category` (always on)

`category` is required. Produce a DataFrame indexed by `category × pillar`
with all per-pillar metrics + row counts + label availability.

### `by_reviewer` (auto-enabled when reviewer columns exist)

See `reviewer_analytics.md`. Includes inter-reviewer disagreement when
multiple reviewers overlap on the same `record_id`.

### `by_intent` / `by_topic` (opt-in)

When present on the normalized row, expose as additional slice helpers.

## Dashboard views

Pages consume the DataFrames returned by `evaluation/` through reusable
helpers in `src/dashboard/`:

- `04_dashboard.py` — overall + per-category + per-reviewer tabs.
- `05_disagreements.py` — sortable table of largest judge-vs-human
  disagreements; drill into the row (prompt, evidence, Langfuse trace).
- `06_compare_runs.py` — diff MLflow runs side-by-side.

## North-star thresholds (`configs/evaluation_thresholds.yaml`)

Per-pillar **pass / warn / fail** gates drive the **Risk evidence** page and
MLflow `threshold/*` metrics. Bounds cover:

- `within_1_rate`, `weighted_kappa`, `mean_absolute_error`,
  `severity_aware_alignment`, `large_miss_rate` (defined as
  `off_by_2_rate + off_by_3_plus_rate`).

Higher-is-better metrics use a target (pass) and warn floor; lower-is-better
metrics use a target (pass) and warn ceiling. See `src/evaluation/thresholds.py`.

## Diagnostics & drift (`src/evaluation/diagnostics.py`)

Stdlib-only helpers for risk analytics (no dependency on `metrics.py`):

- **Histogram PMF**: ordinal 1–5 probability mass for judge and human scores.
- **Jensen–Shannon divergence**: symmetric measure of difference between two PMFs
  (e.g. current vs baseline judge distribution).
- **PSI-style stability**: population stability–style index on the same
  5-bin ordinal histograms (baseline vs current).
- **Residuals**: `judge - human`; mean residual and fraction of positive bias.
- **OLS regression**: **human ~ judge** (human as dependent variable) with
  closed-form slope, intercept, and **R²**. The identity line (`human = judge`)
  is shown alongside the fit for “SME-like scale” interpretation.

Baseline comparison uses a pinned `BaselineSnapshot` and requires a matching
`dataset_fingerprint`; see `docs/OBSERVABILITY.md`.

## Tests

Every metric has a golden-fixture unit test in `tests/unit/evaluation/`.
Must-cover cases: all-agree, all-disagree, monotonic, partial labels,
ties in kappa, ordinal edges, empty input, single-row input.

Diagnostics and thresholds have dedicated unit tests under
`tests/unit/test_evaluation_diagnostics.py` and
`tests/unit/test_evaluation_thresholds.py`.
