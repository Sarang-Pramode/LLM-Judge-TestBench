# Metrics

All metrics are computed in `src/evaluation/` as pure functions of a per-row
results DataFrame + metadata. No plotting, no IO, no Streamlit. Charts live
in `src/dashboard/charts.py` and consume the outputs of this module.

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

## Tests

Every metric has a golden-fixture unit test in `tests/unit/evaluation/`.
Must-cover cases: all-agree, all-disagree, monotonic, partial labels,
ties in kappa, ordinal edges, empty input, single-row input.
