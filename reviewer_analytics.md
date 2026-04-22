# Reviewer Analytics

Auto-enabled whenever the normalized dataset contains reviewer metadata
(`reviewer_name` or `reviewer_id`). Computed in
`src/evaluation/reviewer_analysis.py`. Rendered on the dashboard in a
reviewer tab — do NOT show this tab if no reviewer metadata is present.

## Inputs

- Normalized rows with at least `reviewer_name` or `reviewer_id`.
- Judge outputs (from a completed run).
- Optional: multiple rows per `record_id` with different reviewers (enables
  inter-reviewer analysis).

## Metrics (per reviewer, per pillar)

- `n_samples` — rows reviewed.
- `avg_human_score` — mean `label_<pillar>` for this reviewer.
- `judge_vs_reviewer`:
  - `exact_match_rate`
  - `within_1_rate`
  - `off_by_2_rate`
  - `off_by_3_plus_rate`
  - `mean_absolute_error`
  - `weighted_kappa`
  - `severity_aware_alignment`
- `large_miss_rate` — |judge - reviewer| >= 2.

## Inter-reviewer metrics (when multi-review exists)

If the same `record_id` appears with >1 reviewer:

- `overlap_count` per pair of reviewers.
- `pairwise_exact_match_rate`
- `pairwise_weighted_kappa`
- `pairwise_severity_aware_alignment`

This yields an N×N reviewer disagreement matrix per pillar.

## Recommended plots (in `src/dashboard/charts.py`)

- Per-reviewer score distributions (violin or box per pillar).
- Heatmap of average reviewer score by pillar.
- Judge-vs-reviewer agreement bar chart per pillar.
- Reviewer-pair disagreement matrix per pillar, when overlap is sufficient.

## Single-reviewer-per-row case

If `reviewer_*` columns exist but each `record_id` has exactly one
reviewer, still slice **judge disagreement by reviewer** to surface
reviewer-level systematic differences with the judge.

## UI behavior

- Show a "Reviewers" tab only when `has_reviewer` is true on any row.
- Default sort: descending `n_samples`, then descending `large_miss_rate`.
- Expose a filter for minimum `n_samples` to avoid noisy small-N cells.
- Inter-reviewer matrix is hidden until at least one reviewer pair has
  overlap ≥ a configurable threshold (default 20 rows).

## Cross-references

- Dataset contract: `dataset_contract.md` (reviewer columns).
- Metrics: `docs/METRICS.md` (base per-pillar metrics).
- Rules: `.cursor/rules/evaluation-metrics.mdc`,
  `.cursor/rules/streamlit-ui.mdc`.
