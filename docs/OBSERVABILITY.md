# Observability

Two complementary tools, used together. Every row-level judge call must be
traceable; every run must have a `run_id`.

## Identifiers & fingerprints (per run)

Produced by `observability/run_metadata.py` and logged to both MLflow and
Langfuse:

- `run_id` (uuid4).
- `dataset_fingerprint` — hash of normalized rows + resolved mapping preset.
- `rubric_version_<pillar>` per pillar.
- `prompt_version_<pillar>` per pillar.
- `model_alias` and resolved `model_version` (vendor + concrete id).
- `kb_version` (when completeness is in the run).
- `run_config_hash` — hash of merged resolved configs.

## MLflow — experiments and aggregates

One **run** per evaluation execution (dataset × judge set × model config).

**Params** (always): all fingerprints above.

**Metrics per pillar** (see `METRICS.md`): `exact_match_rate`,
`within_1_rate`, `off_by_2_rate`, `off_by_3_plus_rate`,
`mean_absolute_error`, `weighted_kappa`, `spearman_correlation`,
`severity_aware_alignment`.

**Run-level metrics**:

- `total_rows`
- `rows_successfully_scored_<pillar>`
- `rows_failed_parsing_<pillar>`
- `avg_latency_ms_<pillar>`
- `tokens_in_<pillar>`, `tokens_out_<pillar>`
- `cost_estimate_<pillar>` (USD)

**Agreement, slices, and reviewer depth** (logged from **Run evaluation** after a successful run, inside the same MLflow active run):

- Existing helpers on `MLflowLogger`: `log_agreement_report`, `log_slice_report`, `log_reviewer_analytics` (when reviewer signal exists).
- **Threshold gates**: metrics under keys like `threshold/pillar/<pillar>/overall_status` (numeric encoding) plus per-metric gate detail from `log_threshold_report`.
- **Risk diagnostics**: PMF summaries, Jensen–Shannon / PSI vs baseline (when compatible), OLS and residuals via `log_diagnostics`.
- **Artifacts**: `diagnostics.json` (structured run diagnostics + serializable threshold report); optional `plotly/` HTML bundles for the combined risk-evidence figure when Plotly export succeeds.

**Baseline for drift (product contract)**:

- The Streamlit **Risk evidence** page can **pin the current run** as a session baseline (`BaselineSnapshot`: judge score PMFs per pillar + `dataset_fingerprint` + optional `run_id`).
- Drift metrics (judge PMF JS divergence, PSI) are only interpreted when the current run’s `dataset_fingerprint` matches the baseline; otherwise the UI and diagnostics surface a compatibility flag instead of silent comparison.
- Optional future extension: load a baseline from MLflow by `run_id` or a `role=baseline` tag (not required for the MVP).

**Artifacts** (run bundle):

- Merged resolved run config (`run_config.yaml`).
- Per-row results (parquet) including judge outputs + labels + flags.
- Confusion matrix per pillar.
- Score distribution plots (judge & human) per pillar.
- Failure-tag frequency table per pillar.
- (Optional) completeness KB snapshot used by the run.

## Langfuse — per-row traces

- **Trace**: one per `(record_id, run_id)`, named by `record_id`.
- **Span**: one per judge call with:
  - Input prompt (after redaction if enabled).
  - Raw model response + parsed `JudgeResult`.
  - Token usage, latency, cost estimate, model fingerprint.
  - `prompt_version`, `rubric_version`, `pillar`.
- **Trace metadata**: `run_id`, `dataset_fingerprint`, `kb_version`,
  `category`, `reviewer_id` (if present).
- Langfuse trace URL surfaces on every row in the results table and in the
  disagreement explorer.

## Redaction & secrets

- API keys ONLY from env vars.
- Config flag `observability.redact_pii: bool`. When `true`, run
  `observability/redaction.py` over prompts and outputs before sending to
  Langfuse. Off by default for internal datasets; turn on for external ones.
- Never log full environment dumps or secret material.

## Failure handling

- Observability failures MUST NOT break a run. Wrap writes in try/except,
  log a single warning per failure type, keep processing rows.
- If MLflow is unreachable, continue with local parquet artifact writes.
- If Langfuse is unreachable, continue and emit a run-level warning metric.

## Ownership

- Runners (`orchestration/runner.py`) own MLflow run lifecycle and Langfuse
  trace creation.
- Judges receive the span handle via `RunContext` and only add attributes
  — they never create traces themselves.
- `evaluation/` never touches observability directly; the runner logs
  metrics after `evaluation/` returns.
