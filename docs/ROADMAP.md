# Roadmap & Definition of Done

## Implementation philosophy

The codebase should optimize for **modularity, auditability, reproducibility,
extensibility, clear failure analysis, and future judge expansion**. This
repo is an **evaluation platform**, not a prompt playground.

## Implementation workflow (per feature)

When implementing any non-trivial feature, follow this order:

1. **Update types/models** in `src/core/types.py` (or per-pillar `schema.py`).
2. **Implement core logic** in its owning module.
3. **Add tests** in `tests/unit/` (and `tests/integration/` if cross-module).
4. **Wire into UI** via `src/app/pages/` and `src/dashboard/`.
5. **Add observability hooks** (MLflow metrics, Langfuse span attributes).

## Milestones

### Milestone 1 — Foundations

- Repo scaffold matching `docs/ARCHITECTURE.md`.
- Common types in `src/core/types.py` (`NormalizedRow`, `Turn`, `JudgeResult`,
  `RunContext`, exceptions).
- Provider abstraction: `src/llm/base.py` + `factory.py` + `mock_client.py`
  (+ at least one real client behind env-gated credentials).
- Dataset normalization: `ingestion/loaders.py`, `schema_mapper.py`,
  `normalizer.py`, `validators.py`.
- Required-column validation with per-column error reporting.
- Streamlit upload page (`01_upload.py`) with preset persistence.

### Milestone 2 — Judges

- Implement all six v1 judge modules: `factual_accuracy`, `hallucination`,
  `relevance`, `completeness`, `toxicity`, `bias_discrimination`.
- `judges/base.py` + `registry.py` + `prompt_builder.py` + `output_parser.py`
  with strict output + one repair attempt.
- Orchestration runner (`orchestration/runner.py`) with concurrency and
  batching.
- Configure + Run pages (`02_configure.py`, `03_run_eval.py`).
- Basic per-row results table.

### Milestone 3 — Metrics & Dashboards

- `evaluation/metrics.py` implementing all per-pillar metrics from
  `docs/METRICS.md` (including severity-aware alignment).
- `evaluation/agreement.py` (kappa, spearman, confusion matrices).
- `evaluation/slices.by_category` — first-class.
- Dashboard page (`04_dashboard.py`) with overall + per-category tabs.
- Disagreement explorer (`05_disagreements.py`).

### Milestone 4 — Reviewer & Completeness

- `evaluation/reviewer_analysis.py` + reviewer tab on the dashboard when
  reviewer columns exist.
- Completeness KB end-to-end: `completeness/kb_loader.py`,
  `kb_matcher.py`, `task_profile_builder.py`.
- Completeness judge uses KB task profiles; falls back with `kb_miss` tag.
- Prompt/rubric version tracking surfaced in results and run metadata.

### Milestone 5 — Observability & Exports

- MLflow wiring (`observability/mlflow_logger.py`) — params, metrics,
  artifacts.
- Langfuse wiring (`observability/langfuse_tracer.py`) — trace per row,
  span per judge call.
- Run comparison page (`06_compare_runs.py`) — diff MLflow runs.
- Exports (`exports/writers.py`, `exports/reports.py`): normalized dataset
  + judge outputs, per-run report.

## Definition of Done (v1)

v1 ships when **all** of the following are true:

1. A user can upload a dataset with custom source columns.
2. User maps source columns to normalized columns; mappings are persistable.
3. Validation enforces required fields (`record_id`, `user_input`,
   `agent_output`, `category`) and blocks eval on failure.
4. All six pillar judges run successfully through the provider abstraction.
5. Outputs are **strict structured JSON** per `docs/JUDGE_OUTPUT_CONTRACT.md`
   and stored per row.
6. Judge-vs-SME metrics are computed per pillar using the full metric set in
   `docs/METRICS.md`, including `severity_aware_alignment`.
7. Dashboard shows **overall**, **per-category** (always), and
   **per-reviewer** (auto when reviewer columns exist) views.
8. Completeness judge can load and use SME knowledge bank entries, and
   falls back with `kb_miss` when no match exists.
9. All runs are reproducible: `run_id`, dataset fingerprint, rubric
   version, prompt version, model version, KB version recorded.
10. All runs are observable: MLflow params/metrics/artifacts + Langfuse
    traces linked from each result row.
11. Tests cover normalization, parsing, metrics, judge output validation,
    and at least one integration test end-to-end with `mock_client`.
