# Architecture

## Repository layout

```
repo/
  dataset_contract.md          # normalized schema spec (repo-root, first-class)
  reviewer_analytics.md        # reviewer analytics spec (repo-root, first-class)
  AGENTS.md                    # coding-agent instructions
  README.md
  docs/                        # deeper design docs
  .cursor/rules/               # enforced agent rules
  configs/
    models.yaml                # model aliases → vendor/model id
    judges/<pillar>.yaml       # per-pillar judge config
    rubrics/<pillar>.yaml      # rubric text + version (or pointer to md)
    mappings/<preset>.yaml     # saved column mappings
    runs/defaults.yaml         # default run settings
    knowledge/completeness.*   # completeness KB (versioned)
  data/                        # samples committed; real data gitignored
  src/
    app/
      streamlit_app.py         # entry. Wiring only.
      pages/
        01_upload.py
        02_configure.py
        03_run_eval.py
        04_dashboard.py
        05_disagreements.py
        06_compare_runs.py
    core/
      types.py                 # NormalizedRow, Turn, JudgeResult base, RunContext
      settings.py              # env + config loading
      constants.py
      exceptions.py            # SchemaValidationError, ProviderError, JudgeExecutionError, ...
    ingestion/
      loaders.py               # csv / xlsx / json / parquet
      schema_mapper.py         # source → normalized column mapping
      validators.py            # enforce required columns, raise SchemaValidationError
      normalizer.py            # parse JSON-ish fields, derive flags, build NormalizedRow
    llm/
      base.py                  # LLMClient protocol + StructuredResponse
      factory.py               # dispatcher: alias → provider client
      openai_client.py
      anthropic_client.py
      vertex_client.py
      mock_client.py           # deterministic offline testing
    judges/
      base.py                  # BaseJudge, JudgeResult, shared validators
      registry.py              # pillar name → judge class
      prompt_builder.py        # shared builders / helpers
      output_parser.py         # strict JSON parse + repair
      factual_accuracy.py
      hallucination.py
      relevance.py
      completeness.py
      toxicity.py
      bias_discrimination.py
    rubrics/
      loader.py                # load + version rubric configs
      models.py                # Pydantic models for rubric content
    completeness/
      kb_loader.py             # load + validate the completeness KB
      kb_matcher.py            # intent/topic/pattern match → KB entries
      task_profile_builder.py  # merged scoring criteria per row
    evaluation/
      metrics.py               # per-pillar metrics (see METRICS.md)
      agreement.py             # kappa, within-k, confusion matrices
      slices.py                # category / reviewer / intent slices
      stability.py             # re-run variance and self-consistency
      reviewer_analysis.py     # reviewer-level analytics
    orchestration/
      runner.py                # executes a run: dataset × judges × config
      batching.py
      concurrency.py           # concurrency cap, semaphores
      caching.py               # optional response cache keyed by prompt hash
    observability/
      mlflow_logger.py
      langfuse_tracer.py
      run_metadata.py          # dataset/rubric/prompt/model/kb fingerprints
    dashboard/
      charts.py                # reusable plot functions — one import point
      tables.py                # results / disagreement tables
      filters.py               # category / reviewer / pillar filters
    exports/
      writers.py               # normalized-data + judge-outputs export
      reports.py               # run reports (HTML / PDF / markdown)
  tests/
    unit/
    integration/
    fixtures/
```

## Dependency direction (strict)

```
app → orchestration → {judges, evaluation, ingestion, exports}
                         ↓         ↓         ↓         ↓
                      core ← llm, rubrics, completeness, observability
dashboard → evaluation, core           # read-only on results; no LLM calls
```

- Never import upward.
- `app/` and `dashboard/` must not import `llm/` or judge internals. They go
  through `orchestration/` and `evaluation/`.
- Vendor SDKs (`openai`, `anthropic`, `google-*`) import only in
  `src/llm/*_client.py`.

## Data flow (happy path)

1. **Upload** (`app/pages/01_upload.py`): `ingestion.loaders` loads the file;
   user maps source → normalized columns in `ingestion.schema_mapper`;
   `ingestion.validators` enforces required columns;
   `ingestion.normalizer` builds `NormalizedRow`s (preserves `source_extras`).
2. **Configure** (`02_configure.py`): pick judges, model alias,
   temperature/retries, KB version, concurrency.
3. **Run** (`03_run_eval.py` → `orchestration.runner`): starts MLflow run,
   creates Langfuse traces, fans out judge calls with `orchestration.concurrency`.
4. **Judge**: builds prompt (pure fn), calls `LLMClient.generate_structured`,
   parses via `judges.output_parser` (retry/repair on bad JSON), returns
   `JudgeResult`.
5. **Persist**: per-row outputs stored as artifacts + parquet; per-run metadata
   fingerprinted by `observability.run_metadata`.
6. **Evaluate**: `evaluation.metrics` + `evaluation.agreement` +
   `evaluation.reviewer_analysis` compute metrics, logged to MLflow.
7. **Dashboard** (`04_dashboard.py`, `05_disagreements.py`,
   `06_compare_runs.py`): read results + metrics only.
8. **Export** (`exports/`): normalized data + judge outputs + reports.

## Reproducibility contract

Every run records, via `observability.run_metadata`:

- `run_id` (uuid).
- `dataset_fingerprint` (hash of normalized rows + mapping preset).
- `rubric_version` per pillar.
- `prompt_version` per pillar.
- `model_alias` and resolved `model_version` (provider-specific id).
- `kb_version` (when completeness is in the run).
- `run_config_hash` (hash of merged resolved configs).

Logged to MLflow params and attached as metadata on every Langfuse trace.

## Error model

- `SchemaValidationError` — ingestion blocks eval; UI surfaces per-column report.
- `ProviderError`, `ProviderTimeoutError`, `ProviderRateLimitError` — raised
  by `llm/` only; orchestration decides retry/skip/fail-fast.
- `JudgeOutputParseError` — raised by `judges.output_parser` after repair
  attempts fail; row is tagged `parse_failed` and included in run with
  status=`error`.
- `JudgeExecutionError` — any other judge failure; row status=`error`,
  reason captured.
- `KBMissError` is **not** raised — completeness judge tags `kb_miss` and
  falls back to generic scoring with lower confidence.
