# Agent Instructions

You are working on a Python + Streamlit repository for evaluating LLM
judges against SME labels. This system is **not** a generic chatbot app.
It is a **judge testbench**. The codebase should feel like an evaluation
platform, not a prompt playground.

Read this file, then the cross-referenced specs below, before making
changes.

## Start here — read order

1. `docs/PROJECT_CONTEXT.md` — product brief.
2. `docs/ARCHITECTURE.md` — module layout, dependency direction.
3. `dataset_contract.md` (repo root) — normalized schema contract.
4. `docs/JUDGE_OUTPUT_CONTRACT.md` — strict judge output schema.
5. `docs/METRICS.md` — per-pillar + run-level metrics + severity-aware alignment.
6. `reviewer_analytics.md` (repo root) — reviewer analytics.
7. `docs/COMPLETENESS_KB.md` — knowledge bank spec.
8. `docs/OBSERVABILITY.md` — MLflow + Langfuse conventions.
9. `docs/ROADMAP.md` — milestones + v1 DoD + implementation workflow.
10. `.cursor/rules/` — enforced rules (always-apply + file-scoped).

## Core constraints

- Keep all LLM / provider calls under `src/llm/`.
- Never call provider SDKs directly from judge modules, UI, or metrics code.
- Each evaluation pillar is its own judge module with its own rubric and
  prompt config (`configs/judges/<pillar>.yaml`,
  `configs/rubrics/<pillar>.yaml`).
- All judge outputs must be strict structured JSON validated by Pydantic
  models (see `docs/JUDGE_OUTPUT_CONTRACT.md`).
- Uploaded datasets may have different source schemas, but the app
  normalizes them into the shared schema in `dataset_contract.md`.
- Minimum required normalized columns: `record_id`, `user_input`,
  `agent_output`, `category`. Block eval if any are missing.
- If reviewer metadata is present, compute reviewer-level agreement
  analytics automatically.
- `category` is required and all dashboards must support per-category
  metrics and plots.
- Completeness scoring must use the SME completeness knowledge bank when
  possible; fall back with `kb_miss` tag + lower confidence otherwise.
- Favor typed, modular, testable code with strong separation of concerns.
- Prefer extending **configs** and **registries** rather than hardcoding
  new judges.
- Add tests for every non-trivial behavior.

## Implementation workflow (apply to any non-trivial feature)

1. **Update types/models** — `src/core/types.py` or per-pillar schema.
2. **Implement core logic** in its owning module.
3. **Add tests** in `tests/unit/` (+ `tests/integration/` when cross-module).
4. **Wire into UI** via `src/app/pages/` and reusable `src/dashboard/`
   helpers.
5. **Add observability hooks** — MLflow metrics, Langfuse span attributes.

## Module boundaries (enforced by `.cursor/rules/architecture.mdc`)

```
app → orchestration → {judges, evaluation, ingestion, exports}
                        ↓         ↓         ↓         ↓
                     core ← llm, rubrics, completeness, observability
dashboard → evaluation, core
```

Never import upward. Vendor SDKs import ONLY in `src/llm/*_client.py`.

## Project-wide rules

- Preserve modular boundaries.
- Never place raw provider SDK calls inside judge modules.
- Favor typed Python and Pydantic models (or dataclasses where pure-data).
- Prefer config-driven behavior over hardcoding.
- Validate all external inputs.
- Keep functions small and composable.
- Add tests for normalization, parsing, metrics, and judge output validation.
- Avoid hidden global state.
- All plots must come from reusable chart functions in
  `src/dashboard/charts.py`.
- Log enough metadata to reproduce runs (see
  `observability/run_metadata.py`).

## Judge-specific rules

- One judge class per pillar.
- One rubric config per pillar (`configs/rubrics/<pillar>.yaml`).
- One prompt template per pillar version.
- Return strict structured outputs only.
- Include `confidence` and `failure_tags`.
- Include `why_not_higher` and `why_not_lower`.
- Include `rubric_anchor`, `raw_model_name`, `prompt_version`,
  `rubric_version`.
- Judges MUST NOT mutate source data.

## Completeness-specific rules

- Do not treat completeness as generic only.
- Always attempt to use the completeness KB / task profile first.
- Explicitly mark fallback mode if KB match is weak or absent
  (`match_strength = weak` or `miss`, failure tag `kb_miss`, reduced
  `confidence`).
- Return which required elements were present vs missing.

## UI rules

- Block runs if minimum required columns are not mapped.
- Always surface validation errors clearly.
- Always show category breakdowns on the dashboard.
- If reviewer metadata exists, automatically enable reviewer analytics.
- Allow export of normalized dataset + judge outputs.

## Observability rules

- Every run must have a `run_id`.
- Every row-level judge call must be traceable (Langfuse trace per
  `(record_id, run_id)`; span per judge call).
- Persist `prompt_version`, `rubric_version`, `model_alias`,
  `model_version`, `dataset_fingerprint`, `kb_version`, `run_config_hash`.

## Never do these

- Do NOT create one mega-judge that scores all pillars in one prompt.
- Do NOT mix plotting code with metric computation.
- Do NOT bury schema assumptions in ad hoc code.
- Do NOT hardcode one vendor client into judge logic.
- Do NOT commit secrets — API keys come from env vars only.
- Do NOT let observability failures break a run — wrap, warn, continue.
- Do NOT silently coerce bad data during ingestion — fail loudly.
- Do NOT call judges directly from UI pages — always go through
  `orchestration/`.

## Adding a new pillar (must remain pluggable)

1. New file `src/judges/<pillar>.py` implementing `BaseJudge`.
2. New config `configs/judges/<pillar>.yaml` and rubric
   `configs/rubrics/<pillar>.yaml` (or `.md`).
3. Register the judge in `src/judges/registry.py`.
4. Optional label column `label_<pillar>` is auto-detected by ingestion
   and the UI.
5. No changes required in `orchestration/`, `evaluation/`, or `app/`.

## Final philosophy

Optimize for **modularity, auditability, reproducibility, extensibility,
clear failure analysis, and future judge expansion**. If a change makes
the system harder to audit or reproduce, reject it.
