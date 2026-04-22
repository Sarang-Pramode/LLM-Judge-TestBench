# LLM Judge Testbench — Project Context

Canonical brief for this repo. If anything in code or other docs conflicts
with this file, update this file or the code — don't let them drift.

## What this repo is

A Python + Streamlit evaluation workbench for **LLM judges** that score
chatbot responses against SME-style rubrics. This repo evaluates **the
judge**, not the chatbot. It should feel like an **evaluation platform**,
not a prompt playground.

## Primary goal

Make LLM judges behave as closely as possible to SME reviewers:

- Measure judge quality against human labels.
- Show where and why the judge disagrees.
- Support future custom judges without redesigning the system.

## Product goals

- Upload datasets with varying source schemas.
- Map source columns into a **normalized internal schema**
  (`dataset_contract.md`).
- Enforce a minimum required set of columns.
- Run multiple judge modules in parallel.
- Score chatbot answers on **1–5 ordinal** scales (5 = best, across all
  pillars).
- Compare judge outputs against SME labels; surface disagreements.
- Dashboard views, plots, per-category and per-reviewer analysis.
- Support completeness judging using an **SME knowledge/policy bank**.
- Keep all LLM calls isolated behind a **provider abstraction**.

## Non-goals

- Do not tightly couple to one model vendor.
- Do not hardcode dataset schemas.
- Do not build one monolithic judge prompt for all pillars.
- Do not mix UI code, judge logic, and provider code together.

## v1 judge pillars

`factual_accuracy`, `hallucination`, `relevance`, `completeness`,
`toxicity`, `bias_discrimination`. See `docs/JUDGE_PILLARS.md`.

## Future pluggable pillars

`tone`, `instruction_following`, `policy_compliance`, `brand_voice`,
`answer_groundedness`, `citation_quality`.

## Core architectural principles

1. **LLM provider isolation** — SDKs live only in `src/llm/`.
2. **Config-driven judges & rubrics** — YAML/JSON; no magic constants.
3. **Normalized dataset schema** — single internal schema; ingestion
   normalizes.
4. **Strong observability** — MLflow (experiments) + Langfuse (traces).
5. **Strict structured outputs** — machine-validated via Pydantic.
6. **Reproducibility** — `run_id`, `dataset_fingerprint`, `rubric_version`,
   `prompt_version`, `model_version`, `kb_version`, `run_config_hash`
   logged per run.

## Required metric set (per pillar)

`exact_match_rate`, `within_1_rate`, `off_by_2_rate`, `off_by_3_plus_rate`,
`mean_absolute_error`, `weighted_kappa`, `spearman_correlation`,
`score_distribution`, `confusion_matrix`, `severity_aware_alignment`.

Plus run-level: `total_rows`, `rows_successfully_scored`,
`rows_failed_parsing`, `avg_latency_ms`, `tokens_in/out`, `cost_estimate`
per judge.

All per-category; reviewer-level when reviewer metadata exists.

## Related docs

- `dataset_contract.md` (repo root) — normalized schema.
- `reviewer_analytics.md` (repo root) — reviewer analytics spec.
- `docs/ARCHITECTURE.md` — module layout, data flow, error model.
- `docs/JUDGE_PILLARS.md` — rubric orientation per pillar.
- `docs/JUDGE_OUTPUT_CONTRACT.md` — strict judge output schema.
- `docs/METRICS.md` — per-pillar + run-level metrics + severity-aware score.
- `docs/COMPLETENESS_KB.md` — knowledge bank spec.
- `docs/OBSERVABILITY.md` — MLflow + Langfuse conventions.
- `docs/ROADMAP.md` — milestones + v1 Definition of Done.
- `AGENTS.md` — coding-agent instructions.
- `.cursor/rules/` — enforced rules.
