# LLM Judge Testbench

Python + Streamlit evaluation workbench for **LLM judges** that score
chatbot responses against SME-style rubrics. This repo evaluates **the
judge**, not the chatbot. It should feel like an **evaluation platform**,
not a prompt playground.

## Why

Make LLM judges behave as closely as possible to SME reviewers. Measure
judge quality against human labels, show where and why the judge
disagrees, and support future custom judges without redesign.

## Features (target v1)

- Upload CSV / XLSX / JSON / Parquet with arbitrary source schemas.
- Map source columns → a normalized internal schema; persist presets.
- Six v1 judge pillars, run in parallel on a 1–5 scale (5 = best):
  `factual_accuracy`, `hallucination`, `relevance`, `completeness`,
  `toxicity`, `bias_discrimination`.
- Plug in future pillars without touching runner, metrics, or UI.
- Completeness judging backed by an SME-maintained knowledge bank.
- Per-pillar + run-level metrics, including a **severity-aware alignment
  score** for business-friendly reporting.
- Dashboards for overall, per-category (always), and per-reviewer (auto
  when reviewer metadata exists) analysis.
- Observability: **MLflow** for experiments and aggregates, **Langfuse**
  for per-row traces.
- All LLM calls isolated behind a provider abstraction.

## Getting started

Requires **Python 3.12**.

```bash
# 1. Create the virtualenv and install runtime + dev deps
make install-dev

# 2. Run the test suite (658 tests as of Stage 10)
make test

# 3. Lint + typecheck
make lint
make typecheck

# 4. Launch the full Streamlit app (Upload -> Configure -> Run -> Dashboard)
make run
```

Without `make`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pytest -q
streamlit run src/app/streamlit_app.py
```

Copy `.env.example` to `.env` when credentials are needed (later stages).
All app env vars use the `JTB_` prefix (see `src/core/settings.py`).

## Architecture at a glance

```
app → orchestration → {judges, evaluation, ingestion, exports}
                        ↓         ↓         ↓         ↓
                     core ← llm, rubrics, completeness, observability
dashboard → evaluation, core
```

Modules live under `src/`: `app`, `core`, `ingestion`, `llm`, `judges`,
`rubrics`, `completeness`, `evaluation`, `orchestration`, `observability`,
`dashboard`, `exports`. See `docs/ARCHITECTURE.md`.

## Start here

For contributors **and** coding agents:

1. `AGENTS.md` — rules for coding agents (read first).
2. `docs/PROJECT_CONTEXT.md` — product brief.
3. `docs/ARCHITECTURE.md` — module layout + data flow.
4. `dataset_contract.md` — normalized schema (repo-root, canonical).
5. `docs/JUDGE_OUTPUT_CONTRACT.md` — strict judge output schema.
6. `docs/JUDGE_PILLARS.md` — per-pillar rubric orientation.
7. `docs/METRICS.md` — per-pillar + run-level metrics + severity-aware alignment.
8. `reviewer_analytics.md` — reviewer analytics (repo-root).
9. `docs/COMPLETENESS_KB.md` — knowledge bank spec.
10. `docs/OBSERVABILITY.md` — MLflow + Langfuse conventions.
11. `docs/ROADMAP.md` — milestones + v1 DoD.
12. `.cursor/rules/` — enforced rules.

## Status

Stages 1-10 complete. The repo now supports:

- Ingestion + schema mapping + validation (Stage 2).
- An LLM provider abstraction with a LangChain-backed Google GenAI
  client plus a mock client (Stage 3).
- A config-driven judge framework and registry (Stage 4).
- All six v1 pillar judges wired end-to-end through the registry
  (Stage 5).
- A completeness knowledge bank: models, loader, matcher, task-profile
  builder, and a KB-aware `CompletenessJudge` that reports
  `kb_informed` or `generic_fallback` mode in `JudgeOutcome.extras`
  (Stage 6). A seed KB ships at `configs/completeness_kb/seed.yaml`.
- A parallel **evaluation runner** (`src/orchestration/`) that fans
  `(row, pillar)` tasks across a thread pool with per-provider
  rate-limit throttles, content-addressed outcome caching, progress
  and per-outcome callbacks, deterministic output ordering, and full
  failure isolation (Stage 7). Public API: `EvaluationRunner`,
  `RunPlan`, `RunResult`, `ConcurrencyPolicy`, `InMemoryOutcomeCache`.
- A **metric engine** (`src/evaluation/`) that joins runner outcomes
  to SME labels and produces per-pillar agreement (exact match,
  within-1, off-by-2, off-by-3+, MAE, severity-aware alignment,
  Cohen's weighted kappa, Spearman, score distributions, confusion
  matrices) plus sliced reports by category / reviewer / intent /
  topic / model and standalone reviewer analytics that activate only
  when reviewer metadata exists (Stage 8). Public API:
  `join_outcomes_with_labels`, `compute_agreement_report`,
  `compute_sliced_report`, `compute_reviewer_analytics`,
  `has_reviewer_signal`.
- A **Streamlit dashboard + disagreement explorer** (`src/app/pages/`
  and `src/dashboard/`) with six pages - Upload, Configure, Run,
  Dashboard, Disagreements, Reviewer analytics (Stage 9). Category
  is a first-class dimension in every view; reviewer analytics
  auto-activates when reviewer metadata is present. All plotting is
  centralized in `src/dashboard/charts.py` (Altair) and pages only
  orchestrate widgets - no inline chart construction. Session state
  keys are namespaced under `jtb.` and documented in
  `src/app/state.py`. Provider is selectable at run time (`mock` for
  offline demos, `google` for real Gemini calls).
- **Observability** (`src/observability/`, Stage 10). Every run gets a
  content-based dataset fingerprint and run-config hash, aggregate
  metrics (agreement, slices, reviewer analytics) flow to **MLflow**,
  and each `(row, pillar)` judge call opens a **Langfuse** generation
  trace. Both backends are optional dependencies and degrade gracefully
  to no-ops when credentials or libraries are missing - observability
  failures never break a run. See `src/observability/__init__.py` for
  the public API (`RunMetadata`, `build_run_metadata`,
  `MLflowLogger`, `LangfuseTracer`, `build_observability_callbacks`)
  and `docs/OBSERVABILITY.md` for environment variables.

Remaining stages (exports, hardening) are tracked in
`docs/ROADMAP.md`.
