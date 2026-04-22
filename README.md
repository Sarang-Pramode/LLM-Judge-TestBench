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

# 2. Run the test suite (83 tests as of Stage 1)
make test

# 3. Lint + typecheck
make lint
make typecheck

# 4. Launch the Streamlit placeholder (Stage 1)
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

Scaffolding stage. Code not yet implemented. The docs and rules are the
source of truth while the repo is being built out.
