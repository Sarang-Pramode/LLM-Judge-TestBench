# Judge Pillars

Each pillar is a self-contained judge module at `src/judges/<pillar>.py`
with its own rubric config, prompt template, output schema, failure tags,
and metrics.

**Output schema** is specified in `docs/JUDGE_OUTPUT_CONTRACT.md` — this
document focuses on the rubric orientation and per-pillar extensions.

Every pillar scores on a **1–5 ordinal** scale where **5 = best**, including
pillars where "higher is worse" is natural (toxicity, bias_discrimination).
This keeps all cross-pillar metrics (MAE, kappa, severity-aware alignment)
directionally consistent.

## v1 pillars

### `factual_accuracy`

Whether claims in `agent_output` are factually correct given
`retrieved_context` and/or `ground_truth_answer`.

- Extra output fields: `unsupported_claims`, `contradicted_claims`.
- Failure tags: `fabricated_fact`, `wrong_number`, `misattribution`,
  `outdated_info`, `unsupported_claim`.

### `hallucination`

Stricter than factual_accuracy — even "true" claims count as hallucination
if unsupported by provided context.

- Extra output fields: `ungrounded_spans`.
- Failure tags: `ungrounded_claim`, `invented_citation`,
  `context_contradiction`.

### `relevance`

Whether the answer addresses the user's actual question / intent.

- Extra output fields: `off_topic_spans`, `addresses_intent`.
- Failure tags: `off_topic`, `partial_answer`, `answered_different_question`.

### `completeness` (uses Completeness KB)

Whether the answer contains all required elements for this kind of question.
Pipeline: match KB entry → build task profile → score against required /
optional / forbidden elements.

- Extra output fields: `matched_kb_id`, `match_strength`,
  `required_elements_present`, `required_elements_missing`,
  `forbidden_elements_present`.
- Failure tags: `missing_steps`, `missing_timeline`, `missing_escalation`,
  `forbidden_content`, `kb_miss`.

See `docs/COMPLETENESS_KB.md`.

### `toxicity`

Whether the answer contains toxic, harmful, or unsafe content. 5 = least
toxic.

- Extra output fields: `toxic_spans`, `categories` (`harassment`, `hate`,
  `self_harm`, `violence`, `sexual`, `other`).

### `bias_discrimination`

Whether the answer exhibits bias or discrimination. 5 = least biased.

- Extra output fields: `biased_spans`, `affected_groups`.
- Failure tags: `stereotype`, `unfair_generalization`,
  `exclusionary_language`.

## Future pluggable pillars

`tone`, `instruction_following`, `policy_compliance`, `brand_voice`,
`answer_groundedness`, `citation_quality`.

Adding a pillar must only require:

1. New file `src/judges/<pillar>.py` (one class).
2. New `configs/judges/<pillar>.yaml` + `configs/rubrics/<pillar>.yaml`.
3. Registration line in `src/judges/registry.py`.
4. Optional label column `label_<pillar>` auto-detected by ingestion/UI.

No changes to `orchestration/`, `evaluation/`, `app/`, or `dashboard/`
should be required.

## Rubric & prompt versioning

- `configs/rubrics/<pillar>.yaml` (or `.md`) is the human source of truth
  per pillar.
- `rubric_version` and `prompt_version` are bumped on every substantive
  change and logged to MLflow params and every Langfuse span for
  reproducibility.
