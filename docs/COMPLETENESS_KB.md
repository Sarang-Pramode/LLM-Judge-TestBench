# Completeness Knowledge Bank

Completeness cannot be judged generically alone. This repo ships an
SME-maintained **knowledge bank** that tells the completeness judge what a
complete answer must contain for certain kinds of user questions.

## Storage

- Source of truth: `configs/knowledge/completeness.yaml` (or `.json`) — or a
  pointer to a managed data file under `data/knowledge/completeness.*`.
- Versioned via a top-level `version` field. `kb_version` is logged on every
  run that uses the KB.
- A Streamlit admin page (`Completeness KB Admin`) lets SMEs browse/edit
  entries with validation.

## Required entry fields

- `kb_id` — stable unique id.
- `question_or_utterance_pattern` — canonical question or utterance pattern.
- `topic_list` — list of topic tags.
- `intent` — intent label.
- `example_agent_response` — a good answer (for reference, not auto-graded).
- `completeness_notes` — SME guidance describing what a complete answer needs.

## Recommended entry fields

- `required_elements` — list of must-have elements.
- `optional_elements` — nice-to-have elements.
- `forbidden_elements` — things an answer must NOT include.
- `policy_refs` — policy document references.
- `priority_level` — e.g. `high` / `medium` / `low`.
- `domain` — business domain.
- `version` — entry-level version.
- `author`, `last_updated`.

## Example entry

```json
{
  "kb_id": "cmp_001",
  "question_or_utterance_pattern": "How do I dispute a transaction?",
  "topic_list": ["transactions", "disputes", "card"],
  "intent": "transaction_dispute",
  "example_agent_response": "You can start a dispute by...",
  "completeness_notes": "A complete answer should explain eligibility, the steps, expected timeline, and escalation path.",
  "required_elements": [
    "clear direct answer",
    "dispute initiation steps",
    "timeline or expectation",
    "required information or documents",
    "escalation/help path"
  ],
  "optional_elements": [
    "brief caveat about pending charges"
  ],
  "forbidden_elements": [
    "promising guaranteed refund"
  ],
  "policy_refs": ["policy_disputes_v3"],
  "priority_level": "high",
  "domain": "consumer_banking",
  "version": "1.0"
}
```

## Retrieval & use in the judge

The completeness judge runs this pipeline:

1. **Identify likely topic/intent/category** from the row. Use
   `row.intent`, `row.topic`, and `row.category` if present; otherwise derive
   via a lightweight LLM classification call (also behind `LLMClient`).
2. **Retrieve candidate KB entries.** First try exact match on `intent`, then
   topic overlap, then fuzzy match on `question_or_utterance_pattern`
   (embedding similarity is optional; keep fuzzy matching pluggable).
3. **Score** against the matched entry's `required_elements`,
   `optional_elements`, `forbidden_elements`, and `completeness_notes`.
4. **Explain** which required elements were present vs missing; surface
   forbidden elements if present.

## Behavior on KB miss

If no KB entry matches:

- Fall back to **generic completeness** scoring based on the rubric alone.
- Reduce `confidence` in the output.
- Add failure tag `kb_miss`.
- Note in `rationale` that task-specific completeness criteria were
  unavailable.

## Validation

- KB entries are validated via Pydantic in `src/knowledge/schema.py`.
- Duplicate `kb_id` values are rejected.
- `required_elements` and `forbidden_elements` must not overlap.
