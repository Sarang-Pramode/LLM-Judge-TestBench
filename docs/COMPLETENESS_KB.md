# Completeness Knowledge Bank

Completeness cannot be judged generically alone. This repo ships an
SME-maintained **knowledge bank** that tells the completeness judge what a
complete answer must contain for certain kinds of user questions.

## Storage

- Source of truth: `configs/completeness_kb/seed.yaml` (single-file mode) or
  a directory of one-entry-per-file YAML/JSON files with an optional
  `_meta.yaml` (directory mode — see `src.completeness.load_kb_dir`).
- Versioned via a top-level `version` field. `kb_version` is logged on every
  run that uses the KB; callers should use `CompletenessKB.fingerprint()`
  when they need a stable content hash rather than an authored version string.
- A Streamlit admin page (`Completeness KB Admin`) will let SMEs browse/edit
  entries with validation (later milestone).

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

The completeness judge runs this pipeline
(`src.completeness.kb_matcher.KBMatcher` +
`src.completeness.task_profile.build_task_profile`):

1. **Score each KB entry** against the row using three additive signals:
   - `intent` exact match (weight 0.7).
   - `row.topic` in the entry's `topic_list` (weight 0.4); falls back to
     `row.category` in `topic_list` (weight 0.3).
   - Keyword overlap between `row.user_input` and the entry's
     `question_or_utterance_pattern` (max weight 0.2, proportional to
     overlap ratio).
   Scores clamp to `[0.0, 1.0]`.
2. **Pick the highest-scoring entry**; if its score is below the matcher's
   `threshold` (default `0.5`), treat it as a no-match.
3. **Build a `TaskProfile`** from the winning entry — `required_elements`,
   `optional_elements`, `forbidden_elements`, and SME notes are surfaced
   to the judge prompt. The matcher's `example_agent_response` is NOT
   injected, to avoid anchoring the judge to a specific phrasing.
4. **Score** the answer against the profile. The LLM output contract
   requires the judge to split the profile's required elements into
   `elements_present` (covered by the answer) and `elements_missing`
   (not covered); together they must cover the full required list.

## Behavior on KB miss

If no KB entry matches (or no KB is wired):

- Fall back to **generic completeness** scoring based on the rubric alone.
- `elements_present` / `elements_missing` stay empty in this mode.
- `JudgeOutcome.extras[completeness_mode]` is set to `generic_fallback`;
  `kb_match` is `"none"`; `kb_match_confidence` still reports the best
  candidate score (useful for tuning the threshold).

In KB-informed mode, `extras` additionally carries `kb_id` (which entry
won) and `kb_match_reason` (a breakdown of the winning signals).

## Validation

- KB entries are validated via Pydantic in `src.completeness.models`.
- Duplicate `kb_id` values are rejected both per entry and across a
  merged directory-mode load.
- `required_elements` must not overlap with `optional_elements` or
  `forbidden_elements` (case- and whitespace-insensitive comparison).
- `topic_list` must be non-empty (the matcher needs a signal to score).
