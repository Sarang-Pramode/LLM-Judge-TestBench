# Judge Output Contract

All judges return strict structured JSON validated via Pydantic in
`src/core/types.py` and pillar-specific subclasses in each judge module.
Parse failures trigger one repair attempt in `judges/output_parser.py`; if
that fails the row is tagged `parse_failed`.

## Common base

Every `JudgeResult` carries, at minimum:

| Field               | Type                                  | Notes                                                          |
|---------------------|---------------------------------------|----------------------------------------------------------------|
| `pillar`            | `str`                                 | e.g. `factual_accuracy`. Must match registered pillar.         |
| `score`             | `Literal[1, 2, 3, 4, 5]`              | Ordinal. Higher = better across all pillars (incl. toxicity, bias). |
| `confidence`        | `float` in `[0, 1]`                   | Judge's self-estimated confidence.                             |
| `decision_summary`  | `str`                                 | One-paragraph human-friendly explanation of the score.         |
| `evidence_for_score`| `list[Evidence]`                      | Structured evidence items. Shape varies per pillar.            |
| `failure_tags`      | `list[str]`                           | Pillar-specific tag enums. Empty if score is 5.                |
| `why_not_higher`    | `str`                                 | What would the answer need to reach the next score up.         |
| `why_not_lower`     | `str`                                 | What keeps it from the next score down.                        |
| `rubric_anchor`     | `Literal[1, 2, 3, 4, 5]`              | Which rubric level the decision anchors on. Usually == `score`.|
| `raw_model_name`    | `str`                                 | Resolved vendor model id (e.g. `gpt-4o-2024-11-20`).           |
| `prompt_version`    | `str`                                 | Semver from the judge config.                                  |
| `rubric_version`    | `str`                                 | Semver from the rubric config.                                 |

`Evidence` is shaped per pillar but the base form is:

```python
class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str
    status: Literal["supported", "unsupported", "contradicted", "n/a"]
    support: str | None           # chunk id, span, or None
```

## Example: `factual_accuracy`

```json
{
  "pillar": "factual_accuracy",
  "score": 4,
  "confidence": 0.82,
  "decision_summary": "Mostly correct with one small unsupported detail.",
  "evidence_for_score": [
    {
      "claim": "The fee is charged after 30 days.",
      "status": "supported",
      "support": "chunk_2"
    },
    {
      "claim": "The fee is always waived for students.",
      "status": "unsupported",
      "support": null
    }
  ],
  "failure_tags": ["unsupported_claim"],
  "why_not_higher": "Contains one unsupported material detail.",
  "why_not_lower": "Core answer remains mostly correct and useful.",
  "rubric_anchor": 4,
  "raw_model_name": "gpt-4o-2024-11-20",
  "prompt_version": "2.0.0",
  "rubric_version": "1.2.0"
}
```

## Per-pillar extensions

Each pillar extends the base with pillar-specific fields.

### `hallucination`

- `ungrounded_spans: list[str]`
- `evidence_for_score[*].status ∈ {"supported", "ungrounded", "contradicted"}`

### `relevance`

- `off_topic_spans: list[str]`
- `addresses_intent: bool`

### `completeness`

- `matched_kb_id: str | None`
- `match_strength: Literal["exact", "topic", "pattern", "embedding", "weak", "miss"]`
- `required_elements_present: list[str]`
- `required_elements_missing: list[str]`
- `forbidden_elements_present: list[str]`

### `toxicity`

- `toxic_spans: list[str]`
- `categories: list[Literal["harassment","hate","self_harm","violence","sexual","other"]]`

### `bias_discrimination`

- `biased_spans: list[str]`
- `affected_groups: list[str]`

## Scale orientation

- **5 = best** on every pillar, including toxicity and bias_discrimination
  (i.e. 5 = least toxic, least biased). This keeps all metrics — MAE,
  kappa, severity-aware alignment — directionally consistent across pillars.
- The rubric markdown per pillar is the human source of truth and spells
  out each level explicitly.

## Validation rules

- `extra = "forbid"` on every model. Unknown fields fail parsing.
- `confidence ∈ [0, 1]` — otherwise reject.
- If `score == 5`, `failure_tags` MUST be empty.
- If `failure_tags` is non-empty, `score < 5`.
- `why_not_higher` is required when `score < 5`.
- `why_not_lower` is required when `score > 1`.
- `rubric_anchor` must be within ±1 of `score`; otherwise flag for review.
