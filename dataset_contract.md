# Dataset Contract

The **canonical, repo-level contract** for the normalized dataset schema.
Any uploaded file must be mappable to this schema before evaluation runs.

Source of truth. If anything in code, rules, or other docs drifts from this
file, fix the drift.

## Formats supported

CSV, XLSX, JSON, Parquet. Loaded by `src/ingestion/loaders.py`. Mapping is
done interactively in the upload page and persisted to
`configs/mappings/<preset>.yaml`.

## Required columns (block eval if missing)

| Column         | Type     | Notes                                                 |
|----------------|----------|-------------------------------------------------------|
| `record_id`    | `str`    | Unique per row.                                       |
| `user_input`   | `str`    | Raw user utterance / current-turn question.           |
| `agent_output` | `str`    | Chatbot answer to evaluate.                           |
| `category`     | `str`    | First-class slice for filtering and dashboards.       |

## Strongly recommended

| Column              | Type                 | Notes                                  |
|---------------------|----------------------|----------------------------------------|
| `retrieved_context` | `list[str]`          | JSON / list / string tolerated on input.|
| `chat_history`      | `list[Turn]`         | JSON / list / string tolerated on input.|
| `metadata`          | `dict[str, Any]`     | JSON / dict / string tolerated on input.|

`Turn = {role: "user"|"assistant"|"system"|"tool", content: str}`.

## Optional per-pillar labels (ordinal 1–5)

`label_factual_accuracy`, `label_hallucination`, `label_relevance`,
`label_completeness`, `label_toxicity`, `label_bias_discrimination`.

## Optional per-pillar rationales (free text)

`rationale_factual_accuracy`, `rationale_hallucination`,
`rationale_relevance`, `rationale_completeness`, `rationale_toxicity`,
`rationale_bias_discrimination`.

## Optional reviewer columns

`reviewer_name`, `reviewer_id`.

## Optional metadata columns

`intent`, `topic`, `model_name`, `conversation_id`, `turn_index`,
`ground_truth_answer`, `policy_reference`.

## Parsing rules (in `ingestion/normalizer.py`)

- JSON-like strings: `json.loads` → `ast.literal_eval` fallback → `None`
  with a per-row warning flag. Never raise from a single bad cell.
- Normalize empty-ish values to `None`: `""`, `"null"`, `"NaN"`, `"None"`.
- Preserve ALL original columns in `row.source_extras` for debugging.
- Derive flags on every row: `has_reviewer`, `has_context`, `has_labels`,
  `has_history`, `has_ground_truth`.

## Validation (in `ingestion/validators.py`)

- Missing required column → `SchemaValidationError` with a per-column report.
- UI must **block evaluation** on validation errors and surface them clearly.
- Do not silently coerce types. Fail loudly, show a preview table, let the
  user fix the mapping.

## Internal representation (`src/core/types.py`)

```python
class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant", "system", "tool"]
    content: str

class NormalizedRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str
    user_input: str
    agent_output: str
    category: str

    retrieved_context: list[str] | None = None
    chat_history: list[Turn] | None = None
    metadata: dict[str, Any] | None = None

    # Optional per-pillar labels
    label_factual_accuracy: int | None = None
    label_hallucination: int | None = None
    label_relevance: int | None = None
    label_completeness: int | None = None
    label_toxicity: int | None = None
    label_bias_discrimination: int | None = None

    # Optional per-pillar rationales
    rationale_factual_accuracy: str | None = None
    rationale_hallucination: str | None = None
    rationale_relevance: str | None = None
    rationale_completeness: str | None = None
    rationale_toxicity: str | None = None
    rationale_bias_discrimination: str | None = None

    # Optional reviewer + misc
    reviewer_name: str | None = None
    reviewer_id: str | None = None
    intent: str | None = None
    topic: str | None = None
    model_name: str | None = None
    conversation_id: str | None = None
    turn_index: int | None = None
    ground_truth_answer: str | None = None
    policy_reference: str | None = None

    source_extras: dict[str, Any] = Field(default_factory=dict)

    # Derived flags
    has_reviewer: bool = False
    has_context: bool = False
    has_labels: bool = False
    has_history: bool = False
    has_ground_truth: bool = False
```

## Cross-references

- Agent rules: `.cursor/rules/dataset-schema.mdc`.
- Architecture: `docs/ARCHITECTURE.md` (dependency direction).
- Evaluation: `docs/METRICS.md` (how labels feed metrics).
- Reviewer: `reviewer_analytics.md`.
