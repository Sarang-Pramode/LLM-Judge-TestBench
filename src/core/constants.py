"""Project-wide constants shared across modules.

These values must match the docs in `docs/METRICS.md`, `dataset_contract.md`,
`docs/JUDGE_PILLARS.md`, and `docs/JUDGE_OUTPUT_CONTRACT.md`. If any of those
documents change, update this module first and let tests flag downstream
drift.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

# ---------------------------------------------------------------------------
# Pillar identifiers
# ---------------------------------------------------------------------------

PILLAR_FACTUAL_ACCURACY: Final[str] = "factual_accuracy"
PILLAR_HALLUCINATION: Final[str] = "hallucination"
PILLAR_RELEVANCE: Final[str] = "relevance"
PILLAR_COMPLETENESS: Final[str] = "completeness"
PILLAR_TOXICITY: Final[str] = "toxicity"
PILLAR_BIAS_DISCRIMINATION: Final[str] = "bias_discrimination"

PILLARS: Final[tuple[str, ...]] = (
    PILLAR_FACTUAL_ACCURACY,
    PILLAR_HALLUCINATION,
    PILLAR_RELEVANCE,
    PILLAR_COMPLETENESS,
    PILLAR_TOXICITY,
    PILLAR_BIAS_DISCRIMINATION,
)

# ---------------------------------------------------------------------------
# Score scale. All pillars use 1-5 ordinal with 5 == best.
# ---------------------------------------------------------------------------

SCORE_MIN: Final[int] = 1
SCORE_MAX: Final[int] = 5

#: Maximum possible absolute distance between judge and human on a 1-5 scale.
MAX_SCORE_DISTANCE: Final[int] = SCORE_MAX - SCORE_MIN  # == 4

# ---------------------------------------------------------------------------
# Severity-aware alignment weights. See docs/METRICS.md.
# ---------------------------------------------------------------------------

DISTANCE_WEIGHTS: Final[Mapping[int, float]] = MappingProxyType(
    {
        0: 1.00,
        1: 0.75,
        2: 0.40,
        3: 0.10,
        4: 0.00,
    }
)

# ---------------------------------------------------------------------------
# Normalized dataset columns. Canonical source: dataset_contract.md.
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "record_id",
    "user_input",
    "agent_output",
    "category",
)

RECOMMENDED_COLUMNS: Final[tuple[str, ...]] = (
    "retrieved_context",
    "chat_history",
    "metadata",
)

OPTIONAL_REVIEWER_COLUMNS: Final[tuple[str, ...]] = (
    "reviewer_name",
    "reviewer_id",
)

OPTIONAL_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "intent",
    "topic",
    "model_name",
    "conversation_id",
    "turn_index",
    "ground_truth_answer",
    "policy_reference",
)

OPTIONAL_LABEL_COLUMNS: Final[tuple[str, ...]] = tuple(f"label_{p}" for p in PILLARS)
OPTIONAL_RATIONALE_COLUMNS: Final[tuple[str, ...]] = tuple(f"rationale_{p}" for p in PILLARS)

# ---------------------------------------------------------------------------
# Judge output invariants (see docs/JUDGE_OUTPUT_CONTRACT.md).
# ---------------------------------------------------------------------------

#: When score == SCORE_MAX (5), failure_tags must be empty.
#: When failure_tags is non-empty, score must be < SCORE_MAX.
#: rubric_anchor must be within MAX_RUBRIC_ANCHOR_DELTA of score.
MAX_RUBRIC_ANCHOR_DELTA: Final[int] = 1
