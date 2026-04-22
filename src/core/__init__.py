"""Core contracts: types, settings, constants, exceptions. No deps on llm/judges."""

from src.core.constants import (
    DISTANCE_WEIGHTS,
    OPTIONAL_LABEL_COLUMNS,
    OPTIONAL_RATIONALE_COLUMNS,
    PILLARS,
    REQUIRED_COLUMNS,
    SCORE_MAX,
    SCORE_MIN,
)
from src.core.exceptions import (
    ConfigLoadError,
    JudgeExecutionError,
    JudgeOutputParseError,
    JudgeTestbenchError,
    KnowledgeBankError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    SchemaValidationError,
)
from src.core.types import (
    Evidence,
    EvidenceStatus,
    JudgeResult,
    NormalizedRow,
    RunContext,
    Turn,
    TurnRole,
)

__all__ = [
    "DISTANCE_WEIGHTS",
    "OPTIONAL_LABEL_COLUMNS",
    "OPTIONAL_RATIONALE_COLUMNS",
    "PILLARS",
    "REQUIRED_COLUMNS",
    "SCORE_MAX",
    "SCORE_MIN",
    "ConfigLoadError",
    "Evidence",
    "EvidenceStatus",
    "JudgeExecutionError",
    "JudgeOutputParseError",
    "JudgeResult",
    "JudgeTestbenchError",
    "KnowledgeBankError",
    "NormalizedRow",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "RunContext",
    "SchemaValidationError",
    "Turn",
    "TurnRole",
]
