"""Judge framework.

Public API used by orchestration / UI / Stage 5 pillar judges. The
concrete pillar classes live in sibling modules (``factual_accuracy``,
``hallucination``, ...) and register themselves via
:func:`register_judge` at import time.
"""

from __future__ import annotations

from src.judges.base import BaseJudge, JudgeCoreOutput, JudgeOutcome
from src.judges.config import (
    JudgeBundle,
    JudgeConfig,
    load_judge_bundle,
    load_judge_config,
)
from src.judges.output_parser import validate_against_rubric
from src.judges.prompt_builder import (
    PromptPair,
    build_default_prompt,
    render_row_block,
    render_rubric_block,
)
from src.judges.registry import (
    build_judge,
    is_registered,
    list_registered_pillars,
    register_judge,
    registered_judges,
    reset_registry,
    resolve_judge,
)

__all__ = [
    "BaseJudge",
    "JudgeBundle",
    "JudgeConfig",
    "JudgeCoreOutput",
    "JudgeOutcome",
    "PromptPair",
    "build_default_prompt",
    "build_judge",
    "is_registered",
    "list_registered_pillars",
    "load_judge_bundle",
    "load_judge_config",
    "register_judge",
    "registered_judges",
    "render_row_block",
    "render_rubric_block",
    "reset_registry",
    "resolve_judge",
    "validate_against_rubric",
]
