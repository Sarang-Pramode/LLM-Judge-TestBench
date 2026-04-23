"""Judge framework.

Public API used by orchestration / UI / Stage 5 pillar judges. The
concrete pillar classes live in sibling modules (``factual_accuracy``,
``hallucination``, ...) and register themselves via
:func:`register_judge` at import time.
"""

from __future__ import annotations

from src.judges.base import BaseJudge, JudgeCoreOutput, JudgeOutcome

# ---------------------------------------------------------------------------
# Stage 5 pillar modules. Imported for their side-effect of registering
# their judge class via ``@register_judge``. Keep the imports at the
# bottom so the base classes / registry are fully defined first.
# ---------------------------------------------------------------------------
from src.judges.bias_discrimination import BiasDiscriminationJudge
from src.judges.completeness import (
    COMPLETENESS_MODE_EXTRA_KEY,
    COMPLETENESS_MODE_GENERIC_FALLBACK,
    COMPLETENESS_MODE_KB_INFORMED,
    CompletenessJudge,
)
from src.judges.config import (
    JudgeBundle,
    JudgeConfig,
    load_judge_bundle,
    load_judge_config,
)
from src.judges.factual_accuracy import FactualAccuracyJudge
from src.judges.hallucination import HallucinationJudge
from src.judges.output_parser import validate_against_rubric
from src.judges.prompt_builder import (
    PromptPair,
    build_default_prompt,
    render_row_block,
    render_rubric_block,
    render_task_profile_block,
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
from src.judges.relevance import RelevanceJudge
from src.judges.toxicity import ToxicityJudge

__all__ = [
    "COMPLETENESS_MODE_EXTRA_KEY",
    "COMPLETENESS_MODE_GENERIC_FALLBACK",
    "COMPLETENESS_MODE_KB_INFORMED",
    "BaseJudge",
    "BiasDiscriminationJudge",
    "CompletenessJudge",
    "FactualAccuracyJudge",
    "HallucinationJudge",
    "JudgeBundle",
    "JudgeConfig",
    "JudgeCoreOutput",
    "JudgeOutcome",
    "PromptPair",
    "RelevanceJudge",
    "ToxicityJudge",
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
    "render_task_profile_block",
    "reset_registry",
    "resolve_judge",
    "validate_against_rubric",
]
