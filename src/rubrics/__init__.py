"""Rubric models and loaders.

Rubrics are SME-authored YAML that describe how a pillar is scored.
This package owns only the data contract - judge logic lives in
:mod:`src.judges`.
"""

from __future__ import annotations

from src.rubrics.loader import load_rubric, load_rubrics_dir
from src.rubrics.models import (
    ALLOWED_RUBRIC_INPUTS,
    Rubric,
    ScoreAnchor,
    is_known_pillar,
)

__all__ = [
    "ALLOWED_RUBRIC_INPUTS",
    "Rubric",
    "ScoreAnchor",
    "is_known_pillar",
    "load_rubric",
    "load_rubrics_dir",
]
