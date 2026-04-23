"""Completeness knowledge bank (Stage 6).

Public API for the SME-maintained completeness KB that informs the
:class:`src.judges.completeness.CompletenessJudge` when it has a
task-specific profile available. See ``docs/ROADMAP.md`` Stage 6 for
the design rationale.
"""

from __future__ import annotations

from src.completeness.kb_loader import load_kb, load_kb_dir
from src.completeness.kb_matcher import KBMatcher, MatchResult
from src.completeness.models import (
    ALLOWED_PRIORITY_LEVELS,
    CompletenessEntry,
    CompletenessKB,
)
from src.completeness.task_profile import TaskProfile, build_task_profile

__all__ = [
    "ALLOWED_PRIORITY_LEVELS",
    "CompletenessEntry",
    "CompletenessKB",
    "KBMatcher",
    "MatchResult",
    "TaskProfile",
    "build_task_profile",
    "load_kb",
    "load_kb_dir",
]
