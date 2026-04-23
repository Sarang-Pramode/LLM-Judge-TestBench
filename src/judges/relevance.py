"""Relevance judge.

Measures whether the agent's answer addresses the user's question.
Does not judge factuality, completeness, or tone - those live in
their own pillar judges.
"""

from __future__ import annotations

from src.core.constants import PILLAR_RELEVANCE
from src.judges.base import BaseJudge
from src.judges.registry import register_judge

__all__ = ["RelevanceJudge"]


@register_judge
class RelevanceJudge(BaseJudge):
    """Judge for the ``relevance`` pillar."""

    pillar = PILLAR_RELEVANCE

    def _marker(self) -> None:  # pragma: no cover - structural only
        return None
