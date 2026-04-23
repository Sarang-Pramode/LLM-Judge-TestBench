"""Match normalized rows to the best-matching :class:`CompletenessEntry`.

Matching is deliberately a pure, dependency-free scoring function -
it runs for every row at judge time, so it must be cheap and
deterministic. No external embeddings; the signals are:

1. ``intent`` exact match (strongest signal; an explicit intent label
   is rare ground truth).
2. ``topic`` overlap (``row.topic`` or ``row.category`` present in the
   entry's ``topic_list``).
3. ``question_or_utterance_pattern`` keyword overlap against
   ``row.user_input``.

Scores are clamped to ``[0.0, 1.0]``. If the best candidate's score is
below :attr:`KBMatcher.threshold`, the matcher returns a no-match
result and the judge will fall back to generic mode.

Thresholds are intentionally conservative. A false KB match in a
fallback-eligible case is worse than a missed match: the judge would
score against the wrong required_elements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final

from src.completeness.models import CompletenessEntry, CompletenessKB
from src.core.types import NormalizedRow

__all__ = [
    "DEFAULT_MATCH_THRESHOLD",
    "KBMatcher",
    "MatchResult",
]

# Default acceptance threshold. Calibrated so that an ``intent`` exact
# match alone clears it, as does ``topic + keyword`` together.
DEFAULT_MATCH_THRESHOLD: Final[float] = 0.5

# Per-signal caps. Signals compose additively and the total clamps to
# 1.0; these constants live as module-level symbols so tests can
# reference them and future tuning doesn't need magic numbers in
# multiple places.
_INTENT_MATCH_SCORE: Final[float] = 0.7
_TOPIC_ROW_TOPIC_SCORE: Final[float] = 0.4
_TOPIC_ROW_CATEGORY_SCORE: Final[float] = 0.3
_KEYWORD_MAX_SCORE: Final[float] = 0.2

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9']+")


@dataclass(frozen=True)
class MatchResult:
    """Outcome of matching a row against a KB.

    A ``MatchResult`` always exists (never ``None``); when no entry
    scores above the threshold, ``entry`` is ``None`` and ``confidence``
    is ``0.0``. This keeps the judge contract crisp: the judge always
    receives a match object and can inspect it.
    """

    entry: CompletenessEntry | None
    confidence: float
    match_reason: str
    signals: dict[str, float] = field(default_factory=dict)

    @property
    def is_hit(self) -> bool:
        return self.entry is not None


class KBMatcher:
    """Scores rows against a KB and returns the best candidate.

    The matcher is stateless with respect to rows; a single instance
    can be reused across a run (or shared across judge instances).
    """

    def __init__(
        self,
        kb: CompletenessKB,
        *,
        threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"KBMatcher: threshold must be in [0.0, 1.0]; got {threshold!r}.")
        self.kb = kb
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def match(self, row: NormalizedRow) -> MatchResult:
        """Return the best matching :class:`MatchResult` for ``row``.

        Returns a no-match result when the KB is empty, or when the
        best candidate scores below ``threshold``.
        """
        if not self.kb.entries:
            return MatchResult(
                entry=None,
                confidence=0.0,
                match_reason="kb_empty",
                signals={},
            )

        best_entry: CompletenessEntry | None = None
        best_score: float = 0.0
        best_signals: dict[str, float] = {}

        row_keywords = _extract_keywords(row.user_input)

        for entry in self.kb.entries:
            signals = _score_entry(row, row_keywords=row_keywords, entry=entry)
            score = min(sum(signals.values()), 1.0)
            if score > best_score:
                best_entry = entry
                best_score = score
                best_signals = signals

        if best_entry is None or best_score < self.threshold:
            return MatchResult(
                entry=None,
                confidence=best_score,
                match_reason=(
                    "no_candidate_cleared_threshold"
                    if best_entry is not None
                    else "no_candidate_scored"
                ),
                signals=best_signals,
            )

        return MatchResult(
            entry=best_entry,
            confidence=round(best_score, 4),
            match_reason=_describe_match(best_signals),
            signals=best_signals,
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_entry(
    row: NormalizedRow,
    *,
    row_keywords: set[str],
    entry: CompletenessEntry,
) -> dict[str, float]:
    """Compute the (additive) signal breakdown for one entry."""
    signals: dict[str, float] = {}

    # Intent: strongest signal.
    if row.intent and _normalize(row.intent) == _normalize(entry.intent):
        signals["intent"] = _INTENT_MATCH_SCORE

    # Topic: either row.topic or row.category in the entry's topic_list.
    topic_list_norm = {_normalize(t) for t in entry.topic_list}
    if row.topic and _normalize(row.topic) in topic_list_norm:
        signals["row_topic"] = _TOPIC_ROW_TOPIC_SCORE
    elif row.category and _normalize(row.category) in topic_list_norm:
        signals["row_category"] = _TOPIC_ROW_CATEGORY_SCORE

    # Keyword overlap between user_input and the SME-authored pattern.
    pattern_keywords = _extract_keywords(entry.question_or_utterance_pattern)
    if pattern_keywords and row_keywords:
        overlap = row_keywords & pattern_keywords
        if overlap:
            ratio = len(overlap) / len(pattern_keywords)
            signals["keyword_overlap"] = round(min(ratio, 1.0) * _KEYWORD_MAX_SCORE, 4)

    return signals


def _describe_match(signals: dict[str, float]) -> str:
    """Return a short human-readable reason for the winning match."""
    if not signals:
        return "no_signal"
    parts = sorted(signals.items(), key=lambda kv: -kv[1])
    return "+".join(f"{name}={score:.2f}" for name, score in parts)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "will",
        "with",
        "you",
        "your",
    }
)


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _extract_keywords(text: str) -> set[str]:
    """Return the set of content-bearing lowercase word stems in ``text``."""
    tokens = _WORD_RE.findall(text.lower())
    return {tok for tok in tokens if tok not in _STOPWORDS and len(tok) > 2}
