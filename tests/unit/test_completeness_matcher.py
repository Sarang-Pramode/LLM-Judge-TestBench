"""Unit tests for :class:`KBMatcher`."""

from __future__ import annotations

import pytest

from src.completeness.kb_matcher import (
    DEFAULT_MATCH_THRESHOLD,
    KBMatcher,
    MatchResult,
)
from src.completeness.models import CompletenessEntry, CompletenessKB
from src.core.types import NormalizedRow


def _entry(**overrides: object) -> CompletenessEntry:
    payload: dict[str, object] = {
        "kb_id": "cmp_dispute_001",
        "question_or_utterance_pattern": "How do I dispute a transaction on my card?",
        "topic_list": ["disputes", "transactions", "card"],
        "intent": "transaction_dispute",
        "example_agent_response": "Answer.",
        "completeness_notes": "Notes.",
    }
    payload.update(overrides)
    return CompletenessEntry.model_validate(payload)


def _row(**overrides: object) -> NormalizedRow:
    payload: dict[str, object] = {
        "record_id": "R1",
        "user_input": "How do I dispute a charge on my card?",
        "agent_output": "Open a dispute in the app.",
        "category": "disputes",
        "intent": "transaction_dispute",
        "topic": "disputes",
    }
    payload.update(overrides)
    return NormalizedRow.model_validate(payload)


def _kb(*entries: CompletenessEntry, version: str = "v1") -> CompletenessKB:
    return CompletenessKB(version=version, entries=list(entries))


# ---------------------------------------------------------------------------
# Basic scoring
# ---------------------------------------------------------------------------


def test_intent_exact_match_clears_threshold_on_its_own() -> None:
    entry = _entry()
    matcher = KBMatcher(_kb(entry))
    # No topic/keyword signal - only intent.
    row = _row(topic=None, category="other_category", user_input="unrelated text")
    result = matcher.match(row)
    assert result.is_hit
    assert result.entry == entry
    assert result.signals == {"intent": pytest.approx(0.7)}
    assert result.confidence >= DEFAULT_MATCH_THRESHOLD


def test_topic_plus_keyword_clears_threshold() -> None:
    entry = _entry(intent="some_other_intent")
    matcher = KBMatcher(_kb(entry))
    row = _row(intent="mismatched")
    result = matcher.match(row)
    assert result.is_hit
    # topic match via row.topic == "disputes" (in topic_list) -> 0.3
    # keyword overlap {dispute, charge, card, transaction} vs
    # {dispute, transaction, card} gives enough signal to pass 0.5.
    assert "row_topic" in result.signals
    assert result.confidence >= DEFAULT_MATCH_THRESHOLD


def test_category_fallback_used_when_row_topic_missing() -> None:
    entry = _entry(intent="mismatch")
    matcher = KBMatcher(_kb(entry))
    row = _row(intent=None, topic=None, category="disputes")
    result = matcher.match(row)
    # Category in topic_list contributes 0.25 + small keyword match.
    assert "row_category" in result.signals


def test_no_match_returns_threshold_reason() -> None:
    entry = _entry()
    matcher = KBMatcher(_kb(entry))
    row = _row(
        intent="totally_unrelated",
        topic="weather",
        category="weather",
        user_input="what is the weather today?",
    )
    result = matcher.match(row)
    assert not result.is_hit
    assert result.entry is None
    assert result.confidence < DEFAULT_MATCH_THRESHOLD


def test_empty_kb_returns_no_match() -> None:
    matcher = KBMatcher(_kb())
    row = _row()
    result = matcher.match(row)
    assert not result.is_hit
    assert result.match_reason == "kb_empty"


def test_matcher_picks_highest_scoring_candidate() -> None:
    winner = _entry(kb_id="cmp_dispute_001")  # intent matches
    also_valid = _entry(
        kb_id="cmp_generic_cards_001",
        intent="card_info",
        question_or_utterance_pattern="Tell me about my card.",
        topic_list=["card"],
    )
    matcher = KBMatcher(_kb(also_valid, winner))
    result = matcher.match(_row())
    assert result.is_hit
    assert result.entry is not None
    assert result.entry.kb_id == "cmp_dispute_001"


def test_bad_threshold_rejected() -> None:
    with pytest.raises(ValueError):
        KBMatcher(_kb(_entry()), threshold=1.5)


def test_custom_threshold_can_disable_match() -> None:
    """Raising the threshold above a weak-signal match discards it."""
    entry = _entry(intent="mismatched_intent")
    matcher = KBMatcher(_kb(entry), threshold=0.85)
    # Topic + small keyword overlap only; below 0.85.
    row = _row(
        intent=None,
        topic="disputes",
        user_input="What should I do about this?",
    )
    result = matcher.match(row)
    assert not result.is_hit
    assert result.confidence < 0.85


def test_match_reason_includes_signal_breakdown() -> None:
    entry = _entry()
    matcher = KBMatcher(_kb(entry))
    row = _row()
    result = matcher.match(row)
    assert isinstance(result, MatchResult)
    assert "intent" in result.match_reason


def test_case_insensitive_intent_match() -> None:
    entry = _entry(intent="transaction_dispute")
    matcher = KBMatcher(_kb(entry))
    row = _row(intent="Transaction_Dispute ")
    result = matcher.match(row)
    assert result.is_hit
