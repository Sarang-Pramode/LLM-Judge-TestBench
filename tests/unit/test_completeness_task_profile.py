"""Unit tests for ``TaskProfile`` and ``build_task_profile``."""

from __future__ import annotations

from src.completeness.kb_matcher import MatchResult
from src.completeness.models import CompletenessEntry
from src.completeness.task_profile import TaskProfile, build_task_profile


def _entry() -> CompletenessEntry:
    return CompletenessEntry.model_validate(
        {
            "kb_id": "cmp_x",
            "question_or_utterance_pattern": "pattern",
            "topic_list": ["x"],
            "intent": "i",
            "example_agent_response": "example",
            "completeness_notes": "notes",
            "required_elements": ["a", "b"],
            "optional_elements": ["c"],
            "forbidden_elements": ["d"],
            "policy_refs": ["pol_1"],
            "priority_level": "high",
            "domain": "test",
        }
    )


def test_build_task_profile_returns_none_when_no_entry() -> None:
    match = MatchResult(entry=None, confidence=0.0, match_reason="no_match")
    assert build_task_profile(match) is None


def test_build_task_profile_copies_entry_fields() -> None:
    match = MatchResult(
        entry=_entry(),
        confidence=0.85,
        match_reason="intent=0.70",
        signals={"intent": 0.7, "row_topic": 0.15},
    )
    profile = build_task_profile(match)
    assert isinstance(profile, TaskProfile)
    assert profile.source_kb_id == "cmp_x"
    assert profile.required_elements == ["a", "b"]
    assert profile.optional_elements == ["c"]
    assert profile.forbidden_elements == ["d"]
    assert profile.policy_refs == ["pol_1"]
    assert profile.priority_level == "high"
    assert profile.domain == "test"
    assert profile.match_confidence == 0.85
    assert profile.match_reason == "intent=0.70"
    assert profile.signals == {"intent": 0.7, "row_topic": 0.15}


def test_task_profile_lists_are_independent_copies() -> None:
    entry = _entry()
    match = MatchResult(entry=entry, confidence=0.9, match_reason="intent")
    profile = build_task_profile(match)
    assert profile is not None
    profile.required_elements.append("mutated")
    # Underlying entry must NOT be affected (frozen Pydantic model).
    assert entry.required_elements == ["a", "b"]
