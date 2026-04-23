"""Unit tests for ``CompletenessEntry`` and ``CompletenessKB``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.completeness.models import (
    ALLOWED_PRIORITY_LEVELS,
    CompletenessEntry,
    CompletenessKB,
)


def _valid_entry_kwargs(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "kb_id": "cmp_test_001",
        "question_or_utterance_pattern": "How do I reset my password?",
        "topic_list": ["account", "password"],
        "intent": "password_reset",
        "example_agent_response": "You can reset your password via Settings > Security.",
        "completeness_notes": "Should describe where to click and what happens next.",
    }
    payload.update(overrides)
    return payload


def test_entry_accepts_minimal_required_fields() -> None:
    entry = CompletenessEntry.model_validate(_valid_entry_kwargs())
    assert entry.kb_id == "cmp_test_001"
    assert entry.topic_list == ["account", "password"]
    assert entry.required_elements == []
    assert entry.version == "1.0"


def test_entry_rejects_bad_kb_id_pattern() -> None:
    with pytest.raises(ValidationError):
        CompletenessEntry.model_validate(_valid_entry_kwargs(kb_id="Bad-ID"))


def test_entry_rejects_empty_topic_list() -> None:
    with pytest.raises(ValidationError, match="topic_list"):
        CompletenessEntry.model_validate(_valid_entry_kwargs(topic_list=[]))


def test_entry_rejects_duplicate_topic() -> None:
    with pytest.raises(ValidationError, match="duplicate topic"):
        CompletenessEntry.model_validate(_valid_entry_kwargs(topic_list=["Account", "account"]))


def test_entry_rejects_unknown_priority() -> None:
    with pytest.raises(ValidationError, match="priority_level"):
        CompletenessEntry.model_validate(_valid_entry_kwargs(priority_level="very-high"))


@pytest.mark.parametrize("level", sorted(ALLOWED_PRIORITY_LEVELS))
def test_entry_accepts_all_allowed_priorities(level: str) -> None:
    entry = CompletenessEntry.model_validate(_valid_entry_kwargs(priority_level=level))
    assert entry.priority_level == level


def test_entry_rejects_required_forbidden_overlap() -> None:
    with pytest.raises(ValidationError, match="forbidden_elements"):
        CompletenessEntry.model_validate(
            _valid_entry_kwargs(
                required_elements=["Clear answer", "timeline"],
                forbidden_elements=["clear answer"],
            )
        )


def test_entry_rejects_required_optional_overlap() -> None:
    with pytest.raises(ValidationError, match="optional_elements"):
        CompletenessEntry.model_validate(
            _valid_entry_kwargs(
                required_elements=["escalation path"],
                optional_elements=["  Escalation  Path  "],
            )
        )


def test_entry_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        CompletenessEntry.model_validate(_valid_entry_kwargs(unknown_field="x"))


# ---------------------------------------------------------------------------
# CompletenessKB
# ---------------------------------------------------------------------------


def _make_entry(kb_id: str) -> CompletenessEntry:
    return CompletenessEntry.model_validate(_valid_entry_kwargs(kb_id=kb_id))


def test_kb_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate kb_id"):
        CompletenessKB.model_validate(
            {
                "version": "v1",
                "entries": [
                    _make_entry("cmp_a").model_dump(),
                    _make_entry("cmp_a").model_dump(),
                ],
            }
        )


def test_kb_by_id_returns_entry() -> None:
    kb = CompletenessKB(version="v1", entries=[_make_entry("cmp_a"), _make_entry("cmp_b")])
    assert kb.by_id("cmp_b").kb_id == "cmp_b"


def test_kb_by_id_raises_keyerror_when_missing() -> None:
    kb = CompletenessKB(version="v1", entries=[_make_entry("cmp_a")])
    with pytest.raises(KeyError):
        kb.by_id("missing")


def test_kb_len_and_iter() -> None:
    kb = CompletenessKB(version="v1", entries=[_make_entry("cmp_a"), _make_entry("cmp_b")])
    assert len(kb) == 2
    assert [e.kb_id for e in kb] == ["cmp_a", "cmp_b"]


def test_kb_fingerprint_stable_under_reordering() -> None:
    a = _make_entry("cmp_a")
    b = _make_entry("cmp_b")
    kb1 = CompletenessKB(version="v1", entries=[a, b])
    kb2 = CompletenessKB(version="v1", entries=[b, a])
    assert kb1.fingerprint() == kb2.fingerprint()


def test_kb_fingerprint_changes_with_version() -> None:
    a = _make_entry("cmp_a")
    kb1 = CompletenessKB(version="v1", entries=[a])
    kb2 = CompletenessKB(version="v2", entries=[a])
    assert kb1.fingerprint() != kb2.fingerprint()


def test_kb_fingerprint_changes_with_content() -> None:
    a = _make_entry("cmp_a")
    b = _make_entry("cmp_b")
    kb1 = CompletenessKB(version="v1", entries=[a])
    kb2 = CompletenessKB(version="v1", entries=[a, b])
    assert kb1.fingerprint() != kb2.fingerprint()
