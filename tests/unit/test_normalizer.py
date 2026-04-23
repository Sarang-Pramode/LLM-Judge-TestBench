"""Tests for :mod:`src.ingestion.normalizer`."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from src.core.types import NormalizedRow, Turn
from src.ingestion.normalizer import (
    is_empty_ish,
    normalize_rows,
    parse_chat_history,
    parse_json_like,
    parse_metadata,
    parse_retrieved_context,
)

# ---------------------------------------------------------------------------
# is_empty_ish
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        " ",
        "   ",
        "null",
        "NULL",
        "None",
        "none",
        "NaN",
        "nan",
        "na",
        "N/A",
        math.nan,
    ],
)
def test_is_empty_ish_recognizes_empty_values(value: Any) -> None:
    assert is_empty_ish(value) is True


@pytest.mark.parametrize("value", ["hello", "0", "false", 0, 1, ["a"], {"k": "v"}])
def test_is_empty_ish_preserves_real_values(value: Any) -> None:
    assert is_empty_ish(value) is False


# ---------------------------------------------------------------------------
# parse_json_like
# ---------------------------------------------------------------------------


def test_parse_json_like_json_array() -> None:
    assert parse_json_like("[1, 2, 3]") == [1, 2, 3]


def test_parse_json_like_json_object() -> None:
    assert parse_json_like('{"a": 1}') == {"a": 1}


def test_parse_json_like_falls_back_to_literal_eval() -> None:
    # Single-quoted Python repr is not valid JSON but is valid literal_eval.
    assert parse_json_like("['a', 'b']") == ["a", "b"]


def test_parse_json_like_none_passes_through() -> None:
    assert parse_json_like(None) is None


def test_parse_json_like_non_string_returns_unchanged() -> None:
    value = [1, 2, 3]
    assert parse_json_like(value) is value


def test_parse_json_like_raises_on_garbage() -> None:
    with pytest.raises(ValueError):
        parse_json_like("not valid anything <>")


# ---------------------------------------------------------------------------
# parse_retrieved_context
# ---------------------------------------------------------------------------


def test_parse_retrieved_context_from_json_list() -> None:
    result = parse_retrieved_context('["chunk_a", "chunk_b"]')
    assert result == ["chunk_a", "chunk_b"]


def test_parse_retrieved_context_from_python_list_passthrough() -> None:
    result = parse_retrieved_context(["a", "b"])
    assert result == ["a", "b"]


def test_parse_retrieved_context_lone_string_wraps_into_list() -> None:
    assert parse_retrieved_context('"just a string"') == ["just a string"]


def test_parse_retrieved_context_empty_like_returns_none() -> None:
    assert parse_retrieved_context("null") is None
    assert parse_retrieved_context(None) is None
    assert parse_retrieved_context("") is None


def test_parse_retrieved_context_wraps_lone_dict() -> None:
    """A single structured chunk is wrapped into a one-element list."""
    result = parse_retrieved_context('{"text": "chunk", "doc_id": "d-1"}')
    assert result == [{"text": "chunk", "doc_id": "d-1"}]


def test_parse_retrieved_context_preserves_mixed_shapes() -> None:
    """A list of mixed strings + dicts keeps item types intact."""
    raw = [
        "plain text chunk",
        {"text": "structured chunk", "doc_id": "d-2", "score": 0.91},
        {"text": "another", "metadata": {"source": "kb"}},
    ]
    result = parse_retrieved_context(raw)
    assert result == raw


def test_parse_retrieved_context_stringifies_weird_items() -> None:
    """Unusual scalar items are coerced to strings without raising."""
    result = parse_retrieved_context([123, "ok"])
    assert result == ["123", "ok"]


def test_parse_retrieved_context_drops_emptyish_items() -> None:
    result = parse_retrieved_context(["a", "", None, "n/a", "b"])
    assert result == ["a", "b"]


def test_parse_retrieved_context_accepts_raw_text_blob() -> None:
    """A non-JSON string is treated as a single chunk of document text."""
    blob = "This is a large free-form document.\nMultiple lines OK."
    result = parse_retrieved_context(blob)
    assert result == [blob]


def test_parse_retrieved_context_empty_list_becomes_none() -> None:
    assert parse_retrieved_context([]) is None
    assert parse_retrieved_context(["", None]) is None


# ---------------------------------------------------------------------------
# parse_chat_history
# ---------------------------------------------------------------------------


def test_parse_chat_history_roundtrip() -> None:
    raw = json.dumps(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    result = parse_chat_history(raw)
    assert result is not None
    assert len(result) == 2
    assert all(isinstance(t, Turn) for t in result)
    assert result[0].role == "user"


def test_parse_chat_history_rejects_bad_role() -> None:
    with pytest.raises(ValueError):
        parse_chat_history(json.dumps([{"role": "robot", "content": "beep"}]))


def test_parse_chat_history_rejects_non_list() -> None:
    with pytest.raises(ValueError):
        parse_chat_history('{"role": "user"}')


def test_parse_chat_history_empty_is_none() -> None:
    assert parse_chat_history(None) is None
    assert parse_chat_history("") is None
    assert parse_chat_history("null") is None


# ---------------------------------------------------------------------------
# parse_metadata
# ---------------------------------------------------------------------------


def test_parse_metadata_roundtrip() -> None:
    assert parse_metadata('{"a": 1, "b": "two"}') == {"a": 1, "b": "two"}


def test_parse_metadata_rejects_list() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        parse_metadata("[1, 2, 3]")


def test_parse_metadata_empty_returns_none() -> None:
    assert parse_metadata(None) is None
    assert parse_metadata("") is None


# ---------------------------------------------------------------------------
# normalize_rows - happy paths
# ---------------------------------------------------------------------------


def test_normalize_rows_builds_normalized_rows_with_required_fields() -> None:
    source = [
        {"id": "r1", "prompt": "hi", "response": "hello", "topic": "greetings"},
        {"id": "r2", "prompt": "thanks", "response": "you're welcome", "topic": "greetings"},
    ]
    mapping = {
        "record_id": "id",
        "user_input": "prompt",
        "agent_output": "response",
        "category": "topic",
    }
    result = normalize_rows(source, mapping=mapping)
    assert result.failures == []
    assert result.success_count == 2
    assert result.success_rate == 1.0
    assert all(isinstance(row, NormalizedRow) for row in result.rows)
    assert result.rows[0].record_id == "r1"
    assert result.rows[0].category == "greetings"


def test_normalize_rows_preserves_source_extras() -> None:
    source = [{"id": "r1", "prompt": "hi", "response": "hi", "topic": "x", "extra": "preserve-me"}]
    mapping = {
        "record_id": "id",
        "user_input": "prompt",
        "agent_output": "response",
        "category": "topic",
    }
    result = normalize_rows(source, mapping=mapping)
    row = result.rows[0]
    assert "extra" in row.source_extras
    assert row.source_extras["extra"] == "preserve-me"


def test_normalize_rows_derives_has_flags() -> None:
    source = [
        {
            "id": "r1",
            "prompt": "hi",
            "response": "hello",
            "topic": "greetings",
            "context": '["ctx1", "ctx2"]',
            "reviewer": "alice",
            "label_fa": "4",
        }
    ]
    mapping = {
        "record_id": "id",
        "user_input": "prompt",
        "agent_output": "response",
        "category": "topic",
        "retrieved_context": "context",
        "reviewer_name": "reviewer",
        "label_factual_accuracy": "label_fa",
    }
    row = normalize_rows(source, mapping=mapping).rows[0]
    assert row.has_reviewer is True
    assert row.has_context is True
    assert row.has_labels is True
    assert row.label_factual_accuracy == 4


def test_normalize_rows_coerces_int_labels() -> None:
    source: list[dict[str, Any]] = [
        {"id": "r1", "prompt": "p", "response": "r", "topic": "t", "lab": "3"},
        {"id": "r2", "prompt": "p", "response": "r", "topic": "t", "lab": 2},
        {"id": "r3", "prompt": "p", "response": "r", "topic": "t", "lab": 5.0},
    ]
    mapping = {
        "record_id": "id",
        "user_input": "prompt",
        "agent_output": "response",
        "category": "topic",
        "label_toxicity": "lab",
    }
    result = normalize_rows(source, mapping=mapping)
    assert [r.label_toxicity for r in result.rows] == [3, 2, 5]


def test_normalize_rows_empty_values_become_none() -> None:
    source = [
        {
            "id": "r1",
            "prompt": "p",
            "response": "r",
            "topic": "t",
            "rev": "null",
            "lab": "",
            "gt": "N/A",
        }
    ]
    mapping = {
        "record_id": "id",
        "user_input": "prompt",
        "agent_output": "response",
        "category": "topic",
        "reviewer_name": "rev",
        "label_toxicity": "lab",
        "ground_truth_answer": "gt",
    }
    row = normalize_rows(source, mapping=mapping).rows[0]
    assert row.reviewer_name is None
    assert row.label_toxicity is None
    assert row.ground_truth_answer is None


# ---------------------------------------------------------------------------
# normalize_rows - failure paths
# ---------------------------------------------------------------------------


def test_normalize_rows_missing_required_field_becomes_row_failure() -> None:
    source = [{"id": "r1", "prompt": "hi", "response": "hello"}]  # no topic
    mapping = {
        "record_id": "id",
        "user_input": "prompt",
        "agent_output": "response",
        "category": "topic",  # points at a column not in the row
    }
    result = normalize_rows(source, mapping=mapping)
    assert result.success_count == 0
    assert result.failure_count == 1
    failure = result.failures[0]
    assert failure.record_id == "r1"
    assert "category" in failure.details


def test_normalize_rows_accepts_nonjson_context_as_free_text_blob() -> None:
    """Non-JSON text in retrieved_context is accepted as a single-chunk
    document blob.

    The system deliberately does not require retrieved_context to be
    structured (RAG systems dump raw documents, chunk lists, or
    metadata-bearing dicts interchangeably). See
    ``parse_retrieved_context`` for the full contract.
    """
    source = [
        {"id": "r1", "prompt": "p", "response": "r", "topic": "t", "ctx": '["a", "b"]'},
        {"id": "r2", "prompt": "p", "response": "r", "topic": "t", "ctx": "free text doc <>"},
    ]
    mapping = {
        "record_id": "id",
        "user_input": "prompt",
        "agent_output": "response",
        "category": "topic",
        "retrieved_context": "ctx",
    }
    result = normalize_rows(source, mapping=mapping)
    assert result.success_count == 2
    assert result.failure_count == 0
    by_id = {row.record_id: row for row in result.rows}
    assert by_id["r1"].retrieved_context == ["a", "b"]
    assert by_id["r2"].retrieved_context == ["free text doc <>"]


def test_normalize_rows_label_out_of_range_is_per_row_failure() -> None:
    source = [{"id": "r1", "prompt": "p", "response": "r", "topic": "t", "lab": "9"}]
    mapping = {
        "record_id": "id",
        "user_input": "prompt",
        "agent_output": "response",
        "category": "topic",
        "label_toxicity": "lab",
    }
    result = normalize_rows(source, mapping=mapping)
    assert result.success_count == 0
    assert result.failure_count == 1


def test_normalize_rows_non_integer_float_label_fails() -> None:
    source = [{"id": "r1", "prompt": "p", "response": "r", "topic": "t", "lab": 2.5}]
    mapping = {
        "record_id": "id",
        "user_input": "prompt",
        "agent_output": "response",
        "category": "topic",
        "label_toxicity": "lab",
    }
    result = normalize_rows(source, mapping=mapping)
    assert result.failure_count == 1
    assert "label_toxicity" in result.failures[0].details


def test_normalize_rows_empty_input_returns_empty_result() -> None:
    empty: list[dict[str, Any]] = []
    result = normalize_rows(empty, mapping={})
    assert result.total == 0
    assert result.success_count == 0
    assert result.failure_count == 0
    assert result.success_rate == 0.0
