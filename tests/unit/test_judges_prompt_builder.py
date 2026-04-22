"""Tests for :mod:`src.judges.prompt_builder`."""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.core.types import NormalizedRow, Turn
from src.judges.prompt_builder import (
    PromptPair,
    build_default_prompt,
    render_row_block,
    render_rubric_block,
)
from src.rubrics.models import Rubric


def _row_with_context(**overrides: Any) -> NormalizedRow:
    defaults: dict[str, Any] = {
        "record_id": "r-001",
        "user_input": "How do I dispute a transaction?",
        "agent_output": "Open the app, tap the charge, then choose Dispute.",
        "category": "transactions",
        "retrieved_context": [
            "To dispute a transaction, follow the in-app flow.",
            "Escalation path: chat -> phone -> branch.",
        ],
        "chat_history": [
            Turn(role="user", content="Hi"),
            Turn(role="assistant", content="Hello"),
        ],
        "metadata": {"device": "ios", "app_version": "3.2.1"},
        "intent": "transaction_dispute",
        "topic": "disputes",
        "ground_truth_answer": "Initiate dispute from the app.",
        "policy_reference": "policy_disputes_v3",
    }
    defaults.update(overrides)
    return NormalizedRow(**defaults)


# ---------------------------------------------------------------------------
# Rubric block
# ---------------------------------------------------------------------------


def test_rubric_block_lists_every_anchor(factual_accuracy_rubric: Rubric) -> None:
    block = render_rubric_block(factual_accuracy_rubric)
    for score in factual_accuracy_rubric.scores():
        assert f" {score} " in block or block.startswith(f"{score} ") or f"\n  {score} " in block
    # Every failure tag is listed:
    for tag in factual_accuracy_rubric.failure_tags:
        assert f"- {tag}" in block


def test_rubric_block_mentions_score_scale(factual_accuracy_rubric: Rubric) -> None:
    block = render_rubric_block(factual_accuracy_rubric)
    assert "Score scale" in block
    assert "1 (worst)" in block
    assert "5 (best)" in block


# ---------------------------------------------------------------------------
# Row block
# ---------------------------------------------------------------------------


def test_row_block_only_renders_rubric_required_fields(
    factual_accuracy_rubric: Rubric,
) -> None:
    row = _row_with_context()
    # Rubric requires user_input + agent_output + retrieved_context.
    block = render_row_block(row, rubric=factual_accuracy_rubric)
    assert "User input" in block
    assert "Agent output" in block
    assert "Retrieved context" in block
    # Not requested by this rubric, so must NOT be rendered:
    assert "Chat history" not in block
    assert "Row metadata" not in block
    assert "Ground-truth answer" not in block


def test_row_block_renders_optional_sections_when_required(
    factual_accuracy_rubric: Rubric,
) -> None:
    # Add chat_history + metadata to rubric requirements.
    rubric = factual_accuracy_rubric.model_copy(
        update={
            "required_inputs": [
                "user_input",
                "agent_output",
                "retrieved_context",
                "chat_history",
                "metadata",
                "ground_truth_answer",
                "policy_reference",
            ]
        }
    )
    row = _row_with_context()
    block = render_row_block(row, rubric=rubric)
    assert "Chat history (2 turns)" in block
    assert "[user] Hi" in block
    assert "Row metadata" in block
    # Metadata rendered as pretty JSON with sorted keys:
    assert '"app_version": "3.2.1"' in block
    assert "Ground-truth answer" in block
    assert "policy_disputes_v3" in block


def test_row_block_skips_optional_sections_when_row_lacks_them(
    factual_accuracy_rubric: Rubric,
) -> None:
    rubric = factual_accuracy_rubric.model_copy(
        update={
            "required_inputs": [
                "user_input",
                "agent_output",
                "retrieved_context",
                "chat_history",
                "metadata",
            ]
        }
    )
    bare = NormalizedRow(
        record_id="r-002",
        user_input="hi",
        agent_output="hello",
        category="general",
    )
    block = render_row_block(bare, rubric=rubric)
    # retrieved_context is rubric-required but the row has none -> section omitted.
    assert "Retrieved context" not in block
    assert "Chat history" not in block
    assert "Row metadata" not in block


# ---------------------------------------------------------------------------
# Full prompt pair
# ---------------------------------------------------------------------------


def test_build_default_prompt_returns_system_and_user_strings(
    factual_accuracy_rubric: Rubric,
) -> None:
    row = _row_with_context()
    pair = build_default_prompt(
        rubric=factual_accuracy_rubric,
        row=row,
        prompt_version="factual_accuracy.v1",
    )
    assert isinstance(pair, PromptPair)
    assert "impartial evaluation judge" in pair.system
    assert "factual_accuracy" in pair.system
    # System prompt carries the hard invariants.
    assert "score == 5" in pair.system
    # User prompt contains the row fields that this rubric cares about.
    assert row.user_input in pair.user
    assert row.agent_output in pair.user
    # And references the prompt version so humans can grep logs:
    assert "factual_accuracy.v1" in pair.system


def test_build_default_prompt_is_deterministic(
    factual_accuracy_rubric: Rubric,
) -> None:
    row = _row_with_context()
    a = build_default_prompt(rubric=factual_accuracy_rubric, row=row, prompt_version="v1")
    b = build_default_prompt(rubric=factual_accuracy_rubric, row=row, prompt_version="v1")
    assert a == b


def test_rubric_block_flags_failure_tag_usage_rule(
    factual_accuracy_rubric: Rubric,
) -> None:
    block = render_rubric_block(factual_accuracy_rubric)
    assert "empty list is valid" in block


def test_metadata_falls_back_to_str_when_unserializable(
    factual_accuracy_rubric: Rubric,
) -> None:
    rubric = factual_accuracy_rubric.model_copy(
        update={"required_inputs": ["user_input", "agent_output", "metadata"]}
    )

    class _Weird:
        def __repr__(self) -> str:
            return "<Weird>"

    row = NormalizedRow(
        record_id="r-weird",
        user_input="hi",
        agent_output="ok",
        category="misc",
        metadata={"obj": _Weird()},
    )
    block = render_row_block(row, rubric=rubric)
    # The default JSON encoder with default=str should serialise it.
    assert "<Weird>" in block
    # And it still looks like a metadata section:
    assert "Row metadata" in block
    # Valid JSON shape (or plain str fallback):
    rendered_section = block.split("Row metadata:\n", 1)[1]
    try:
        json.loads(rendered_section.strip())
    except json.JSONDecodeError:
        # Plain str fallback is also acceptable.
        pass


def test_build_default_prompt_accepts_run_context_without_leaking_ids(
    factual_accuracy_rubric: Rubric, run_context: Any
) -> None:
    row = _row_with_context()
    pair = build_default_prompt(
        rubric=factual_accuracy_rubric,
        row=row,
        prompt_version="v1",
        run_context=run_context,
    )
    # run_id / dataset_fingerprint must NOT appear in prompts.
    assert run_context.run_id not in pair.system
    assert run_context.run_id not in pair.user
    assert run_context.dataset_fingerprint not in pair.system


def test_row_block_renders_empty_user_input(factual_accuracy_rubric: Rubric) -> None:
    row = NormalizedRow(
        record_id="r-empty",
        user_input="",
        agent_output="answer",
        category="x",
    )
    rubric = factual_accuracy_rubric.model_copy(
        update={"required_inputs": ["user_input", "agent_output"]}
    )
    block = render_row_block(row, rubric=rubric)
    assert "(empty)" in block


def test_row_block_header_singular_for_one_chunk(
    factual_accuracy_rubric: Rubric,
) -> None:
    rubric = factual_accuracy_rubric.model_copy(
        update={"required_inputs": ["user_input", "agent_output", "retrieved_context"]}
    )
    row = NormalizedRow(
        record_id="r-1chunk",
        user_input="q",
        agent_output="a",
        category="x",
        retrieved_context=["only one"],
    )
    block = render_row_block(row, rubric=rubric)
    assert "Retrieved context (1 chunk)" in block


@pytest.mark.parametrize(
    ("field", "value", "expected_substring"),
    [
        ("intent", "transaction_dispute", "Intent: transaction_dispute"),
        ("topic", "disputes", "Topic: disputes"),
        ("policy_reference", "policy_disputes_v3", "Policy reference: policy_disputes_v3"),
    ],
)
def test_inline_optional_fields_render(
    factual_accuracy_rubric: Rubric,
    field: str,
    value: str,
    expected_substring: str,
) -> None:
    rubric = factual_accuracy_rubric.model_copy(
        update={"required_inputs": ["user_input", "agent_output", field]}
    )
    row = NormalizedRow(
        record_id="r",
        user_input="q",
        agent_output="a",
        category="x",
        **{field: value},
    )
    block = render_row_block(row, rubric=rubric)
    assert expected_substring in block
