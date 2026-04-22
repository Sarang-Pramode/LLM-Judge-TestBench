"""Tests for :mod:`src.ingestion.schema_mapper`."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.constants import REQUIRED_COLUMNS
from src.ingestion.schema_mapper import (
    ALLOWED_NORMALIZED_FIELDS,
    ColumnMapping,
    MappingSaveLoadError,
    auto_suggest_mapping,
    load_mapping,
    save_mapping,
)

# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


def test_column_mapping_accepts_valid_fields() -> None:
    mapping = ColumnMapping(
        mappings={
            "record_id": "id",
            "user_input": "prompt",
            "agent_output": "response",
            "category": "topic",
        }
    )
    assert mapping.is_complete_for_evaluation() is True
    assert mapping.missing_required() == []
    assert mapping.source_columns_used() == ["id", "prompt", "response", "topic"]


def test_column_mapping_rejects_unknown_target_field() -> None:
    with pytest.raises(ValueError, match="unknown normalized fields"):
        ColumnMapping(mappings={"not_a_real_field": "foo"})


def test_column_mapping_rejects_duplicate_source() -> None:
    with pytest.raises(ValueError, match="mapped to both"):
        ColumnMapping(mappings={"record_id": "id", "user_input": "id"})  # same source used twice


def test_column_mapping_rejects_blank_source() -> None:
    with pytest.raises(ValueError, match="blank source columns"):
        ColumnMapping(mappings={"record_id": "   "})


def test_column_mapping_rejects_extra_fields() -> None:
    with pytest.raises(ValueError):
        ColumnMapping.model_validate({"mappings": {"record_id": "id"}, "unexpected": "nope"})


def test_missing_required_lists_all_required_when_empty() -> None:
    mapping = ColumnMapping(mappings={})
    assert set(mapping.missing_required()) == set(REQUIRED_COLUMNS)
    assert mapping.is_complete_for_evaluation() is False


def test_allowed_fields_covers_every_required_column() -> None:
    for col in REQUIRED_COLUMNS:
        assert col in ALLOWED_NORMALIZED_FIELDS


# ---------------------------------------------------------------------------
# YAML persistence
# ---------------------------------------------------------------------------


def test_save_and_load_mapping_roundtrip(tmp_path: Path) -> None:
    original = ColumnMapping(
        name="my_preset",
        description="Roundtrip test",
        mappings={
            "record_id": "id",
            "user_input": "prompt",
            "agent_output": "response",
            "category": "topic",
            "label_toxicity": "sme_tox",
        },
    )
    path = tmp_path / "preset.yaml"
    save_mapping(original, path)
    assert path.exists()
    restored = load_mapping(path)
    assert restored == original


def test_load_mapping_missing_file_raises() -> None:
    with pytest.raises(MappingSaveLoadError, match="not found"):
        load_mapping("/tmp/does_not_exist_42.yaml")


def test_load_mapping_empty_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    with pytest.raises(MappingSaveLoadError, match="empty"):
        load_mapping(p)


def test_load_mapping_non_dict_yaml_raises(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(MappingSaveLoadError, match="mapping at top level"):
        load_mapping(p)


def test_load_mapping_invalid_yaml_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(": not: valid: yaml: [", encoding="utf-8")
    with pytest.raises(MappingSaveLoadError, match="Invalid YAML"):
        load_mapping(p)


def test_repo_preset_loads_and_validates() -> None:
    """The checked-in preset under configs/mappings/ must stay valid."""
    repo_root = Path(__file__).resolve().parents[2]
    preset = repo_root / "configs" / "mappings" / "retail_support.yaml"
    if not preset.exists():
        pytest.skip(f"Preset not present: {preset}")
    mapping = load_mapping(preset)
    assert mapping.is_complete_for_evaluation()
    assert mapping.name == "retail_support_v1"


# ---------------------------------------------------------------------------
# auto_suggest_mapping
# ---------------------------------------------------------------------------


def test_auto_suggest_picks_up_direct_names() -> None:
    columns = ["record_id", "user_input", "agent_output", "category", "reviewer_name"]
    suggestion = auto_suggest_mapping(columns)
    assert suggestion.mappings["record_id"] == "record_id"
    assert suggestion.mappings["category"] == "category"
    assert suggestion.mappings["reviewer_name"] == "reviewer_name"


def test_auto_suggest_handles_common_synonyms() -> None:
    columns = ["id", "prompt", "response", "topic"]
    suggestion = auto_suggest_mapping(columns)
    assert suggestion.mappings["record_id"] == "id"
    assert suggestion.mappings["user_input"] == "prompt"
    assert suggestion.mappings["agent_output"] == "response"
    assert suggestion.mappings["category"] == "topic"


def test_auto_suggest_is_conservative_when_no_hit() -> None:
    suggestion = auto_suggest_mapping(["x", "y", "z"])
    assert suggestion.mappings == {}


def test_auto_suggest_is_case_insensitive_and_strips_separators() -> None:
    columns = ["Record-Id", "User_Input", "Agent_Output", "Category"]
    suggestion = auto_suggest_mapping(columns)
    assert "record_id" in suggestion.mappings
    assert "user_input" in suggestion.mappings
    assert suggestion.is_complete_for_evaluation()
