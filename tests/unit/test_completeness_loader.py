"""Unit tests for :mod:`src.completeness.kb_loader`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.completeness.kb_loader import load_kb, load_kb_dir
from src.completeness.models import CompletenessKB
from src.core.exceptions import ConfigLoadError


def _valid_entry_dict(kb_id: str, intent: str = "x") -> dict[str, object]:
    return {
        "kb_id": kb_id,
        "question_or_utterance_pattern": f"How do I {intent}?",
        "topic_list": [intent],
        "intent": intent,
        "example_agent_response": "An example answer.",
        "completeness_notes": "Notes",
    }


# ---------------------------------------------------------------------------
# Single-file mode
# ---------------------------------------------------------------------------


def test_load_kb_reads_yaml(tmp_path: Path) -> None:
    payload = {
        "version": "v1",
        "description": "test",
        "entries": [_valid_entry_dict("cmp_a")],
    }
    path = tmp_path / "kb.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    kb = load_kb(path)

    assert isinstance(kb, CompletenessKB)
    assert kb.version == "v1"
    assert [e.kb_id for e in kb.entries] == ["cmp_a"]


def test_load_kb_reads_json(tmp_path: Path) -> None:
    payload = {
        "version": "v1",
        "entries": [_valid_entry_dict("cmp_a"), _valid_entry_dict("cmp_b", intent="y")],
    }
    path = tmp_path / "kb.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    kb = load_kb(path)

    assert len(kb) == 2


def test_load_kb_raises_when_file_missing(tmp_path: Path) -> None:
    with pytest.raises(ConfigLoadError, match="not found"):
        load_kb(tmp_path / "nope.yaml")


def test_load_kb_raises_on_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "kb.yaml"
    path.write_text("- just_a_list\n- of_items\n", encoding="utf-8")
    with pytest.raises(ConfigLoadError, match="mapping at the top level"):
        load_kb(path)


def test_load_kb_raises_on_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "kb.yaml"
    path.write_text("not: valid: yaml:\n  - [\n", encoding="utf-8")
    with pytest.raises(ConfigLoadError, match="could not be parsed"):
        load_kb(path)


def test_load_kb_raises_on_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "kb.json"
    path.write_text("{not json}", encoding="utf-8")
    with pytest.raises(ConfigLoadError, match="could not be parsed"):
        load_kb(path)


def test_load_kb_raises_on_schema_violation(tmp_path: Path) -> None:
    payload = {
        "version": "v1",
        "entries": [
            # Missing the required `intent` field.
            {
                "kb_id": "cmp_a",
                "question_or_utterance_pattern": "pattern",
                "topic_list": ["t"],
                "example_agent_response": "resp",
                "completeness_notes": "notes",
            }
        ],
    }
    path = tmp_path / "kb.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigLoadError, match="failed schema validation"):
        load_kb(path)


# ---------------------------------------------------------------------------
# Directory mode
# ---------------------------------------------------------------------------


def test_load_kb_dir_merges_entries(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text(
        yaml.safe_dump(_valid_entry_dict("cmp_a", intent="alpha")),
        encoding="utf-8",
    )
    (tmp_path / "b.yaml").write_text(
        yaml.safe_dump(_valid_entry_dict("cmp_b", intent="beta")),
        encoding="utf-8",
    )

    kb = load_kb_dir(tmp_path)
    assert len(kb) == 2
    assert {e.kb_id for e in kb.entries} == {"cmp_a", "cmp_b"}
    assert kb.version == "dir.v1"


def test_load_kb_dir_uses_meta_version(tmp_path: Path) -> None:
    (tmp_path / "_meta.yaml").write_text(
        yaml.safe_dump({"version": "seed.v7", "description": "from meta"}),
        encoding="utf-8",
    )
    (tmp_path / "a.yaml").write_text(
        yaml.safe_dump(_valid_entry_dict("cmp_a")),
        encoding="utf-8",
    )

    kb = load_kb_dir(tmp_path)
    assert kb.version == "seed.v7"
    assert kb.description == "from meta"


def test_load_kb_dir_rejects_duplicate_ids_across_files(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text(
        yaml.safe_dump(_valid_entry_dict("cmp_a", intent="alpha")),
        encoding="utf-8",
    )
    (tmp_path / "b.yaml").write_text(
        yaml.safe_dump(_valid_entry_dict("cmp_a", intent="beta")),
        encoding="utf-8",
    )
    with pytest.raises(ConfigLoadError, match="Duplicate kb_id"):
        load_kb_dir(tmp_path)


def test_load_kb_dir_ignores_dotfiles(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text(
        yaml.safe_dump(_valid_entry_dict("cmp_a")),
        encoding="utf-8",
    )
    (tmp_path / ".DS_Store.yaml").write_text("ignored", encoding="utf-8")

    kb = load_kb_dir(tmp_path)
    assert len(kb) == 1


def test_load_kb_dir_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(ConfigLoadError, match="not found"):
        load_kb_dir(tmp_path / "missing_dir")


def test_load_kb_dir_raises_when_meta_missing_version(tmp_path: Path) -> None:
    (tmp_path / "_meta.yaml").write_text(
        yaml.safe_dump({"description": "no version"}), encoding="utf-8"
    )
    (tmp_path / "a.yaml").write_text(
        yaml.safe_dump(_valid_entry_dict("cmp_a")),
        encoding="utf-8",
    )
    with pytest.raises(ConfigLoadError, match="non-empty `version`"):
        load_kb_dir(tmp_path)


def test_shipped_seed_kb_loads() -> None:
    """The KB shipped with the repo must always stay valid."""
    path = Path(__file__).parents[2] / "configs" / "completeness_kb" / "seed.yaml"
    kb = load_kb(path)
    assert kb.version.startswith("seed")
    assert {"transaction_dispute", "fraud_protection", "policy_lookup"} <= {
        e.intent for e in kb.entries
    }
