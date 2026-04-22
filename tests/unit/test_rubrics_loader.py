"""Tests for :mod:`src.rubrics.loader`."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.exceptions import ConfigLoadError
from src.rubrics.loader import load_rubric, load_rubrics_dir

_GOOD_YAML = """
pillar: factual_accuracy
version: "v1.0"
description: "test rubric"
required_inputs:
  - user_input
  - agent_output
anchors:
  - score: 1
    name: worst
    description: bad
  - score: 2
    name: poor
    description: bad
  - score: 3
    name: mid
    description: mid
  - score: 4
    name: good
    description: ok
  - score: 5
    name: perfect
    description: great
failure_tags:
  - unsupported_claim
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_rubric_happy_path(tmp_path: Path) -> None:
    p = _write(tmp_path / "r.yaml", _GOOD_YAML)
    r = load_rubric(p)
    assert r.pillar == "factual_accuracy"
    assert r.version == "v1.0"
    assert list(r.scores()) == [1, 2, 3, 4, 5]
    assert "unsupported_claim" in r.failure_tag_set


def test_load_rubric_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigLoadError, match="not found"):
        load_rubric(tmp_path / "nope.yaml")


def test_load_rubric_invalid_yaml(tmp_path: Path) -> None:
    p = _write(tmp_path / "bad.yaml", "key: value\n  another: : :")
    with pytest.raises(ConfigLoadError, match="could not be parsed"):
        load_rubric(p)


def test_load_rubric_top_level_must_be_mapping(tmp_path: Path) -> None:
    p = _write(tmp_path / "list.yaml", "- a\n- b\n")
    with pytest.raises(ConfigLoadError, match="must contain a mapping"):
        load_rubric(p)


def test_load_rubric_fails_schema_validation(tmp_path: Path) -> None:
    # Remove the score=3 anchor entry in full. YAML remains well-formed;
    # only the schema rule (every score must have an anchor) is violated.
    broken = _GOOD_YAML.replace("  - score: 3\n    name: mid\n    description: mid\n", "")
    p = _write(tmp_path / "r.yaml", broken)
    with pytest.raises(ConfigLoadError, match="failed schema validation"):
        load_rubric(p)


def test_load_rubrics_dir_loads_multiple(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", _GOOD_YAML)
    _write(
        tmp_path / "b.yml",
        _GOOD_YAML.replace("factual_accuracy", "relevance"),
    )
    # Non-YAML files are ignored.
    _write(tmp_path / "readme.md", "# stuff")
    loaded = load_rubrics_dir(tmp_path)
    assert set(loaded) == {"factual_accuracy", "relevance"}


def test_load_rubrics_dir_rejects_duplicate_pillar(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", _GOOD_YAML)
    _write(tmp_path / "b.yaml", _GOOD_YAML)  # same pillar
    with pytest.raises(ConfigLoadError, match="Duplicate rubric"):
        load_rubrics_dir(tmp_path)


def test_load_rubrics_dir_missing(tmp_path: Path) -> None:
    with pytest.raises(ConfigLoadError, match="not found"):
        load_rubrics_dir(tmp_path / "nope")


def test_shipped_factual_accuracy_rubric_loads() -> None:
    """The reference YAML under configs/ must always load cleanly."""
    repo_root = Path(__file__).resolve().parents[2]
    r = load_rubric(repo_root / "configs" / "rubrics" / "factual_accuracy.yaml")
    assert r.pillar == "factual_accuracy"
    assert r.version == "v1.0"
