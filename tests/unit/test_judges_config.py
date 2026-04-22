"""Tests for :mod:`src.judges.config`."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.exceptions import ConfigLoadError
from src.judges.config import (
    JudgeBundle,
    JudgeConfig,
    load_judge_bundle,
    load_judge_config,
)
from src.rubrics.models import Rubric, ScoreAnchor

_RUBRIC_YAML = """
pillar: factual_accuracy
version: "v1.0"
description: r
required_inputs: []
anchors:
  - {score: 1, name: a, description: a}
  - {score: 2, name: b, description: b}
  - {score: 3, name: c, description: c}
  - {score: 4, name: d, description: d}
  - {score: 5, name: e, description: e}
failure_tags: []
"""

_CONFIG_YAML_TEMPLATE = """
pillar: factual_accuracy
prompt_version: "factual_accuracy.v1"
rubric_path: "{rubric_path}"
rubric_version: "{rubric_version}"
model_alias: judge-default
temperature: 0.0
max_output_tokens: 1024
timeout_s: 30.0
retry:
  max_attempts: 2
  initial_backoff_s: 0.1
  max_backoff_s: 1.0
  backoff_multiplier: 2.0
  jitter_s: 0.0
"""


def _write(p: Path, text: str) -> Path:
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# JudgeConfig model
# ---------------------------------------------------------------------------


def test_judge_config_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        JudgeConfig.model_validate(
            {
                "pillar": "factual_accuracy",
                "prompt_version": "v1",
                "rubric_path": "x",
                "rubric_version": "v1",
                "unknown": "field",
            }
        )


def test_judge_config_rejects_non_lowercase_pillar() -> None:
    with pytest.raises(ValidationError, match="lowercase"):
        JudgeConfig(
            pillar="Factual_Accuracy",
            prompt_version="v1",
            rubric_path="x",
            rubric_version="v1",
        )


def test_judge_config_defaults() -> None:
    c = JudgeConfig(
        pillar="factual_accuracy",
        prompt_version="v1",
        rubric_path="x",
        rubric_version="v1",
    )
    assert c.model_alias == "judge-default"
    assert c.temperature == 0.0
    assert c.retry.max_attempts == 3


# ---------------------------------------------------------------------------
# YAML loaders
# ---------------------------------------------------------------------------


def test_load_judge_config_happy_path(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "j.yaml",
        _CONFIG_YAML_TEMPLATE.format(rubric_path="rubrics/x.yaml", rubric_version="v1.0"),
    )
    c = load_judge_config(p)
    assert c.pillar == "factual_accuracy"
    assert c.retry.max_attempts == 2


def test_load_judge_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigLoadError, match="not found"):
        load_judge_config(tmp_path / "nope.yaml")


def test_load_judge_config_invalid_yaml(tmp_path: Path) -> None:
    p = _write(tmp_path / "j.yaml", ":\n:::")
    with pytest.raises(ConfigLoadError, match="could not be parsed"):
        load_judge_config(p)


def test_load_judge_config_top_level_must_be_mapping(tmp_path: Path) -> None:
    p = _write(tmp_path / "j.yaml", "[1, 2, 3]")
    with pytest.raises(ConfigLoadError, match="must contain a mapping"):
        load_judge_config(p)


def test_load_judge_config_schema_failure(tmp_path: Path) -> None:
    p = _write(tmp_path / "j.yaml", "pillar: factual_accuracy\n")  # missing fields
    with pytest.raises(ConfigLoadError, match="failed schema validation"):
        load_judge_config(p)


# ---------------------------------------------------------------------------
# JudgeBundle + load_judge_bundle
# ---------------------------------------------------------------------------


def test_bundle_cross_file_pillar_mismatch() -> None:
    config = JudgeConfig(
        pillar="relevance",
        prompt_version="v1",
        rubric_path="x",
        rubric_version="v1.0",
    )
    rubric = Rubric(
        pillar="factual_accuracy",
        version="v1.0",
        description="x",
        anchors=[ScoreAnchor(score=s, name=str(s), description="d") for s in range(1, 6)],
    )
    with pytest.raises(ValidationError, match=r"config\.pillar"):
        JudgeBundle(config=config, rubric=rubric)


def test_bundle_cross_file_version_mismatch() -> None:
    config = JudgeConfig(
        pillar="factual_accuracy",
        prompt_version="v1",
        rubric_path="x",
        rubric_version="v0.9",
    )
    rubric = Rubric(
        pillar="factual_accuracy",
        version="v1.0",
        description="x",
        anchors=[ScoreAnchor(score=s, name=str(s), description="d") for s in range(1, 6)],
    )
    with pytest.raises(ValidationError, match="version mismatch"):
        JudgeBundle(config=config, rubric=rubric)


def test_load_judge_bundle_resolves_relative_rubric_path(tmp_path: Path) -> None:
    # Layout:
    #   tmp_path/
    #     judges/fa.yaml
    #     rubrics/fa.yaml
    judges_dir = tmp_path / "judges"
    rubrics_dir = tmp_path / "rubrics"
    judges_dir.mkdir()
    rubrics_dir.mkdir()

    _write(rubrics_dir / "fa.yaml", _RUBRIC_YAML)
    _write(
        judges_dir / "fa.yaml",
        _CONFIG_YAML_TEMPLATE.format(rubric_path="../rubrics/fa.yaml", rubric_version="v1.0"),
    )

    bundle = load_judge_bundle(judges_dir / "fa.yaml")
    assert bundle.config.pillar == "factual_accuracy"
    assert bundle.rubric.version == "v1.0"


def test_load_judge_bundle_uses_rubric_root(tmp_path: Path) -> None:
    rubrics_root = tmp_path / "rubrics_root"
    rubrics_root.mkdir()
    _write(rubrics_root / "fa.yaml", _RUBRIC_YAML)
    cfg_path = _write(
        tmp_path / "j.yaml",
        _CONFIG_YAML_TEMPLATE.format(rubric_path="fa.yaml", rubric_version="v1.0"),
    )
    bundle = load_judge_bundle(cfg_path, rubric_root=rubrics_root)
    assert bundle.rubric.pillar == "factual_accuracy"


def test_load_judge_bundle_accepts_absolute_rubric_path(tmp_path: Path) -> None:
    rubric_file = _write(tmp_path / "any_dir_rubric.yaml", _RUBRIC_YAML)
    cfg_path = _write(
        tmp_path / "j.yaml",
        _CONFIG_YAML_TEMPLATE.format(rubric_path=str(rubric_file), rubric_version="v1.0"),
    )
    bundle = load_judge_bundle(cfg_path)
    assert bundle.rubric.pillar == "factual_accuracy"


def test_load_judge_bundle_surfaces_version_mismatch(tmp_path: Path) -> None:
    judges_dir = tmp_path / "judges"
    rubrics_dir = tmp_path / "rubrics"
    judges_dir.mkdir()
    rubrics_dir.mkdir()
    _write(rubrics_dir / "fa.yaml", _RUBRIC_YAML)
    _write(
        judges_dir / "fa.yaml",
        _CONFIG_YAML_TEMPLATE.format(rubric_path="../rubrics/fa.yaml", rubric_version="v9.9"),
    )
    with pytest.raises(ConfigLoadError, match="incoherent"):
        load_judge_bundle(judges_dir / "fa.yaml")


def test_shipped_reference_bundle_loads() -> None:
    """Reference YAML under configs/ must always load as a valid bundle."""
    repo_root = Path(__file__).resolve().parents[2]
    bundle = load_judge_bundle(repo_root / "configs" / "judges" / "factual_accuracy.yaml")
    assert bundle.config.pillar == "factual_accuracy"
    assert bundle.rubric.version == bundle.config.rubric_version
