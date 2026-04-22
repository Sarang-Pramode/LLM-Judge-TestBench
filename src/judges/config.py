"""Judge configuration model + YAML loader.

A ``JudgeConfig`` is how the *runtime* (model alias, temperature, retry
policy, prompt version) is decoupled from the *scoring criteria* (the
rubric). Keeping them in separate files lets SMEs tune rubrics without
touching model routing, and vice versa.

A judge config points at a rubric file. The loader optionally verifies
the referenced rubric version matches ``rubric_version`` in the config -
this catches the classic "rubric was updated but judge config still
declares v0.9" drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.core.exceptions import ConfigLoadError
from src.llm.base import RetryPolicy
from src.rubrics.loader import load_rubric
from src.rubrics.models import Rubric

__all__ = [
    "JudgeBundle",
    "JudgeConfig",
    "load_judge_bundle",
    "load_judge_config",
]


class JudgeConfig(BaseModel):
    """Runtime configuration for a single pillar's judge."""

    model_config = ConfigDict(extra="forbid")

    pillar: str = Field(..., min_length=1)
    prompt_version: str = Field(..., min_length=1)
    rubric_path: str = Field(..., min_length=1)
    rubric_version: str = Field(..., min_length=1)

    #: Model alias resolved by :func:`src.llm.build_client`. Defaults to
    #: the project-wide default so individual judges don't have to
    #: declare it unless they want a bigger/smaller model.
    model_alias: str = "judge-default"

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=1024, ge=1, le=32_768)
    timeout_s: float = Field(default=60.0, gt=0.0)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)

    #: Optional free-form notes SMEs can leave in the config file - never
    #: sent to the model; purely documentary.
    notes: str | None = None

    @model_validator(mode="after")
    def _check_pillar_shape(self) -> JudgeConfig:
        if self.pillar != self.pillar.lower().strip():
            raise ValueError(
                f"JudgeConfig.pillar must be lowercase and untrimmed; got {self.pillar!r}."
            )
        return self


class JudgeBundle(BaseModel):
    """A fully-resolved judge config + its rubric.

    :func:`load_judge_bundle` returns this so the judge factory (Stage
    5) doesn't have to worry about reading two files in the right
    order. Cross-file invariants (pillar match, version match) are
    checked here.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    config: JudgeConfig
    rubric: Rubric

    @model_validator(mode="after")
    def _check_coherence(self) -> JudgeBundle:
        if self.config.pillar != self.rubric.pillar:
            raise ValueError(
                f"JudgeBundle mismatch: config.pillar={self.config.pillar!r} "
                f"but rubric.pillar={self.rubric.pillar!r}."
            )
        if self.config.rubric_version != self.rubric.version:
            raise ValueError(
                f"JudgeBundle version mismatch for pillar {self.rubric.pillar!r}: "
                f"config.rubric_version={self.config.rubric_version!r} vs "
                f"rubric.version={self.rubric.version!r}."
            )
        return self


# ---------------------------------------------------------------------------
# YAML loaders
# ---------------------------------------------------------------------------


def load_judge_config(path: Path | str) -> JudgeConfig:
    """Parse a single judge YAML into a :class:`JudgeConfig`."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigLoadError(f"Judge config file not found: {file_path}")
    try:
        payload: Any = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigLoadError(
            f"Judge config YAML could not be parsed: {file_path} ({exc})"
        ) from exc

    if not isinstance(payload, dict):
        raise ConfigLoadError(
            f"Judge config {file_path} must contain a mapping at the top level; "
            f"got {type(payload).__name__}."
        )

    try:
        return JudgeConfig.model_validate(payload)
    except ValidationError as exc:
        raise ConfigLoadError(f"Judge config {file_path} failed schema validation: {exc}") from exc


def load_judge_bundle(
    config_path: Path | str,
    *,
    rubric_root: Path | str | None = None,
) -> JudgeBundle:
    """Load a judge config *and* its rubric as a coherent bundle.

    ``config.rubric_path`` is interpreted relative to ``rubric_root`` if
    provided, otherwise relative to the judge config file's parent
    directory. Absolute paths are used verbatim.
    """
    config = load_judge_config(config_path)
    rubric_file = _resolve_rubric_path(
        rubric_path=config.rubric_path,
        config_file=Path(config_path),
        rubric_root=Path(rubric_root) if rubric_root is not None else None,
    )
    rubric = load_rubric(rubric_file)
    try:
        return JudgeBundle(config=config, rubric=rubric)
    except ValidationError as exc:
        raise ConfigLoadError(
            f"Judge bundle for pillar {config.pillar!r} is incoherent: {exc}"
        ) from exc


def _resolve_rubric_path(
    *,
    rubric_path: str,
    config_file: Path,
    rubric_root: Path | None,
) -> Path:
    raw = Path(rubric_path)
    if raw.is_absolute():
        return raw
    if rubric_root is not None:
        return rubric_root / raw
    return config_file.parent / raw
