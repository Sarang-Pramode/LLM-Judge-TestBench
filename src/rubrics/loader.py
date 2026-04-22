"""YAML loader for :class:`src.rubrics.Rubric` files.

Loaders are deliberately dumb - parse YAML, hand to Pydantic, translate
errors to :class:`ConfigLoadError`. Any "rubric search" / "rubric
catalog" behaviour is handled by the judge layer; this module is only
responsible for the file <-> model round-trip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from src.core.exceptions import ConfigLoadError
from src.rubrics.models import Rubric

__all__ = ["load_rubric", "load_rubrics_dir"]


def load_rubric(path: Path | str) -> Rubric:
    """Load a single rubric YAML file into a :class:`Rubric`.

    Raises:
        ConfigLoadError: if the file is missing, not YAML, or does not
            satisfy the rubric schema.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigLoadError(f"Rubric file not found: {file_path}")
    try:
        payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"Rubric YAML could not be parsed: {file_path} ({exc})") from exc

    if not isinstance(payload, dict):
        raise ConfigLoadError(
            f"Rubric {file_path} must contain a mapping at the top level; "
            f"got {type(payload).__name__}."
        )

    try:
        return Rubric.model_validate(payload)
    except ValidationError as exc:
        raise ConfigLoadError(f"Rubric {file_path} failed schema validation: {exc}") from exc


def load_rubrics_dir(directory: Path | str) -> dict[str, Rubric]:
    """Load every ``*.yaml`` / ``*.yml`` file in ``directory`` into a
    ``{pillar: Rubric}`` map.

    Two rubrics for the same pillar in the same directory is an error -
    the caller should use separate directories or version suffixes in
    file names if they want to stage multiple versions.
    """
    base = Path(directory)
    if not base.is_dir():
        raise ConfigLoadError(f"Rubric directory not found: {base}")

    out: dict[str, Rubric] = {}
    yaml_paths: list[Path] = sorted(
        [p for p in base.iterdir() if p.suffix.lower() in {".yaml", ".yml"}]
    )
    for p in yaml_paths:
        rubric = load_rubric(p)
        if rubric.pillar in out:
            raise ConfigLoadError(
                f"Duplicate rubric for pillar {rubric.pillar!r} in {base}: "
                f"{p.name} conflicts with a previously loaded file."
            )
        out[rubric.pillar] = rubric
    return out


# Internal helper kept as a module symbol so tests can monkeypatch the
# underlying YAML parser if we ever need to test loader-specific error
# handling without writing real files.
def _parse_yaml(text: str) -> Any:  # pragma: no cover - thin wrapper
    return yaml.safe_load(text)
