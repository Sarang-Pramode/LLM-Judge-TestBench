"""YAML / JSON loaders for the completeness knowledge bank.

Mirrors the conventions in :mod:`src.rubrics.loader`: loaders are
deliberately dumb - parse bytes, hand to Pydantic, translate errors to
:class:`ConfigLoadError`. Matching / ranking logic lives in
:mod:`src.completeness.kb_matcher`.

Two loading modes:

1. :func:`load_kb` - a single file describing the whole KB
   (``{"version": "...", "entries": [...]}``).
2. :func:`load_kb_dir` - one directory, one file per entry. Files at
   the top level are merged into a single KB. The directory's KB
   version comes from an optional ``_meta.yaml`` / ``_meta.yml`` file
   (``{"version": "..."}``) or defaults to ``"dir.v1"``.

Both paths enforce unique ``kb_id`` across the merged KB.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from src.completeness.models import CompletenessEntry, CompletenessKB
from src.core.exceptions import ConfigLoadError

__all__ = ["load_kb", "load_kb_dir"]


def load_kb(path: Path | str) -> CompletenessKB:
    """Load a single KB file (YAML or JSON) into a :class:`CompletenessKB`.

    Args:
        path: Path to a ``.yaml`` / ``.yml`` / ``.json`` file whose
            top-level object has keys ``version`` and ``entries``.

    Returns:
        The validated :class:`CompletenessKB`.

    Raises:
        ConfigLoadError: file missing, unparseable, wrong shape, or
            any entry fails schema validation.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigLoadError(f"Completeness KB file not found: {file_path}")

    payload = _parse_structured_file(file_path)

    if not isinstance(payload, dict):
        raise ConfigLoadError(
            f"Completeness KB {file_path} must contain a mapping at the top "
            f"level; got {type(payload).__name__}."
        )

    try:
        return CompletenessKB.model_validate(payload)
    except ValidationError as exc:
        raise ConfigLoadError(
            f"Completeness KB {file_path} failed schema validation: {exc}"
        ) from exc


def load_kb_dir(directory: Path | str) -> CompletenessKB:
    """Load every KB entry file in ``directory`` into one
    :class:`CompletenessKB`.

    Conventions:

    - Each ``*.yaml`` / ``*.yml`` / ``*.json`` file at the top level
      of ``directory`` describes ONE entry (``CompletenessEntry`` shape).
    - A ``_meta.yaml`` / ``_meta.yml`` / ``_meta.json`` file (optional)
      provides the KB-level ``version`` (and optional ``description``).
      Without it the loader uses ``"dir.v1"`` as the KB version.
    - Filenames starting with ``.`` or ``_`` (other than ``_meta.*``)
      are skipped so SMEs can stash drafts alongside the live files.

    Raises:
        ConfigLoadError: directory missing, any file is bad, or two
            files declare the same ``kb_id``.
    """
    base = Path(directory)
    if not base.is_dir():
        raise ConfigLoadError(f"Completeness KB directory not found: {base}")

    meta_version = "dir.v1"
    meta_description: str | None = None

    meta_path = _find_meta_file(base)
    if meta_path is not None:
        meta_payload = _parse_structured_file(meta_path)
        if not isinstance(meta_payload, dict):
            raise ConfigLoadError(
                f"KB meta file {meta_path} must be a mapping; got {type(meta_payload).__name__}."
            )
        version = meta_payload.get("version")
        description = meta_payload.get("description")
        if not isinstance(version, str) or not version:
            raise ConfigLoadError(f"KB meta file {meta_path} must provide a non-empty `version`.")
        meta_version = version
        if description is not None:
            if not isinstance(description, str):
                raise ConfigLoadError(f"KB meta file {meta_path}: `description` must be a string.")
            meta_description = description

    entries: list[CompletenessEntry] = []
    seen_ids: dict[str, Path] = {}
    entry_paths = sorted(
        p
        for p in base.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".yaml", ".yml", ".json"}
        and not _is_meta_file(p)
        and not p.name.startswith(".")
    )

    for p in entry_paths:
        payload = _parse_structured_file(p)
        if not isinstance(payload, dict):
            raise ConfigLoadError(
                f"Completeness entry {p} must be a mapping; got {type(payload).__name__}."
            )
        try:
            entry = CompletenessEntry.model_validate(payload)
        except ValidationError as exc:
            raise ConfigLoadError(
                f"Completeness entry {p} failed schema validation: {exc}"
            ) from exc
        if entry.kb_id in seen_ids:
            raise ConfigLoadError(
                f"Duplicate kb_id {entry.kb_id!r}: {p.name} conflicts with "
                f"{seen_ids[entry.kb_id].name}."
            )
        seen_ids[entry.kb_id] = p
        entries.append(entry)

    try:
        return CompletenessKB.model_validate(
            {
                "version": meta_version,
                "description": meta_description,
                "entries": [e.model_dump() for e in entries],
            }
        )
    except ValidationError as exc:  # pragma: no cover - impossible given above
        raise ConfigLoadError(f"KB directory {base} produced an invalid merged KB: {exc}") from exc


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


_META_STEMS = {"_meta"}


def _is_meta_file(path: Path) -> bool:
    return path.stem in _META_STEMS and path.suffix.lower() in {".yaml", ".yml", ".json"}


def _find_meta_file(base: Path) -> Path | None:
    for name in ("_meta.yaml", "_meta.yml", "_meta.json"):
        p = base / name
        if p.is_file():
            return p
    return None


def _parse_structured_file(path: Path) -> Any:
    """Parse a YAML or JSON file into a Python object, tagged with a
    clear error if the content is malformed.
    """
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            return json.loads(text)
        return yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ConfigLoadError(f"Completeness KB file {path} could not be parsed: {exc}") from exc
