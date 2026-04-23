"""Streamlit upload + column mapping page.

Responsibilities (UI only - no judge / metrics / provider imports):
1. Accept a CSV / XLSX / JSON / Parquet upload.
2. Show the detected source columns.
3. Let the user map source columns to normalized fields, starting from an
   auto-suggested mapping. Optionally load a saved preset.
4. Validate the mapping against the required normalized columns.
5. Normalize a preview of rows and render them.
6. Persist normalized rows + mapping into ``st.session_state`` under
   stable keys so downstream pages (configure, run_eval) can read them.

Downstream session keys written here:
- ``jtb.source_columns``: ``list[str]``
- ``jtb.source_rows``: raw ``list[dict]`` (for debug / re-normalization)
- ``jtb.mapping``: ``ColumnMapping``
- ``jtb.normalized_rows``: ``list[NormalizedRow]``
- ``jtb.dataset_name``: uploaded filename (for run metadata)
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.core.constants import (
    OPTIONAL_LABEL_COLUMNS,
    OPTIONAL_METADATA_COLUMNS,
    OPTIONAL_RATIONALE_COLUMNS,
    OPTIONAL_REVIEWER_COLUMNS,
    RECOMMENDED_COLUMNS,
    REQUIRED_COLUMNS,
)
from src.core.exceptions import SchemaValidationError
from src.ingestion import (
    ColumnMapping,
    LoadedFile,
    auto_suggest_mapping,
    detect_format,
    load_from_bytes,
    normalize_rows,
    validate_mapping,
    validate_normalization,
)

_UNMAPPED_LABEL = "-- none --"
_ALL_NORMALIZED_ORDER: tuple[str, ...] = (
    REQUIRED_COLUMNS
    + RECOMMENDED_COLUMNS
    + OPTIONAL_REVIEWER_COLUMNS
    + OPTIONAL_METADATA_COLUMNS
    + OPTIONAL_LABEL_COLUMNS
    + OPTIONAL_RATIONALE_COLUMNS
)


def _mapping_seed_for_upload(
    source_columns: list[str],
    *,
    previous: ColumnMapping | None,
    previous_fingerprint: tuple[str, ...] | None,
) -> tuple[ColumnMapping, tuple[str, ...]]:
    """Same logic as :func:`src.ingestion.schema_mapper.select_mapping_seed`.

    Implemented here with :func:`auto_suggest_mapping` only so the Upload
    page does not depend on re-exporting ``select_mapping_seed`` from
    ``src.ingestion`` (a partial checkout can otherwise break every
    ``from src.ingestion import …`` at package import time).
    """
    fingerprint = tuple(sorted(source_columns))
    suggested = auto_suggest_mapping(source_columns)
    if previous_fingerprint is None or previous_fingerprint != fingerprint:
        return suggested, fingerprint
    return (previous if previous is not None else suggested), fingerprint


def render() -> None:
    """Render the upload page. Called by Streamlit's multipage runtime."""
    st.set_page_config(page_title="Upload dataset", layout="wide")
    st.title("Upload and map dataset")
    st.caption(
        "Upload a file, map its source columns to the normalized schema, "
        "and preview the result. Required normalized columns must be mapped "
        "before evaluation can run."
    )

    uploaded = st.file_uploader(
        "Upload dataset file",
        type=["csv", "xlsx", "json", "parquet"],
        accept_multiple_files=False,
    )
    if uploaded is None:
        st.info("Pick a CSV / XLSX / JSON / Parquet file to begin.")
        return

    try:
        fmt = detect_format(uploaded.name)
        loaded = load_from_bytes(uploaded.getvalue(), fmt=fmt)
    except SchemaValidationError as exc:
        st.error(f"Could not load file: {exc}")
        return

    st.session_state["jtb.source_columns"] = loaded.source_columns
    st.session_state["jtb.source_rows"] = loaded.rows
    st.session_state["jtb.dataset_name"] = uploaded.name

    _render_source_summary(loaded)
    mapping = _render_mapping_editor(loaded.source_columns)
    st.session_state["jtb.mapping"] = mapping

    _render_validation_and_preview(loaded, mapping)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _render_source_summary(loaded: LoadedFile) -> None:
    with st.expander(f"Source file summary ({len(loaded.rows)} rows)", expanded=True):
        st.write(f"Detected format: `{loaded.format}`")
        st.write(f"Source columns ({len(loaded.source_columns)}):")
        st.code(", ".join(loaded.source_columns) or "<none>", language=None)


def _render_mapping_editor(source_columns: list[str]) -> ColumnMapping:
    """Render selectboxes for each normalized field, return the mapping."""
    st.subheader("Column mapping")
    st.caption(
        "Each normalized field listed below can be left unmapped except "
        "the required ones (highlighted)."
    )

    previous: ColumnMapping | None = st.session_state.get("jtb.mapping")
    fp_prev: tuple[str, ...] | None = st.session_state.get("jtb.mapping_columns_fingerprint")
    seed, fp_new = _mapping_seed_for_upload(
        source_columns,
        previous=previous,
        previous_fingerprint=fp_prev,
    )
    st.session_state["jtb.mapping_columns_fingerprint"] = fp_new

    options = [_UNMAPPED_LABEL, *source_columns]
    groups = (
        ("Required", REQUIRED_COLUMNS, True),
        ("Recommended", RECOMMENDED_COLUMNS, False),
        ("Reviewer", OPTIONAL_REVIEWER_COLUMNS, False),
        ("Metadata", OPTIONAL_METADATA_COLUMNS, False),
        ("Labels", OPTIONAL_LABEL_COLUMNS, False),
        ("Rationales", OPTIONAL_RATIONALE_COLUMNS, False),
    )

    chosen: dict[str, str] = {}
    for heading, fields, required in groups:
        st.markdown(f"**{heading}**")
        cols = st.columns(2)
        for i, normalized_field in enumerate(fields):
            default_source = seed.mappings.get(normalized_field)
            default_index = options.index(default_source) if default_source in options else 0
            label = f":red[{normalized_field} *]" if required else normalized_field
            picked = cols[i % 2].selectbox(
                label,
                options=options,
                index=default_index,
                key=f"jtb.map.{normalized_field}",
                help=_help_for(normalized_field),
            )
            if picked != _UNMAPPED_LABEL:
                chosen[normalized_field] = picked

    try:
        mapping = ColumnMapping(mappings=chosen, name=seed.name)
    except ValueError as exc:
        st.error(f"Mapping is invalid: {exc}")
        # Fallback: return an empty mapping so the validator below can
        # still run and report missing required columns.
        return ColumnMapping(mappings={})
    return mapping


def _render_validation_and_preview(loaded: LoadedFile, mapping: ColumnMapping) -> None:
    st.subheader("Validation")
    mapping_report = validate_mapping(mapping, source_columns=loaded.source_columns)
    if not mapping_report.ok:
        st.error("The mapping has errors. Fix them before evaluation.")
        st.table([_issue_to_dict(i) for i in mapping_report.issues])
        return
    if not mapping.is_complete_for_evaluation():
        st.warning(f"Required columns are not yet mapped: {mapping.missing_required()}")
        return

    result = normalize_rows(loaded.rows, mapping=mapping.mappings)
    norm_report = validate_normalization(result)

    cols = st.columns(3)
    cols[0].metric("Rows in file", result.total)
    cols[1].metric("Normalized", result.success_count)
    cols[2].metric("Failures", result.failure_count)

    if not norm_report.ok:
        st.warning(
            "Some rows failed to normalize. You can still proceed with "
            "the successful rows; fix the source data or mapping to "
            "reduce failures."
        )
        st.table([_issue_to_dict(i) for i in norm_report.issues])
        with st.expander("Per-row failure details"):
            st.json([_failure_to_dict(f) for f in result.failures])

    if result.rows:
        st.success(f"Preview of the first {min(10, len(result.rows))} normalized rows:")
        preview = [row.model_dump(mode="json") for row in result.rows[:10]]
        st.dataframe(preview, use_container_width=True)
        st.session_state["jtb.normalized_rows"] = result.rows
    else:
        st.error("No rows could be normalized.")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _issue_to_dict(issue: Any) -> dict[str, str]:
    return {
        "column": str(issue.column),
        "issue": str(issue.issue),
        "detail": str(issue.detail) if issue.detail else "",
    }


def _failure_to_dict(failure: Any) -> dict[str, Any]:
    return {
        "row_index": failure.row_index,
        "record_id": failure.record_id,
        "reason": failure.reason,
        "details": dict(failure.details),
    }


def _help_for(normalized_field: str) -> str:
    if normalized_field in REQUIRED_COLUMNS:
        return f"{normalized_field} is required before evaluation."
    if normalized_field in RECOMMENDED_COLUMNS:
        return f"{normalized_field} is recommended for richer metrics."
    if normalized_field in OPTIONAL_LABEL_COLUMNS:
        return (
            f"{normalized_field} is an SME label (1-5). Enables judge vs "
            "human agreement for this pillar."
        )
    if normalized_field in OPTIONAL_REVIEWER_COLUMNS:
        return f"{normalized_field} enables reviewer-level analytics."
    return f"{normalized_field} is optional."


# Streamlit executes each page file top-to-bottom on every run, so render()
# is invoked unconditionally at module level. The ``if __name__ ...`` guard
# is intentionally omitted - pages are never imported; they are executed.
render()
