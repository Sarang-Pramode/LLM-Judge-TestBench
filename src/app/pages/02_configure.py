"""Streamlit Configure page.

Lets the user choose pillars, provider, and concurrency for the next
run. Writes a :class:`RunConfig` into session state for the Run page
to pick up.

Hard constraints (per ``.cursor/rules/streamlit-ui.mdc``):

- No LLM SDK imports here. Provider selection is an *alias* - the
  factory (`src.llm.factory`) resolves it to a concrete client when
  the Run page executes.
- No judge logic here. We only surface what's available and let the
  Run page drive execution.
- Requires the Upload page to have produced normalized rows. If they
  aren't there yet, the page renders a helpful pointer instead of
  silently failing.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import streamlit as st

from src.app.state import (
    SS_DATASET_NAME,
    SS_NORMALIZED_ROWS,
    SS_RUN_CONFIG,
    RunConfig,
)
from src.core.constants import PILLARS
from src.core.settings import get_settings

JUDGES_DIR = Path(__file__).resolve().parents[3] / "configs" / "judges"


def render() -> None:
    st.set_page_config(page_title="Configure run", layout="wide")
    st.title("Configure evaluation run")
    st.caption(
        "Choose pillars, provider, and concurrency. The selected "
        "configuration is stored in session state for the Run page."
    )

    rows = st.session_state.get(SS_NORMALIZED_ROWS)
    if not rows:
        st.warning(
            "No normalized rows in session. Go to **Upload** first to load and map a dataset."
        )
        return

    dataset_name = st.session_state.get(SS_DATASET_NAME, "uploaded dataset")
    st.write(f"Dataset: `{dataset_name}` - {len(rows)} normalized rows")

    previous: RunConfig | None = st.session_state.get(SS_RUN_CONFIG)
    seed = previous if previous is not None else RunConfig()

    pillars = _render_pillar_picker(seed)
    provider = _render_provider_picker(seed)
    max_workers, per_provider_limit, use_cache = _render_concurrency_controls(seed)
    enable_mlflow, enable_langfuse = _render_observability_controls(seed)

    if not pillars:
        st.error("Pick at least one pillar to evaluate.")
        return

    try:
        new_config = RunConfig(
            pillars=tuple(pillars),
            provider=provider,
            max_workers=max_workers,
            per_provider_limit=per_provider_limit,
            use_cache=use_cache,
            enable_mlflow=enable_mlflow,
            enable_langfuse=enable_langfuse,
            extra=dict(seed.extra),
        )
    except ValueError as exc:
        st.error(f"Invalid configuration: {exc}")
        return

    # Store immediately - users expect their picks to persist when
    # navigating away. Use ``replace()`` for the extra dict to avoid
    # sharing mutable state with the previous config.
    st.session_state[SS_RUN_CONFIG] = replace(new_config, extra=dict(new_config.extra))

    st.success("Configuration saved. Navigate to **Run evaluation** to execute.")
    with st.expander("Resolved configuration", expanded=False):
        st.write(
            {
                "pillars": list(new_config.pillars),
                "provider": new_config.provider,
                "max_workers": new_config.max_workers,
                "per_provider_limit": new_config.per_provider_limit,
                "use_cache": new_config.use_cache,
                "enable_mlflow": new_config.enable_mlflow,
                "enable_langfuse": new_config.enable_langfuse,
            }
        )


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _render_pillar_picker(seed: RunConfig) -> list[str]:
    st.subheader("Pillars")
    st.caption("Evaluating more pillars increases cost and latency linearly.")
    chosen: list[str] = []
    cols = st.columns(3)
    for i, pillar in enumerate(PILLARS):
        config_path = JUDGES_DIR / f"{pillar}.yaml"
        default = pillar in seed.pillars
        label = pillar
        if not config_path.exists():
            label = f"{pillar} (no config found)"
        selected = cols[i % 3].checkbox(
            label,
            value=default and config_path.exists(),
            key=f"jtb.cfg.pillar.{pillar}",
            disabled=not config_path.exists(),
        )
        if selected:
            chosen.append(pillar)
    return chosen


def _render_provider_picker(seed: RunConfig) -> str:
    st.subheader("LLM provider")
    settings = get_settings()
    choices = ["mock", "google"]
    help_text = (
        "mock: deterministic offline client for demos / tests. "
        "google: real Gemini calls via LangChain (requires JTB_GOOGLE_API_KEY)."
    )
    idx = choices.index(seed.provider) if seed.provider in choices else 0
    provider = st.selectbox(
        "Provider",
        options=choices,
        index=idx,
        help=help_text,
    )
    if provider == "google" and settings.google_api_key is None:
        st.warning(
            "JTB_GOOGLE_API_KEY is not configured. Set it in the environment "
            "or fall back to the mock provider."
        )
    return provider


def _render_concurrency_controls(seed: RunConfig) -> tuple[int, int | None, bool]:
    st.subheader("Concurrency")
    col_a, col_b, col_c = st.columns(3)
    max_workers = col_a.slider(
        "Max parallel workers",
        min_value=1,
        max_value=32,
        value=max(1, min(32, seed.max_workers)),
        help="Thread-pool size. Scaled with the number of (row, pillar) tasks.",
    )
    per_provider_default = seed.per_provider_limit if seed.per_provider_limit else 0
    per_provider_raw = col_b.number_input(
        "Per-provider rate limit (0 = unlimited)",
        min_value=0,
        max_value=max_workers,
        value=min(per_provider_default, max_workers),
        help="Cap concurrent calls against any single LLM client.",
    )
    per_provider_limit = int(per_provider_raw) if per_provider_raw else None
    use_cache = col_c.checkbox(
        "Use in-memory outcome cache",
        value=seed.use_cache,
        help=(
            "Skip work for identical (pillar, bundle, model, row, KB) "
            "keys across re-runs within the same session."
        ),
    )
    return max_workers, per_provider_limit, use_cache


def _render_observability_controls(seed: RunConfig) -> tuple[bool, bool]:
    """Opt-out toggles for MLflow + Langfuse logging.

    Both loggers gracefully no-op when their credentials aren't set, so
    the defaults are intentionally "on". Users who want to skip logging
    for a particular run (e.g. quick smoke test) disable here; a hint
    below the toggles shows whether the env actually has the creds.
    """
    st.subheader("Observability")
    settings = get_settings()
    has_mlflow = settings.mlflow_tracking_uri is not None
    has_langfuse = (
        settings.langfuse_host is not None
        and settings.langfuse_public_key is not None
        and settings.langfuse_secret_key is not None
    )

    col_a, col_b = st.columns(2)
    enable_mlflow = col_a.checkbox(
        "Log run metrics to MLflow",
        value=seed.enable_mlflow,
        help=(
            "Requires JTB_MLFLOW_TRACKING_URI (and optionally "
            "JTB_MLFLOW_EXPERIMENT_NAME). Off = logger is disabled."
        ),
    )
    if enable_mlflow and not has_mlflow:
        col_a.caption(":warning: JTB_MLFLOW_TRACKING_URI is not set; logger will stay disabled.")
    elif enable_mlflow:
        col_a.caption(f"Tracking URI: `{settings.mlflow_tracking_uri}`")

    enable_langfuse = col_b.checkbox(
        "Log per-row traces to Langfuse",
        value=seed.enable_langfuse,
        help=(
            "Requires JTB_LANGFUSE_HOST, JTB_LANGFUSE_PUBLIC_KEY, and "
            "JTB_LANGFUSE_SECRET_KEY. Off = tracer is disabled."
        ),
    )
    if enable_langfuse and not has_langfuse:
        col_b.caption(
            ":warning: Langfuse credentials are not fully configured; tracer will stay disabled."
        )
    elif enable_langfuse:
        col_b.caption(f"Host: `{settings.langfuse_host}`")

    return enable_mlflow, enable_langfuse


render()
