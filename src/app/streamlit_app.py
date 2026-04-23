"""Streamlit entry point.

Renders the landing page and describes the upload -> configure ->
run -> dashboard flow. Pages live under ``src/app/pages/`` and are
picked up automatically by Streamlit's multipage runtime.

Run with::

    streamlit run src/app/streamlit_app.py

Design rule (see ``.cursor/rules/streamlit-ui.mdc``): this file must
not import from ``src.llm`` or ``src.judges`` directly. It talks to
``src.orchestration`` / ``src.evaluation`` / ``src.dashboard`` only.
"""

from __future__ import annotations


def main() -> None:
    """Render the landing page.

    Streamlit is imported inside ``main()`` so the module stays
    importable without a Streamlit runtime, which matters for the
    unit tests that just assert the module imports cleanly.
    """
    import streamlit as st

    from src.app.state import (
        SS_LAST_RUN_RESULT,
        SS_NORMALIZED_ROWS,
        SS_RUN_CONFIG,
    )
    from src.core import PILLARS, REQUIRED_COLUMNS
    from src.core.settings import get_settings

    settings = get_settings()

    st.set_page_config(
        page_title="LLM Judge Testbench",
        page_icon=":bar_chart:",
        layout="wide",
    )

    st.title("LLM Judge Testbench")
    st.caption(
        "Evaluation workbench for LLM judges against SME rubrics. "
        "Upload a dataset, map columns, configure a run, execute, then "
        "explore dashboards and disagreements."
    )

    with st.container(border=True):
        st.subheader("Flow")
        st.markdown(
            """
            1. **Upload** - CSV / XLSX / JSON / Parquet; map to the
               normalized schema.
            2. **Configure** - pick pillars, provider, concurrency.
            3. **Run evaluation** - parallel judge execution with
               progress and partial-failure handling.
            4. **Dashboard** - overall + per-category agreement, score
               distributions, confusion matrices.
            5. **Disagreements** - filter + inspect per-row misses.
            6. **Reviewer analytics** - activates when reviewer
               metadata is present.
            7. **Risk evidence** - north-star gates, Plotly deep dives,
               drift vs a pinned baseline; pairs with MLflow run comparison.
            """
        )

    with st.container(border=True):
        st.subheader("Session status")
        rows = st.session_state.get(SS_NORMALIZED_ROWS)
        run_config = st.session_state.get(SS_RUN_CONFIG)
        last_result = st.session_state.get(SS_LAST_RUN_RESULT)
        st.write(
            {
                "dataset_loaded": rows is not None and len(rows) > 0,
                "normalized_rows": len(rows) if rows else 0,
                "run_configured": run_config is not None,
                "last_run_id": last_result.run_id if last_result is not None else None,
                "pillars": list(PILLARS),
                "required_columns": list(REQUIRED_COLUMNS),
                "provider_default": settings.default_model_alias,
                "mlflow_configured": settings.mlflow_tracking_uri is not None,
                "langfuse_configured": settings.langfuse_host is not None,
            }
        )

    st.info(
        "Navigate using the left sidebar. Each page gates on the "
        "previous one's output, so work through them in order."
    )


if __name__ == "__main__":
    main()
