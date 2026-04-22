"""Streamlit entry point.

Stage 1 is a placeholder: it imports the core settings + constants to prove
the wiring works end-to-end, and renders a short status block. Page
registration + the Upload / Configure / Run / Dashboard pages land in
Stage 2 and later.

Run with::

    streamlit run src/app/streamlit_app.py

Design rule (see .cursor/rules/streamlit-ui.mdc): this file must not import
from ``src.llm`` or ``src.judges`` directly. It goes through
``src.orchestration`` / ``src.evaluation`` once those exist.
"""

from __future__ import annotations


def main() -> None:
    """Render the placeholder landing page.

    Streamlit is imported inside ``main()`` so the module stays importable
    without a Streamlit runtime (needed for unit tests that just check
    imports and for environments that run linters/typecheckers without
    Streamlit installed).
    """
    import streamlit as st

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
        "Stage 1 scaffold - upload, run, and dashboard pages arrive in "
        "later stages."
    )

    with st.container(border=True):
        st.subheader("Status")
        st.write(
            {
                "stage": 1,
                "pillars": list(PILLARS),
                "required_columns": list(REQUIRED_COLUMNS),
                "configs_dir": str(settings.configs_dir),
                "data_dir": str(settings.data_dir),
                "model_alias": settings.default_model_alias,
                "mlflow_configured": settings.mlflow_tracking_uri is not None,
                "langfuse_configured": settings.langfuse_host is not None,
            }
        )

    st.info(
        "Next stage: dataset ingestion + schema mapping + upload page. "
        "See docs/ROADMAP.md for the full plan."
    )


if __name__ == "__main__":
    main()
