"""Observability: MLflow (aggregates) + Langfuse (per-row traces).

Both backends are optional at runtime. Builders return loggers that
are safe to call even without credentials configured - they no-op and
warn once per failure type so a missing observability backend never
breaks a run.
"""

from __future__ import annotations

from src.observability.adapters import (
    ObservabilityCallbacks,
    build_observability_callbacks,
)
from src.observability.langfuse_tracer import LangfuseTracer, build_langfuse_tracer
from src.observability.mlflow_logger import MLflowLogger, build_mlflow_logger
from src.observability.run_metadata import (
    RunMetadata,
    build_run_metadata,
    dataset_fingerprint,
    run_config_hash,
    to_langfuse_metadata,
    to_mlflow_params,
    to_mlflow_tags,
)

__all__ = [
    "LangfuseTracer",
    "MLflowLogger",
    "ObservabilityCallbacks",
    "RunMetadata",
    "build_langfuse_tracer",
    "build_mlflow_logger",
    "build_observability_callbacks",
    "build_run_metadata",
    "dataset_fingerprint",
    "run_config_hash",
    "to_langfuse_metadata",
    "to_mlflow_params",
    "to_mlflow_tags",
]
