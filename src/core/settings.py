"""Application settings loaded from environment (and optional .env).

``AppSettings`` is the single source of truth for runtime configuration
knobs that live outside of YAML configs (credentials, URIs, redaction
toggles, log level). Per-judge and per-rubric configuration lives under
``configs/`` and is loaded in later stages.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT: Path = Path(__file__).resolve().parents[2]


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class AppSettings(BaseSettings):
    """Environment-backed settings.

    Defaults are chosen so a fresh checkout can import this module without
    any env setup. All secrets use ``SecretStr`` so they are not printed
    accidentally via ``repr`` or logging.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="JTB_",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Paths ------------------------------------------------------------
    configs_dir: Path = Field(default_factory=lambda: REPO_ROOT / "configs")
    data_dir: Path = Field(default_factory=lambda: REPO_ROOT / "data")

    # --- Model routing ----------------------------------------------------
    default_model_alias: str = "judge-default"

    # --- Provider credentials --------------------------------------------
    # Vendor SDK keys. Accessed only from src/llm/. These are optional so
    # offline/mock runs succeed even without any real provider configured.
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None

    # --- Observability ---------------------------------------------------
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "llm-judge-testbench"

    langfuse_host: str | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None

    # --- Behavior --------------------------------------------------------
    redact_pii: bool = False
    log_level: LogLevel = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a process-wide cached ``AppSettings``.

    Tests that need to mutate settings should call ``get_settings.cache_clear()``
    after tweaking env vars or instantiate ``AppSettings(...)`` directly.
    """
    return AppSettings()
