"""Unit tests for AppSettings / get_settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.settings import AppSettings, get_settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scrub any JTB_* env the host machine may have set.

    Without this, a developer with a real OpenAI key in their shell would
    see tests that expected ``None`` fail intermittently.
    """
    for var in list(
        k
        for k in (
            "JTB_CONFIGS_DIR",
            "JTB_DATA_DIR",
            "JTB_DEFAULT_MODEL_ALIAS",
            "JTB_OPENAI_API_KEY",
            "JTB_ANTHROPIC_API_KEY",
            "JTB_GOOGLE_API_KEY",
            "JTB_MLFLOW_TRACKING_URI",
            "JTB_MLFLOW_EXPERIMENT_NAME",
            "JTB_LANGFUSE_HOST",
            "JTB_LANGFUSE_PUBLIC_KEY",
            "JTB_LANGFUSE_SECRET_KEY",
            "JTB_REDACT_PII",
            "JTB_LOG_LEVEL",
        )
    ):
        monkeypatch.delenv(var, raising=False)


def test_defaults_with_no_env(clean_settings_cache: None, tmp_path: Path) -> None:
    # Point to a nonexistent .env so pydantic-settings does not pick up any
    # real file sitting next to the repo during development.
    settings = AppSettings(_env_file=tmp_path / "missing.env")  # type: ignore[call-arg]
    assert settings.default_model_alias == "judge-default"
    assert settings.redact_pii is False
    assert settings.log_level == "INFO"
    assert settings.openai_api_key is None
    assert settings.anthropic_api_key is None
    assert settings.mlflow_experiment_name == "llm-judge-testbench"
    # Paths default into the repo root via the factory.
    assert isinstance(settings.configs_dir, Path)
    assert settings.configs_dir.name == "configs"
    assert settings.data_dir.name == "data"


def test_reads_env_vars(
    monkeypatch: pytest.MonkeyPatch,
    clean_settings_cache: None,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("JTB_DEFAULT_MODEL_ALIAS", "gpt-4o-mini")
    monkeypatch.setenv("JTB_REDACT_PII", "true")
    monkeypatch.setenv("JTB_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("JTB_OPENAI_API_KEY", "sk-test-xyz")
    monkeypatch.setenv("JTB_CONFIGS_DIR", str(tmp_path / "cfg"))

    settings = AppSettings(_env_file=tmp_path / "missing.env")  # type: ignore[call-arg]
    assert settings.default_model_alias == "gpt-4o-mini"
    assert settings.redact_pii is True
    assert settings.log_level == "DEBUG"
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-test-xyz"
    assert settings.configs_dir == tmp_path / "cfg"


def test_reads_dotenv_file(
    clean_settings_cache: None,
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "JTB_DEFAULT_MODEL_ALIAS=claude-haiku\n"
        "JTB_MLFLOW_TRACKING_URI=http://localhost:5000\n"
        "JTB_LANGFUSE_HOST=https://cloud.langfuse.com\n"
    )
    settings = AppSettings(_env_file=env_path)  # type: ignore[call-arg]
    assert settings.default_model_alias == "claude-haiku"
    assert settings.mlflow_tracking_uri == "http://localhost:5000"
    assert settings.langfuse_host == "https://cloud.langfuse.com"


def test_secrets_do_not_leak_in_repr(
    monkeypatch: pytest.MonkeyPatch,
    clean_settings_cache: None,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("JTB_OPENAI_API_KEY", "sk-super-secret-123")
    settings = AppSettings(_env_file=tmp_path / "missing.env")  # type: ignore[call-arg]
    rendered = repr(settings)
    assert "sk-super-secret-123" not in rendered
    # Pydantic's SecretStr prints as ** or similar; accept any redaction.
    assert "SecretStr" in rendered or "**" in rendered


def test_rejects_invalid_log_level(
    monkeypatch: pytest.MonkeyPatch,
    clean_settings_cache: None,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("JTB_LOG_LEVEL", "VERBOSE")
    with pytest.raises(Exception):  # pydantic ValidationError
        AppSettings(_env_file=tmp_path / "missing.env")  # type: ignore[call-arg]


def test_get_settings_is_cached(clean_settings_cache: None) -> None:
    a = get_settings()
    b = get_settings()
    assert a is b


def test_cache_clear_returns_new_instance(clean_settings_cache: None) -> None:
    a = get_settings()
    get_settings.cache_clear()
    b = get_settings()
    assert a is not b


def test_extra_env_vars_ignored(
    monkeypatch: pytest.MonkeyPatch,
    clean_settings_cache: None,
    tmp_path: Path,
) -> None:
    # Foreign JTB_* variable must not blow up model construction.
    monkeypatch.setenv("JTB_SOME_FUTURE_SETTING", "hello")
    settings = AppSettings(_env_file=tmp_path / "missing.env")  # type: ignore[call-arg]
    assert settings.default_model_alias == "judge-default"
