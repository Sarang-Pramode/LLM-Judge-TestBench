"""Tests for :mod:`src.llm.factory`."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import SecretStr

from src.core.exceptions import ProviderError
from src.core.settings import AppSettings
from src.llm import (
    BUILTIN_ALIASES,
    LLMClient,
    MockLLMClient,
    build_client,
    list_aliases,
    register_alias,
    reset_registry,
)
from src.llm.google_client import GoogleGenAIClient


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    reset_registry()
    try:
        yield
    finally:
        reset_registry()


@pytest.fixture
def settings_with_google(tmp_path: Path) -> AppSettings:
    return AppSettings(
        _env_file=tmp_path / "missing.env",
        google_api_key=SecretStr("test-google-key"),
    )


@pytest.fixture
def settings_no_google(tmp_path: Path) -> AppSettings:
    return AppSettings(_env_file=tmp_path / "missing.env")


def _fake_chat_factory(**_kwargs: object) -> object:
    """Stand-in so GoogleGenAIClient.__init__ does not call real SDK."""

    class _FakeChat:
        def invoke(self, *_a: object, **_k: object) -> object:  # pragma: no cover
            raise AssertionError("should not be called in factory tests")

        def with_structured_output(self, *_a: object, **_k: object) -> object:  # pragma: no cover
            raise AssertionError("should not be called in factory tests")

    return _FakeChat()


# ---------------------------------------------------------------------------
# list_aliases / built-ins
# ---------------------------------------------------------------------------


def test_list_aliases_includes_builtins() -> None:
    aliases = list_aliases()
    for name in BUILTIN_ALIASES:
        assert name in aliases


def test_builtin_mock_alias_returns_mock_client(settings_no_google: AppSettings) -> None:
    client = build_client("mock", settings=settings_no_google)
    assert isinstance(client, MockLLMClient)


def test_default_alias_resolves_from_settings(
    settings_with_google: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # default_model_alias = "judge-default" in AppSettings; that maps to
    # Gemini under the hood. We monkeypatch the chat factory so no real
    # SDK is touched.
    monkeypatch.setattr(
        "src.llm.google_client._default_chat_model_factory",
        _fake_chat_factory,
    )
    client = build_client(settings=settings_with_google)
    assert isinstance(client, GoogleGenAIClient)


@pytest.mark.parametrize("alias", ["gemini-flash", "gemini-pro", "judge-default"])
def test_gemini_aliases_build_a_google_client(
    alias: str,
    settings_with_google: AppSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.llm.google_client._default_chat_model_factory",
        _fake_chat_factory,
    )
    client = build_client(alias, settings=settings_with_google)
    assert isinstance(client, GoogleGenAIClient)


def test_gemini_alias_without_api_key_raises(
    settings_no_google: AppSettings,
) -> None:
    with pytest.raises(ProviderError, match="JTB_GOOGLE_API_KEY"):
        build_client("gemini-flash", settings=settings_no_google)


def test_unknown_alias_raises(settings_no_google: AppSettings) -> None:
    with pytest.raises(ProviderError, match="Unknown LLM alias"):
        build_client("definitely-not-a-real-alias", settings=settings_no_google)


# ---------------------------------------------------------------------------
# Custom registrations
# ---------------------------------------------------------------------------


def test_register_alias_overrides_builtin_for_this_process(
    settings_no_google: AppSettings,
) -> None:
    sentinel = MockLLMClient(model_name="sentinel-v1")

    def factory(_settings: AppSettings) -> LLMClient:
        return sentinel

    register_alias("judge-default", factory)
    client = build_client("judge-default", settings=settings_no_google)
    assert client is sentinel


def test_register_alias_requires_nonempty_name() -> None:
    with pytest.raises(ValueError):
        register_alias("", lambda _: MockLLMClient())


def test_custom_alias_appears_in_listing() -> None:
    register_alias("my-test-alias", lambda _: MockLLMClient())
    assert "my-test-alias" in list_aliases()


def test_reset_registry_clears_custom_only(
    settings_no_google: AppSettings,
) -> None:
    register_alias("my-temp", lambda _: MockLLMClient())
    assert "my-temp" in list_aliases()
    reset_registry()
    assert "my-temp" not in list_aliases()
    # Built-ins remain:
    assert "mock" in list_aliases()


def test_build_client_passes_overrides_to_builtins(
    settings_no_google: AppSettings,
) -> None:
    client = build_client(
        "mock",
        settings=settings_no_google,
        model_name="custom-mock-name",
    )
    assert isinstance(client, MockLLMClient)
    assert client.model_name == "custom-mock-name"
