"""Factory that maps a model *alias* to a concrete :class:`LLMClient`.

Aliases decouple judges from specific vendor/model strings: a judge
config says ``model_alias: judge-default`` and the factory decides which
provider and which concrete model backs that alias today. Swapping
Gemini Flash for Gemini Pro, or switching to an OpenAI client later, is
a factory-level change; judge code does not move.

Two ways to resolve an alias:

1. **Registered alias** (preferred): call :func:`register_alias` with a
   callable that produces an :class:`LLMClient`. The upload / judges
   layer can ship built-in aliases without editing settings.
2. **Built-in routing**: if no alias is registered, we fall back to a
   small built-in table that maps well-known aliases (``"mock"``,
   ``"gemini-flash"``, ``"gemini-pro"``, ``"judge-default"``) to default
   configurations, reading credentials from :class:`AppSettings`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.core.exceptions import ProviderError
from src.core.settings import AppSettings, get_settings
from src.llm.base import LLMClient
from src.llm.google_client import DEFAULT_GEMINI_MODEL, GoogleGenAIClient
from src.llm.mock_client import MockLLMClient

__all__ = [
    "BUILTIN_ALIASES",
    "build_client",
    "list_aliases",
    "register_alias",
    "reset_registry",
]

AliasFactory = Callable[[AppSettings], LLMClient]

# Well-known aliases that ship out of the box. Keep this list small and
# intentional - the judge configs in ``configs/judges/`` reference these.
BUILTIN_ALIASES: frozenset[str] = frozenset(
    {
        "mock",
        "judge-default",
        "gemini-flash",
        "gemini-pro",
    }
)

_REGISTRY: dict[str, AliasFactory] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_alias(alias: str, factory: AliasFactory) -> None:
    """Register or override an alias.

    Callers are expected to register at import time (e.g. inside a
    plugin's ``__init__``). Double-registration silently overrides -
    this is intentional so tests can swap implementations per-call.
    """
    if not alias:
        raise ValueError("alias must be a non-empty string.")
    _REGISTRY[alias] = factory


def reset_registry() -> None:
    """Clear custom alias registrations. Built-ins are unaffected."""
    _REGISTRY.clear()


def list_aliases() -> list[str]:
    """Return every alias the factory can currently resolve."""
    return sorted(set(_REGISTRY) | BUILTIN_ALIASES)


def build_client(
    alias: str | None = None,
    *,
    settings: AppSettings | None = None,
    **overrides: Any,
) -> LLMClient:
    """Resolve ``alias`` to a concrete :class:`LLMClient`.

    Args:
        alias: Alias name. ``None`` means use
            :attr:`AppSettings.default_model_alias`.
        settings: Optional settings object. Defaults to the cached
            process-wide ``AppSettings``.
        **overrides: Passed through to the alias factory for the
            built-in routes. Unknown keys are ignored by factories
            that don't use them. Example: ``model_name="gemini-1.5-pro"``.
    """
    cfg = settings if settings is not None else get_settings()
    resolved = alias or cfg.default_model_alias

    if resolved in _REGISTRY:
        return _REGISTRY[resolved](cfg)

    if resolved in BUILTIN_ALIASES:
        return _build_builtin(resolved, cfg, overrides)

    available = list_aliases()
    raise ProviderError(
        f"Unknown LLM alias {resolved!r}. Known aliases: {available}. "
        "Register a custom alias with register_alias() or use one of the "
        "built-ins."
    )


# ---------------------------------------------------------------------------
# Built-in routing
# ---------------------------------------------------------------------------


def _build_builtin(
    alias: str,
    settings: AppSettings,
    overrides: dict[str, Any],
) -> LLMClient:
    if alias == "mock":
        return MockLLMClient(
            model_name=str(overrides.get("model_name", "mock-llm-v0")),
        )

    if alias == "judge-default":
        # The default alias resolves to Gemini Flash in v1 - fast enough
        # for batch evaluation and with strong structured-output support
        # via LangChain. Future stages may switch this based on settings.
        return _build_gemini(
            settings=settings,
            model_name=str(overrides.get("model_name", DEFAULT_GEMINI_MODEL)),
            default_timeout_s=float(overrides.get("default_timeout_s", 60.0)),
        )

    if alias == "gemini-flash":
        return _build_gemini(
            settings=settings,
            model_name=str(overrides.get("model_name", "gemini-2.0-flash")),
            default_timeout_s=float(overrides.get("default_timeout_s", 60.0)),
        )

    if alias == "gemini-pro":
        return _build_gemini(
            settings=settings,
            model_name=str(overrides.get("model_name", "gemini-1.5-pro")),
            default_timeout_s=float(overrides.get("default_timeout_s", 120.0)),
        )

    # Unreachable because ``alias in BUILTIN_ALIASES`` was checked above.
    raise ProviderError(f"Internal: built-in alias {alias!r} has no implementation.")


def _build_gemini(
    *,
    settings: AppSettings,
    model_name: str,
    default_timeout_s: float,
) -> LLMClient:
    if settings.google_api_key is None:
        raise ProviderError(
            "Gemini aliases require JTB_GOOGLE_API_KEY (or settings.google_api_key). "
            "Either set the env var, or use the 'mock' alias for offline runs."
        )
    return GoogleGenAIClient(
        model_name=model_name,
        api_key=settings.google_api_key,
        default_timeout_s=default_timeout_s,
    )
