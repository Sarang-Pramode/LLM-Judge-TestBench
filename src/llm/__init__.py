"""LLM provider abstraction layer.

All vendor/model calls are confined to this package. Judges,
orchestration, dashboards and metrics import from here - never from
``langchain_*`` or a vendor SDK directly.
"""

from __future__ import annotations

from src.llm.base import (
    LLMClient,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    RetryPolicy,
    StructuredResponse,
)
from src.llm.factory import (
    BUILTIN_ALIASES,
    build_client,
    list_aliases,
    register_alias,
    reset_registry,
)
from src.llm.google_client import DEFAULT_GEMINI_MODEL, GoogleGenAIClient
from src.llm.mock_client import MockCall, MockLLMClient

__all__ = [
    "BUILTIN_ALIASES",
    "DEFAULT_GEMINI_MODEL",
    "GoogleGenAIClient",
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "LLMUsage",
    "MockCall",
    "MockLLMClient",
    "RetryPolicy",
    "StructuredResponse",
    "build_client",
    "list_aliases",
    "register_alias",
    "reset_registry",
]
