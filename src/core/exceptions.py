"""Domain exceptions.

All project-owned errors inherit from :class:`JudgeTestbenchError` so callers
can catch broadly when needed. Vendor SDK exceptions must be translated at
the provider boundary (``src/llm/``) into the ``Provider*`` subtypes below -
they must never leak into judges, orchestration, or UI code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class JudgeTestbenchError(Exception):
    """Base class for all project-owned errors."""


# ---------------------------------------------------------------------------
# Ingestion / schema
# ---------------------------------------------------------------------------


@dataclass
class ColumnIssue:
    """Single column-level problem surfaced by ingestion validation."""

    column: str
    issue: str
    detail: str | None = None


class SchemaValidationError(JudgeTestbenchError):
    """Raised when normalized dataset validation fails.

    Carries a per-column report that the Streamlit UI renders back to the
    user so they can fix the mapping.
    """

    def __init__(self, message: str, issues: list[ColumnIssue] | None = None) -> None:
        super().__init__(message)
        self.issues: list[ColumnIssue] = list(issues) if issues else []

    def add(self, column: str, issue: str, detail: str | None = None) -> None:
        self.issues.append(ColumnIssue(column=column, issue=issue, detail=detail))

    @property
    def report(self) -> list[dict[str, str | None]]:
        """Plain dict representation suitable for Streamlit tables/JSON."""
        return [{"column": i.column, "issue": i.issue, "detail": i.detail} for i in self.issues]


# ---------------------------------------------------------------------------
# Config / rubrics / KB
# ---------------------------------------------------------------------------


class ConfigLoadError(JudgeTestbenchError):
    """Raised when a YAML/JSON config fails to parse or validate."""


class KnowledgeBankError(JudgeTestbenchError):
    """Raised when the completeness knowledge bank fails to load/validate."""


# ---------------------------------------------------------------------------
# Provider layer (raised by src/llm/ only)
# ---------------------------------------------------------------------------


class ProviderError(JudgeTestbenchError):
    """Generic provider-layer failure."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider call times out."""


class ProviderRateLimitError(ProviderError):
    """Raised when the provider reports throttling / 429."""


# ---------------------------------------------------------------------------
# Judge execution
# ---------------------------------------------------------------------------


@dataclass
class ParseFailure:
    """Diagnostic payload for a failed structured-output parse attempt."""

    attempt: int
    raw_response: str
    reason: str
    details: dict[str, str] = field(default_factory=dict)


class JudgeOutputParseError(JudgeTestbenchError):
    """Raised when a judge's structured output cannot be validated.

    ``failures`` contains one entry per attempt (initial parse + any repair
    attempts performed by ``judges/output_parser``).
    """

    def __init__(self, message: str, failures: list[ParseFailure] | None = None) -> None:
        super().__init__(message)
        self.failures: list[ParseFailure] = list(failures) if failures else []


class JudgeExecutionError(JudgeTestbenchError):
    """Raised for non-parse judge failures (unexpected exceptions, guards)."""
