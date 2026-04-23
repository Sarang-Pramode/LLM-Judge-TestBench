"""Judge registry.

Each pillar judge (Stage 5) registers itself at import time via
:func:`register_judge`. Orchestration then calls :func:`build_judge`
with a pillar name + wiring; it does not need to know which concrete
class backs the pillar. This is what makes "add a new pillar without
changing the runner" true.

Registration rules:

- Keys are pillar strings (matching :attr:`BaseJudge.pillar`).
- Re-registering the same pillar is an error unless ``force=True`` -
  accidental duplicate imports should fail loudly.
- Tests that want to swap in a fake judge should use ``force=True`` or
  call :func:`reset_registry` on teardown.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, overload

from src.core.exceptions import JudgeExecutionError
from src.judges.base import BaseJudge
from src.judges.config import JudgeBundle
from src.llm.base import LLMClient

if TYPE_CHECKING:
    from src.completeness.models import CompletenessKB

__all__ = [
    "build_judge",
    "is_registered",
    "list_registered_pillars",
    "register_judge",
    "registered_judges",
    "reset_registry",
    "resolve_judge",
]


_REGISTRY: dict[str, type[BaseJudge]] = {}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@overload
def register_judge(cls: type[BaseJudge], /) -> type[BaseJudge]: ...
@overload
def register_judge(cls: type[BaseJudge], *, force: bool) -> type[BaseJudge]: ...
@overload
def register_judge(
    cls: None = None, *, force: bool = ...
) -> Callable[[type[BaseJudge]], type[BaseJudge]]: ...


def register_judge(
    cls: type[BaseJudge] | None = None,
    *,
    force: bool = False,
) -> type[BaseJudge] | Callable[[type[BaseJudge]], type[BaseJudge]]:
    """Register a :class:`BaseJudge` subclass.

    Usable as a bare decorator (``@register_judge``) or with kwargs
    (``@register_judge(force=True)``).

    Raises:
        JudgeExecutionError: if the class has an empty ``pillar``, or
            when re-registering a pillar without ``force=True``.
    """
    if cls is None:

        def _wrap(inner: type[BaseJudge]) -> type[BaseJudge]:
            return _register(inner, force=force)

        return _wrap

    return _register(cls, force=force)


def _register(cls: type[BaseJudge], *, force: bool) -> type[BaseJudge]:
    if not isinstance(cls, type) or not issubclass(cls, BaseJudge):
        raise JudgeExecutionError(f"register_judge expected a BaseJudge subclass, got {cls!r}.")
    pillar = getattr(cls, "pillar", "")
    if not pillar:
        raise JudgeExecutionError(
            f"Cannot register {cls.__name__}: its `pillar` class attribute is empty."
        )

    existing = _REGISTRY.get(pillar)
    if existing is not None and existing is not cls and not force:
        raise JudgeExecutionError(
            f"Pillar {pillar!r} is already registered to {existing.__name__}; "
            f"pass force=True to override with {cls.__name__}."
        )
    _REGISTRY[pillar] = cls
    return cls


def reset_registry() -> None:
    """Clear the registry. Intended for test teardown."""
    _REGISTRY.clear()


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def is_registered(pillar: str) -> bool:
    return pillar in _REGISTRY


def list_registered_pillars() -> list[str]:
    return sorted(_REGISTRY)


def registered_judges() -> Iterator[tuple[str, type[BaseJudge]]]:
    """Iterate over ``(pillar, cls)`` pairs in a stable alphabetical order."""
    for pillar in sorted(_REGISTRY):
        yield pillar, _REGISTRY[pillar]


def resolve_judge(pillar: str) -> type[BaseJudge]:
    """Return the registered class for ``pillar`` or raise."""
    try:
        return _REGISTRY[pillar]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise JudgeExecutionError(
            f"No judge registered for pillar {pillar!r}. Registered pillars: {available}."
        ) from exc


# ---------------------------------------------------------------------------
# Instantiation helper
# ---------------------------------------------------------------------------


def build_judge(
    pillar: str,
    *,
    bundle: JudgeBundle,
    llm: LLMClient,
    kb: CompletenessKB | None = None,
) -> BaseJudge:
    """Resolve ``pillar`` and instantiate the judge with ``bundle`` + ``llm``.

    This is the one-call form orchestration will use once per run per
    pillar. The registry never holds on to instances - callers own the
    lifetimes.

    The optional ``kb`` is forwarded to judge classes whose
    ``__init__`` declares a ``kb`` parameter (currently only the
    completeness pillar). Other pillars ignore it, so orchestration
    can thread one KB object through to every ``build_judge`` call
    without conditional logic.
    """
    if bundle.config.pillar != pillar:
        raise JudgeExecutionError(
            f"JudgeBundle declares pillar {bundle.config.pillar!r} but "
            f"build_judge was asked for {pillar!r}."
        )
    cls = resolve_judge(pillar)
    kwargs: dict[str, Any] = {
        "config": bundle.config,
        "rubric": bundle.rubric,
        "llm": llm,
    }
    # Only forward `kb` to classes that accept it. Using inspect keeps
    # the registry decoupled from the concrete CompletenessJudge class
    # (avoiding a circular import at module import time).
    if kb is not None:
        init_params = inspect.signature(cls.__init__).parameters
        if "kb" in init_params:
            kwargs["kb"] = kb
    return cls(**kwargs)
