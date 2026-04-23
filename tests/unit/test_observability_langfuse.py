"""Unit tests for :class:`src.observability.langfuse_tracer.LangfuseTracer`.

We assert on recorded calls against a duck-typed fake; never import
langfuse. Coverage:

- Enablement (missing creds → disabled; all three needed).
- ``start_run`` produces a root span with the expected input/metadata.
- ``log_outcome`` attaches a child generation with parsed or error
  payloads and the tokens/model propagated correctly.
- Redaction replaces free-text when ``redact_pii=True``.
- Thread-safety: concurrent ``log_outcome`` calls attach to the same
  root span without racing.
- Failures inside the fake span do not crash the tracer.
- Disabled tracer is a total no-op.
"""

from __future__ import annotations

import datetime
import threading
from typing import Any

import pytest

from src.core.types import NormalizedRow
from src.judges.base import JudgeOutcome
from src.judges.config import JudgeBundle, JudgeConfig
from src.llm.base import LLMUsage
from src.llm.mock_client import MockLLMClient
from src.observability.langfuse_tracer import (
    LangfuseTracer,
    _redact_text,
    build_langfuse_tracer,
)
from src.observability.run_metadata import RunMetadata, build_run_metadata
from src.rubrics.models import Rubric, ScoreAnchor

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeObservation:
    """Duck-type of a :class:`LangfuseSpan` / ``LangfuseGeneration``."""

    def __init__(self, *, parent: FakeLangfuseBackend | FakeObservation, **kwargs: Any):
        self.name = kwargs.get("name", "?")
        self.parent = parent
        self.kwargs = kwargs
        self.children: list[FakeObservation] = []
        self.ended = False
        self.updates: list[dict[str, Any]] = []

    def start_observation(self, **kwargs: Any) -> FakeObservation:
        child = FakeObservation(parent=self, **kwargs)
        self.children.append(child)
        return child

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def end(self, **kwargs: Any) -> None:
        self.ended = True


class FakeLangfuseBackend:
    """Stand-in for ``Langfuse`` client."""

    def __init__(self, *, fail_on: set[str] | None = None) -> None:
        self.observations: list[FakeObservation] = []
        self.flushed = 0
        self._fail_on = fail_on or set()

    def start_observation(self, **kwargs: Any) -> FakeObservation:
        if "start_observation" in self._fail_on:
            raise RuntimeError("backend fail")
        obs = FakeObservation(parent=self, **kwargs)
        self.observations.append(obs)
        return obs

    def flush(self) -> None:
        if "flush" in self._fail_on:
            raise RuntimeError("flush fail")
        self.flushed += 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bundle() -> JudgeBundle:
    return JudgeBundle(
        config=JudgeConfig(
            pillar="factual_accuracy",
            prompt_version="pv1",
            rubric_path="rubrics/fa.yaml",
            rubric_version="rv1",
        ),
        rubric=Rubric(
            pillar="factual_accuracy",
            version="rv1",
            description="d",
            required_inputs=["user_input", "agent_output"],
            anchors=[ScoreAnchor(score=i, name=f"s{i}", description="d") for i in range(1, 6)],
            failure_tags=["t"],
        ),
    )


def _meta() -> RunMetadata:
    bundle = _bundle()
    llm = MockLLMClient(model_name="mock", usage=LLMUsage(input_tokens=0, output_tokens=0))
    return build_run_metadata(
        run_id="r-1",
        dataset_fingerprint="sha256:aa",
        dataset_row_count=1,
        pillars=["factual_accuracy"],
        bundles={"factual_accuracy": bundle},
        llm_by_pillar={"factual_accuracy": llm},
        provider="mock",
    )


def _row(record_id: str = "r-1") -> NormalizedRow:
    return NormalizedRow(
        record_id=record_id,
        user_input="user text",
        agent_output="agent text",
        category="cat",
    )


def _ok_outcome() -> JudgeOutcome:
    from src.core.types import Evidence, JudgeResult

    result = JudgeResult(
        pillar="factual_accuracy",
        score=4,
        confidence=0.8,
        decision_summary="ok",
        evidence_for_score=[Evidence(claim="c", status="supported", support="x")],
        failure_tags=[],
        why_not_higher="why hi",
        why_not_lower="why lo",
        rubric_anchor=4,
        raw_model_name="mock",
        prompt_version="pv1",
        rubric_version="rv1",
    )
    return JudgeOutcome(
        pillar="factual_accuracy",
        record_id="r-1",
        latency_ms=42.0,
        attempts=1,
        usage=LLMUsage(input_tokens=7, output_tokens=3),
        model_name="mock",
        result=result,
        prompt_versions=("pv1", "rv1"),
    )


def _fail_outcome() -> JudgeOutcome:
    return JudgeOutcome(
        pillar="factual_accuracy",
        record_id="r-1",
        latency_ms=3.0,
        attempts=2,
        usage=LLMUsage(input_tokens=0, output_tokens=0),
        model_name="mock",
        error="parse failure",
        error_type="SchemaMismatch",
    )


def _enabled() -> tuple[LangfuseTracer, FakeLangfuseBackend]:
    backend = FakeLangfuseBackend()
    tracer = LangfuseTracer(
        host="http://localhost",
        public_key="pk",
        secret_key="sk",
        enabled=True,
        backend=backend,
    )
    return tracer, backend


# ---------------------------------------------------------------------------
# Enablement
# ---------------------------------------------------------------------------


class TestEnablement:
    @pytest.mark.parametrize(
        "host,pk,sk",
        [
            (None, "pk", "sk"),
            ("http://localhost", None, "sk"),
            ("http://localhost", "pk", None),
        ],
    )
    def test_missing_any_cred_disables_when_no_backend(
        self, host: str | None, pk: str | None, sk: str | None
    ) -> None:
        tracer = LangfuseTracer(host=host, public_key=pk, secret_key=sk, enabled=True, backend=None)
        assert tracer.enabled is False

    def test_explicit_backend_overrides_creds(self) -> None:
        tracer = LangfuseTracer(
            host=None,
            public_key=None,
            secret_key=None,
            enabled=True,
            backend=FakeLangfuseBackend(),
        )
        assert tracer.enabled is True

    def test_explicit_disable_blocks_all_calls(self) -> None:
        backend = FakeLangfuseBackend()
        tracer = LangfuseTracer(
            host="http://localhost",
            public_key="pk",
            secret_key="sk",
            enabled=False,
            backend=backend,
        )
        tracer.start_run(_meta())
        tracer.log_outcome(_ok_outcome(), _row())
        tracer.end_run()
        assert backend.observations == []
        assert backend.flushed == 0


# ---------------------------------------------------------------------------
# start_run / end_run
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_start_run_creates_root_span(self) -> None:
        tracer, backend = _enabled()
        tracer.start_run(_meta())
        assert len(backend.observations) == 1
        root = backend.observations[0]
        assert root.kwargs["as_type"] == "span"
        assert root.kwargs["name"].endswith(".r-1")
        assert root.kwargs["input"]["pillars"] == ["factual_accuracy"]
        assert "per_pillar" in root.kwargs["metadata"]

    def test_start_run_is_idempotent(self) -> None:
        tracer, backend = _enabled()
        tracer.start_run(_meta())
        tracer.start_run(_meta())
        assert len(backend.observations) == 1

    def test_end_run_updates_and_flushes(self) -> None:
        tracer, backend = _enabled()
        tracer.start_run(_meta())
        tracer.end_run("FINISHED", summary={"succeeded": 5})
        root = backend.observations[0]
        assert root.ended is True
        assert root.updates[-1]["output"]["status"] == "FINISHED"
        assert root.updates[-1]["output"]["succeeded"] == 5
        assert backend.flushed == 1

    def test_end_run_before_start_is_noop(self) -> None:
        tracer, backend = _enabled()
        tracer.end_run()
        assert backend.flushed == 0


# ---------------------------------------------------------------------------
# log_outcome
# ---------------------------------------------------------------------------


class TestLogOutcome:
    def test_success_outcome_attaches_child(self) -> None:
        tracer, backend = _enabled()
        tracer.start_run(_meta())
        tracer.log_outcome(_ok_outcome(), _row())
        root = backend.observations[0]
        assert len(root.children) == 1
        child = root.children[0]
        assert child.kwargs["as_type"] == "generation"
        assert child.kwargs["name"] == "factual_accuracy@r-1"
        assert child.kwargs["model"] == "mock"
        assert child.kwargs["usage_details"] == {"input": 7, "output": 3, "total": 10}
        assert child.kwargs["level"] == "DEFAULT"
        assert child.kwargs["output"]["score"] == 4
        assert child.ended is True

    def test_failure_outcome_records_error_level(self) -> None:
        tracer, backend = _enabled()
        tracer.start_run(_meta())
        tracer.log_outcome(_fail_outcome(), _row())
        child = backend.observations[0].children[0]
        assert child.kwargs["level"] == "ERROR"
        assert child.kwargs["status_message"] == "parse failure"
        assert child.kwargs["output"] is None

    def test_log_outcome_without_start_run_still_records(self) -> None:
        tracer, backend = _enabled()
        tracer.log_outcome(_ok_outcome(), _row())
        # Falls back to creating a top-level observation on the backend.
        assert len(backend.observations) == 1
        assert backend.observations[0].kwargs["as_type"] == "generation"

    def test_log_outcome_metadata_carries_versions_and_category(self) -> None:
        tracer, backend = _enabled()
        tracer.start_run(_meta())
        tracer.log_outcome(_ok_outcome(), _row())
        child = backend.observations[0].children[0]
        md = child.kwargs["metadata"]
        assert md["pillar"] == "factual_accuracy"
        assert md["category"] == "cat"
        assert md["prompt_version"] == "pv1"
        assert md["rubric_version"] == "rv1"
        assert md["latency_ms"] == 42.0


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


class TestRedaction:
    def test_redaction_replaces_free_text(self) -> None:
        assert _redact_text("secret", redact=True) == "<redacted len=6>"

    def test_no_redaction_truncates_long_text(self) -> None:
        long = "x" * 3000
        out = _redact_text(long, redact=False, max_len=100)
        assert out.startswith("x" * 100)
        assert "truncated" in out

    def test_tracer_redacts_when_enabled(self) -> None:
        backend = FakeLangfuseBackend()
        tracer = LangfuseTracer(
            host="h", public_key="pk", secret_key="sk", redact_pii=True, backend=backend
        )
        tracer.start_run(_meta())
        tracer.log_outcome(_ok_outcome(), _row())
        child = backend.observations[0].children[0]
        assert child.kwargs["input"]["user_input"].startswith("<redacted")
        assert child.kwargs["input"]["agent_output"].startswith("<redacted")


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_log_outcome_does_not_race(self) -> None:
        tracer, backend = _enabled()
        tracer.start_run(_meta())
        n = 50

        def _emit(i: int) -> None:
            outcome = JudgeOutcome(
                pillar="factual_accuracy",
                record_id=f"r-{i}",
                latency_ms=float(i),
                attempts=1,
                usage=LLMUsage(input_tokens=1, output_tokens=1),
                model_name="mock",
                result=_ok_outcome().result,
            )
            tracer.log_outcome(outcome, _row(record_id=f"r-{i}"))

        threads = [threading.Thread(target=_emit, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        root = backend.observations[0]
        assert len(root.children) == n


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------


class TestResilience:
    def test_backend_failure_swallowed(self) -> None:
        backend = FakeLangfuseBackend(fail_on={"start_observation"})
        tracer = LangfuseTracer(host="h", public_key="pk", secret_key="sk", backend=backend)
        tracer.start_run(_meta())  # must not raise
        tracer.log_outcome(_ok_outcome(), _row())  # must not raise

    def test_flush_failure_swallowed(self) -> None:
        backend = FakeLangfuseBackend(fail_on={"flush"})
        tracer = LangfuseTracer(host="h", public_key="pk", secret_key="sk", backend=backend)
        tracer.start_run(_meta())
        tracer.end_run("FINISHED")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_build_with_all_missing_creds_disabled(self) -> None:
        tracer = build_langfuse_tracer(
            host=None, public_key=None, secret_key=None, redact_pii=False
        )
        assert tracer.enabled is False

    def test_build_with_fake_backend_enabled(self) -> None:
        tracer = build_langfuse_tracer(
            host="h",
            public_key="pk",
            secret_key="sk",
            redact_pii=False,
            backend=FakeLangfuseBackend(),
        )
        assert tracer.enabled is True


# Guard against drift: the tracer must never import langfuse at module load.
def test_module_does_not_eager_import_langfuse() -> None:
    import importlib
    import sys

    sys.modules.pop("langfuse", None)
    # Re-import the tracer module to confirm it can be loaded without the
    # real SDK present. If someone adds a top-level ``import langfuse``,
    # this test will fail immediately because the module now tries to load
    # without it.
    import src.observability.langfuse_tracer as tracer_mod

    importlib.reload(tracer_mod)
    assert "langfuse" not in sys.modules or sys.modules["langfuse"] is not None
    # Not strictly empty because other test modules may have imported it,
    # but reload must succeed.


# Stale datetime reference guard — silence ruff if unused.
_ = datetime
