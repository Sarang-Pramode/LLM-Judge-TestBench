"""Unit tests for :class:`src.orchestration.runner.EvaluationRunner`.

These exercise the runner in isolation using scripted
:class:`MockLLMClient` responses - no real provider, no KB yet. KB
integration lives in the Stage 7 integration tests.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from src.core.exceptions import JudgeExecutionError, ProviderError
from src.core.types import NormalizedRow, RunContext
from src.judges.base import JudgeCoreOutput
from src.judges.config import load_judge_bundle
from src.llm.base import LLMUsage
from src.llm.mock_client import MockLLMClient
from src.orchestration.batching import plan_tasks
from src.orchestration.caching import InMemoryOutcomeCache
from src.orchestration.concurrency import ConcurrencyPolicy
from src.orchestration.runner import EvaluationRunner, RunPlan

REPO_ROOT = Path(__file__).resolve().parents[2]
JUDGES_DIR = REPO_ROOT / "configs" / "judges"
RUBRICS_DIR = REPO_ROOT / "configs" / "rubrics"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _register_pillars(all_pillars_registered: None) -> None:
    """Auto-register production pillar judges for every test in the module."""
    return None


def _row(rid: str, category: str = "disputes", topic: str | None = None) -> NormalizedRow:
    # factual_accuracy + hallucination require retrieved_context, so
    # supply a minimal value to keep rows valid across all pillars we
    # test here.
    return NormalizedRow(
        record_id=rid,
        user_input=f"user asks {rid}",
        agent_output=f"agent replies for {rid}",
        category=category,
        topic=topic,
        retrieved_context=["chunk-1 reference material"],
    )


def _run_ctx() -> RunContext:
    return RunContext(
        run_id="rid-test",
        dataset_fingerprint="sha256:testds",
        model_alias="mock",
    )


def _core(pillar: str, *, score: int = 4, failure_tag: str | None = None) -> JudgeCoreOutput:
    payload: dict[str, Any] = {
        "pillar": pillar,
        "score": score,
        "confidence": 0.8,
        "decision_summary": f"{pillar} judgement",
        "evidence_for_score": [],
        "failure_tags": [failure_tag] if failure_tag else [],
        "why_not_higher": "hh" if score < 5 else None,
        "why_not_lower": "ll" if score > 1 else None,
        "rubric_anchor": score,
    }
    return JudgeCoreOutput(**payload)


def _fa_payload() -> JudgeCoreOutput:
    return _core("factual_accuracy", failure_tag="unsupported_claim")


def _rel_payload() -> JudgeCoreOutput:
    return _core("relevance", failure_tag="off_topic")


def _tox_payload() -> JudgeCoreOutput:
    return _core("toxicity", score=5)


def _fa_bundle() -> Any:
    return load_judge_bundle(JUDGES_DIR / "factual_accuracy.yaml", rubric_root=RUBRICS_DIR)


def _rel_bundle() -> Any:
    return load_judge_bundle(JUDGES_DIR / "relevance.yaml", rubric_root=RUBRICS_DIR)


def _tox_bundle() -> Any:
    return load_judge_bundle(JUDGES_DIR / "toxicity.yaml", rubric_root=RUBRICS_DIR)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_rows_raises(self) -> None:
        llm = MockLLMClient(model_name="m")
        plan = RunPlan(
            rows=[],
            pillars=["factual_accuracy"],
            bundles={"factual_accuracy": _fa_bundle()},
            llm_by_pillar={"factual_accuracy": llm},
            run_context=_run_ctx(),
        )
        with pytest.raises(JudgeExecutionError, match="rows is empty"):
            EvaluationRunner().run(plan)

    def test_missing_bundle_raises(self) -> None:
        llm = MockLLMClient(model_name="m")
        plan = RunPlan(
            rows=[_row("r1")],
            pillars=["factual_accuracy"],
            bundles={},
            llm_by_pillar={"factual_accuracy": llm},
            run_context=_run_ctx(),
        )
        with pytest.raises(JudgeExecutionError, match="bundles is missing"):
            EvaluationRunner().run(plan)

    def test_duplicate_pillar_raises(self) -> None:
        llm = MockLLMClient(model_name="m")
        plan = RunPlan(
            rows=[_row("r1")],
            pillars=["factual_accuracy", "factual_accuracy"],
            bundles={"factual_accuracy": _fa_bundle()},
            llm_by_pillar={"factual_accuracy": llm},
            run_context=_run_ctx(),
        )
        with pytest.raises(JudgeExecutionError, match="twice"):
            EvaluationRunner().run(plan)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_single_pillar_single_row(self) -> None:
        llm = MockLLMClient(
            model_name="mock-fa",
            structured_script=[_fa_payload()],
            usage=LLMUsage(input_tokens=10, output_tokens=5),
        )
        plan = RunPlan(
            rows=[_row("r1")],
            pillars=["factual_accuracy"],
            bundles={"factual_accuracy": _fa_bundle()},
            llm_by_pillar={"factual_accuracy": llm},
            run_context=_run_ctx(),
        )
        result = EvaluationRunner().run(plan)
        assert len(result.outcomes) == 1
        assert result.outcomes[0].ok
        assert result.summary.succeeded == 1
        assert result.summary.failed == 0
        assert result.summary.total_input_tokens == 10
        assert result.summary.total_output_tokens == 5
        assert result.run_id == "rid-test"

    def test_multi_pillar_multi_row_deterministic_order(self) -> None:
        llm_fa = MockLLMClient(
            model_name="mock-fa", structured_script=[_fa_payload() for _ in range(3)]
        )
        llm_rel = MockLLMClient(
            model_name="mock-rel", structured_script=[_rel_payload() for _ in range(3)]
        )
        llm_tox = MockLLMClient(
            model_name="mock-tox", structured_script=[_tox_payload() for _ in range(3)]
        )
        rows = [_row(f"r{i}") for i in range(3)]
        plan = RunPlan(
            rows=rows,
            pillars=["factual_accuracy", "relevance", "toxicity"],
            bundles={
                "factual_accuracy": _fa_bundle(),
                "relevance": _rel_bundle(),
                "toxicity": _tox_bundle(),
            },
            llm_by_pillar={
                "factual_accuracy": llm_fa,
                "relevance": llm_rel,
                "toxicity": llm_tox,
            },
            run_context=_run_ctx(),
            concurrency=ConcurrencyPolicy(max_workers=4),
        )
        result = EvaluationRunner().run(plan)

        # Outcomes are sorted by (row_index, pillar). With the pillars
        # above that means (r0, factual_accuracy), (r0, relevance), ...
        ordered = [(o.record_id, o.pillar) for o in result.outcomes]
        expected = plan_tasks(rows, ["factual_accuracy", "relevance", "toxicity"])
        assert ordered == [(t.record_id, t.pillar) for t in expected]
        assert result.summary.total_tasks == 9
        assert result.summary.succeeded == 9
        assert result.summary.pillar_stats == {
            "factual_accuracy": (3, 0),
            "relevance": (3, 0),
            "toxicity": (3, 0),
        }


# ---------------------------------------------------------------------------
# Partial failure
# ---------------------------------------------------------------------------


class TestPartialFailure:
    def test_provider_error_surfaces_on_outcome_not_exception(self) -> None:
        llm = MockLLMClient(
            model_name="mock-fa",
            structured_script=[
                _fa_payload(),
                ProviderError("simulated rate limit"),
                _fa_payload(),
            ],
        )
        rows = [_row(f"r{i}") for i in range(3)]
        plan = RunPlan(
            rows=rows,
            pillars=["factual_accuracy"],
            bundles={"factual_accuracy": _fa_bundle()},
            llm_by_pillar={"factual_accuracy": llm},
            run_context=_run_ctx(),
            concurrency=ConcurrencyPolicy(max_workers=1),  # determinism
        )
        result = EvaluationRunner().run(plan)
        assert result.summary.succeeded == 2
        assert result.summary.failed == 1
        errored = [o for o in result.outcomes if not o.ok]
        assert len(errored) == 1
        assert errored[0].error_type == "ProviderError"
        # A provider failure in the middle of a run must not stop the
        # other rows from completing.
        assert all(o.record_id in {"r0", "r1", "r2"} for o in result.outcomes)


# ---------------------------------------------------------------------------
# Cache hits
# ---------------------------------------------------------------------------


class TestCaching:
    def test_cache_hit_skips_llm_call(self) -> None:
        llm = MockLLMClient(
            model_name="mock-fa",
            structured_script=[_fa_payload()],  # ONLY one scripted response
        )
        cache = InMemoryOutcomeCache()
        rows = [_row("r1"), _row("r1")]  # identical content
        # Use two row objects with the same content. The cache key is
        # content-based, so the second call should hit the cache; if
        # it didn't, MockLLMClient would raise (no more scripted
        # responses).
        plan = RunPlan(
            rows=rows,
            pillars=["factual_accuracy"],
            bundles={"factual_accuracy": _fa_bundle()},
            llm_by_pillar={"factual_accuracy": llm},
            run_context=_run_ctx(),
            cache=cache,
            concurrency=ConcurrencyPolicy(max_workers=1),
        )
        result = EvaluationRunner().run(plan)
        assert result.summary.cache_hits == 1
        assert result.summary.succeeded == 2
        # Only one LLM call actually happened.
        structured_calls = [c for c in llm.calls if c.kind == "structured"]
        assert len(structured_calls) == 1


# ---------------------------------------------------------------------------
# Callbacks + parallelism
# ---------------------------------------------------------------------------


class TestCallbacks:
    def test_on_progress_reports_full_total(self) -> None:
        llm = MockLLMClient(model_name="m", structured_script=[_fa_payload() for _ in range(4)])
        rows = [_row(f"r{i}") for i in range(4)]

        seen: list[tuple[int, int]] = []
        seen_lock = threading.Lock()

        def _cb(done: int, total: int) -> None:
            with seen_lock:
                seen.append((done, total))

        plan = RunPlan(
            rows=rows,
            pillars=["factual_accuracy"],
            bundles={"factual_accuracy": _fa_bundle()},
            llm_by_pillar={"factual_accuracy": llm},
            run_context=_run_ctx(),
            on_progress=_cb,
            concurrency=ConcurrencyPolicy(max_workers=2),
        )
        EvaluationRunner().run(plan)
        # Every task produces one callback; totals constant.
        assert len(seen) == 4
        assert all(total == 4 for _done, total in seen)
        assert sorted(d for d, _t in seen) == [1, 2, 3, 4]

    def test_on_outcome_sees_every_outcome(self) -> None:
        llm = MockLLMClient(model_name="m", structured_script=[_fa_payload() for _ in range(4)])
        rows = [_row(f"r{i}") for i in range(4)]
        seen: list[str] = []
        lock = threading.Lock()

        def _on(o: Any) -> None:
            with lock:
                seen.append(o.record_id)

        plan = RunPlan(
            rows=rows,
            pillars=["factual_accuracy"],
            bundles={"factual_accuracy": _fa_bundle()},
            llm_by_pillar={"factual_accuracy": llm},
            run_context=_run_ctx(),
            on_outcome=_on,
            concurrency=ConcurrencyPolicy(max_workers=2),
        )
        EvaluationRunner().run(plan)
        assert sorted(seen) == ["r0", "r1", "r2", "r3"]

    def test_parallelism_respects_per_provider_limit(self) -> None:
        """With 4 workers and per_provider_limit=1, the script queue
        must NOT race - a bounded semaphore should serialize calls to
        the same client.
        """
        n = 20
        llm = MockLLMClient(
            model_name="m-single",
            structured_script=[_fa_payload() for _ in range(n)],
        )
        rows = [_row(f"r{i}") for i in range(n)]
        plan = RunPlan(
            rows=rows,
            pillars=["factual_accuracy"],
            bundles={"factual_accuracy": _fa_bundle()},
            llm_by_pillar={"factual_accuracy": llm},
            run_context=_run_ctx(),
            concurrency=ConcurrencyPolicy(max_workers=4, per_provider_limit=1),
        )
        result = EvaluationRunner().run(plan)
        assert result.summary.succeeded == n
        # n tasks -> n structured calls despite 4 workers.
        assert len([c for c in llm.calls if c.kind == "structured"]) == n


# ---------------------------------------------------------------------------
# Wall-clock sanity
# ---------------------------------------------------------------------------


class TestWallClock:
    def test_summary_latency_percentiles_sane(self) -> None:
        llm = MockLLMClient(model_name="m", structured_script=[_fa_payload() for _ in range(5)])
        rows = [_row(f"r{i}") for i in range(5)]
        plan = RunPlan(
            rows=rows,
            pillars=["factual_accuracy"],
            bundles={"factual_accuracy": _fa_bundle()},
            llm_by_pillar={"factual_accuracy": llm},
            run_context=_run_ctx(),
            concurrency=ConcurrencyPolicy(max_workers=3),
        )
        t0 = time.perf_counter()
        result = EvaluationRunner().run(plan)
        elapsed = time.perf_counter() - t0
        assert result.summary.duration_s <= elapsed + 0.05
        assert result.summary.latency_ms_p50 >= 0.0
        assert result.summary.latency_ms_p95 >= result.summary.latency_ms_p50
