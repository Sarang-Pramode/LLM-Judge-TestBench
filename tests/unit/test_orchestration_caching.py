"""Unit tests for :mod:`src.orchestration.caching`."""

from __future__ import annotations

import threading
from pathlib import Path

from src.core.types import JudgeResult, NormalizedRow
from src.judges.base import JudgeOutcome
from src.judges.config import JudgeBundle, JudgeConfig
from src.llm.base import LLMUsage
from src.llm.mock_client import MockLLMClient
from src.orchestration.caching import (
    InMemoryOutcomeCache,
    NoCache,
    compute_cache_key,
)
from src.rubrics.loader import load_rubric

REPO_ROOT = Path(__file__).resolve().parents[2]
RUBRICS_DIR = REPO_ROOT / "configs" / "rubrics"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(rid: str = "r-1", category: str = "c") -> NormalizedRow:
    return NormalizedRow(
        record_id=rid,
        user_input="u",
        agent_output="a",
        category=category,
    )


def _bundle(pillar: str = "factual_accuracy") -> JudgeBundle:
    rubric = load_rubric(RUBRICS_DIR / f"{pillar}.yaml")
    config = JudgeConfig(
        pillar=pillar,
        prompt_version=f"{pillar}.v1",
        rubric_path=f"rubrics/{pillar}.yaml",
        rubric_version=rubric.version,
        model_alias="mock-judge-default",
        temperature=0.0,
        max_output_tokens=512,
    )
    return JudgeBundle(config=config, rubric=rubric)


def _outcome(pillar: str = "factual_accuracy", rid: str = "r-1", ok: bool = True) -> JudgeOutcome:
    if ok:
        result = JudgeResult(
            pillar=pillar,
            score=4,
            confidence=0.8,
            decision_summary="ok",
            evidence_for_score=[],
            failure_tags=[],
            why_not_higher="hh",
            why_not_lower="ll",
            rubric_anchor=4,
            raw_model_name="mock",
            prompt_version="p.v1",
            rubric_version="v1",
        )
        return JudgeOutcome(
            pillar=pillar,
            record_id=rid,
            latency_ms=10.0,
            attempts=1,
            usage=LLMUsage(input_tokens=10, output_tokens=5),
            model_name="mock",
            run_id="rid",
            result=result,
        )
    return JudgeOutcome(
        pillar=pillar,
        record_id=rid,
        latency_ms=0.0,
        attempts=0,
        usage=LLMUsage(),
        error="boom",
        error_type="ProviderError",
    )


# ---------------------------------------------------------------------------
# NoCache
# ---------------------------------------------------------------------------


class TestNoCache:
    def test_get_always_none(self) -> None:
        cache = NoCache()
        cache.set("k", _outcome())
        assert cache.get("k") is None


# ---------------------------------------------------------------------------
# InMemoryOutcomeCache
# ---------------------------------------------------------------------------


class TestInMemoryOutcomeCache:
    def test_roundtrip(self) -> None:
        cache = InMemoryOutcomeCache()
        outcome = _outcome()
        cache.set("k1", outcome)
        got = cache.get("k1")
        assert got is outcome

    def test_errors_are_not_cached(self) -> None:
        cache = InMemoryOutcomeCache()
        cache.set("k-err", _outcome(ok=False))
        assert cache.get("k-err") is None
        assert len(cache) == 0

    def test_clear(self) -> None:
        cache = InMemoryOutcomeCache()
        cache.set("a", _outcome(rid="a"))
        cache.set("b", _outcome(rid="b"))
        assert len(cache) == 2
        cache.clear()
        assert len(cache) == 0

    def test_is_thread_safe_under_concurrent_writes(self) -> None:
        cache = InMemoryOutcomeCache()
        errors: list[Exception] = []
        errors_lock = threading.Lock()

        def writer(tid: int) -> None:
            try:
                for i in range(200):
                    outcome = _outcome(rid=f"r{tid}-{i}")
                    cache.set(f"k{tid}-{i}", outcome)
                    got = cache.get(f"k{tid}-{i}")
                    assert got is outcome
            except Exception as exc:  # pragma: no cover - failure path
                with errors_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(cache) == 8 * 200


# ---------------------------------------------------------------------------
# compute_cache_key
# ---------------------------------------------------------------------------


class TestComputeCacheKey:
    def test_key_is_deterministic(self) -> None:
        bundle = _bundle()
        llm = MockLLMClient(model_name="mock-v1")
        row = _row()
        k1 = compute_cache_key(
            pillar="factual_accuracy", bundle=bundle, llm=llm, row=row, kb_fingerprint=None
        )
        k2 = compute_cache_key(
            pillar="factual_accuracy", bundle=bundle, llm=llm, row=row, kb_fingerprint=None
        )
        assert k1 == k2
        assert len(k1) == 64  # sha256 hex

    def test_key_changes_with_pillar(self) -> None:
        bundle_fa = _bundle("factual_accuracy")
        bundle_rel = _bundle("relevance")
        llm = MockLLMClient(model_name="mock-v1")
        row = _row()
        k_fa = compute_cache_key(
            pillar="factual_accuracy",
            bundle=bundle_fa,
            llm=llm,
            row=row,
            kb_fingerprint=None,
        )
        k_rel = compute_cache_key(
            pillar="relevance", bundle=bundle_rel, llm=llm, row=row, kb_fingerprint=None
        )
        assert k_fa != k_rel

    def test_key_changes_with_model_name(self) -> None:
        bundle = _bundle()
        row = _row()
        k1 = compute_cache_key(
            pillar="factual_accuracy",
            bundle=bundle,
            llm=MockLLMClient(model_name="m-1"),
            row=row,
            kb_fingerprint=None,
        )
        k2 = compute_cache_key(
            pillar="factual_accuracy",
            bundle=bundle,
            llm=MockLLMClient(model_name="m-2"),
            row=row,
            kb_fingerprint=None,
        )
        assert k1 != k2

    def test_key_changes_with_row_content(self) -> None:
        bundle = _bundle()
        llm = MockLLMClient(model_name="m")
        k1 = compute_cache_key(
            pillar="factual_accuracy",
            bundle=bundle,
            llm=llm,
            row=_row("r-1"),
            kb_fingerprint=None,
        )
        k2 = compute_cache_key(
            pillar="factual_accuracy",
            bundle=bundle,
            llm=llm,
            row=_row("r-2"),
            kb_fingerprint=None,
        )
        assert k1 != k2

    def test_key_changes_with_kb_fingerprint(self) -> None:
        bundle = _bundle()
        llm = MockLLMClient(model_name="m")
        row = _row()
        k1 = compute_cache_key(
            pillar="completeness",
            bundle=_bundle("completeness"),
            llm=llm,
            row=row,
            kb_fingerprint="kb-a",
        )
        k2 = compute_cache_key(
            pillar="completeness",
            bundle=_bundle("completeness"),
            llm=llm,
            row=row,
            kb_fingerprint="kb-b",
        )
        assert k1 != k2
        # Bundle unused warning guard - ensure we referenced it.
        assert bundle is not None
