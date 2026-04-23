"""Unit tests for :mod:`src.observability.run_metadata`.

The module is pure, so these tests are straightforward: build, hash,
flatten. We cover:

- Fingerprint stability (same rows → same hash, regardless of object
  identity).
- Fingerprint sensitivity (different required-column content changes
  the hash).
- Fingerprint insensitivity to optional fields (a rule we deliberately
  enforce to avoid noisy fingerprints).
- Builder error paths (missing bundle / client).
- Flattener correctness (MLflow param keys, Langfuse metadata shape).
"""

from __future__ import annotations

from typing import Any

import pytest

from src.core.types import NormalizedRow
from src.judges.config import JudgeBundle, JudgeConfig
from src.llm.base import LLMUsage
from src.llm.mock_client import MockLLMClient
from src.observability.run_metadata import (
    RunMetadata,
    build_run_metadata,
    dataset_fingerprint,
    run_config_hash,
    to_langfuse_metadata,
    to_mlflow_params,
    to_mlflow_tags,
)
from src.rubrics.models import Rubric, ScoreAnchor

# ---------------------------------------------------------------------------
# Fixtures local to this module
# ---------------------------------------------------------------------------


def _row(record_id: str, **overrides: Any) -> NormalizedRow:
    payload: dict[str, Any] = {
        "record_id": record_id,
        "user_input": "u",
        "agent_output": "a",
        "category": "c",
    }
    payload.update(overrides)
    return NormalizedRow(**payload)


def _bundle(pillar: str, *, prompt_version: str = "v1", rubric_version: str = "v1") -> JudgeBundle:
    return JudgeBundle(
        config=JudgeConfig(
            pillar=pillar,
            prompt_version=prompt_version,
            rubric_path=f"rubrics/{pillar}.yaml",
            rubric_version=rubric_version,
        ),
        rubric=Rubric(
            pillar=pillar,
            version=rubric_version,
            description="desc",
            required_inputs=["user_input", "agent_output"],
            anchors=[
                ScoreAnchor(score=1, name="s1", description="d"),
                ScoreAnchor(score=2, name="s2", description="d"),
                ScoreAnchor(score=3, name="s3", description="d"),
                ScoreAnchor(score=4, name="s4", description="d"),
                ScoreAnchor(score=5, name="s5", description="d"),
            ],
            failure_tags=["tag"],
        ),
    )


# ---------------------------------------------------------------------------
# dataset_fingerprint
# ---------------------------------------------------------------------------


class TestDatasetFingerprint:
    def test_empty_dataset_fingerprint_is_valid(self) -> None:
        fp = dataset_fingerprint([])
        assert fp.startswith("sha256:")
        assert len(fp) == len("sha256:") + 16

    def test_same_rows_same_fingerprint(self) -> None:
        rows_a = [_row("r1"), _row("r2")]
        rows_b = [_row("r1"), _row("r2")]
        assert dataset_fingerprint(rows_a) == dataset_fingerprint(rows_b)

    def test_row_reorder_changes_fingerprint(self) -> None:
        a = [_row("r1"), _row("r2")]
        b = [_row("r2"), _row("r1")]
        # Order matters because the runner preserves order too.
        assert dataset_fingerprint(a) != dataset_fingerprint(b)

    def test_content_change_changes_fingerprint(self) -> None:
        base = [_row("r1")]
        changed = [_row("r1", agent_output="different")]
        assert dataset_fingerprint(base) != dataset_fingerprint(changed)

    def test_fingerprint_ignores_optional_columns(self) -> None:
        base = [_row("r1")]
        with_extras = [_row("r1", metadata={"topic": "billing"}, chat_history=None)]
        # Metadata / chat_history are deliberately excluded so trivial
        # ETL reshaping doesn't bust the fingerprint.
        assert dataset_fingerprint(base) == dataset_fingerprint(with_extras)


# ---------------------------------------------------------------------------
# run_config_hash
# ---------------------------------------------------------------------------


class TestRunConfigHash:
    def test_key_order_stable(self) -> None:
        a = {"pillars": ["x", "y"], "provider": "mock"}
        b = {"provider": "mock", "pillars": ["x", "y"]}
        assert run_config_hash(a) == run_config_hash(b)

    def test_list_order_matters(self) -> None:
        a = {"pillars": ["x", "y"]}
        b = {"pillars": ["y", "x"]}
        assert run_config_hash(a) != run_config_hash(b)

    def test_handles_non_jsonable_values(self) -> None:
        # Decimal / Path / etc. are coerced via default=str.
        from pathlib import Path

        h1 = run_config_hash({"root": Path("/tmp/x")})
        h2 = run_config_hash({"root": "/tmp/x"})
        assert h1 == h2


# ---------------------------------------------------------------------------
# build_run_metadata
# ---------------------------------------------------------------------------


class TestBuildRunMetadata:
    @pytest.fixture
    def components(self) -> dict[str, Any]:
        bundles = {
            "factual_accuracy": _bundle(
                "factual_accuracy", prompt_version="fa.v1", rubric_version="fa-rubric.v1"
            ),
            "relevance": _bundle(
                "relevance", prompt_version="rel.v2", rubric_version="rel-rubric.v3"
            ),
        }
        llms = {
            "factual_accuracy": MockLLMClient(
                model_name="mock-fa", usage=LLMUsage(input_tokens=1, output_tokens=1)
            ),
            "relevance": MockLLMClient(
                model_name="mock-rel", usage=LLMUsage(input_tokens=1, output_tokens=1)
            ),
        }
        return {"bundles": bundles, "llms": llms}

    def test_builder_populates_per_pillar_maps(self, components: dict[str, Any]) -> None:
        meta = build_run_metadata(
            run_id="r-1",
            dataset_fingerprint="sha256:aaaa",
            dataset_row_count=10,
            pillars=["factual_accuracy", "relevance"],
            bundles=components["bundles"],
            llm_by_pillar=components["llms"],
            provider="mock",
            run_config={"use_cache": True},
            kb_version="kb-v1",
        )
        assert isinstance(meta, RunMetadata)
        assert meta.pillars == ("factual_accuracy", "relevance")
        assert meta.model_alias_by_pillar == {
            "factual_accuracy": "judge-default",
            "relevance": "judge-default",
        }
        assert meta.model_name_by_pillar == {
            "factual_accuracy": "mock-fa",
            "relevance": "mock-rel",
        }
        assert meta.prompt_version_by_pillar == {
            "factual_accuracy": "fa.v1",
            "relevance": "rel.v2",
        }
        assert meta.rubric_version_by_pillar == {
            "factual_accuracy": "fa-rubric.v1",
            "relevance": "rel-rubric.v3",
        }
        assert meta.kb_version == "kb-v1"
        assert meta.run_config_hash is not None
        assert meta.run_config_hash.startswith("sha256:")

    def test_missing_bundle_raises(self, components: dict[str, Any]) -> None:
        with pytest.raises(KeyError, match="bundles missing"):
            build_run_metadata(
                run_id="r-1",
                dataset_fingerprint="sha256:aaaa",
                dataset_row_count=1,
                pillars=["factual_accuracy", "toxicity"],
                bundles=components["bundles"],
                llm_by_pillar=components["llms"],
            )

    def test_missing_client_raises(self, components: dict[str, Any]) -> None:
        bundles = {**components["bundles"], "toxicity": _bundle("toxicity")}
        with pytest.raises(KeyError, match="llm_by_pillar missing"):
            build_run_metadata(
                run_id="r-1",
                dataset_fingerprint="sha256:aaaa",
                dataset_row_count=1,
                pillars=["factual_accuracy", "toxicity"],
                bundles=bundles,
                llm_by_pillar=components["llms"],
            )

    def test_no_run_config_produces_no_hash(self, components: dict[str, Any]) -> None:
        meta = build_run_metadata(
            run_id="r-1",
            dataset_fingerprint="sha256:aaaa",
            dataset_row_count=1,
            pillars=["factual_accuracy"],
            bundles=components["bundles"],
            llm_by_pillar=components["llms"],
        )
        assert meta.run_config_hash is None


# ---------------------------------------------------------------------------
# Flatteners
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_meta() -> RunMetadata:
    bundles = {
        "factual_accuracy": _bundle("factual_accuracy", prompt_version="pv1", rubric_version="rv1")
    }
    llms = {
        "factual_accuracy": MockLLMClient(
            model_name="mock-fa", usage=LLMUsage(input_tokens=1, output_tokens=1)
        )
    }
    return build_run_metadata(
        run_id="run-42",
        dataset_fingerprint="sha256:deadbeef",
        dataset_row_count=3,
        pillars=["factual_accuracy"],
        bundles=bundles,
        llm_by_pillar=llms,
        provider="mock",
        run_config={"max_workers": 4},
        kb_version="kb-v3",
        extra_tags={"dataset_name": "sample"},
    )


class TestToMLflowParams:
    def test_core_keys_present(self, sample_meta: RunMetadata) -> None:
        params = to_mlflow_params(sample_meta)
        assert params["run_id"] == "run-42"
        assert params["dataset_fingerprint"] == "sha256:deadbeef"
        assert params["dataset_row_count"] == "3"
        assert params["pillars"] == "factual_accuracy"
        assert params["provider"] == "mock"
        assert params["kb_version"] == "kb-v3"
        assert params["run_config_hash"].startswith("sha256:")

    def test_per_pillar_keys(self, sample_meta: RunMetadata) -> None:
        params = to_mlflow_params(sample_meta)
        assert params["prompt_version_factual_accuracy"] == "pv1"
        assert params["rubric_version_factual_accuracy"] == "rv1"
        assert params["model_alias_factual_accuracy"] == "judge-default"
        assert params["model_name_factual_accuracy"] == "mock-fa"

    def test_all_values_are_strings(self, sample_meta: RunMetadata) -> None:
        params = to_mlflow_params(sample_meta)
        assert all(isinstance(v, str) for v in params.values())


class TestToMLflowTags:
    def test_namespace_prefix(self, sample_meta: RunMetadata) -> None:
        tags = to_mlflow_tags(sample_meta)
        assert all(k.startswith("jtb.") for k in tags)

    def test_extra_tags_merged(self, sample_meta: RunMetadata) -> None:
        tags = to_mlflow_tags(sample_meta)
        assert tags["jtb.dataset_name"] == "sample"


class TestToLangfuseMetadata:
    def test_per_pillar_nested(self, sample_meta: RunMetadata) -> None:
        meta = to_langfuse_metadata(sample_meta)
        assert "per_pillar" in meta
        fa = meta["per_pillar"]["factual_accuracy"]
        assert fa["prompt_version"] == "pv1"
        assert fa["rubric_version"] == "rv1"
        assert fa["model_name"] == "mock-fa"

    def test_pillars_preserves_order(self, sample_meta: RunMetadata) -> None:
        meta = to_langfuse_metadata(sample_meta)
        assert meta["pillars"] == ["factual_accuracy"]

    def test_optional_keys_only_when_set(self) -> None:
        bundles = {"toxicity": _bundle("toxicity")}
        llms = {
            "toxicity": MockLLMClient(
                model_name="m", usage=LLMUsage(input_tokens=0, output_tokens=0)
            )
        }
        meta = build_run_metadata(
            run_id="r",
            dataset_fingerprint="sha256:ff",
            dataset_row_count=0,
            pillars=["toxicity"],
            bundles=bundles,
            llm_by_pillar=llms,
        )
        encoded = to_langfuse_metadata(meta)
        assert "kb_version" not in encoded
        assert "run_config_hash" not in encoded
        assert "extra_tags" not in encoded
