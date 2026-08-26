"""Evaluation: dataset validation, summaries, baseline, serializability."""

from __future__ import annotations

import json

import pytest

from research_system.adapters.llm import FakeLLMClient
from research_system.core.state import RetrievedDocument
from research_system.errors import EvaluationError
from research_system.evaluation.benchmark import SingleAgentBaseline
from research_system.evaluation.evaluate import (
    RunRecord,
    environment_metadata,
    evaluate_dataset,
    load_dataset,
    summarize,
)
from research_system.schemas import GeneratorOutput


def write_dataset(tmp_path, payload):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- dataset validation ----------------------------------------------------
def test_valid_dataset_loads(tmp_path):
    path = write_dataset(tmp_path, [{"question": "q1", "ground_truth": "g1"}])
    items = load_dataset(path)

    assert items[0].question == "q1"
    assert items[0].ground_truth == "g1"


def test_missing_file_raises(tmp_path):
    with pytest.raises(EvaluationError, match="not found"):
        load_dataset(tmp_path / "nope.json")


def test_malformed_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(EvaluationError, match="not valid JSON"):
        load_dataset(path)


def test_non_array_dataset_raises(tmp_path):
    path = write_dataset(tmp_path, {"question": "q"})

    with pytest.raises(EvaluationError, match="must be a JSON array"):
        load_dataset(path)


@pytest.mark.parametrize("bad", [[{"question": ""}], [{"question": "   "}], [{}]])
def test_empty_question_raises(tmp_path, bad):
    path = write_dataset(tmp_path, bad)

    with pytest.raises(EvaluationError, match="empty 'question'"):
        load_dataset(path)


def test_non_object_entry_raises(tmp_path):
    path = write_dataset(tmp_path, ["just a string"])

    with pytest.raises(EvaluationError, match="not an object"):
        load_dataset(path)


def test_missing_ground_truth_defaults_to_empty(tmp_path):
    path = write_dataset(tmp_path, [{"question": "q"}])
    assert load_dataset(path)[0].ground_truth == ""


def test_empty_dataset_loads_as_empty(tmp_path):
    assert load_dataset(write_dataset(tmp_path, [])) == []


def test_the_shipped_sample_dataset_is_valid():
    from pathlib import Path

    dataset = Path(__file__).resolve().parents[2] / "eval/datasets/sample_research_qa.json"
    items = load_dataset(dataset)

    assert len(items) >= 5
    assert all(item.question and item.ground_truth for item in items)


# --- summaries -------------------------------------------------------------
def test_empty_records_do_not_divide_by_zero():
    summary = summarize([])

    assert summary["total_questions"] == 0
    assert summary["avg_latency_s"] == 0.0
    assert summary["p95_latency_s"] == 0.0
    assert summary["retry_rate"] == 0.0
    assert summary["verification_pass_rate"] == 0.0
    assert summary["failed_questions"] == 0


def test_summary_computes_rates():
    records = [
        RunRecord("q1", "g", "a", latency_s=1.0, retry_count=0, is_verified=True),
        RunRecord("q2", "g", "a", latency_s=3.0, retry_count=2, is_verified=False),
        RunRecord("q3", "g", "", latency_s=2.0, retry_count=0, is_verified=False, error="boom"),
    ]
    summary = summarize(records)

    assert summary["total_questions"] == 3
    assert summary["failed_questions"] == 1
    assert summary["avg_latency_s"] == pytest.approx(2.0)
    assert summary["retry_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert summary["verification_pass_rate"] == pytest.approx(1 / 3, abs=1e-4)


def test_p95_of_a_single_record_is_that_record():
    assert summarize([RunRecord("q", "g", "a", latency_s=4.2)])["p95_latency_s"] == 4.2


# --- environment metadata --------------------------------------------------
def test_environment_records_model_and_dataset_hash(tmp_path, settings):
    path = write_dataset(tmp_path, [{"question": "q"}])
    meta = environment_metadata(settings, path)

    assert meta["model"] == settings.default_llm
    assert meta["embedding_model"] == settings.embedding_model
    assert len(meta["dataset_sha256"]) == 64
    assert "dependency_versions" in meta
    assert meta["python"]


def test_environment_omits_dataset_fields_when_absent(settings):
    assert "dataset_sha256" not in environment_metadata(settings, None)


def test_environment_contains_no_secrets(tmp_path, settings):
    path = write_dataset(tmp_path, [{"question": "q"}])
    rendered = json.dumps(environment_metadata(settings, path))

    assert "test-openai-key" not in rendered
    assert "test-anthropic-key" not in rendered


# --- baseline --------------------------------------------------------------
def _seed_web(deps, web_search, query, content="baseline evidence"):
    deps.vector_store = None
    deps.embeddings = None
    web_search.results = {
        query: [
            RetrievedDocument(
                content=content,
                source="https://example.com",
                score=0.9,
                metadata={"source_type": "web"},
            )
        ]
    }


def test_baseline_runs_a_real_query(deps, web_search):
    _seed_web(deps, web_search, "what is rrf?")
    deps.llm = FakeLLMClient(
        [GeneratorOutput(answer="RRF fuses lists [Source 1]", confidence=0.7, sources_used=[1])]
    )

    result = SingleAgentBaseline(deps).run("what is rrf?")

    assert result["response"] == "RRF fuses lists [Source 1]"
    assert result["sources"][0]["index"] == 1
    assert result["_contexts"] == ["baseline evidence"]


def test_baseline_does_not_decompose_the_question(deps, web_search):
    _seed_web(deps, web_search, "one question")
    deps.llm = FakeLLMClient([GeneratorOutput(answer="a", confidence=0.5, sources_used=[1])])

    SingleAgentBaseline(deps).run("one question")

    assert web_search.queries == ["one question"]


def test_baseline_makes_exactly_one_model_call(deps, web_search):
    _seed_web(deps, web_search, "q")
    deps.llm = FakeLLMClient([GeneratorOutput(answer="a", confidence=0.5, sources_used=[1])])

    SingleAgentBaseline(deps).run("q")

    assert len(deps.llm.calls) == 1


def test_baseline_never_claims_verification(deps, web_search):
    _seed_web(deps, web_search, "q")
    deps.llm = FakeLLMClient([GeneratorOutput(answer="a", confidence=0.9, sources_used=[1])])

    result = SingleAgentBaseline(deps).run("q")

    assert result["is_verified"] is False
    assert result["retry_count"] == 0
    assert "verification" not in result


def test_baseline_validates_citations(deps, web_search):
    _seed_web(deps, web_search, "q")
    deps.llm = FakeLLMClient(
        [GeneratorOutput(answer="a [Source 9]", confidence=0.5, sources_used=[9])]
    )

    result = SingleAgentBaseline(deps).run("q")

    assert [entry["index"] for entry in result["sources"]] == [1]


def test_baseline_handles_empty_retrieval(deps, web_search):
    deps.vector_store = None
    deps.embeddings = None
    deps.llm = FakeLLMClient([])

    result = SingleAgentBaseline(deps).run("nothing indexed")

    assert "could not find any evidence" in result["response"]
    assert deps.llm.calls == []


def test_baseline_rejects_an_empty_query(deps):
    assert SingleAgentBaseline(deps).run("   ")["error"] == "Query was empty."


def test_baseline_reports_a_retrieval_failure(deps):
    deps.vector_store = None
    deps.embeddings = None
    deps.web_search = None

    result = SingleAgentBaseline(deps).run("q")

    assert "No retrieval backend" in result["error"]


# --- full report -----------------------------------------------------------
def test_empty_dataset_produces_a_serializable_report(tmp_path, settings, deps):
    path = write_dataset(tmp_path, [])
    out = tmp_path / "out.json"

    report = evaluate_dataset(path, settings=settings, deps=deps, output_path=out)

    assert report["summary"]["total_questions"] == 0
    assert report["ragas_scores"]["skipped"] == "dataset is empty"
    json.dumps(report, default=str)
    assert json.loads(out.read_text(encoding="utf-8"))["environment"]["model"]


def test_baseline_flag_actually_runs_the_baseline(tmp_path, settings, deps, web_search):
    path = write_dataset(tmp_path, [{"question": "q", "ground_truth": "g"}])
    _seed_web(deps, web_search, "q")
    deps.llm = FakeLLMClient(
        [
            {"sub_queries": ["q"], "retrieval_strategy": "web"},
            GeneratorOutput(answer="answer [Source 1]", confidence=0.9, sources_used=[1]),
            {
                "claims": ["c"],
                "supported_claims": ["c"],
                "unsupported_claims": [],
                "faithfulness_score": 1.0,
            },
            GeneratorOutput(answer="baseline answer [Source 1]", confidence=0.6, sources_used=[1]),
        ]
    )

    report = evaluate_dataset(path, settings=settings, deps=deps, include_baseline=True)

    assert "baseline" in report
    assert report["baseline"]["timings"][0]["answer"] == "baseline answer [Source 1]"
    assert report["baseline"]["summary"]["total_questions"] == 1
    assert "comparison" in report
    assert "caveat" in report["comparison"]
    assert report["comparison"]["sample_size"] == 1
    json.dumps(report, default=str)


def test_report_without_the_flag_has_no_baseline(tmp_path, settings, deps, web_search):
    path = write_dataset(tmp_path, [{"question": "q", "ground_truth": "g"}])
    _seed_web(deps, web_search, "q")
    deps.llm = FakeLLMClient(
        [
            {"sub_queries": ["q"], "retrieval_strategy": "web"},
            GeneratorOutput(answer="answer [Source 1]", confidence=0.9, sources_used=[1]),
            {
                "claims": ["c"],
                "supported_claims": ["c"],
                "unsupported_claims": [],
                "faithfulness_score": 1.0,
            },
        ]
    )

    report = evaluate_dataset(path, settings=settings, deps=deps, include_baseline=False)

    assert "baseline" not in report
    assert "comparison" not in report
    assert report["summary"]["total_questions"] == 1


def test_report_records_per_question_timings(tmp_path, settings, deps, web_search):
    path = write_dataset(tmp_path, [{"question": "q", "ground_truth": "g"}])
    _seed_web(deps, web_search, "q")
    deps.llm = FakeLLMClient(
        [
            {"sub_queries": ["q"], "retrieval_strategy": "web"},
            GeneratorOutput(answer="answer [Source 1]", confidence=0.9, sources_used=[1]),
            {
                "claims": ["c"],
                "supported_claims": ["c"],
                "unsupported_claims": [],
                "faithfulness_score": 1.0,
            },
        ]
    )

    report = evaluate_dataset(path, settings=settings, deps=deps)
    record = report["timings"][0]

    assert record["question"] == "q"
    assert record["is_verified"] is True
    assert record["num_sources"] == 1
    assert record["latency_s"] >= 0.0
    assert record["contexts"] == ["baseline evidence"]


def test_a_failing_question_does_not_abort_the_run(tmp_path, settings, deps, web_search):
    path = write_dataset(
        tmp_path, [{"question": "q", "ground_truth": "g"}, {"question": "q2", "ground_truth": "g"}]
    )
    _seed_web(deps, web_search, "q")
    deps.llm = FakeLLMClient([])  # exhausted immediately -> every node errors

    report = evaluate_dataset(path, settings=settings, deps=deps)

    assert report["summary"]["total_questions"] == 2
    json.dumps(report, default=str)
