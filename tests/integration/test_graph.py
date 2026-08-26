"""Graph integration: full paths through plan -> retrieve -> generate -> verify.

Offline: every adapter is a fake, so these run in CI with no credentials.
"""

from __future__ import annotations

import pytest

from research_system.adapters.llm import FakeLLMClient
from research_system.core.pipeline import ResearchPipeline
from research_system.core.state import RetrievedDocument
from research_system.errors import StructuredOutputError
from research_system.schemas import GeneratorOutput, PlannerOutput, RefinedQueries, VerifierOutput


def web(content: str, url: str = "https://example.com/a") -> RetrievedDocument:
    return RetrievedDocument(
        content=content, source=url, score=0.9, metadata={"source_type": "web", "title": "T"}
    )


@pytest.fixture
def web_only(deps, web_search):
    """Web-only pipeline so retrieval is fully scripted."""
    deps.vector_store = None
    deps.embeddings = None
    return deps


def plan(strategy="web", queries=("q1",)):
    return PlannerOutput(sub_queries=list(queries), retrieval_strategy=strategy, reasoning="r")


def verdict(supported, unsupported):
    claims = list(supported) + list(unsupported)
    return VerifierOutput(
        claims=claims,
        supported_claims=list(supported),
        unsupported_claims=list(unsupported),
        faithfulness_score=len(supported) / len(claims) if claims else 0.0,
        reasoning="r",
    )


# --- happy path ------------------------------------------------------------
def test_happy_path_visits_every_stage_once(web_only, web_search):
    web_search.results = {"q1": [web("solid evidence")]}
    web_only.llm = FakeLLMClient(
        [
            plan(),
            GeneratorOutput(answer="The answer [Source 1]", confidence=0.9, sources_used=[1]),
            verdict(["c1", "c2", "c3"], []),
        ]
    )

    result = ResearchPipeline(web_only).run("what is rrf?")

    assert result["is_verified"] is True
    assert result["retry_count"] == 0
    assert result["response"] == "The answer [Source 1]"
    assert result["verification"]["faithfulness_score"] == 1.0
    assert len(web_only.llm.calls) == 3


def test_public_contract_is_preserved(web_only, web_search):
    web_search.results = {"q1": [web("evidence")]}
    web_only.llm = FakeLLMClient(
        [
            plan(),
            GeneratorOutput(answer="A [Source 1]", confidence=0.8, sources_used=[1]),
            verdict(["c"], []),
        ]
    )

    result = ResearchPipeline(web_only).run("q")

    for key in ("response", "sources", "confidence", "is_verified", "retry_count"):
        assert key in result
    assert isinstance(result["sources"], list)
    entry = result["sources"][0]
    assert entry["index"] == 1
    assert entry["source"] == "https://example.com/a"
    assert "score" in entry
    assert result["verification"]["total_claims"] == 1


def test_every_citation_index_exists(web_only, web_search):
    web_search.results = {"q1": [web("e1", "https://a.com"), web("e2", "https://b.com")]}
    web_only.llm = FakeLLMClient(
        [
            plan(),
            GeneratorOutput(answer="A [Source 1][Source 2]", confidence=0.8, sources_used=[1, 2]),
            verdict(["c"], []),
        ]
    )

    result = ResearchPipeline(web_only).run("q")

    indices = [entry["index"] for entry in result["sources"]]
    assert indices == [1, 2]


# --- one retry -------------------------------------------------------------
def test_one_failed_verification_retries_retrieval_with_refined_queries(web_only, web_search):
    web_search.results = {
        "q1": [web("weak evidence", "https://weak.com")],
        "refined query": [web("strong evidence", "https://strong.com")],
    }
    web_only.llm = FakeLLMClient(
        [
            plan(),
            GeneratorOutput(answer="First attempt [Source 1]", confidence=0.5, sources_used=[1]),
            verdict(["ok"], ["unsupported claim"]),
            RefinedQueries(queries=["refined query"]),
            GeneratorOutput(answer="Second attempt [Source 1]", confidence=0.9, sources_used=[1]),
            verdict(["ok", "now supported"], []),
        ]
    )

    result = ResearchPipeline(web_only).run("q")

    assert result["retry_count"] == 1
    assert result["is_verified"] is True
    assert result["response"] == "Second attempt [Source 1]"
    # The retry issued the refined query, not the original again.
    assert web_search.queries == ["q1", "refined query"]


def test_retry_keeps_prior_evidence(web_only, web_search):
    web_search.results = {
        "q1": [web("original evidence", "https://old.com")],
        "refined": [web("new evidence", "https://new.com")],
    }
    web_only.llm = FakeLLMClient(
        [
            plan(),
            GeneratorOutput(answer="First [Source 1]", confidence=0.5, sources_used=[1]),
            verdict([], ["bad claim"]),
            RefinedQueries(queries=["refined"]),
            GeneratorOutput(answer="Second [Source 1]", confidence=0.9, sources_used=[1, 2]),
            verdict(["good"], []),
        ]
    )

    result = ResearchPipeline(web_only).run("q")

    sources = {entry["source"] for entry in result["sources"]}
    assert sources == {"https://old.com", "https://new.com"}


# --- retry exhaustion ------------------------------------------------------
def test_two_failures_end_unverified_at_the_cap(web_only, web_search):
    """Two failed verifications exhaust the default cap and end the run.

    The router stops when retry_count reaches max_verification_retries (2), so
    the sequence is: verify fails (count 1) -> retry -> verify fails (count 2)
    -> stop. The second refinement is never requested.
    """
    web_search.results = {"q1": [web("evidence")], "r1": [web("evidence")]}
    web_only.llm = FakeLLMClient(
        [
            plan(),
            GeneratorOutput(answer="Attempt 1 [Source 1]", confidence=0.5, sources_used=[1]),
            verdict([], ["bad"]),
            RefinedQueries(queries=["r1"]),
            GeneratorOutput(answer="Attempt 2 [Source 1]", confidence=0.5, sources_used=[1]),
            verdict([], ["still bad"]),
        ]
    )

    result = ResearchPipeline(web_only).run("q")

    assert result["is_verified"] is False
    assert result["retry_count"] == 2
    assert result["response"] == "Attempt 2 [Source 1]"
    assert any("remains unverified" in w for w in result["warnings"])
    assert web_search.queries == ["q1", "r1"]
    assert not web_only.llm.responses  # every scripted response was consumed


def test_a_custom_retry_cap_is_respected(web_only, web_search):
    web_only.settings.max_verification_retries = 0
    web_search.results = {"q1": [web("evidence")]}
    web_only.llm = FakeLLMClient(
        [
            plan(),
            GeneratorOutput(answer="Only attempt [Source 1]", confidence=0.5, sources_used=[1]),
            verdict([], ["bad"]),
        ]
    )

    result = ResearchPipeline(web_only).run("q")

    assert result["is_verified"] is False
    assert result["retry_count"] == 1
    assert web_search.queries == ["q1"]


# --- failure handling ------------------------------------------------------
def test_no_evidence_yields_an_honest_answer_without_looping(web_only, web_search):
    """An explicit "no evidence" answer is faithful, so it scores 1.0.

    Per the score contract, zero claims means 1.0 only for an explicit
    insufficient-evidence response. The answer cites nothing and the warning
    makes the empty retrieval visible, so a verified status here does not imply
    the question was answered.
    """
    web_search.results = {}
    web_only.llm = FakeLLMClient([plan(), verdict([], [])])

    result = ResearchPipeline(web_only).run("q")

    assert "could not find any evidence" in result["response"]
    assert result["sources"] == []
    assert result["confidence"] == 0.0
    assert result["retry_count"] == 0
    assert result["verification"]["total_claims"] == 0
    assert any("no documents" in w for w in result["warnings"])


def test_planner_failure_still_produces_an_answer(web_only, web_search):
    web_search.results = {"q": [web("evidence")]}
    web_only.llm = FakeLLMClient(
        [
            StructuredOutputError("planner broke"),
            GeneratorOutput(answer="Answer [Source 1]", confidence=0.8, sources_used=[1]),
            verdict(["c"], []),
        ]
    )

    result = ResearchPipeline(web_only).run("q")

    assert result["response"] == "Answer [Source 1]"
    assert any("Planner output was unusable" in w for w in result["warnings"])


def test_an_unexpected_node_crash_becomes_an_error_not_a_hang(web_only, web_search, monkeypatch):
    web_search.results = {"q1": [web("evidence")]}
    web_only.llm = FakeLLMClient([plan()])

    def boom(*args, **kwargs):
        raise RuntimeError("unexpected bug")

    monkeypatch.setattr("research_system.core.graph.generator_node", boom)

    result = ResearchPipeline(web_only).run("q")

    assert result["error"]
    assert "unexpected bug" in result["error"]
    assert result["is_verified"] is False


def test_retrieval_backend_error_is_reported(deps):
    deps.vector_store = None
    deps.web_search = None
    deps.llm = FakeLLMClient([plan(strategy="hybrid")])

    result = ResearchPipeline(deps).run("q")

    assert result["error"]
    assert "No retrieval backend" in result["error"]


def test_empty_query_is_rejected_without_calling_a_model(web_only):
    web_only.llm = FakeLLMClient([])

    result = ResearchPipeline(web_only).run("   ")

    assert result["error"] == "Query was empty."
    assert web_only.llm.calls == []


def test_overlong_query_is_truncated(web_only, web_search):
    web_only.settings.max_query_chars = 50
    web_search.results = {"q1": [web("evidence")]}
    web_only.llm = FakeLLMClient(
        [
            plan(),
            GeneratorOutput(answer="A [Source 1]", confidence=0.8, sources_used=[1]),
            verdict(["c"], []),
        ]
    )

    result = ResearchPipeline(web_only).run("x" * 500)

    assert any("truncated" in w for w in result["warnings"])


# --- memory ---------------------------------------------------------------
def test_memory_is_recalled_and_stored_per_user(memory_deps, web_search):
    memory_deps.vector_store = None
    memory_deps.embeddings = None
    web_search.results = {"q1": [web("evidence")]}
    memory_deps.memory.add([{"role": "user", "content": "alice prefers concise answers"}], "alice")
    memory_deps.llm = FakeLLMClient(
        [
            plan(),
            GeneratorOutput(answer="Concise [Source 1]", confidence=0.9, sources_used=[1]),
            verdict(["c"], []),
        ]
    )

    pipeline = ResearchPipeline(memory_deps)
    result = pipeline.run("prefers what?", user_id="alice", use_memory=True)

    planner_prompt = memory_deps.llm.calls[0]["user"]
    assert "<memory>" in planner_prompt
    assert "concise answers" in planner_prompt
    # The answer was written back for this user only.
    assert any("Concise" in entry for entry in memory_deps.memory.get_all("alice"))
    assert memory_deps.memory.get_all("bob") == []
    assert result["is_verified"] is True


def test_memory_is_not_used_when_disabled(memory_deps, web_search):
    memory_deps.vector_store = None
    memory_deps.embeddings = None
    web_search.results = {"q1": [web("evidence")]}
    memory_deps.memory.add([{"role": "user", "content": "secret note"}], "alice")
    memory_deps.llm = FakeLLMClient(
        [
            plan(),
            GeneratorOutput(answer="A [Source 1]", confidence=0.9, sources_used=[1]),
            verdict(["c"], []),
        ]
    )

    ResearchPipeline(memory_deps).run("q", user_id="alice", use_memory=False)

    assert "secret note" not in memory_deps.llm.calls[0]["user"]


def test_a_broken_memory_backend_does_not_block_the_query(memory_deps, web_search, monkeypatch):
    memory_deps.vector_store = None
    memory_deps.embeddings = None
    web_search.results = {"q1": [web("evidence")]}
    monkeypatch.setattr(
        memory_deps.memory,
        "search",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("memory down")),
    )
    memory_deps.llm = FakeLLMClient(
        [
            plan(),
            GeneratorOutput(answer="A [Source 1]", confidence=0.9, sources_used=[1]),
            verdict(["c"], []),
        ]
    )

    result = ResearchPipeline(memory_deps).run("q", user_id="alice", use_memory=True)

    assert result["response"] == "A [Source 1]"
    assert any("Memory lookup failed" in w for w in result["warnings"])
