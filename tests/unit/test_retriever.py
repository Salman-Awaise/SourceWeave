"""Retriever: backend selection, degradation, and retry evidence merging."""

from __future__ import annotations

import pytest

from research_system.adapters.qdrant_store import VectorPoint
from research_system.agents.retriever import retriever_node
from research_system.core.state import RetrievedDocument, create_initial_state
from research_system.errors import RetrievalError, VectorStoreError, WebSearchError


def web_doc(content: str, url: str = "https://example.com/a") -> RetrievedDocument:
    return RetrievedDocument(
        content=content, source=url, score=0.9, metadata={"source_type": "web"}
    )


def seed_vector_store(store, embeddings, texts):
    points = [
        VectorPoint(
            id=f"p{i}",
            vector=embeddings.embed(text),
            content=text,
            source=f"doc{i}.pdf",
            metadata={"source_type": "document"},
        )
        for i, text in enumerate(texts)
    ]
    store.upsert("test_collection", points)


def state_for(strategy: str, queries: list[str]):
    state = create_initial_state("original question")
    state["retrieval_strategy"] = strategy
    state["sub_queries"] = queries
    return state


def test_hybrid_calls_both_backends(deps, embeddings, vector_store, web_search):
    seed_vector_store(vector_store, embeddings, ["indexed evidence"])
    web_search.results = {"q1": [web_doc("web evidence")]}

    update = retriever_node(state_for("hybrid", ["q1"]), deps)

    contents = {doc.content for doc in update["retrieved_documents"]}
    assert "web evidence" in contents
    assert web_search.queries == ["q1"]


def test_vector_strategy_does_not_call_web(deps, embeddings, vector_store, web_search):
    seed_vector_store(vector_store, embeddings, ["indexed evidence"])

    update = retriever_node(state_for("vector", ["q1"]), deps)

    assert web_search.queries == []
    assert update["retrieved_documents"]


def test_web_strategy_does_not_touch_the_vector_store(deps, web_search):
    web_search.results = {"q1": [web_doc("web evidence")]}

    update = retriever_node(state_for("web", ["q1"]), deps)

    assert [d.content for d in update["retrieved_documents"]] == ["web evidence"]


def test_explicit_vector_without_a_backend_raises(deps):
    deps.vector_store = None

    with pytest.raises(RetrievalError, match="Strategy 'vector'"):
        retriever_node(state_for("vector", ["q1"]), deps)


def test_explicit_web_without_a_key_raises(deps):
    deps.web_search = None

    with pytest.raises(RetrievalError, match="Strategy 'web'"):
        retriever_node(state_for("web", ["q1"]), deps)


def test_hybrid_degrades_with_a_warning_when_web_is_missing(deps, embeddings, vector_store):
    deps.web_search = None
    seed_vector_store(vector_store, embeddings, ["indexed evidence"])

    update = retriever_node(state_for("hybrid", ["q1"]), deps)

    assert update["retrieved_documents"]
    assert any("web search unavailable" in w for w in update["warnings"])


def test_hybrid_degrades_when_vector_is_missing(deps, web_search):
    deps.vector_store = None
    web_search.results = {"q1": [web_doc("web evidence")]}

    update = retriever_node(state_for("hybrid", ["q1"]), deps)

    assert update["retrieved_documents"]
    assert any("document search unavailable" in w for w in update["warnings"])


def test_hybrid_with_no_backends_raises(deps):
    deps.vector_store = None
    deps.web_search = None

    with pytest.raises(RetrievalError, match="No retrieval backend"):
        retriever_node(state_for("hybrid", ["q1"]), deps)


def test_a_failing_backend_warns_instead_of_killing_a_hybrid_run(deps, web_search, monkeypatch):
    web_search.results = {"q1": [web_doc("web evidence")]}

    def boom(*args, **kwargs):
        raise VectorStoreError("collection 'test_collection' does not exist")

    monkeypatch.setattr(deps.vector_store, "search", boom)

    update = retriever_node(state_for("hybrid", ["q1"]), deps)

    assert [d.content for d in update["retrieved_documents"]] == ["web evidence"]
    assert any("Document search failed" in w for w in update["warnings"])


def test_web_failure_warns_and_continues(deps, embeddings, vector_store, web_search, monkeypatch):
    seed_vector_store(vector_store, embeddings, ["indexed evidence"])

    def boom(*args, **kwargs):
        raise WebSearchError("tavily 503")

    monkeypatch.setattr(web_search, "search", boom)

    update = retriever_node(state_for("hybrid", ["q1"]), deps)

    assert update["retrieved_documents"]
    assert any("Web search failed" in w for w in update["warnings"])


def test_empty_retrieval_warns(deps, web_search):
    deps.vector_store = None

    update = retriever_node(state_for("hybrid", ["q1"]), deps)

    assert update["retrieved_documents"] == []
    assert any("no documents" in w for w in update["warnings"])


# --- broadening when the chosen strategy finds nothing ---------------------
def test_empty_vector_search_broadens_to_web(deps, web_search):
    """An indexed corpus that doesn't cover the question must not dead-end.

    Vector search is configured and working, it just has no relevant content.
    Web search is available and unused, so it should be tried rather than
    answering "insufficient evidence".
    """
    web_search.results = {"q1": [web_doc("web evidence")]}
    # vector_store exists but the collection holds nothing relevant

    update = retriever_node(state_for("vector", ["q1"]), deps)

    assert [d.content for d in update["retrieved_documents"]] == ["web evidence"]
    assert any("broadening to web search" in w for w in update["warnings"])


def test_empty_web_search_broadens_to_documents(deps, embeddings, vector_store):
    seed_vector_store(vector_store, embeddings, ["indexed evidence"])
    # web_search returns nothing for this query

    update = retriever_node(state_for("web", ["q1"]), deps)

    assert update["retrieved_documents"]
    assert any("broadening to document search" in w for w in update["warnings"])


def test_broadening_does_not_happen_when_the_strategy_found_something(
    deps, embeddings, vector_store, web_search
):
    seed_vector_store(vector_store, embeddings, ["indexed evidence"])

    update = retriever_node(state_for("vector", ["indexed evidence"]), deps)

    assert update["retrieved_documents"]
    assert web_search.queries == []  # web never touched
    assert not any("broadening" in w for w in update["warnings"])


def test_no_broadening_when_both_backends_were_already_used(deps, web_search):
    """Hybrid already tried everything, so there is nothing to broaden to."""
    deps.vector_store = None  # hybrid degrades to web-only, which finds nothing

    update = retriever_node(state_for("hybrid", ["q1"]), deps)

    assert update["retrieved_documents"] == []
    assert not any("broadening" in w for w in update["warnings"])


def test_broadening_is_skipped_when_no_spare_backend_exists(deps):
    deps.web_search = None
    # only vector configured, and it finds nothing

    update = retriever_node(state_for("vector", ["q1"]), deps)

    assert update["retrieved_documents"] == []
    assert not any("broadening" in w for w in update["warnings"])


def test_retry_merges_prior_evidence_with_new(deps, web_search):
    deps.vector_store = None
    web_search.results = {"refined": [web_doc("new evidence", "https://example.com/new")]}

    state = state_for("hybrid", ["refined"])
    state["prior_documents"] = [web_doc("old evidence", "https://example.com/old")]

    update = retriever_node(state, deps)

    contents = {doc.content for doc in update["retrieved_documents"]}
    assert contents == {"new evidence", "old evidence"}


def test_prior_evidence_is_deduplicated_against_new(deps, web_search):
    deps.vector_store = None
    same = web_doc("same evidence", "https://example.com/same")
    web_search.results = {"refined": [same]}

    state = state_for("hybrid", ["refined"])
    state["prior_documents"] = [same]

    update = retriever_node(state, deps)

    assert len(update["retrieved_documents"]) == 1
    assert update["retrieved_documents"][0].metadata["rrf_appearances"] == 2


def test_results_are_capped_at_max_retrieval_docs(deps, web_search):
    deps.vector_store = None
    deps.settings.max_retrieval_docs = 3
    deps.settings.web_results_per_query = 10
    web_search.results = {
        "q1": [web_doc(f"evidence {i}", f"https://example.com/{i}") for i in range(10)]
    }

    update = retriever_node(state_for("hybrid", ["q1"]), deps)

    assert len(update["retrieved_documents"]) == 3


def test_queries_used_accumulates_without_duplicates(deps, web_search):
    deps.vector_store = None
    state = state_for("hybrid", ["q1", "q2"])
    state["retrieval_queries_used"] = ["q1"]

    update = retriever_node(state, deps)

    assert update["retrieval_queries_used"] == ["q1", "q2"]


def test_missing_sub_queries_falls_back_to_the_original_question(deps, web_search):
    deps.vector_store = None
    web_search.results = {"original question": [web_doc("evidence")]}

    update = retriever_node(state_for("hybrid", []), deps)

    assert web_search.queries == ["original question"]
    assert update["retrieved_documents"]


def test_sub_queries_are_capped(deps, web_search):
    deps.vector_store = None
    update = retriever_node(state_for("hybrid", [f"q{i}" for i in range(10)]), deps)

    assert len(web_search.queries) == deps.settings.max_sub_queries
    assert update["retrieval_queries_used"] == web_search.queries
