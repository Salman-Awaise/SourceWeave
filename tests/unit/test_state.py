"""Domain records: dedup identity, citation entries, public result contract."""

from __future__ import annotations

from research_system.core.state import (
    RetrievedDocument,
    VerificationResult,
    build_public_result,
    canonical_source,
    create_initial_state,
    normalize_text,
)


def test_initial_state_populates_every_field():
    state = create_initial_state("what is rrf?")

    assert state["query"] == "what is rrf?"
    assert state["original_query"] == "what is rrf?"
    assert state["retrieval_strategy"] == "hybrid"
    assert state["confidence"] == 0.0
    assert state["retry_count"] == 0
    assert state["is_verified"] is False
    assert state["verification"] is None
    assert state["error"] is None
    for key in (
        "chat_history",
        "memory_context",
        "sub_queries",
        "retrieved_documents",
        "prior_documents",
        "retrieval_queries_used",
        "sources",
        "unsupported_claims",
        "gaps",
        "warnings",
    ):
        assert state[key] == [], key


def test_memory_context_is_separate_from_the_query():
    state = create_initial_state("q", memory_context=["user likes short answers"])
    assert state["original_query"] == "q"
    assert "short answers" not in state["query"]


def test_normalize_text_collapses_whitespace_and_case():
    assert normalize_text("  Hello   WORLD \n ") == "hello world"


def test_canonical_source_normalizes_urls_and_paths():
    assert canonical_source("https://www.Example.com/page/") == "example.com/page"
    assert canonical_source("http://example.com/page#frag") == "example.com/page"
    assert canonical_source("/long/path/to/report.pdf") == "report.pdf"
    assert canonical_source("") == "unknown"


def test_same_content_and_source_share_a_dedup_key():
    a = RetrievedDocument("Some  text", "https://www.example.com/a", 0.5)
    b = RetrievedDocument("some text", "http://example.com/a/", 0.9)
    assert a.dedup_key == b.dedup_key


def test_shared_prefix_from_different_sources_does_not_collide():
    prefix = "The quick brown fox jumps over the lazy dog. " * 5
    a = RetrievedDocument(prefix + "Ending A.", "a.pdf", 0.5)
    b = RetrievedDocument(prefix + "Ending B.", "a.pdf", 0.5)
    c = RetrievedDocument(prefix + "Ending A.", "b.pdf", 0.5)

    assert a.dedup_key != b.dedup_key
    assert a.dedup_key != c.dedup_key


def test_copy_with_does_not_mutate_the_original():
    original = RetrievedDocument("text", "a.pdf", 0.5, {"k": "v"})
    updated = original.copy_with(score=0.9)
    updated.metadata["k"] = "changed"

    assert original.score == 0.5
    assert original.metadata["k"] == "v"
    assert updated.score == 0.9


def test_source_entry_carries_available_metadata():
    doc = RetrievedDocument(
        "text", "https://example.com", 0.0164, {"title": "T", "source_type": "web"}
    )
    entry = doc.to_source_entry(1)

    assert entry["index"] == 1
    assert entry["source"] == "https://example.com"
    assert entry["score"] == 0.0164
    assert entry["title"] == "T"
    assert entry["source_type"] == "web"


def test_public_result_keeps_required_keys_and_omits_absent_ones():
    state = create_initial_state("q")
    state["response"] = "answer"
    result = build_public_result(state)

    assert set(result) == {
        "response",
        "sources",
        "confidence",
        "is_verified",
        "retry_count",
        "answered",
        "documents_retrieved",
    }
    assert "verification" not in result
    assert "error" not in result


def test_public_result_separates_answered_from_verified():
    """A declined answer can be verified: "no evidence covers this" is truthful.

    Callers must therefore read `answered`, not `is_verified`, to know whether
    the question was actually answered.
    """
    state = create_initial_state("q")
    state["response"] = "The evidence does not cover this."
    state["is_verified"] = True
    state["answered"] = False
    state["retrieved_documents"] = [RetrievedDocument("irrelevant", "a.pdf", 0.5)]

    result = build_public_result(state)

    assert result["is_verified"] is True
    assert result["answered"] is False
    assert result["documents_retrieved"] == 1


def test_public_result_includes_verification_and_dedupes_warnings():
    state = create_initial_state("q")
    state["response"] = "answer"
    state["verification"] = VerificationResult(0.9, ["a"] * 10, ["a"] * 9, ["a"], "why")
    state["warnings"] = ["same", "same", "other"]
    state["error"] = "boom"

    result = build_public_result(state)

    assert result["verification"] == {
        "faithfulness_score": 0.9,
        "total_claims": 10,
        "supported_claims": 9,
        "unsupported_claims": 1,
    }
    assert result["warnings"] == ["same", "other"]
    assert result["error"] == "boom"
