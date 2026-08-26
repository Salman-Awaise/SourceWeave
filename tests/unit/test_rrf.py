"""Reciprocal Rank Fusion and deduplication."""

from __future__ import annotations

import pytest

from research_system.agents.retriever import deduplicate, reciprocal_rank_fusion
from research_system.core.state import RetrievedDocument


def doc(content: str, source: str = "a.pdf", score: float = 0.5) -> RetrievedDocument:
    return RetrievedDocument(content=content, source=source, score=score)


def test_empty_input_returns_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_single_list_preserves_order():
    docs = [doc("one"), doc("two"), doc("three")]
    fused = reciprocal_rank_fusion([docs], k=60)

    assert [d.content for d in fused] == ["one", "two", "three"]


def test_scores_match_the_exact_rrf_formula():
    docs = [doc("first"), doc("second")]
    fused = reciprocal_rank_fusion([docs], k=60)

    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[1].score == pytest.approx(1 / 62)


def test_duplicate_across_lists_accumulates_score():
    shared = doc("shared")
    list_a = [shared, doc("only-a")]
    list_b = [doc("only-b"), shared]

    fused = reciprocal_rank_fusion([list_a, list_b], k=60)

    top = fused[0]
    assert top.content == "shared"
    # rank 0 in list A, rank 1 in list B
    assert top.score == pytest.approx(1 / 61 + 1 / 62)
    assert top.metadata["rrf_appearances"] == 2


def test_agreement_beats_a_single_high_rank():
    """A doc ranked 2nd in both lists outranks one ranked 1st in only one."""
    both = doc("agreed")
    list_a = [doc("solo-a"), both]
    list_b = [doc("solo-b"), both]

    fused = reciprocal_rank_fusion([list_a, list_b], k=60)

    assert fused[0].content == "agreed"


def test_similar_prefixes_from_different_sources_do_not_fuse():
    prefix = "Shared opening sentence. " * 8
    a = doc(prefix + "Tail A", source="a.pdf")
    b = doc(prefix + "Tail B", source="b.pdf")

    fused = reciprocal_rank_fusion([[a], [b]], k=60)

    assert len(fused) == 2


def test_native_score_is_preserved_and_rrf_score_replaces_it():
    original = doc("text", score=0.87)
    fused = reciprocal_rank_fusion([[original]], k=60)

    assert fused[0].metadata["native_score"] == 0.87
    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[0].metadata["rrf_score"] == pytest.approx(1 / 61)


def test_inputs_are_not_mutated():
    original = doc("text", score=0.87)
    reciprocal_rank_fusion([[original]], k=60)

    assert original.score == 0.87
    assert "rrf_score" not in original.metadata


def test_top_n_caps_the_result():
    docs = [doc(f"d{i}") for i in range(10)]
    assert len(reciprocal_rank_fusion([docs], top_n=3)) == 3


def test_k_changes_the_score_scale():
    docs = [doc("one")]
    assert reciprocal_rank_fusion([docs], k=1)[0].score == pytest.approx(0.5)
    assert reciprocal_rank_fusion([docs], k=60)[0].score == pytest.approx(1 / 61)


def test_deduplicate_keeps_first_occurrence():
    a = doc("same", source="x.pdf", score=0.9)
    b = doc("same", source="x.pdf", score=0.1)
    out = deduplicate([a, b, doc("other")])

    assert len(out) == 2
    assert out[0].score == 0.9
