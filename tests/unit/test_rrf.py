"""Reciprocal Rank Fusion and deduplication."""

from __future__ import annotations

from collections import Counter

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


# --- weighted fusion: primary sources outrank summaries --------------------
def typed(content: str, source_type: str, source: str = "s") -> RetrievedDocument:
    return RetrievedDocument(
        content=content, source=source, score=0.5, metadata={"source_type": source_type}
    )


def test_no_weights_behaves_exactly_like_plain_rrf():
    docs = [typed("a", "document"), typed("b", "web", source="t")]
    fused = reciprocal_rank_fusion([docs], k=60)

    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[1].score == pytest.approx(1 / 62)


def test_a_weighted_document_outranks_a_higher_placed_web_result():
    """The exact case from the live run: both appear once, web listed first."""
    web_list = [typed("web summary", "web", source="blog")]
    doc_list = [typed("primary paper", "document", source="paper.pdf")]

    fused = reciprocal_rank_fusion(
        [web_list, doc_list], k=60, weights={"document": 1.5, "web": 1.0}
    )

    assert fused[0].content == "primary paper"
    assert fused[0].score == pytest.approx(1.5 / 61)
    assert fused[1].score == pytest.approx(1.0 / 61)


def test_weighting_is_decisive_not_marginal():
    """Documents the real behaviour: a weight beats a large rank advantage.

    The 1/(k+rank) curve is nearly flat, so at k=60 a 1.5x weight still wins
    from 20 places below. This is deliberate but worth pinning down, because it
    means any weight above 1.0 is a preference rather than a nudge.
    """
    web_list = [typed("web first", "web", source="w")]
    doc_list = [typed(f"filler {i}", "web", source=f"f{i}") for i in range(20)]
    doc_list.append(typed("late document", "document", source="paper.pdf"))

    fused = reciprocal_rank_fusion(
        [web_list, doc_list], k=60, weights={"document": 1.5, "web": 1.0}
    )

    assert fused[0].content == "late document"
    assert fused[0].score == pytest.approx(1.5 / 81)


def test_a_large_enough_rank_gap_still_beats_the_weight():
    """The weight is finite: far enough down, rank wins again."""
    web_list = [typed("web first", "web", source="w")]
    doc_list = [typed(f"filler {i}", "web", source=f"f{i}") for i in range(40)]
    doc_list.append(typed("very late document", "document", source="paper.pdf"))

    fused = reciprocal_rank_fusion(
        [web_list, doc_list], k=60, weights={"document": 1.2, "web": 1.0}
    )

    assert fused[0].content == "web first"


def test_unlisted_source_types_default_to_weight_one():
    docs = [RetrievedDocument(content="x", source="s", score=0.5)]  # no source_type
    fused = reciprocal_rank_fusion([docs], k=60, weights={"document": 1.5})

    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[0].metadata["rrf_weight"] == 1.0


def test_applied_weight_is_recorded_for_traceability():
    fused = reciprocal_rank_fusion([[typed("a", "document")]], k=60, weights={"document": 1.5})

    assert fused[0].metadata["rrf_weight"] == 1.5


def test_zero_weight_suppresses_a_source_type():
    fused = reciprocal_rank_fusion(
        [[typed("web", "web", source="w"), typed("doc", "document", source="d")]],
        k=60,
        weights={"web": 0.0, "document": 1.0},
    )

    assert fused[0].content == "doc"
    assert fused[-1].score == 0.0


# --- the cap must not erase an entire source type --------------------------
def test_weighting_alone_can_erase_a_source_type():
    """The problem being solved: without a floor, web disappears completely."""
    docs = [typed(f"doc {i}", "document", source=f"d{i}") for i in range(10)]
    webs = [typed(f"web {i}", "web", source=f"w{i}") for i in range(10)]

    fused = reciprocal_rank_fusion(
        [docs, webs], k=60, top_n=10, weights={"document": 1.2, "web": 1.0}, min_per_type=0
    )

    assert {d.source_type for d in fused} == {"document"}


def test_a_floor_keeps_both_source_types_in_the_cap():
    docs = [typed(f"doc {i}", "document", source=f"d{i}") for i in range(10)]
    webs = [typed(f"web {i}", "web", source=f"w{i}") for i in range(10)]

    fused = reciprocal_rank_fusion(
        [docs, webs], k=60, top_n=10, weights={"document": 1.2, "web": 1.0}, min_per_type=2
    )

    counts = Counter(d.source_type for d in fused)
    assert len(fused) == 10
    assert counts["web"] >= 2
    assert counts["document"] == 8


def test_the_promoted_web_results_are_the_highest_scoring_ones():
    docs = [typed(f"doc {i}", "document", source=f"d{i}") for i in range(10)]
    webs = [typed(f"web {i}", "web", source=f"w{i}") for i in range(10)]

    fused = reciprocal_rank_fusion(
        [docs, webs], k=60, top_n=10, weights={"document": 1.2, "web": 1.0}, min_per_type=2
    )

    kept = [d.content for d in fused if d.source_type == "web"]
    assert kept == ["web 0", "web 1"]  # top of their own ranking, not arbitrary


def test_floor_is_a_noop_when_only_one_source_type_exists():
    docs = [typed(f"doc {i}", "document", source=f"d{i}") for i in range(10)]

    fused = reciprocal_rank_fusion([docs], k=60, top_n=5, weights={"document": 1.2}, min_per_type=2)

    assert len(fused) == 5
    assert all(d.source_type == "document" for d in fused)


def test_floor_does_not_pad_beyond_what_a_type_actually_returned():
    docs = [typed(f"doc {i}", "document", source=f"d{i}") for i in range(10)]
    webs = [typed("only web", "web", source="w0")]

    fused = reciprocal_rank_fusion(
        [docs, webs], k=60, top_n=10, weights={"document": 1.2, "web": 1.0}, min_per_type=2
    )

    counts = Counter(d.source_type for d in fused)
    assert counts["web"] == 1  # asked for 2, only 1 exists
    assert len(fused) == 10


def test_results_stay_score_ordered_after_promotion():
    docs = [typed(f"doc {i}", "document", source=f"d{i}") for i in range(10)]
    webs = [typed(f"web {i}", "web", source=f"w{i}") for i in range(10)]

    fused = reciprocal_rank_fusion(
        [docs, webs], k=60, top_n=10, weights={"document": 1.2, "web": 1.0}, min_per_type=2
    )

    scores = [d.score for d in fused]
    assert scores == sorted(scores, reverse=True)
