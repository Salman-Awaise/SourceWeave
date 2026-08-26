"""Vector store: collection lifecycle, filters, deterministic IDs."""

from __future__ import annotations

import pytest

from research_system.adapters.qdrant_store import (
    InMemoryVectorStore,
    VectorPoint,
    _build_filter,
    _missing_index_fields,
    deterministic_point_id,
)
from research_system.errors import VectorStoreError


def point(pid: str, vector: list[float], content: str = "text", **metadata) -> VectorPoint:
    return VectorPoint(id=pid, vector=vector, content=content, source="a.pdf", metadata=metadata)


# --- deterministic IDs -----------------------------------------------------
def test_same_inputs_give_the_same_id():
    assert deterministic_point_id("a.pdf", 0, "text") == deterministic_point_id("a.pdf", 0, "text")


def test_whitespace_differences_do_not_change_the_id():
    assert deterministic_point_id("a.pdf", 0, "the  text") == deterministic_point_id(
        "a.pdf", 0, "The text"
    )


def test_position_source_and_content_all_affect_the_id():
    base = deterministic_point_id("a.pdf", 0, "text")
    assert deterministic_point_id("a.pdf", 1, "text") != base
    assert deterministic_point_id("b.pdf", 0, "text") != base
    assert deterministic_point_id("a.pdf", 0, "other") != base


def test_ids_are_valid_uuids():
    import uuid

    uuid.UUID(deterministic_point_id("a.pdf", 0, "text"))  # raises if malformed


# --- collection lifecycle --------------------------------------------------
def test_ensure_collection_creates_once():
    store = InMemoryVectorStore(dimension=8)

    assert store.ensure_collection("c", 8) is True
    assert store.ensure_collection("c", 8) is False


def test_dimension_mismatch_is_detected():
    store = InMemoryVectorStore(dimension=8)
    store.ensure_collection("c", 8)

    with pytest.raises(VectorStoreError, match="vector size"):
        store.ensure_collection("c", 1536)


def test_search_on_a_missing_collection_raises_actionably():
    store = InMemoryVectorStore(dimension=8)

    with pytest.raises(VectorStoreError, match="does not exist"):
        store.search("missing", [0.0] * 8)


# --- payload and search ----------------------------------------------------
def test_payload_reserves_content_and_source():
    payload = point("p1", [1.0], content="body", page_label="3").payload()

    assert payload["content"] == "body"
    assert payload["source"] == "a.pdf"
    assert payload["page_label"] == "3"


def test_metadata_cannot_shadow_content():
    payload = VectorPoint(
        id="p", vector=[1.0], content="real", source="a.pdf", metadata={"content": "fake"}
    ).payload()

    assert payload["content"] == "real"


def test_search_ranks_by_similarity_and_caps_at_top_k():
    store = InMemoryVectorStore(dimension=2)
    store.ensure_collection("c", 2)
    store.upsert(
        "c",
        [
            point("near", [1.0, 0.0], content="near"),
            point("far", [0.0, 1.0], content="far"),
        ],
    )

    results = store.search("c", [1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].content == "near"


def test_score_threshold_excludes_weak_hits():
    store = InMemoryVectorStore(dimension=2)
    store.ensure_collection("c", 2)
    store.upsert("c", [point("orthogonal", [0.0, 1.0])])

    assert store.search("c", [1.0, 0.0], score_threshold=0.3) == []


def test_metadata_filter_restricts_results():
    store = InMemoryVectorStore(dimension=2)
    store.ensure_collection("c", 2)
    store.upsert(
        "c",
        [
            point("a", [1.0, 0.0], content="keep", kind="report"),
            point("b", [1.0, 0.0], content="drop", kind="memo"),
        ],
    )

    results = store.search("c", [1.0, 0.0], metadata_filter={"kind": "report"})

    assert [r.content for r in results] == ["keep"]


def test_search_results_carry_native_score_and_type():
    store = InMemoryVectorStore(dimension=2)
    store.ensure_collection("c", 2)
    store.upsert("c", [point("a", [1.0, 0.0])])

    result = store.search("c", [1.0, 0.0])[0]

    assert result.metadata["source_type"] == "document"
    assert result.metadata["native_score"] == pytest.approx(result.score)


# --- idempotency -----------------------------------------------------------
def test_reupserting_the_same_ids_does_not_grow_the_collection():
    store = InMemoryVectorStore(dimension=2)
    store.ensure_collection("c", 2)
    points = [
        VectorPoint(
            id=deterministic_point_id("a.pdf", i, f"chunk {i}"),
            vector=[1.0, 0.0],
            content=f"chunk {i}",
            source="a.pdf",
            metadata={},
        )
        for i in range(5)
    ]

    store.upsert("c", points)
    assert store.count("c") == 5

    store.upsert("c", points)
    assert store.count("c") == 5


# --- filter translation ----------------------------------------------------
def test_empty_filter_is_none():
    assert _build_filter(None) is None
    assert _build_filter({}) is None


def test_filter_builds_one_condition_per_key():
    pytest.importorskip("qdrant_client")

    built = _build_filter({"kind": "report", "year": 2026})

    assert built is not None
    assert len(built.must) == 2
    assert {c.key for c in built.must} == {"kind", "year"}


# --- Qdrant Cloud requires an index before filtering -----------------------
CLOUD_400 = (
    "Unexpected Response: 400 (Bad Request) Raw response content: "
    'b\'{"status":{"error":"Bad request: Index required but not found for '
    '\\\\"kind\\\\" of one of the following types: [keyword]."}}\''
)


def test_missing_index_field_is_detected_from_the_cloud_error():
    assert _missing_index_fields(CLOUD_400, {"kind": "memo"}) == ["kind"]


def test_only_fields_we_filtered_on_are_reported():
    """An unrelated field named in the error must not trigger index creation."""
    assert _missing_index_fields(CLOUD_400, {"year": 2026}) == []


def test_unrelated_400s_are_not_treated_as_a_missing_index():
    assert _missing_index_fields("400 Bad Request: malformed vector", {"kind": "memo"}) == []


def test_no_filter_means_nothing_to_index():
    assert _missing_index_fields(CLOUD_400, None) == []
    assert _missing_index_fields(CLOUD_400, {}) == []


def test_multiple_missing_fields_are_all_reported():
    msg = (
        "Index required but not found for kind and also year of one of the "
        "following types: [keyword]"
    )
    assert sorted(_missing_index_fields(msg, {"kind": "a", "year": 1})) == ["kind", "year"]
