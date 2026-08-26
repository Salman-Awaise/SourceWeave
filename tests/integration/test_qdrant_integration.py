"""Opt-in tests against a real Qdrant.

Run with a live Qdrant from docker-compose:

    docker compose up -d qdrant
    RUN_INTEGRATION=1 pytest tests/integration/test_qdrant_integration.py

Embeddings stay fake, so these need no paid credentials -- only a reachable
Qdrant. Each test uses a uniquely-named collection and deletes it afterwards.
"""

from __future__ import annotations

import contextlib
import os
import uuid

import pytest

from research_system.adapters.embeddings import FakeEmbeddings
from research_system.adapters.qdrant_store import (
    QdrantVectorStore,
    VectorPoint,
    deterministic_point_id,
)
from research_system.config import Settings
from research_system.errors import VectorStoreError
from research_system.ingestion import ingest_directory

pytestmark = pytest.mark.integration

DIMENSION = 8

if os.getenv("RUN_INTEGRATION") != "1":
    pytest.skip(
        "set RUN_INTEGRATION=1 and start Qdrant (docker compose up -d qdrant)",
        allow_module_level=True,
    )


@pytest.fixture
def live_settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="unused-embeddings-are-faked",
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
        embedding_dim=DIMENSION,
        langchain_tracing_v2=False,
    )


@pytest.fixture
def store(live_settings) -> QdrantVectorStore:
    store = QdrantVectorStore(live_settings)
    try:
        store.collection_exists("probe")
    except VectorStoreError as exc:
        pytest.skip(f"Qdrant is not reachable: {exc}")
    return store


@pytest.fixture
def collection(store):
    """A uniquely-named collection, removed on teardown."""
    name = f"itest_{uuid.uuid4().hex[:12]}"
    yield name
    # Cleanup must never fail an otherwise-passing test.
    with contextlib.suppress(Exception):
        store._get_client().delete_collection(collection_name=name)


def points(count: int, source: str = "a.pdf") -> list[VectorPoint]:
    return [
        VectorPoint(
            id=deterministic_point_id(source, i, f"chunk {i}"),
            vector=[float((i + j) % 3) / 2.0 for j in range(DIMENSION)],
            content=f"chunk {i}",
            source=source,
            metadata={"chunk_index": i, "source_type": "document", "kind": "report"},
        )
        for i in range(count)
    ]


# --- health and lifecycle --------------------------------------------------
def test_qdrant_is_reachable(store):
    assert store.collection_exists("definitely_not_a_real_collection") is False


def test_collection_is_created_once_with_the_configured_dimension(store, collection):
    assert store.ensure_collection(collection, DIMENSION) is True
    assert store.ensure_collection(collection, DIMENSION) is False
    assert store.collection_exists(collection) is True


def test_dimension_mismatch_is_reported_not_recreated(store, collection):
    store.ensure_collection(collection, DIMENSION)

    with pytest.raises(VectorStoreError, match="vector size"):
        store.ensure_collection(collection, 1536)

    # The existing collection survives the failed call.
    assert store.collection_exists(collection) is True


def test_search_on_a_missing_collection_is_actionable(store):
    with pytest.raises(VectorStoreError, match="does not exist"):
        store.search(f"missing_{uuid.uuid4().hex[:8]}", [0.0] * DIMENSION)


# --- round trip -----------------------------------------------------------
def test_upsert_then_search_returns_the_payload(store, collection):
    store.ensure_collection(collection, DIMENSION)
    batch = points(5)
    assert store.upsert(collection, batch) == 5

    results = store.search(collection, batch[0].vector, top_k=3, score_threshold=0.0)

    assert results
    top = results[0]
    assert top.content.startswith("chunk")
    assert top.source == "a.pdf"
    assert top.metadata["source_type"] == "document"
    assert "native_score" in top.metadata


def test_metadata_filter_is_applied(store, collection):
    store.ensure_collection(collection, DIMENSION)
    store.upsert(collection, points(3, source="report.pdf"))
    store.upsert(
        collection,
        [
            VectorPoint(
                id=deterministic_point_id("memo.pdf", 0, "memo chunk"),
                vector=[1.0] * DIMENSION,
                content="memo chunk",
                source="memo.pdf",
                metadata={"kind": "memo", "source_type": "document"},
            )
        ],
    )

    results = store.search(
        collection,
        [1.0] * DIMENSION,
        top_k=10,
        score_threshold=0.0,
        metadata_filter={"kind": "memo"},
    )

    assert results
    assert {doc.source for doc in results} == {"memo.pdf"}


def test_score_threshold_filters_weak_hits(store, collection):
    store.ensure_collection(collection, DIMENSION)
    store.upsert(collection, points(5))

    strict = store.search(collection, [1.0] * DIMENSION, top_k=10, score_threshold=0.999)
    loose = store.search(collection, [1.0] * DIMENSION, top_k=10, score_threshold=0.0)

    assert len(strict) <= len(loose)


def test_batched_upsert_exceeding_one_batch(store, collection):
    store.ensure_collection(collection, DIMENSION)
    batch = points(250)  # UPSERT_BATCH_SIZE is 100

    assert store.upsert(collection, batch) == 250
    assert store.count(collection) == 250


def test_reupserting_identical_points_does_not_grow_the_collection(store, collection):
    store.ensure_collection(collection, DIMENSION)
    batch = points(20)

    store.upsert(collection, batch)
    first = store.count(collection)
    store.upsert(collection, batch)

    assert store.count(collection) == first == 20


# --- ingestion end to end -------------------------------------------------
@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "a.md").write_text(
        "# Fusion\n\n" + ("Reciprocal rank fusion merges ranked lists. " * 40), encoding="utf-8"
    )
    (tmp_path / "b.txt").write_text("Qdrant uses cosine distance. " * 40, encoding="utf-8")
    return tmp_path


def test_ingesting_twice_is_idempotent_against_live_qdrant(
    corpus, live_settings, store, collection
):
    embeddings = FakeEmbeddings(dimension=DIMENSION)
    kwargs = {
        "settings": live_settings,
        "collection": collection,
        "embeddings": embeddings,
        "vector_store": store,
    }

    first = ingest_directory(corpus, **kwargs)
    assert first.chunks_created > 0
    assert first.collection_created is True
    assert store.count(collection) == first.total_points_in_collection

    second = ingest_directory(corpus, **kwargs)

    assert second.collection_created is False
    assert second.total_points_in_collection == first.total_points_in_collection
    assert store.count(collection) == first.total_points_in_collection


def test_ingested_documents_are_searchable(corpus, live_settings, store, collection):
    embeddings = FakeEmbeddings(dimension=DIMENSION)
    ingest_directory(
        corpus,
        settings=live_settings,
        collection=collection,
        embeddings=embeddings,
        vector_store=store,
    )

    query_vector = embeddings.embed("Reciprocal rank fusion merges ranked lists.")
    results = store.search(collection, query_vector, top_k=5, score_threshold=0.0)

    assert results
    assert all(doc.content for doc in results)
    assert {doc.source for doc in results} <= {"a.md", "b.txt"}
