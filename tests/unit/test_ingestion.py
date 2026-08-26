"""Ingestion: validation, batching, and idempotency."""

from __future__ import annotations

import pytest

from research_system.adapters.qdrant_store import InMemoryVectorStore
from research_system.errors import IngestionError
from research_system.ingestion import Chunk, discover_files, ingest_directory


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "a.md").write_text(
        "# Retrieval\n\n" + ("Reciprocal rank fusion merges ranked lists. " * 40),
        encoding="utf-8",
    )
    (tmp_path / "b.txt").write_text(
        "Qdrant stores vectors with cosine distance. " * 40, encoding="utf-8"
    )
    (tmp_path / "notes.xlsx").write_bytes(b"binary junk")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.txt").write_text("Nested evidence about verification. " * 40, encoding="utf-8")
    return tmp_path


# --- validation ------------------------------------------------------------
def test_missing_path_raises(tmp_path, settings):
    with pytest.raises(IngestionError, match="does not exist"):
        ingest_directory(tmp_path / "nope", settings=settings)


def test_file_instead_of_directory_raises(tmp_path, settings):
    target = tmp_path / "f.txt"
    target.write_text("x", encoding="utf-8")

    with pytest.raises(IngestionError, match="not a directory"):
        ingest_directory(target, settings=settings)


def test_directory_without_supported_files_raises(tmp_path, settings):
    (tmp_path / "x.xlsx").write_bytes(b"junk")

    with pytest.raises(IngestionError, match="No supported documents"):
        ingest_directory(tmp_path, settings=settings)


@pytest.mark.parametrize("overlap,size", [(512, 512), (600, 512), (-1, 512)])
def test_invalid_overlap_is_rejected(corpus, settings, overlap, size):
    with pytest.raises(IngestionError, match="chunk_overlap"):
        ingest_directory(corpus, settings=settings, chunk_size=size, chunk_overlap=overlap)


# --- discovery -------------------------------------------------------------
def test_discovery_is_recursive_and_reports_skipped(corpus):
    supported, skipped = discover_files(corpus)

    names = {path.name for path in supported}
    assert names == {"a.md", "b.txt", "c.txt"}
    assert any("notes.xlsx" in entry for entry in skipped)


def test_hidden_files_are_ignored(tmp_path):
    (tmp_path / ".hidden.txt").write_text("x", encoding="utf-8")
    supported, _ = discover_files(tmp_path)

    assert supported == []


# --- chunk identity --------------------------------------------------------
def test_chunk_id_is_stable_across_instances():
    a = Chunk(content="text", source="a.pdf", chunk_index=0)
    b = Chunk(content="text", source="a.pdf", chunk_index=0)

    assert a.point_id == b.point_id


# --- end to end ------------------------------------------------------------
def test_ingestion_reports_real_counts(corpus, settings, embeddings):
    store = InMemoryVectorStore(dimension=8)

    report = ingest_directory(
        corpus,
        settings=settings,
        collection="c",
        embeddings=embeddings,
        vector_store=store,
    )

    assert report.documents_loaded == 3
    assert report.chunks_created > 0
    assert report.points_indexed == report.chunks_created
    assert report.collection_created is True
    assert report.total_points_in_collection == report.points_indexed
    assert any("notes.xlsx" in entry for entry in report.skipped_files)


def test_second_identical_ingestion_does_not_duplicate(corpus, settings, embeddings):
    store = InMemoryVectorStore(dimension=8)
    kwargs = {
        "settings": settings,
        "collection": "c",
        "embeddings": embeddings,
        "vector_store": store,
    }

    first = ingest_directory(corpus, **kwargs)
    second = ingest_directory(corpus, **kwargs)

    assert second.total_points_in_collection == first.total_points_in_collection
    assert second.collection_created is False


def test_embeddings_are_batched_not_per_chunk(corpus, settings, embeddings):
    store = InMemoryVectorStore(dimension=8)

    report = ingest_directory(
        corpus, settings=settings, collection="c", embeddings=embeddings, vector_store=store
    )

    # One batch call for all chunks, not one call each.
    batch_calls = [call for call in embeddings.calls if len(call) > 1]
    assert batch_calls, "expected a batched embedding call"
    assert len(embeddings.calls) < report.chunks_created


def test_chunks_carry_provenance_metadata(corpus, settings, embeddings):
    store = InMemoryVectorStore(dimension=8)
    ingest_directory(
        corpus, settings=settings, collection="c", embeddings=embeddings, vector_store=store
    )

    stored = list(store.collections["c"].values())
    assert stored
    for point in stored:
        assert point.source
        assert "file_path" in point.metadata
        assert "chunk_index" in point.metadata
        assert point.metadata["source_type"] == "document"


def test_adding_a_file_only_adds_its_points(corpus, settings, embeddings):
    store = InMemoryVectorStore(dimension=8)
    kwargs = {
        "settings": settings,
        "collection": "c",
        "embeddings": embeddings,
        "vector_store": store,
    }

    first = ingest_directory(corpus, **kwargs)
    (corpus / "d.txt").write_text("Brand new evidence about planning. " * 40, encoding="utf-8")
    second = ingest_directory(corpus, **kwargs)

    assert second.total_points_in_collection > first.total_points_in_collection
    assert second.documents_loaded == 4
