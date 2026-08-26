"""Document ingestion: folder of files -> chunks -> embeddings -> Qdrant.

Idempotent by construction. A point's ID is a hash of (file, chunk position,
chunk text), so re-ingesting an unchanged corpus overwrites the same IDs and
the collection size does not grow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters.embeddings import EmbeddingProvider, get_embedding_provider
from .adapters.qdrant_store import (
    VectorPoint,
    VectorStore,
    deterministic_point_id,
    get_vector_store,
)
from .config import Settings, get_settings
from .errors import IngestionError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ".html", ".htm"}
EMBED_BATCH_SIZE = 100


@dataclass
class Chunk:
    """One indexable unit of text."""

    content: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def point_id(self) -> str:
        return deterministic_point_id(self.source, self.chunk_index, self.content)


@dataclass
class IngestionReport:
    """What actually happened, for the CLI to print."""

    documents_loaded: int = 0
    chunks_created: int = 0
    points_indexed: int = 0
    collection: str = ""
    collection_created: bool = False
    total_points_in_collection: int = 0
    skipped_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def discover_files(source: Path) -> tuple[list[Path], list[str]]:
    """Find readable files under `source`. Returns (supported, skipped labels)."""
    supported: list[Path] = []
    skipped: list[str] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            supported.append(path)
        else:
            skipped.append(f"{path.name} (unsupported type {path.suffix or 'none'})")
    return supported, skipped


def load_and_chunk(
    source: Path, settings: Settings, *, chunk_size: int, chunk_overlap: int
) -> tuple[list[Chunk], int, list[str]]:
    """Read every supported file and split it. Returns (chunks, docs, skipped)."""
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.readers import SimpleDirectoryReader

    supported, skipped = discover_files(source)
    if not supported:
        raise IngestionError(
            f"No supported documents found under {source}. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    try:
        reader = SimpleDirectoryReader(
            input_files=[str(path) for path in supported],
            recursive=False,
            filename_as_id=True,
        )
        documents = reader.load_data()
    except Exception as exc:
        raise IngestionError(f"could not read documents from {source}: {exc}") from exc

    if not documents:
        raise IngestionError(f"No document content could be extracted from {source}.")

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    try:
        nodes = splitter.get_nodes_from_documents(documents)
    except Exception as exc:
        raise IngestionError(f"could not split documents: {exc}") from exc

    # Chunk index is per source file, so IDs stay stable when other files change.
    per_source_counter: dict[str, int] = {}
    chunks: list[Chunk] = []
    for node in nodes:
        content = (node.get_content() or "").strip()
        if not content:
            continue
        node_metadata = dict(getattr(node, "metadata", {}) or {})
        file_name = str(node_metadata.get("file_name") or "unknown")
        index = per_source_counter.get(file_name, 0)
        per_source_counter[file_name] = index + 1

        metadata: dict[str, Any] = {
            "file_path": str(node_metadata.get("file_path") or ""),
            "page_label": str(node_metadata.get("page_label") or ""),
            "chunk_index": index,
            "source_type": "document",
        }
        chunks.append(
            Chunk(content=content, source=file_name, chunk_index=index, metadata=metadata)
        )

    if not chunks:
        raise IngestionError(
            f"Documents under {source} produced no non-empty chunks. "
            "They may be scanned images without extractable text."
        )
    return chunks, len(documents), skipped


def ingest_directory(
    source: str | Path,
    *,
    settings: Settings | None = None,
    collection: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embeddings: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
) -> IngestionReport:
    """Index a directory tree into Qdrant."""
    settings = settings or get_settings()
    source_path = Path(source).expanduser()

    if not source_path.exists():
        raise IngestionError(f"Source path does not exist: {source_path}")
    if not source_path.is_dir():
        raise IngestionError(f"Source path is not a directory: {source_path}")

    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    if not 0 <= overlap < size:
        raise IngestionError(
            f"chunk_overlap ({overlap}) must satisfy 0 <= overlap < chunk_size ({size})"
        )

    target_collection = collection or settings.qdrant_collection
    embeddings = embeddings or get_embedding_provider(settings)
    vector_store = vector_store or get_vector_store(settings)

    chunks, document_count, skipped = load_and_chunk(
        source_path, settings, chunk_size=size, chunk_overlap=overlap
    )
    logger.info("loaded %d documents -> %d chunks", document_count, len(chunks))

    created = vector_store.ensure_collection(target_collection, embeddings.dimension)

    # One request per 100 chunks, not one per chunk.
    vectors = embeddings.embed_batch(
        [chunk.content for chunk in chunks], batch_size=EMBED_BATCH_SIZE
    )
    if len(vectors) != len(chunks):
        raise IngestionError(
            f"embedding provider returned {len(vectors)} vectors for {len(chunks)} chunks"
        )

    points = [
        VectorPoint(
            id=chunk.point_id,
            vector=vector,
            content=chunk.content,
            source=chunk.source,
            metadata=chunk.metadata,
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]

    # Identical chunks within one run collapse to one point; without this the
    # batch would contain duplicate IDs and the reported count would lie.
    unique_points = list({point.id: point for point in points}.values())
    duplicates = len(points) - len(unique_points)
    warnings: list[str] = []
    if duplicates:
        warnings.append(f"Collapsed {duplicates} identical chunk(s) into existing points.")

    indexed = vector_store.upsert(target_collection, unique_points)

    return IngestionReport(
        documents_loaded=document_count,
        chunks_created=len(chunks),
        points_indexed=indexed,
        collection=target_collection,
        collection_created=created,
        total_points_in_collection=vector_store.count(target_collection),
        skipped_files=skipped,
        warnings=warnings,
    )
