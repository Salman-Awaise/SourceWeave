"""Qdrant vector store adapter.

Point IDs are deterministic (UUIDv5 over source + chunk index + content hash),
which is what makes re-ingesting the same corpus an update rather than a
duplicate insert.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import Settings, get_settings
from ..core.state import RetrievedDocument, normalize_text
from ..errors import VectorStoreError

logger = logging.getLogger(__name__)

UPSERT_BATCH_SIZE = 100

# Fixed namespace so IDs are stable across machines and runs.
_ID_NAMESPACE = uuid.UUID("6f1d5a2e-6c4f-4f7a-9a4b-2c1d0e8f3b57")


def deterministic_point_id(source: str, chunk_index: int, content: str) -> str:
    """Same (source, position, content) -> same point ID, always."""
    payload = f"{source}\x00{chunk_index}\x00{normalize_text(content)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(_ID_NAMESPACE, digest))


@dataclass
class VectorPoint:
    """A chunk ready to be indexed."""

    id: str
    vector: list[float]
    content: str
    source: str
    metadata: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        # `content` and `source` are reserved; metadata may not shadow them.
        payload: dict[str, Any] = dict(self.metadata)
        payload["content"] = self.content
        payload["source"] = self.source
        return payload


class VectorStore(Protocol):
    """Vector search + indexing."""

    def ensure_collection(self, collection: str, dimension: int) -> bool: ...

    def collection_exists(self, collection: str) -> bool: ...

    def upsert(self, collection: str, points: list[VectorPoint]) -> int: ...

    def search(
        self,
        collection: str,
        vector: list[float],
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]: ...

    def count(self, collection: str) -> int: ...


class QdrantVectorStore:
    """Production adapter. Client construction is lazy and cached."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: Any = None

    def _get_client(self):  # type: ignore[no-untyped-def]
        if self._client is None:
            from qdrant_client import QdrantClient

            kwargs: dict[str, Any] = {"url": self._settings.qdrant_url, "timeout": 30}
            # Passing api_key=None is fine, but passing "" makes some versions
            # send an empty auth header, which local Qdrant rejects.
            if self._settings.qdrant_api_key.strip():
                kwargs["api_key"] = self._settings.qdrant_api_key.strip()
            try:
                self._client = QdrantClient(**kwargs)
            except Exception as exc:
                raise VectorStoreError(
                    f"could not connect to Qdrant at {self._settings.qdrant_url}: {exc}. "
                    "Start it with `docker compose up -d qdrant`."
                ) from exc
        return self._client

    # --- collection management ------------------------------------------
    def collection_exists(self, collection: str) -> bool:
        client = self._get_client()
        try:
            return bool(client.collection_exists(collection_name=collection))
        except Exception as exc:
            raise VectorStoreError(
                f"could not reach Qdrant at {self._settings.qdrant_url}: {exc}. "
                "Start it with `docker compose up -d qdrant`."
            ) from exc

    def ensure_collection(self, collection: str, dimension: int) -> bool:
        """Create the collection if absent; validate it if present.

        Returns True when a collection was created. An existing collection is
        never dropped or recreated -- a dimension mismatch is an error the
        operator must resolve, not data we may silently destroy.
        """
        from qdrant_client.models import Distance, VectorParams

        client = self._get_client()
        if self.collection_exists(collection):
            self._validate_collection(collection, dimension)
            return False

        try:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
        except Exception as exc:
            # A concurrent ingest may have won the race; that is fine.
            if self.collection_exists(collection):
                self._validate_collection(collection, dimension)
                return False
            raise VectorStoreError(f"could not create collection {collection!r}: {exc}") from exc

        logger.info("created Qdrant collection %r (dim=%d, cosine)", collection, dimension)
        return True

    def _validate_collection(self, collection: str, dimension: int) -> None:
        client = self._get_client()
        try:
            info = client.get_collection(collection_name=collection)
            params = info.config.params.vectors
        except Exception as exc:
            raise VectorStoreError(f"could not inspect collection {collection!r}: {exc}") from exc

        # Named-vector collections expose a dict; ours uses the default vector.
        if isinstance(params, dict):
            if len(params) != 1:
                raise VectorStoreError(
                    f"collection {collection!r} uses named vectors {sorted(params)}; "
                    "this system expects a single default vector."
                )
            params = next(iter(params.values()))

        existing_size = getattr(params, "size", None)
        existing_distance = getattr(params, "distance", None)
        if existing_size is not None and int(existing_size) != int(dimension):
            raise VectorStoreError(
                f"collection {collection!r} has vector size {existing_size} but "
                f"EMBEDDING_DIM is {dimension}. Use a different QDRANT_COLLECTION, "
                "or delete the existing one deliberately."
            )
        if (
            existing_distance is not None
            and str(existing_distance).lower().endswith("cosine") is False
        ):
            raise VectorStoreError(
                f"collection {collection!r} uses distance {existing_distance}; "
                "this system indexes and searches with cosine distance."
            )

    # --- writes -----------------------------------------------------------
    def upsert(self, collection: str, points: list[VectorPoint]) -> int:
        if not points:
            return 0
        from qdrant_client.models import PointStruct

        client = self._get_client()
        written = 0
        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            batch = points[start : start + UPSERT_BATCH_SIZE]
            structs = [
                PointStruct(id=point.id, vector=point.vector, payload=point.payload())
                for point in batch
            ]
            try:
                client.upsert(collection_name=collection, points=structs, wait=True)
            except Exception as exc:
                raise VectorStoreError(
                    f"upsert into {collection!r} failed at offset {start}: {exc}"
                ) from exc
            written += len(batch)
            logger.debug("upserted %d/%d points", written, len(points))
        return written

    # --- reads ------------------------------------------------------------
    def search(
        self,
        collection: str,
        vector: list[float],
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        client = self._get_client()
        if not self.collection_exists(collection):
            raise VectorStoreError(
                f"Qdrant collection {collection!r} does not exist. "
                f"Index documents first: `research-system ingest <dir> --collection {collection}`."
            )

        query_filter = _build_filter(metadata_filter)
        try:
            # query_points is the current API; older clients only have search().
            if hasattr(client, "query_points"):
                response = client.query_points(
                    collection_name=collection,
                    query=vector,
                    limit=top_k,
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                    with_payload=True,
                )
                hits = response.points
            else:  # pragma: no cover - legacy client fallback
                hits = client.search(
                    collection_name=collection,
                    query_vector=vector,
                    limit=top_k,
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                    with_payload=True,
                )
        except Exception as exc:
            raise VectorStoreError(f"search in {collection!r} failed: {exc}") from exc

        return [_hit_to_document(hit) for hit in hits]

    def count(self, collection: str) -> int:
        client = self._get_client()
        if not self.collection_exists(collection):
            return 0
        try:
            return int(client.count(collection_name=collection, exact=True).count)
        except Exception as exc:
            raise VectorStoreError(f"count on {collection!r} failed: {exc}") from exc


def _build_filter(metadata_filter: dict[str, Any] | None):  # type: ignore[no-untyped-def]
    """Translate a flat equality dict into a Qdrant `must` filter."""
    if not metadata_filter:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    return Filter(
        must=[
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in metadata_filter.items()
        ]
    )


def _hit_to_document(hit: Any) -> RetrievedDocument:
    payload = dict(getattr(hit, "payload", None) or {})
    content = str(payload.pop("content", ""))
    source = str(payload.pop("source", "unknown"))
    score = float(getattr(hit, "score", 0.0) or 0.0)
    payload.setdefault("source_type", "document")
    payload["native_score"] = score
    payload["point_id"] = str(getattr(hit, "id", ""))
    return RetrievedDocument(content=content, source=source, score=score, metadata=payload)


class InMemoryVectorStore:
    """Offline `VectorStore` for tests: brute-force cosine over stored points."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.collections: dict[str, dict[str, VectorPoint]] = {}

    def collection_exists(self, collection: str) -> bool:
        return collection in self.collections

    def ensure_collection(self, collection: str, dimension: int) -> bool:
        if collection in self.collections:
            if dimension != self.dimension:
                raise VectorStoreError(
                    f"collection {collection!r} has vector size {self.dimension} "
                    f"but EMBEDDING_DIM is {dimension}."
                )
            return False
        self.dimension = dimension
        self.collections[collection] = {}
        return True

    def upsert(self, collection: str, points: list[VectorPoint]) -> int:
        store = self.collections.setdefault(collection, {})
        for point in points:
            store[point.id] = point
        return len(points)

    def search(
        self,
        collection: str,
        vector: list[float],
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        if collection not in self.collections:
            raise VectorStoreError(f"Qdrant collection {collection!r} does not exist.")

        results: list[tuple[float, VectorPoint]] = []
        for point in self.collections[collection].values():
            if metadata_filter and any(
                point.metadata.get(key) != value for key, value in metadata_filter.items()
            ):
                continue
            score = _cosine(vector, point.vector)
            if score_threshold is not None and score < score_threshold:
                continue
            results.append((score, point))

        results.sort(key=lambda item: item[0], reverse=True)
        documents = []
        for score, point in results[:top_k]:
            metadata = dict(point.metadata)
            metadata.setdefault("source_type", "document")
            metadata["native_score"] = score
            metadata["point_id"] = point.id
            documents.append(
                RetrievedDocument(
                    content=point.content, source=point.source, score=score, metadata=metadata
                )
            )
        return documents

    def count(self, collection: str) -> int:
        return len(self.collections.get(collection, {}))


def _cosine(a: list[float], b: list[float]) -> float:
    import math

    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_vector_store(settings: Settings | None = None) -> VectorStore:
    return QdrantVectorStore(settings)
