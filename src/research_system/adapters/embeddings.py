"""Embedding provider behind a protocol.

Ingestion goes through `embed_batch`, never one request per chunk -- a 5k-chunk
corpus would otherwise be 5k round trips.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from ..config import Settings, get_settings
from ..errors import EmbeddingError

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100


class EmbeddingProvider(Protocol):
    """Text -> vector."""

    @property
    def dimension(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(
        self, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> list[list[float]]:
        """Embed many texts, preserving input order exactly."""
        ...


class OpenAIEmbeddings:
    """Production provider. The client is built lazily so import stays cheap."""

    def __init__(self, settings: Settings | None = None, model: str | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = model or self._settings.embedding_model
        self._client: Any = None

    @property
    def dimension(self) -> int:
        return self._settings.embedding_dim

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self):  # type: ignore[no-untyped-def]
        if self._client is None:
            self._settings.require_embeddings()
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._settings.openai_api_key, timeout=60.0, max_retries=3
            )
        return self._client

    def _embed_chunk(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        try:
            response = client.embeddings.create(model=self._model, input=texts)
        except Exception as exc:
            raise EmbeddingError(f"embedding request failed ({self._model}): {exc}") from exc

        if len(response.data) != len(texts):
            raise EmbeddingError(
                f"embedding provider returned {len(response.data)} vectors for {len(texts)} inputs"
            )
        # The API documents order preservation, but index is authoritative --
        # sorting makes a silent misalignment impossible.
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [list(item.embedding) for item in ordered]

        expected = self.dimension
        for vector in vectors:
            if len(vector) != expected:
                raise EmbeddingError(
                    f"embedding dimension mismatch: model {self._model!r} returned "
                    f"{len(vector)} but EMBEDDING_DIM is {expected}. Update EMBEDDING_DIM."
                )
        return vectors

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise EmbeddingError("cannot embed empty text")
        return self._embed_chunk([text])[0]

    def embed_batch(
        self, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> list[list[float]]:
        if not texts:
            return []
        blanks = [i for i, text in enumerate(texts) if not (text or "").strip()]
        if blanks:
            raise EmbeddingError(
                f"cannot embed blank text at positions {blanks[:5]}"
                f"{'...' if len(blanks) > 5 else ''}; filter them before embedding"
            )

        size = max(1, batch_size)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), size):
            batch = texts[start : start + size]
            vectors.extend(self._embed_chunk(batch))
            logger.debug("embedded %d/%d texts", len(vectors), len(texts))
        return vectors


class FakeEmbeddings:
    """Deterministic offline provider for tests.

    Vectors are derived from a hash of the text, so identical text always
    embeds identically and different text almost never collides.
    """

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension
        self.calls: list[list[str]] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> list[float]:
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self._dimension)]

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise EmbeddingError("cannot embed empty text")
        self.calls.append([text])
        return self._vector(text)

    def embed_batch(
        self, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> list[list[float]]:
        if not texts:
            return []
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    return OpenAIEmbeddings(settings)
