"""Dependency container shared by every graph node.

Nodes receive their adapters through this object rather than constructing them,
which is what lets the whole graph run offline against fakes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..adapters.embeddings import EmbeddingProvider, get_embedding_provider
from ..adapters.llm import LLMClient, get_llm_client
from ..adapters.memory import MemoryStore, NullMemoryStore, get_memory_store
from ..adapters.qdrant_store import VectorStore, get_vector_store
from ..adapters.web_search import WebSearchProvider, get_web_search_provider
from ..config import Settings, get_settings


@dataclass
class Dependencies:
    """Adapters + settings for one pipeline run.

    `vector_store`/`web_search` are optional: when a backend is unconfigured we
    leave it as None so the retriever can degrade instead of failing at import
    time.
    """

    settings: Settings = field(default_factory=get_settings)
    llm: LLMClient | None = None
    embeddings: EmbeddingProvider | None = None
    vector_store: VectorStore | None = None
    web_search: WebSearchProvider | None = None
    memory: MemoryStore = field(default_factory=NullMemoryStore)

    @property
    def vector_enabled(self) -> bool:
        return self.vector_store is not None and self.embeddings is not None

    @property
    def web_enabled(self) -> bool:
        return self.web_search is not None

    def require_llm(self) -> LLMClient:
        if self.llm is None:
            raise RuntimeError("no LLM client configured on Dependencies")
        return self.llm


def build_dependencies(
    settings: Settings | None = None, *, use_memory: bool = False
) -> Dependencies:
    """Wire real adapters, including only the backends that are configured."""
    settings = settings or get_settings()
    return Dependencies(
        settings=settings,
        llm=get_llm_client(settings),
        embeddings=get_embedding_provider(settings) if settings.openai_api_key.strip() else None,
        vector_store=get_vector_store(settings) if settings.vector_available else None,
        web_search=get_web_search_provider(settings) if settings.web_available else None,
        memory=get_memory_store(settings, enabled=use_memory),
    )
