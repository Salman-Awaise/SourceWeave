"""Shared fixtures. Every test here runs offline against fakes."""

from __future__ import annotations

import os

import pytest

from research_system.adapters.embeddings import FakeEmbeddings
from research_system.adapters.llm import FakeLLMClient
from research_system.adapters.memory import InMemoryMemoryStore, NullMemoryStore
from research_system.adapters.qdrant_store import InMemoryVectorStore
from research_system.adapters.web_search import FakeWebSearch
from research_system.config import Settings
from research_system.core.deps import Dependencies
from research_system.core.state import RetrievedDocument


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Stop a developer's real .env or shell keys from leaking into tests."""
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "TAVILY_API_KEY",
        "MEM0_API_KEY",
        "LANGCHAIN_API_KEY",
        "QDRANT_API_KEY",
        "QDRANT_URL",
        "DEFAULT_LLM",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


@pytest.fixture
def settings() -> Settings:
    """Settings with credentials present so gates pass, pointing at nothing real."""
    return Settings(
        _env_file=None,
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key",
        tavily_api_key="test-tavily-key",
        qdrant_url="http://localhost:6333",
        qdrant_collection="test_collection",
        embedding_dim=8,
        langchain_tracing_v2=False,
    )


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings(dimension=8)


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore(dimension=8)
    store.ensure_collection("test_collection", 8)
    return store


@pytest.fixture
def web_search() -> FakeWebSearch:
    return FakeWebSearch()


@pytest.fixture
def llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def deps(settings, llm, embeddings, vector_store, web_search) -> Dependencies:
    """Fully-wired offline dependency container."""
    return Dependencies(
        settings=settings,
        llm=llm,
        embeddings=embeddings,
        vector_store=vector_store,
        web_search=web_search,
        memory=NullMemoryStore(),
    )


@pytest.fixture
def memory_deps(deps) -> Dependencies:
    deps.memory = InMemoryMemoryStore()
    return deps


def make_doc(
    content: str, source: str = "doc.pdf", score: float = 0.9, **metadata
) -> RetrievedDocument:
    """Terse document builder for tests."""
    return RetrievedDocument(content=content, source=source, score=score, metadata=metadata)


@pytest.fixture
def doc_factory():
    return make_doc


def requires_integration() -> bool:
    return os.getenv("RUN_INTEGRATION") == "1"


def requires_paid() -> bool:
    return os.getenv("RUN_PAID") == "1"
