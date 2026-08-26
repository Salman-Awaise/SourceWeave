"""Application configuration.

Loads from environment / `.env`. Unknown variables are ignored so that a shared
`.env` containing unrelated keys does not break startup. Credentials are only
required at the point where an operation actually needs them -- see
`require_*` helpers -- so that e.g. `--help` and offline tests never need keys.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import ConfigurationError

RetrievalStrategy = Literal["vector", "web", "hybrid"]
VALID_STRATEGIES: tuple[str, ...] = ("vector", "web", "hybrid")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- providers -------------------------------------------------------
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # --- vector store ----------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "research"

    # --- web search ------------------------------------------------------
    tavily_api_key: str = ""

    # --- memory ----------------------------------------------------------
    mem0_api_key: str = ""

    # --- tracing ---------------------------------------------------------
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "multi-agent-research-system"
    trace_content: bool = False
    """Opt-in: send document/memory content to LangSmith. Off by default."""

    # --- models ----------------------------------------------------------
    default_llm: str = "claude-sonnet-4-20250514"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # --- tunables --------------------------------------------------------
    max_retrieval_docs: int = Field(default=10, ge=1, le=100)
    chunk_size: int = Field(default=512, ge=32, le=8192)
    chunk_overlap: int = Field(default=50, ge=0)
    similarity_top_k: int = Field(default=5, ge=1, le=100)
    web_results_per_query: int = Field(default=3, ge=1, le=25)
    faithfulness_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_verification_retries: int = Field(default=2, ge=0, le=10)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, ge=64, le=32768)
    qdrant_score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)
    memory_search_limit: int = Field(default=3, ge=1, le=50)

    # --- hard safety bounds (not env-tuned by design) --------------------
    max_query_chars: int = 4000
    max_sub_queries: int = 4
    max_context_chars: int = 60_000
    max_chat_history_messages: int = 5

    @field_validator("default_llm", "embedding_model", "qdrant_collection")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()

    @model_validator(mode="after")
    def _check_overlap(self) -> Settings:
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must satisfy "
                f"0 <= overlap < CHUNK_SIZE ({self.chunk_size})"
            )
        return self

    # --- derived ---------------------------------------------------------
    @property
    def llm_provider(self) -> str:
        """Best-effort provider inference from the model name."""
        model = self.default_llm.lower()
        if "/" in model:
            return model.split("/", 1)[0]
        if model.startswith("claude"):
            return "anthropic"
        if model.startswith(("gpt", "o1", "o3", "o4", "text-")):
            return "openai"
        return "unknown"

    @property
    def vector_available(self) -> bool:
        return bool(self.qdrant_url.strip()) and bool(self.openai_api_key.strip())

    @property
    def web_available(self) -> bool:
        return bool(self.tavily_api_key.strip())

    @property
    def tracing_enabled(self) -> bool:
        return self.langchain_tracing_v2 and bool(self.langchain_api_key.strip())

    # --- credential gates ------------------------------------------------
    def require_llm(self) -> None:
        provider = self.llm_provider
        if provider == "anthropic" and not self.anthropic_api_key.strip():
            raise ConfigurationError(
                f"DEFAULT_LLM={self.default_llm!r} needs ANTHROPIC_API_KEY. "
                "Set it in .env or choose a different DEFAULT_LLM."
            )
        if provider == "openai" and not self.openai_api_key.strip():
            raise ConfigurationError(
                f"DEFAULT_LLM={self.default_llm!r} needs OPENAI_API_KEY. "
                "Set it in .env or choose a different DEFAULT_LLM."
            )

    def require_embeddings(self) -> None:
        if not self.openai_api_key.strip():
            raise ConfigurationError(
                f"EMBEDDING_MODEL={self.embedding_model!r} needs OPENAI_API_KEY. Set it in .env."
            )

    def require_web(self) -> None:
        if not self.web_available:
            raise ConfigurationError(
                "Web search needs TAVILY_API_KEY. Set it in .env or use "
                "retrieval strategy 'vector'."
            )

    def redacted(self) -> dict[str, object]:
        """Config snapshot safe to log: every secret becomes set/unset."""
        secret_suffixes = ("_api_key", "_key", "_token", "_secret")
        out: dict[str, object] = {}
        for name, value in self.model_dump().items():
            if name.endswith(secret_suffixes):
                out[name] = "<set>" if str(value).strip() else "<unset>"
            else:
                out[name] = value
        return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Test helper: drop the memoized Settings instance."""
    get_settings.cache_clear()
