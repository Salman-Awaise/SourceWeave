"""Configuration: validation, credential gates, redaction."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_system.config import Settings
from research_system.errors import ConfigurationError


def make(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_defaults_match_the_documented_contract():
    s = make()

    assert s.max_retrieval_docs == 10
    assert s.chunk_size == 512
    assert s.chunk_overlap == 50
    assert s.similarity_top_k == 5
    assert s.web_results_per_query == 3
    assert s.faithfulness_threshold == 0.7
    assert s.max_verification_retries == 2
    assert s.temperature == 0.1
    assert s.llm_max_tokens == 2048
    assert s.qdrant_score_threshold == 0.3
    assert s.rrf_k == 60
    assert s.memory_search_limit == 3
    assert s.embedding_dim == 1536
    assert s.qdrant_collection == "research"


def test_unknown_env_vars_are_ignored(monkeypatch):
    monkeypatch.setenv("SOMETHING_UNRELATED", "value")
    make()  # must not raise


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        make(chunk_size=100, chunk_overlap=100)


def test_out_of_range_threshold_is_rejected():
    with pytest.raises(ValidationError):
        make(faithfulness_threshold=1.5)


def test_empty_model_name_is_rejected():
    with pytest.raises(ValidationError):
        make(default_llm="   ")


@pytest.mark.parametrize(
    "model,provider",
    [
        ("claude-sonnet-4-20250514", "anthropic"),
        ("gpt-4o", "openai"),
        ("o3-mini", "openai"),
        ("anthropic/claude-3-5-haiku", "anthropic"),
        ("openai/gpt-4o", "openai"),
        ("some-local-model", "unknown"),
    ],
)
def test_provider_inference(model, provider):
    assert make(default_llm=model).llm_provider == provider


def test_llm_gate_requires_the_selected_provider_key():
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        make(default_llm="claude-sonnet-4-20250514").require_llm()

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        make(default_llm="gpt-4o").require_llm()


def test_llm_gate_passes_with_the_right_key():
    make(default_llm="gpt-4o", openai_api_key="k").require_llm()
    make(default_llm="claude-sonnet-4-20250514", anthropic_api_key="k").require_llm()


def test_embedding_gate_requires_openai():
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        make().require_embeddings()


def test_web_gate_requires_tavily():
    with pytest.raises(ConfigurationError, match="TAVILY_API_KEY"):
        make().require_web()


def test_availability_flags():
    assert make().vector_available is False
    assert make(openai_api_key="k").vector_available is True
    assert make(tavily_api_key="k").web_available is True


def test_tracing_is_off_without_a_key_even_when_enabled():
    assert make(langchain_tracing_v2=True).tracing_enabled is False
    assert make(langchain_tracing_v2=True, langchain_api_key="k").tracing_enabled is True
    assert make(langchain_tracing_v2=False, langchain_api_key="k").tracing_enabled is False


def test_redaction_hides_every_secret():
    s = make(openai_api_key="sk-secret", anthropic_api_key="", tavily_api_key="tvly-secret")
    redacted = s.redacted()

    assert redacted["openai_api_key"] == "<set>"
    assert redacted["anthropic_api_key"] == "<unset>"
    assert redacted["tavily_api_key"] == "<set>"
    assert "sk-secret" not in str(redacted)
    assert "tvly-secret" not in str(redacted)
    # Non-secrets stay visible.
    assert redacted["qdrant_collection"] == "research"
