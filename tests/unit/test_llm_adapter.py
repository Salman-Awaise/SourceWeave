"""Structured output: JSON location, validation, and retry classification."""

from __future__ import annotations

import pytest

from research_system.adapters.llm import (
    _extract_json_object,
    _is_retryable,
    normalize_model_name,
    parse_structured,
)
from research_system.errors import StructuredOutputError
from research_system.schemas import GeneratorOutput, PlannerOutput


# --- JSON location ---------------------------------------------------------
def test_bare_json_is_parsed():
    out = parse_structured('{"sub_queries": ["a"], "retrieval_strategy": "web"}', PlannerOutput)
    assert out.sub_queries == ["a"]


def test_json_wrapped_in_prose_is_located():
    text = (
        "Sure! Here is the plan:\n"
        '{"sub_queries": ["a"], "retrieval_strategy": "web"}\n'
        "Hope that helps."
    )
    assert parse_structured(text, PlannerOutput).retrieval_strategy == "web"


def test_json_in_a_markdown_fence_is_located():
    text = '```json\n{"sub_queries": ["a"], "retrieval_strategy": "vector"}\n```'
    assert parse_structured(text, PlannerOutput).retrieval_strategy == "vector"


def test_braces_inside_strings_do_not_break_extraction():
    text = '{"answer": "use {this} literally", "confidence": 0.5}'
    assert parse_structured(text, GeneratorOutput).answer == "use {this} literally"


def test_escaped_quotes_do_not_break_extraction():
    text = r'{"answer": "she said \"hi\" loudly", "confidence": 0.5}'
    assert "hi" in parse_structured(text, GeneratorOutput).answer


def test_nested_objects_are_extracted_whole():
    extracted = _extract_json_object('prefix {"a": {"b": {"c": 1}}} suffix')
    assert extracted == '{"a": {"b": {"c": 1}}}'


def test_no_json_raises():
    with pytest.raises(StructuredOutputError, match="no JSON object"):
        _extract_json_object("I cannot help with that.")


def test_unterminated_json_raises():
    with pytest.raises(StructuredOutputError, match="unterminated"):
        _extract_json_object('{"a": 1')


def test_schema_violation_raises_rather_than_returning_junk():
    # 5 sub-queries exceeds the maximum of 4.
    text = '{"sub_queries": ["a","b","c","d","e"], "retrieval_strategy": "nope"}'
    with pytest.raises(StructuredOutputError, match="PlannerOutput"):
        parse_structured(text, PlannerOutput)


def test_prose_only_response_raises():
    with pytest.raises(StructuredOutputError):
        parse_structured("I refuse to answer.", PlannerOutput)


# --- model naming ----------------------------------------------------------
@pytest.mark.parametrize(
    "model,provider,expected",
    [
        ("claude-sonnet-4-20250514", "anthropic", "anthropic/claude-sonnet-4-20250514"),
        ("gpt-4o", "openai", "openai/gpt-4o"),
        ("anthropic/claude-3-5-haiku", "anthropic", "anthropic/claude-3-5-haiku"),
        ("local-model", "unknown", "local-model"),
    ],
)
def test_model_names_are_normalized_once(model, provider, expected):
    assert normalize_model_name(model, provider) == expected


# --- retry classification --------------------------------------------------
@pytest.mark.parametrize(
    "message",
    [
        "rate limit exceeded",
        "429 too many requests",
        "503 service unavailable",
        "connection reset",
        "request timed out",
        "server overloaded",
    ],
)
def test_transient_failures_are_retryable(message):
    assert _is_retryable(Exception(message)) is True


@pytest.mark.parametrize(
    "message",
    ["invalid api key", "401 unauthorized", "403 forbidden", "invalid request: bad schema"],
)
def test_auth_and_validation_failures_are_not_retried(message):
    assert _is_retryable(Exception(message)) is False


def test_auth_error_class_is_not_retried():
    class AuthenticationError(Exception):
        pass

    assert _is_retryable(AuthenticationError("rate limit")) is False


# --- fake client -----------------------------------------------------------
def test_fake_client_returns_scripted_models(llm):
    llm.responses = [PlannerOutput(sub_queries=["a"], retrieval_strategy="web")]
    out = llm.complete_structured(system="s", user="u", schema=PlannerOutput)

    assert out.sub_queries == ["a"]
    assert llm.calls[0]["kind"] == "structured"


def test_fake_client_raises_scripted_exceptions(llm):
    llm.responses = [StructuredOutputError("bad")]

    with pytest.raises(StructuredOutputError):
        llm.complete_structured(system="s", user="u", schema=PlannerOutput)


def test_fake_client_fails_loudly_when_exhausted(llm):
    with pytest.raises(AssertionError, match="ran out of responses"):
        llm.complete_structured(system="s", user="u", schema=PlannerOutput)
