"""Planner: valid plans, fallbacks, and history truncation."""

from __future__ import annotations

from research_system.adapters.llm import FakeLLMClient
from research_system.agents.planner import build_planner_prompt, planner_node
from research_system.core.state import create_initial_state
from research_system.errors import StructuredOutputError
from research_system.schemas import PlannerOutput


def test_valid_plan_produces_queries_and_strategy(deps):
    deps.llm = FakeLLMClient(
        [PlannerOutput(sub_queries=["a", "b", "c"], retrieval_strategy="web", reasoning="r")]
    )
    update = planner_node(create_initial_state("q"), deps)

    assert update["sub_queries"] == ["a", "b", "c"]
    assert update["retrieval_strategy"] == "web"
    assert update["warnings"] == []


def test_sub_queries_are_capped_at_four(deps):
    deps.llm = FakeLLMClient(
        [{"sub_queries": ["a", "b", "c", "d", "e", "f"], "retrieval_strategy": "hybrid"}]
    )
    update = planner_node(create_initial_state("q"), deps)

    assert len(update["sub_queries"]) == 4


def test_blank_and_duplicate_sub_queries_are_dropped(deps):
    deps.llm = FakeLLMClient(
        [{"sub_queries": ["a", "  ", "A", "b"], "retrieval_strategy": "hybrid"}]
    )
    update = planner_node(create_initial_state("q"), deps)

    assert update["sub_queries"] == ["a", "b"]


def test_invalid_json_falls_back_to_the_original_question(deps):
    deps.llm = FakeLLMClient([StructuredOutputError("not json")])
    update = planner_node(create_initial_state("what is rrf?"), deps)

    assert update["sub_queries"] == ["what is rrf?"]
    assert update["retrieval_strategy"] == "hybrid"
    assert any("unusable" in w for w in update["warnings"])


def test_invalid_strategy_falls_back(deps):
    deps.llm = FakeLLMClient([StructuredOutputError("bad strategy")])
    update = planner_node(create_initial_state("q"), deps)

    assert update["retrieval_strategy"] in ("vector", "web", "hybrid")
    assert update["warnings"]


def test_fallback_picks_the_only_available_backend(deps):
    deps.web_search = None  # only vector remains
    deps.llm = FakeLLMClient([StructuredOutputError("bad")])
    update = planner_node(create_initial_state("q"), deps)

    assert update["retrieval_strategy"] == "vector"


def test_fallback_picks_web_when_vector_is_unavailable(deps):
    deps.vector_store = None
    deps.llm = FakeLLMClient([StructuredOutputError("bad")])
    update = planner_node(create_initial_state("q"), deps)

    assert update["retrieval_strategy"] == "web"


def test_excess_chat_history_is_truncated(deps):
    history = [{"role": "user", "content": f"m{i}"} for i in range(12)]
    state = create_initial_state("q", chat_history=history)

    prompt = build_planner_prompt(state, deps)

    assert "m11" in prompt
    assert "m0" not in prompt  # only the last max_chat_history_messages survive


def test_memory_is_delimited_and_not_merged_into_the_question(deps):
    state = create_initial_state("q", memory_context=["prefers bullet points"])
    prompt = build_planner_prompt(state, deps)

    assert "<memory>" in prompt
    assert "prefers bullet points" in prompt
    assert "<user_question>\nq\n</user_question>" in prompt


def test_prompt_lists_only_available_backends(deps):
    deps.web_search = None
    prompt = build_planner_prompt(create_initial_state("q"), deps)

    assert "vector" in prompt
    assert "web (live search)" not in prompt


def test_prompt_instructs_the_model_to_resolve_pronouns():
    """Sub-queries are searched independently, so back-references must be resolved.

    A live run decomposed "which retriever does RAG use, and what accuracy does
    it reach" into a sub-query about RAG's accuracy rather than the retriever's,
    so the primary source was never retrieved.
    """
    from research_system.agents.planner import PLANNER_SYSTEM_PROMPT

    assert "resolve every pronoun" in PLANNER_SYSTEM_PROMPT
    assert "stand alone" in PLANNER_SYSTEM_PROMPT
