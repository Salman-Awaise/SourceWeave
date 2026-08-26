"""Planner: decompose the question and choose a retrieval strategy."""

from __future__ import annotations

import logging

from ..core.deps import Dependencies
from ..core.state import AgentState
from ..errors import LLMError
from ..prompts import UNTRUSTED_INPUT_NOTICE, format_chat_history, format_memory_block
from ..schemas import PlannerOutput

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = (
    """You are a research planning agent. Your job is to analyze a user's question and:

1. Decompose it into 1-4 focused sub-questions that, when answered together, fully address the initial user query.
2. Determine the best retrieval strategy:
   - "vector": Question is about a specific domain where we have indexed documents
   - "web": Question requires real-time or recent information
   - "hybrid": Question benefits from both document retrieval and web search

Respond in JSON format:
{
    "sub_queries": ["sub-question 1", "sub-question 2", ...],
    "retrieval_strategy": "vector" | "web" | "hybrid",
    "reasoning": "Brief explanation of your decomposition strategy"
}

Keep sub-questions specific, focused, and non-overlapping. Each should be answerable independently.
"""
    + UNTRUSTED_INPUT_NOTICE
)


def _available_fallback_strategy(deps: Dependencies) -> str:
    """Pick a strategy we can actually serve when the planner output is unusable."""
    if deps.vector_enabled and deps.web_enabled:
        return "hybrid"
    if deps.vector_enabled:
        return "vector"
    if deps.web_enabled:
        return "web"
    return "hybrid"


def build_planner_prompt(state: AgentState, deps: Dependencies) -> str:
    """Assemble the user-side prompt with history and memory clearly delimited."""
    settings = deps.settings
    parts: list[str] = []

    memory_block = format_memory_block(state.get("memory_context") or [])
    if memory_block:
        parts.append(memory_block)

    history = format_chat_history(
        state.get("chat_history") or [], limit=settings.max_chat_history_messages
    )
    if history:
        parts.append(history)

    available: list[str] = []
    if deps.vector_enabled:
        available.append("vector (indexed documents)")
    if deps.web_enabled:
        available.append("web (live search)")
    if available:
        parts.append("Available retrieval backends: " + ", ".join(available) + ".")

    parts.append(f"<user_question>\n{state.get('original_query', '')}\n</user_question>")
    parts.append("Produce the plan as JSON.")
    return "\n\n".join(parts)


def planner_node(state: AgentState, deps: Dependencies) -> AgentState:
    """Produce `sub_queries` and `retrieval_strategy`.

    Invalid model output is never fatal: we fall back to a single sub-query
    equal to the original question, on a strategy the system can actually
    serve, and record a warning.
    """
    warnings = list(state.get("warnings") or [])
    original_query = state.get("original_query") or state.get("query", "")

    try:
        plan = deps.require_llm().complete_structured(
            system=PLANNER_SYSTEM_PROMPT,
            user=build_planner_prompt(state, deps),
            schema=PlannerOutput,
        )
        sub_queries = plan.sub_queries
        strategy = plan.retrieval_strategy
        logger.debug("planner: %d sub-queries, strategy=%s", len(sub_queries), strategy)
    except (LLMError, ValueError) as exc:
        strategy = _available_fallback_strategy(deps)
        sub_queries = [original_query]
        warnings.append(
            f"Planner output was unusable ({exc}); falling back to the original "
            f"question with the {strategy!r} strategy."
        )
        logger.warning("planner fallback: %s", exc)

    return AgentState(
        sub_queries=sub_queries[: deps.settings.max_sub_queries],
        retrieval_strategy=strategy,
        warnings=warnings,
    )
