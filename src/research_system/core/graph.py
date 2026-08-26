"""LangGraph assembly.

    plan -> retrieve -> generate -> verify -> END
                 ^                     |
                 +---------------------+
                      (unverified, retries remain)

The retry edge goes back to `retrieve`, not `plan`: the decomposition was fine,
it was the *evidence* that fell short, and the verifier has already written
sharper queries for it.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

from ..agents.generator import generator_node
from ..agents.planner import planner_node
from ..agents.retriever import retriever_node
from ..agents.verifier import route_after_verification, verifier_node
from .deps import Dependencies
from .state import AgentState

logger = logging.getLogger(__name__)


def _safe_node(name: str, func: Any, deps: Dependencies) -> Any:
    """Wrap a node so an unexpected crash becomes a typed result, not a hang.

    Expected failures are already handled inside each node. This catches genuine
    bugs and records them in `error`, which routes the graph to the end instead
    of leaving the run without an answer.
    """

    def wrapped(state: AgentState) -> AgentState:
        try:
            return func(state, deps)
        except Exception as exc:
            logger.exception("node %r failed", name)
            warnings = list(state.get("warnings") or [])
            return AgentState(
                error=f"{name} failed: {type(exc).__name__}: {exc}",
                is_verified=False,
                warnings=warnings,
            )

    wrapped.__name__ = f"{name}_node"
    return wrapped


def _route(state: AgentState, deps: Dependencies) -> str:
    # A hard error anywhere ends the run; retrying a bug does not fix it.
    if state.get("error"):
        return "respond"
    return route_after_verification(state, deps.settings)


def _continue_unless_error(state: AgentState, *, next_node: str) -> str:
    """Stop the run as soon as a node reports a hard error.

    Without this, a failed retrieval would flow on into generation and
    verification, and the later stages would overwrite the original error with a
    downstream symptom -- hiding the actual cause from the operator.
    """
    return "end" if state.get("error") else next_node


def build_graph(deps: Dependencies) -> Any:
    """Compile the four-stage graph with `deps` bound into every node."""
    from langgraph.graph import END, StateGraph

    builder = StateGraph(AgentState)

    builder.add_node("plan", _safe_node("plan", planner_node, deps))
    builder.add_node("retrieve", _safe_node("retrieve", retriever_node, deps))
    builder.add_node("generate", _safe_node("generate", generator_node, deps))
    builder.add_node("verify", _safe_node("verify", verifier_node, deps))

    builder.set_entry_point("plan")
    for source, target in (("plan", "retrieve"), ("retrieve", "generate"), ("generate", "verify")):
        builder.add_conditional_edges(
            source,
            partial(_continue_unless_error, next_node=target),
            {target: target, "end": END},
        )
    builder.add_conditional_edges(
        "verify",
        partial(_route, deps=deps),
        {"retrieve": "retrieve", "respond": END},
    )

    # The retry cap is enforced by the router; the recursion limit is a
    # backstop so a routing bug can never spin forever.
    return builder.compile()


def graph_recursion_limit(deps: Dependencies) -> int:
    """Steps needed for plan + (retries + 1) * (retrieve, generate, verify), plus slack."""
    return 4 + (deps.settings.max_verification_retries + 1) * 3 + 4
