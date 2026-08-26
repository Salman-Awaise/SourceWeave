"""The high-level entry point: `run_query`.

Wraps the graph with optional per-user memory and returns the stable public
result contract. Memory never blocks a query -- a memory failure degrades to a
warning.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from ..adapters.memory import MemoryStore, NullMemoryStore
from ..config import Settings, get_settings
from .deps import Dependencies, build_dependencies
from .graph import build_graph, graph_recursion_limit
from .state import AgentState, build_public_result, create_initial_state

logger = logging.getLogger(__name__)


class ResearchPipeline:
    """Reusable pipeline. Build once, call `run` many times.

    The compiled graph and every adapter are shared across calls, so the CLI
    does not rebuild clients per question.
    """

    def __init__(self, deps: Dependencies | None = None, *, use_memory: bool = False) -> None:
        self.deps = deps or build_dependencies(use_memory=use_memory)
        self.settings: Settings = self.deps.settings
        self._graph = build_graph(self.deps)
        self._recursion_limit = graph_recursion_limit(self.deps)

    # --- memory ----------------------------------------------------------
    @property
    def memory(self) -> MemoryStore:
        return self.deps.memory

    def _recall(self, query: str, user_id: str, use_memory: bool, warnings: list[str]) -> list[str]:
        if not use_memory or isinstance(self.memory, NullMemoryStore):
            return []
        if not (user_id or "").strip():
            warnings.append("Memory was requested but no user_id was given; skipping memory.")
            return []
        try:
            return self.memory.search(query, user_id, limit=self.settings.memory_search_limit)
        except Exception as exc:
            warnings.append(f"Memory lookup failed ({exc}); continuing without it.")
            logger.warning("memory recall failed: %s", exc)
            return []

    def _remember(
        self, query: str, response: str, user_id: str, use_memory: bool, warnings: list[str]
    ) -> None:
        if not use_memory or isinstance(self.memory, NullMemoryStore):
            return
        if not (user_id or "").strip() or not response.strip():
            return
        try:
            self.memory.add(
                [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response},
                ],
                user_id,
                metadata={"source": "research_system"},
            )
        except Exception as exc:
            warnings.append(f"Memory write failed ({exc}); the answer is unaffected.")
            logger.warning("memory write failed: %s", exc)

    # --- main ------------------------------------------------------------
    def run(
        self,
        query: str,
        *,
        user_id: str = "default",
        chat_history: list[dict[str, str]] | None = None,
        use_memory: bool = False,
    ) -> dict[str, Any]:
        """Answer one question end to end.

        Always returns the public contract, even on failure, so callers never
        have to guess whether they got a result or an exception.
        """
        warnings: list[str] = []
        cleaned = (query or "").strip()
        if not cleaned:
            return {
                "response": "",
                "sources": [],
                "confidence": 0.0,
                "is_verified": False,
                "retry_count": 0,
                "error": "Query was empty.",
            }

        if len(cleaned) > self.settings.max_query_chars:
            cleaned = cleaned[: self.settings.max_query_chars]
            warnings.append(
                f"Query exceeded {self.settings.max_query_chars} characters and was truncated."
            )

        memory_context = self._recall(cleaned, user_id, use_memory, warnings)
        if memory_context:
            logger.debug("recalled %d memories for user %r", len(memory_context), user_id)

        trace_id = str(uuid.uuid4())
        state = create_initial_state(
            cleaned,
            chat_history=chat_history,
            memory_context=memory_context,
            trace_id=trace_id,
        )
        state["warnings"] = warnings

        started = time.perf_counter()
        try:
            final_state: AgentState = self._graph.invoke(
                state, config={"recursion_limit": self._recursion_limit}
            )
        except Exception as exc:
            logger.exception("pipeline failed")
            return {
                "response": "",
                "sources": [],
                "confidence": 0.0,
                "is_verified": False,
                "retry_count": 0,
                "error": f"Pipeline failed: {type(exc).__name__}: {exc}",
                "warnings": warnings,
                "elapsed_s": round(time.perf_counter() - started, 3),
            }

        elapsed = time.perf_counter() - started
        result = build_public_result(final_state)
        result["elapsed_s"] = round(elapsed, 3)
        result["trace_id"] = final_state.get("trace_id") or trace_id

        self._remember(
            cleaned,
            result.get("response", ""),
            user_id,
            use_memory,
            result.setdefault("warnings", []),
        )
        if not result["warnings"]:
            result.pop("warnings")

        # Retrieved text is needed by the evaluator but is not part of the
        # public contract, so it travels under a private key.
        result["_contexts"] = [
            doc.content for doc in (final_state.get("retrieved_documents") or [])
        ]

        logger.info(
            "query complete in %.2fs (verified=%s, retries=%d)",
            elapsed,
            result["is_verified"],
            result["retry_count"],
        )
        return result


def run_query(
    query: str,
    *,
    user_id: str = "default",
    chat_history: list[dict[str, str]] | None = None,
    use_memory: bool = False,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """One-shot convenience wrapper. Builds a pipeline per call.

    Prefer `ResearchPipeline` for repeated queries -- it reuses clients.
    """
    settings = settings or get_settings()
    deps = build_dependencies(settings, use_memory=use_memory)
    pipeline = ResearchPipeline(deps)
    return pipeline.run(query, user_id=user_id, chat_history=chat_history, use_memory=use_memory)
