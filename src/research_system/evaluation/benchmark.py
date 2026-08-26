"""Single-agent baseline: the control group for the multi-agent pipeline.

Deliberately holds everything constant except the architecture -- same model,
same temperature, same generation rules, same retrieval backends, same evidence
cap. What it drops is exactly the multi-agent machinery: no planner
decomposition (one query, the user's own words) and no verification retry loop.
A difference in scores can then be attributed to those two things.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..agents.generator import (
    GENERATOR_SYSTEM_PROMPT,
    INSUFFICIENT_EVIDENCE_ANSWER,
    validate_citations,
)
from ..agents.retriever import retriever_node
from ..core.deps import Dependencies, build_dependencies
from ..core.state import create_initial_state
from ..errors import LLMError, RetrievalError
from ..prompts import format_documents, truncate_context
from ..schemas import GeneratorOutput

logger = logging.getLogger(__name__)


class SingleAgentBaseline:
    """One retrieval pass, one generation call, no verification."""

    def __init__(self, deps: Dependencies | None = None, *, strategy: str = "hybrid") -> None:
        self.deps = deps or build_dependencies()
        self.strategy = strategy

    def run(self, query: str) -> dict[str, Any]:
        """Answer one question. Mirrors the public shape of `run_query`."""
        started = time.perf_counter()
        cleaned = (query or "").strip()
        if not cleaned:
            return {
                "response": "",
                "sources": [],
                "confidence": 0.0,
                "is_verified": False,
                "retry_count": 0,
                "error": "Query was empty.",
                "elapsed_s": 0.0,
                "_contexts": [],
            }

        # Retrieval reuses the real retriever node, so the baseline searches the
        # same backends with the same fusion and the same document cap.
        state = create_initial_state(cleaned)
        state["sub_queries"] = [cleaned]  # no decomposition: that is the point
        state["retrieval_strategy"] = self.strategy

        warnings: list[str] = []
        try:
            retrieval_update = retriever_node(state, self.deps)
        except RetrievalError as exc:
            return {
                "response": "",
                "sources": [],
                "confidence": 0.0,
                "is_verified": False,
                "retry_count": 0,
                "error": f"Retrieval failed: {exc}",
                "elapsed_s": round(time.perf_counter() - started, 3),
                "_contexts": [],
            }

        documents = retrieval_update.get("retrieved_documents") or []
        warnings.extend(retrieval_update.get("warnings") or [])

        if not documents:
            return {
                "response": INSUFFICIENT_EVIDENCE_ANSWER,
                "sources": [],
                "confidence": 0.0,
                "is_verified": False,
                "retry_count": 0,
                "warnings": warnings,
                "elapsed_s": round(time.perf_counter() - started, 3),
                "_contexts": [],
            }

        context = format_documents(documents)
        context, _ = truncate_context(context, self.deps.settings.max_context_chars)
        prompt = (
            f"<user_question>\n{cleaned}\n</user_question>\n\n"
            f"Context documents:\n\n{context}\n\n"
            "Answer the question as a single JSON object."
        )

        try:
            output = self.deps.require_llm().complete_structured(
                system=GENERATOR_SYSTEM_PROMPT,
                user=prompt,
                schema=GeneratorOutput,
            )
        except LLMError as exc:
            return {
                "response": "",
                "sources": [],
                "confidence": 0.0,
                "is_verified": False,
                "retry_count": 0,
                "error": f"Generation failed: {exc}",
                "warnings": warnings,
                "elapsed_s": round(time.perf_counter() - started, 3),
                "_contexts": [doc.content for doc in documents],
            }

        valid_indices, citation_warnings = validate_citations(output.sources_used, documents)
        warnings.extend(citation_warnings)
        if not valid_indices:
            valid_indices = list(range(1, len(documents) + 1))

        result: dict[str, Any] = {
            "response": output.answer,
            "sources": [documents[i - 1].to_source_entry(i) for i in valid_indices],
            "confidence": output.confidence,
            # The baseline has no verifier, so it makes no verification claim.
            "is_verified": False,
            "retry_count": 0,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "_contexts": [doc.content for doc in documents],
        }
        if warnings:
            result["warnings"] = warnings
        return result
