"""Domain records and the LangGraph state.

`RetrievedDocument` and `VerificationResult` are the two records that travel the
whole pipeline, so both carry their own validation. Everything an LLM produces
is validated before it reaches state -- see `research_system.schemas`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from typing import Any, TypedDict

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse whitespace and case so trivially-different copies dedupe."""
    return _WHITESPACE.sub(" ", text or "").strip().lower()


def canonical_source(source: str) -> str:
    """Normalize a source label so the same doc from two paths collapses to one.

    URLs lose their scheme, `www.`, trailing slash and fragment; file paths keep
    only their basename, since the same file reached via a relative and an
    absolute path is the same evidence.
    """
    s = (source or "").strip()
    if not s:
        return "unknown"
    lowered = s.lower()
    if lowered.startswith(("http://", "https://")):
        without_scheme = lowered.split("://", 1)[1]
        without_scheme = without_scheme.split("#", 1)[0]
        if without_scheme.startswith("www."):
            without_scheme = without_scheme[4:]
        return without_scheme.rstrip("/")
    return s.rsplit("/", 1)[-1]


@dataclass
class RetrievedDocument:
    """One piece of evidence.

    `score` holds the backend's native score until fusion, and the RRF score
    afterwards; the native value is preserved under `metadata["native_score"]`
    so provenance survives ranking.
    """

    content: str
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dedup_key(self) -> str:
        """Stable identity: normalized content + canonical source.

        Hashing the *full* normalized content (not a prefix) keeps two chunks
        that merely share an opening sentence from colliding.
        """
        payload = f"{canonical_source(self.source)}\x00{normalize_text(self.content)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def source_type(self) -> str:
        return str(self.metadata.get("source_type", "unknown"))

    def copy_with(self, **changes: Any) -> RetrievedDocument:
        """Non-mutating update; the metadata dict is copied, not shared."""
        base = replace(self, metadata=dict(self.metadata))
        for key, value in changes.items():
            setattr(base, key, value)
        return base

    def to_source_entry(self, index: int) -> dict[str, Any]:
        """Public citation entry for the `sources` list of a result."""
        entry: dict[str, Any] = {
            "index": index,
            "source": self.source,
            "score": round(float(self.score), 6),
        }
        for key in ("title", "page_label", "source_type", "file_path", "native_score"):
            if key in self.metadata and self.metadata[key] not in (None, ""):
                entry[key] = self.metadata[key]
        return entry


@dataclass
class VerificationResult:
    """Claim-level faithfulness assessment of a generated answer."""

    faithfulness_score: float
    claims: list[str]
    supported_claims: list[str]
    unsupported_claims: list[str]
    reasoning: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "faithfulness_score": round(self.faithfulness_score, 4),
            "total_claims": len(self.claims),
            "supported_claims": len(self.supported_claims),
            "unsupported_claims": len(self.unsupported_claims),
        }


class AgentState(TypedDict, total=False):
    """LangGraph state. `total=False` so nodes may return partial updates."""

    # --- owned by the pipeline runner ---
    query: str
    original_query: str
    chat_history: list[dict[str, str]]
    memory_context: list[str]

    # --- owned by the planner (sub_queries refined by the verifier) ---
    sub_queries: list[str]
    retrieval_strategy: str

    # --- owned by the retriever ---
    retrieved_documents: list[RetrievedDocument]
    prior_documents: list[RetrievedDocument]
    retrieval_queries_used: list[str]

    # --- owned by the generator ---
    response: str
    sources: list[dict[str, Any]]
    confidence: float
    gaps: list[str]
    answered: bool
    """False when the response declines for lack of evidence rather than answering."""

    # --- owned by the verifier ---
    verification: VerificationResult | None
    unsupported_claims: list[str]
    is_verified: bool
    retry_count: int

    # --- owned by runner and nodes ---
    error: str | None
    warnings: list[str]
    trace_id: str | None


def create_initial_state(
    query: str,
    *,
    chat_history: list[dict[str, str]] | None = None,
    memory_context: list[str] | None = None,
    trace_id: str | None = None,
) -> AgentState:
    """Build a fully-populated state so no node has to guess a default."""
    return AgentState(
        query=query,
        original_query=query,
        chat_history=list(chat_history or []),
        memory_context=list(memory_context or []),
        sub_queries=[],
        retrieval_strategy="hybrid",
        retrieved_documents=[],
        prior_documents=[],
        retrieval_queries_used=[],
        response="",
        sources=[],
        confidence=0.0,
        gaps=[],
        answered=False,
        verification=None,
        unsupported_claims=[],
        is_verified=False,
        retry_count=0,
        error=None,
        warnings=[],
        trace_id=trace_id,
    )


def build_public_result(state: AgentState) -> dict[str, Any]:
    """Project state onto the stable public contract of `run_query`.

    `response`, `sources`, `confidence`, `is_verified` and `retry_count` are
    always present. `verification` appears only when verification actually ran,
    and `error` only when something failed.
    """
    result: dict[str, Any] = {
        "response": state.get("response", ""),
        "sources": list(state.get("sources", [])),
        "confidence": round(float(state.get("confidence", 0.0)), 4),
        "is_verified": bool(state.get("is_verified", False)),
        "retry_count": int(state.get("retry_count", 0)),
        # Distinguishes a real answer from a declined one. A declined response
        # can still be `is_verified` -- "the evidence does not cover this" is a
        # truthful statement -- so callers must not read verification alone as
        # meaning the question was answered.
        "answered": bool(state.get("answered", False)),
        "documents_retrieved": len(state.get("retrieved_documents") or []),
    }

    verification = state.get("verification")
    if verification is not None:
        result["verification"] = verification.to_public_dict()

    warnings = state.get("warnings") or []
    if warnings:
        # Preserve order, drop repeats from retry loops.
        result["warnings"] = list(dict.fromkeys(warnings))

    error = state.get("error")
    if error:
        result["error"] = error

    return result
