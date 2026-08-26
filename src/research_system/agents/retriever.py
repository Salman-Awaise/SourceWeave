"""Retriever: run the configured backends and fuse their ranked lists.

Fusion is Reciprocal Rank Fusion. Each (backend, sub-query) pair produces its
own ranked list; a document appearing in several lists accumulates score, which
is what makes agreement between backends count for more than any single native
score. Native scores are not comparable across Qdrant and Tavily, so they are
never summed directly -- only ranks are.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

from ..core.deps import Dependencies
from ..core.state import AgentState, RetrievedDocument
from ..errors import RetrievalError, VectorStoreError, WebSearchError

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedDocument]], k: int = 60, top_n: int | None = None
) -> list[RetrievedDocument]:
    """Fuse ranked lists: score = sum over lists of 1 / (k + rank + 1), rank 0-based.

    Inputs are never mutated; the returned documents are copies carrying the
    fused score, with the original backend score preserved in
    `metadata["native_score"]`.
    """
    fused_scores: dict[str, float] = {}
    representatives: OrderedDict[str, RetrievedDocument] = OrderedDict()
    appearances: dict[str, int] = {}

    for ranked in ranked_lists:
        for rank, document in enumerate(ranked):
            key = document.dedup_key
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            appearances[key] = appearances.get(key, 0) + 1
            if key not in representatives:
                representatives[key] = document

    ordered_keys = sorted(
        fused_scores,
        # Ties broken by first-seen order so fusion is deterministic.
        key=lambda key: (-fused_scores[key], list(representatives).index(key)),
    )

    results: list[RetrievedDocument] = []
    for key in ordered_keys:
        source_doc = representatives[key]
        metadata = dict(source_doc.metadata)
        metadata.setdefault("native_score", source_doc.score)
        metadata["rrf_score"] = fused_scores[key]
        metadata["rrf_appearances"] = appearances[key]
        results.append(
            RetrievedDocument(
                content=source_doc.content,
                source=source_doc.source,
                score=fused_scores[key],
                metadata=metadata,
            )
        )

    return results[:top_n] if top_n is not None else results


def deduplicate(documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
    """Keep the first occurrence of each document, preserving order."""
    seen: set[str] = set()
    out: list[RetrievedDocument] = []
    for document in documents:
        key = document.dedup_key
        if key not in seen:
            seen.add(key)
            out.append(document)
    return out


def _resolve_strategy(requested: str, deps: Dependencies, warnings: list[str]) -> tuple[bool, bool]:
    """Decide which backends to actually call.

    Hybrid degrades to whatever is available, with a warning. An explicit
    single-backend strategy whose backend is missing is a hard error -- silently
    answering from the web when the user asked for their indexed documents
    would be worse than failing.
    """
    strategy = (requested or "hybrid").lower()
    vector_ok = deps.vector_enabled
    web_ok = deps.web_enabled

    if strategy == "vector":
        if not vector_ok:
            raise RetrievalError(
                "Strategy 'vector' needs Qdrant plus OPENAI_API_KEY for embeddings, "
                "and one of them is not configured. Set them in .env, or use "
                "strategy 'web' or 'hybrid'."
            )
        return True, False

    if strategy == "web":
        if not web_ok:
            raise RetrievalError(
                "Strategy 'web' needs TAVILY_API_KEY, which is not configured. "
                "Set it in .env, or use strategy 'vector' or 'hybrid'."
            )
        return False, True

    # hybrid
    if not vector_ok and not web_ok:
        raise RetrievalError(
            "No retrieval backend is configured. Set OPENAI_API_KEY plus a reachable "
            "QDRANT_URL for document search, and/or TAVILY_API_KEY for web search."
        )
    if not vector_ok:
        warnings.append(
            "Hybrid retrieval: document search unavailable (Qdrant or OPENAI_API_KEY "
            "not configured); continuing with web search only."
        )
    if not web_ok:
        warnings.append(
            "Hybrid retrieval: web search unavailable (TAVILY_API_KEY not set); "
            "continuing with document search only."
        )
    return vector_ok, web_ok


def _vector_search(query: str, deps: Dependencies, warnings: list[str]) -> list[RetrievedDocument]:
    assert deps.vector_store is not None and deps.embeddings is not None
    settings = deps.settings
    try:
        vector = deps.embeddings.embed(query)
        return deps.vector_store.search(
            settings.qdrant_collection,
            vector,
            top_k=settings.similarity_top_k,
            score_threshold=settings.qdrant_score_threshold,
        )
    except VectorStoreError as exc:
        # A missing collection or a down Qdrant should not kill a hybrid run.
        warnings.append(f"Document search failed for {query!r}: {exc}")
        logger.warning("vector search failed: %s", exc)
        return []


def _web_search(query: str, deps: Dependencies, warnings: list[str]) -> list[RetrievedDocument]:
    assert deps.web_search is not None
    try:
        return deps.web_search.search(query, max_results=deps.settings.web_results_per_query)
    except WebSearchError as exc:
        warnings.append(f"Web search failed for {query!r}: {exc}")
        logger.warning("web search failed: %s", exc)
        return []


def retriever_node(state: AgentState, deps: Dependencies) -> AgentState:
    """Search every configured backend for every sub-query, then fuse.

    On a retry the verifier has already rewritten `sub_queries` and moved the
    previous evidence into `prior_documents`; that evidence is fused back in so
    a retry adds to the picture rather than replacing it.
    """
    settings = deps.settings
    warnings = list(state.get("warnings") or [])

    sub_queries = [q for q in (state.get("sub_queries") or []) if q.strip()]
    if not sub_queries:
        sub_queries = [state.get("original_query") or state.get("query", "")]
    sub_queries = sub_queries[: settings.max_sub_queries]

    use_vector, use_web = _resolve_strategy(
        state.get("retrieval_strategy", "hybrid"), deps, warnings
    )

    def gather(with_vector: bool, with_web: bool) -> list[list[RetrievedDocument]]:
        """One ranked list per (backend, sub-query) pair that returned hits."""
        lists: list[list[RetrievedDocument]] = []
        for query in sub_queries:
            if with_vector:
                hits = _vector_search(query, deps, warnings)
                if hits:
                    lists.append(hits)
            if with_web:
                hits = _web_search(query, deps, warnings)
                if hits:
                    lists.append(hits)
        return lists

    ranked_lists = gather(use_vector, use_web)

    # A strategy can succeed technically and still find nothing -- an indexed
    # corpus that simply does not cover the question returns zero hits once the
    # score threshold is applied. Answering "insufficient evidence" while a
    # configured backend sits unused is worse than trying it, and retrying the
    # same empty search would just repeat the miss.
    if not ranked_lists:
        spare_vector = deps.vector_enabled and not use_vector
        spare_web = deps.web_enabled and not use_web
        if spare_vector or spare_web:
            spare_name = "document search" if spare_vector else "web search"
            warnings.append(
                f"Strategy {state.get('retrieval_strategy', 'hybrid')!r} returned no "
                f"evidence; broadening to {spare_name}."
            )
            logger.info("empty retrieval, broadening to %s", spare_name)
            ranked_lists = gather(spare_vector, spare_web)

    # Prior evidence enters as its own ranked list, so documents that were
    # already relevant keep their standing instead of being discarded.
    prior_documents = list(state.get("prior_documents") or [])
    if prior_documents:
        ranked_lists.append(prior_documents)

    documents = reciprocal_rank_fusion(
        ranked_lists, k=settings.rrf_k, top_n=settings.max_retrieval_docs
    )

    if not documents:
        warnings.append(
            "Retrieval returned no documents. The answer will state that evidence is insufficient."
        )

    queries_used = list(
        dict.fromkeys(list(state.get("retrieval_queries_used") or []) + sub_queries)
    )

    logger.debug(
        "retriever: %d queries, %d lists, %d fused documents",
        len(sub_queries),
        len(ranked_lists),
        len(documents),
    )

    return AgentState(
        retrieved_documents=documents,
        retrieval_queries_used=queries_used,
        warnings=warnings,
    )
