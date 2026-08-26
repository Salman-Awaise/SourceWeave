"""Verifier: claim-level fact-checking, and the retry that follows a failure.

The faithfulness score is always recomputed in code from the claim lists. A
model that says "0.95" while listing 3 of 10 claims as unsupported does not get
to set the score.
"""

from __future__ import annotations

import logging

from ..config import Settings
from ..core.deps import Dependencies
from ..core.state import AgentState, VerificationResult
from ..errors import LLMError
from ..prompts import UNTRUSTED_INPUT_NOTICE, format_documents, truncate_context
from ..schemas import MAX_SUB_QUERIES, RefinedQueries, VerifierOutput

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM_PROMPT = (
    """You are a fact-checking verification agent. Your job is to verify whether a generated answer is faithfully grounded in the provided source documents.

STEPS:
1. Extract individual factual claims from the answer
2. For each claim, check if it is supported by the source documents
3. Classify each claim as "supported" or "unsupported"
4. Compute a faithfulness score = supported_claims / total_claims

Respond in JSON format:
{
    "claims": ["claim 1", "claim 2", ...],
    "supported_claims": ["claim 1", ...],
    "unsupported_claims": ["claim 3", ...],
    "faithfulness_score": 0.0-1.0,
    "reasoning": "Brief explanation of your verification"
}

Be strict: a claim is only "supported" if there is clear evidence in the sources.
Do NOT count general knowledge claims - only source-grounded facts.
Every entry in supported_claims and unsupported_claims must also appear verbatim in claims.
"""
    + UNTRUSTED_INPUT_NOTICE
)

REFINE_SYSTEM_PROMPT = (
    """You are a retrieval refinement agent. An answer contained factual claims that the available sources did not support.

Write focused search queries that would find evidence for or against those specific
claims. Each query must target one claim, be self-contained, and use concrete
keywords rather than referring back to "the claim" or "the answer".

Respond in JSON format:
{
    "queries": ["query 1", "query 2", ...]
}

Return at most 4 queries. Do not repeat the original question verbatim.
"""
    + UNTRUSTED_INPUT_NOTICE
)

NO_RESPONSE_ERROR = "No response to verify"

# Claims about the present tend to fail against a stale document index, so a
# hybrid run that hits these is pushed toward live web search on the retry.
_RECENCY_MARKERS = (
    "current",
    "currently",
    "latest",
    "recent",
    "recently",
    "today",
    "now",
    "this year",
    "as of",
    "202",
    "upcoming",
    "new",
)


def compute_faithfulness(claims: list[str], supported: list[str], no_answer: bool = False) -> float:
    """Recompute the score from claim counts.

    With no claims the score is defined rather than left to the model: an
    explicit "insufficient evidence" answer is perfectly faithful (1.0), while
    an answer that simply produced no checkable claim is not (0.0).
    """
    if not claims:
        return 1.0 if no_answer else 0.0
    return len(supported) / len(claims)


def reconcile_claims(output: VerifierOutput) -> tuple[list[str], list[str], list[str]]:
    """Force the three claim lists into a consistent partition.

    Models routinely list a claim as supported without including it in `claims`,
    or classify a claim twice. Supported wins over unsupported on conflict, and
    any claim never classified counts as unsupported -- unverified is not the
    same as verified.
    """
    claims = list(dict.fromkeys(output.claims))
    supported_reported = {c.lower() for c in output.supported_claims}
    unsupported_reported = {c.lower() for c in output.unsupported_claims}

    # Absorb classified-but-unlisted claims instead of discarding evidence.
    for claim in list(output.supported_claims) + list(output.unsupported_claims):
        if claim not in claims:
            claims.append(claim)
    claims = list(dict.fromkeys(claims))

    supported: list[str] = []
    unsupported: list[str] = []
    for claim in claims:
        key = claim.lower()
        if key in supported_reported and key not in unsupported_reported:
            supported.append(claim)
        elif key in unsupported_reported:
            unsupported.append(claim)
        else:
            unsupported.append(claim)
    return claims, supported, unsupported


def refine_queries(
    unsupported_claims: list[str], original_query: str, deps: Dependencies
) -> tuple[list[str], list[str]]:
    """Turn unsupported claims into new search queries.

    Returns (queries, warnings). Falls back to using the claim text itself as a
    query, which is still strictly better than repeating the original search.
    """
    warnings: list[str] = []
    if not unsupported_claims:
        return [], warnings

    claim_block = "\n".join(f"- {claim}" for claim in unsupported_claims[:MAX_SUB_QUERIES])
    user_prompt = (
        f"<user_question>\n{original_query}\n</user_question>\n\n"
        f"Unsupported claims:\n{claim_block}\n\n"
        "Write the search queries as JSON."
    )

    try:
        output = deps.require_llm().complete_structured(
            system=REFINE_SYSTEM_PROMPT,
            user=user_prompt,
            schema=RefinedQueries,
            temperature=0.0,
        )
        if output.queries:
            return output.queries, warnings
        warnings.append("Query refinement returned no queries; using the claims directly.")
    except LLMError as exc:
        warnings.append(f"Query refinement failed ({exc}); using the unsupported claims directly.")
        logger.warning("refinement failed: %s", exc)

    return list(unsupported_claims[:MAX_SUB_QUERIES]), warnings


def _needs_web(claims: list[str]) -> bool:
    joined = " ".join(claims).lower()
    return any(marker in joined for marker in _RECENCY_MARKERS)


def verifier_node(state: AgentState, deps: Dependencies) -> AgentState:
    """Verify the answer; on failure, prepare a sharper retrieval pass."""
    settings = deps.settings
    warnings = list(state.get("warnings") or [])
    response = (state.get("response") or "").strip()
    documents = state.get("retrieved_documents") or []
    retry_count = int(state.get("retry_count", 0))

    # Nothing to verify: end safely rather than loop.
    if not response:
        return AgentState(
            verification=None,
            unsupported_claims=[],
            is_verified=False,
            retry_count=retry_count,
            error=state.get("error") or NO_RESPONSE_ERROR,
            warnings=warnings,
        )

    context = format_documents(documents)
    context, _ = truncate_context(context, settings.max_context_chars)
    user_prompt = (
        f"<user_question>\n{state.get('original_query', '')}\n</user_question>\n\n"
        f"Answer to verify:\n{response}\n\n"
        f"Source documents:\n\n{context}\n\n"
        "Verify the answer as JSON."
    )

    try:
        output = deps.require_llm().complete_structured(
            system=VERIFIER_SYSTEM_PROMPT,
            user=user_prompt,
            schema=VerifierOutput,
            temperature=0.0,
        )
    except LLMError as exc:
        # A verifier outage must not be reported as a verified answer.
        logger.error("verification failed: %s", exc)
        warnings.append(f"Verification could not run ({exc}); the answer is unverified.")
        return AgentState(
            verification=None,
            unsupported_claims=[],
            is_verified=False,
            retry_count=retry_count,
            warnings=warnings,
        )

    claims, supported, unsupported = reconcile_claims(output)
    no_answer = not documents
    score = compute_faithfulness(claims, supported, no_answer=no_answer)

    if claims and abs(score - output.faithfulness_score) > 0.01:
        warnings.append(
            f"Verifier reported faithfulness {output.faithfulness_score:.2f} but the "
            f"claim counts give {score:.2f}; using the computed value."
        )

    verification = VerificationResult(
        faithfulness_score=score,
        claims=claims,
        supported_claims=supported,
        unsupported_claims=unsupported,
        reasoning=output.reasoning,
    )
    is_verified = score >= settings.faithfulness_threshold

    logger.debug(
        "verifier: score=%.2f threshold=%.2f verified=%s (%d/%d claims)",
        score,
        settings.faithfulness_threshold,
        is_verified,
        len(supported),
        len(claims),
    )

    update = AgentState(
        verification=verification,
        unsupported_claims=unsupported,
        is_verified=is_verified,
        warnings=warnings,
    )

    if is_verified:
        update["retry_count"] = retry_count
        return update

    # Failed. Count the attempt, and if budget remains, aim the next retrieval
    # at the claims that actually failed.
    attempts_used = retry_count + 1
    update["retry_count"] = attempts_used

    # Refine only when the router will actually route back to `retrieve`; it
    # stops once attempts_used reaches the cap, so refining then would spend a
    # model call whose result is discarded.
    if attempts_used < settings.max_verification_retries:
        queries, refine_warnings = refine_queries(
            unsupported, state.get("original_query", ""), deps
        )
        warnings.extend(refine_warnings)
        if queries:
            update["sub_queries"] = queries
            # Carry the current evidence forward so the retry adds to it.
            update["prior_documents"] = list(documents)
            if state.get("retrieval_strategy") == "hybrid" and _needs_web(unsupported):
                warnings.append(
                    "Unsupported claims look time-sensitive; prioritising web search on retry."
                )
        update["warnings"] = warnings
    else:
        warnings.append(
            f"Answer remains unverified after {settings.max_verification_retries} "
            f"retrieval retries (faithfulness {score:.2f} < "
            f"{settings.faithfulness_threshold:.2f}). Treat it with caution."
        )
        update["warnings"] = warnings

    return update


def route_after_verification(state: AgentState, settings: Settings) -> str:
    """Conditional edge: 'respond' to finish, 'retrieve' to try again."""
    if state.get("error") == NO_RESPONSE_ERROR:
        return "respond"
    if state.get("is_verified"):
        return "respond"
    if int(state.get("retry_count", 0)) >= settings.max_verification_retries:
        return "respond"
    if not state.get("unsupported_claims"):
        # Nothing concrete to search for; retrying would repeat the same query.
        return "respond"
    return "retrieve"
