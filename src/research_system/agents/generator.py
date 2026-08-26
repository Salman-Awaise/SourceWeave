"""Generator: synthesize a source-cited answer from the retrieved evidence."""

from __future__ import annotations

import logging

from ..core.deps import Dependencies
from ..core.state import AgentState, RetrievedDocument
from ..errors import LLMError
from ..prompts import (
    UNTRUSTED_INPUT_NOTICE,
    format_chat_history,
    format_documents,
    truncate_context,
)
from ..schemas import GeneratorOutput

logger = logging.getLogger(__name__)

GENERATOR_SYSTEM_PROMPT = (
    """You are a research synthesis agent. Given retrieved context documents, generate a comprehensive answer to the user's question.

RULES:
1. Only use information from the provided context documents
2. Cite sources using [Source N] notation for every factual claim
3. If the context doesn't contain enough information, say so explicitly
4. Do NOT fabricate information not present in the sources
5. When an indexed document and a web page both cover the same fact, cite the
   document: it is the primary source and the web page is usually a summary of
   it. Cite the web page only for facts the documents do not contain.
6. Answer the question that was asked. If a source discusses a related but
   different subject, do not substitute it for the one asked about.

Respond with a single JSON object and nothing else:
{
    "answer": "your source-cited answer text",
    "confidence": 0.0-1.0,
    "sources_used": [1, 2, ...],
    "gaps": ["any information gaps you identified"],
    "answered": true
}

"answer" holds the full prose answer including its [Source N] citations.
"sources_used" lists only source numbers that actually appear in your answer.
"confidence" reflects how well the sources support the answer, not how sure you
are of the topic in general.
"answered" is false when the context does not actually contain the information
needed, and your answer is therefore a statement that the evidence is
insufficient rather than a real answer. Set it false whenever you decline.
"""
    + UNTRUSTED_INPUT_NOTICE
)

INSUFFICIENT_EVIDENCE_ANSWER = (
    "I could not find any evidence to answer this question. No documents were "
    "retrieved from the configured sources, so there is nothing to cite. "
    "Try ingesting relevant documents, enabling web search, or rephrasing the question."
)


def validate_citations(
    indices: list[int], documents: list[RetrievedDocument]
) -> tuple[list[int], list[str]]:
    """Drop citation indices that do not point at real evidence.

    A model-reported source list is never trusted: an out-of-range index would
    otherwise become a citation to a document that does not exist.
    """
    warnings: list[str] = []
    valid: list[int] = []
    seen: set[int] = set()
    invalid: list[int] = []

    for index in indices:
        if 1 <= index <= len(documents):
            if index not in seen:
                seen.add(index)
                valid.append(index)
        else:
            invalid.append(index)

    if invalid:
        warnings.append(
            f"Generator cited non-existent sources {sorted(set(invalid))}; "
            f"only sources 1-{len(documents)} exist. They were dropped."
        )
    return valid, warnings


def memory_not_evidence_note(state: AgentState) -> str:
    """Explain, when declining, that recalled memory is deliberately not evidence.

    Memory reaches the planner only -- it steers *what* gets searched. It is
    never passed to this agent, because mem0 stores a model-written paraphrase
    of a conversation, and asserting facts from that would break the guarantee
    that every claim traces to a retrievable source. Without this note the
    decline looks like memory is broken rather than deliberately excluded.
    """
    count = len(state.get("memory_context") or [])
    if not count:
        return ""
    return (
        f" {count} memory/memories were recalled for this user, but remembered "
        "conversation is not used as evidence and cannot be cited."
    )


def build_generator_prompt(state: AgentState, deps: Dependencies) -> tuple[str, list[str]]:
    """Assemble the evidence prompt. Returns (prompt, warnings)."""
    warnings: list[str] = []
    documents = state.get("retrieved_documents") or []

    context = format_documents(documents)
    context, was_truncated = truncate_context(context, deps.settings.max_context_chars)
    if was_truncated:
        warnings.append(
            f"Evidence exceeded the {deps.settings.max_context_chars}-character budget "
            "and was truncated."
        )

    parts: list[str] = []
    history = format_chat_history(state.get("chat_history") or [], limit=3)
    if history:
        parts.append(history)

    parts.append(f"<user_question>\n{state.get('original_query', '')}\n</user_question>")
    parts.append(f"Context documents:\n\n{context}")
    parts.append("Answer the question as a single JSON object.")
    return "\n\n".join(parts), warnings


def generator_node(state: AgentState, deps: Dependencies) -> AgentState:
    """Write the answer and attach validated citations."""
    warnings = list(state.get("warnings") or [])
    documents = state.get("retrieved_documents") or []

    # No evidence: answer honestly rather than asking the model to improvise.
    if not documents:
        note = memory_not_evidence_note(state)
        if note:
            warnings.append("No documents were retrieved." + note)
        return AgentState(
            response=INSUFFICIENT_EVIDENCE_ANSWER,
            sources=[],
            confidence=0.0,
            gaps=["No evidence was retrieved for this question."],
            answered=False,
            warnings=warnings,
        )

    prompt, prompt_warnings = build_generator_prompt(state, deps)
    warnings.extend(prompt_warnings)

    try:
        output = deps.require_llm().complete_structured(
            system=GENERATOR_SYSTEM_PROMPT,
            user=prompt,
            schema=GeneratorOutput,
        )
    except LLMError as exc:
        logger.error("generation failed: %s", exc)
        return AgentState(
            response="",
            sources=[],
            confidence=0.0,
            gaps=[],
            answered=False,
            error=f"Answer generation failed: {exc}",
            warnings=warnings,
        )

    # The model declined: evidence was retrieved but none of it covers the
    # question. Listing those documents as `sources` would imply they support an
    # answer that does not exist, so they are reported as retrieved-only.
    if not output.answered:
        warnings.append(
            f"The retrieved evidence does not cover this question "
            f"({len(documents)} document(s) retrieved, none relevant)."
            + memory_not_evidence_note(state)
        )
        return AgentState(
            response=output.answer,
            sources=[],
            confidence=min(output.confidence, 0.1),
            gaps=output.gaps or ["The retrieved evidence does not cover this question."],
            answered=False,
            warnings=warnings,
        )

    valid_indices, citation_warnings = validate_citations(output.sources_used, documents)
    warnings.extend(citation_warnings)

    if not valid_indices:
        # The model answered but failed to report usable citations. Cite
        # everything it was shown, so the verifier sees the same evidence set and
        # the answer stays traceable.
        valid_indices = list(range(1, len(documents) + 1))
        warnings.append("Generator reported no usable citations; all retrieved sources are listed.")

    sources = [documents[index - 1].to_source_entry(index) for index in valid_indices]

    logger.debug(
        "generator: %d chars, confidence=%.2f, %d sources",
        len(output.answer),
        output.confidence,
        len(sources),
    )

    return AgentState(
        response=output.answer,
        sources=sources,
        confidence=output.confidence,
        gaps=output.gaps,
        answered=True,
        warnings=warnings,
    )
