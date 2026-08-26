"""Validated schemas for every structured LLM output.

Each agent asks the model for JSON matching one of these models. Bounds are
enforced here (claim counts, confidence range, strategy enum) so that a
malformed or adversarial model response can never reach `AgentState`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .config import VALID_STRATEGIES

MAX_SUB_QUERIES = 4


class PlannerOutput(BaseModel):
    """Planner: decomposition + backend choice."""

    sub_queries: list[str] = Field(min_length=1, max_length=MAX_SUB_QUERIES)
    retrieval_strategy: str
    reasoning: str = ""

    @field_validator("sub_queries", mode="before")
    @classmethod
    def _clean_sub_queries(cls, v: object) -> object:
        """Drop blanks and near-duplicates before length validation.

        Models often pad the list to hit a count, so trimming first turns what
        would be a hard schema failure into a usable plan.
        """
        if not isinstance(v, list):
            return v
        seen: set[str] = set()
        cleaned: list[str] = []
        for item in v:
            if not isinstance(item, str):
                continue
            text = item.strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                cleaned.append(text)
        return cleaned[:MAX_SUB_QUERIES]

    @field_validator("retrieval_strategy", mode="before")
    @classmethod
    def _check_strategy(cls, v: object) -> str:
        text = str(v or "").strip().lower()
        if text not in VALID_STRATEGIES:
            raise ValueError(f"retrieval_strategy must be one of {VALID_STRATEGIES}, got {v!r}")
        return text


class GeneratorOutput(BaseModel):
    """Generator: the answer plus its self-reported citations.

    `answered` separates "I answered from the evidence" from "I declined because
    the evidence does not cover this". Both are legitimate outputs, but only the
    first should be presented as an answer. Defaults to True so that a model
    which omits the field is treated as having answered.
    """

    answer: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    sources_used: list[int] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    answered: bool = True

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: object) -> float:
        """Clamp rather than reject: a 1-5 style score shouldn't lose the answer."""
        try:
            value = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.5
        return min(1.0, max(0.0, value))

    @field_validator("sources_used", mode="before")
    @classmethod
    def _coerce_indices(cls, v: object) -> list[int]:
        if not isinstance(v, list):
            return []
        out: list[int] = []
        for item in v:
            try:
                out.append(int(item))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
        return out

    @field_validator("gaps", mode="before")
    @classmethod
    def _coerce_gaps(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            return []
        return [str(item).strip() for item in v if str(item).strip()]


class VerifierOutput(BaseModel):
    """Verifier: extracted claims and their support status.

    The reported `faithfulness_score` is accepted here but recomputed from the
    claim lists downstream -- see `research_system.agents.verifier`.
    """

    claims: list[str] = Field(default_factory=list)
    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    faithfulness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""

    @field_validator("claims", "supported_claims", "unsupported_claims", mode="before")
    @classmethod
    def _clean_claims(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            return []
        return [str(item).strip() for item in v if str(item).strip()]

    @field_validator("faithfulness_score", mode="before")
    @classmethod
    def _clamp_score(cls, v: object) -> float:
        try:
            value = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0
        return min(1.0, max(0.0, value))


class RefinedQueries(BaseModel):
    """Verifier-driven retrieval refinement."""

    queries: list[str] = Field(default_factory=list, max_length=MAX_SUB_QUERIES)

    @field_validator("queries", mode="before")
    @classmethod
    def _clean(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            return []
        seen: set[str] = set()
        out: list[str] = []
        for item in v:
            text = str(item).strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                out.append(text)
        return out[:MAX_SUB_QUERIES]
