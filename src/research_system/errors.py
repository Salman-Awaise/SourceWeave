"""Typed, actionable errors.

Every error carries a message the operator can act on. The CLI catches
`ResearchSystemError` and prints it without a traceback; anything else is a
genuine bug and keeps its traceback.
"""

from __future__ import annotations


class ResearchSystemError(Exception):
    """Base class for all expected, actionable failures."""


class ConfigurationError(ResearchSystemError):
    """A required credential or setting is missing or invalid."""


class RetrievalError(ResearchSystemError):
    """A retrieval backend required by the selected strategy is unusable."""


class VectorStoreError(RetrievalError):
    """Qdrant is unreachable, or a collection is missing / mismatched."""


class WebSearchError(RetrievalError):
    """The web search backend failed."""


class EmbeddingError(ResearchSystemError):
    """The embedding provider failed or returned an unusable response."""


class LLMError(ResearchSystemError):
    """The LLM provider failed."""


class StructuredOutputError(LLMError):
    """The model returned output that could not be validated against a schema."""


class SchemaValidationError(StructuredOutputError):
    """The model returned well-formed JSON whose *values* violate the schema.

    Distinct from a parse failure: the response format was fine, so asking the
    provider for a different output format will not help. Re-sending the same
    prompt reliably reproduces the same invalid value, so callers should fall
    back rather than retry.
    """


class IngestionError(ResearchSystemError):
    """A document source could not be read or indexed."""


class EvaluationError(ResearchSystemError):
    """An evaluation dataset or metric run is invalid."""
