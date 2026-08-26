"""LLM access behind a narrow protocol.

Everything downstream depends on `LLMClient`, never on LiteLLM directly, so
unit tests can substitute `FakeLLMClient` and run with no network or keys.

Structured output uses a three-tier strategy, because provider support for
native JSON schemas is uneven and changes between releases:

1. ask for the provider's native JSON-schema mode,
2. fall back to plain JSON-object mode,
3. extract the first balanced JSON object from the text.

Every tier ends at the same Pydantic validation, so no tier is "trust the
model" -- tier 3 is a *locator*, not a parser, which is why it is safe in a way
that splitting on Markdown fences is not.
"""

from __future__ import annotations

import contextlib
import logging
import random
import time
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import Settings, get_settings
from ..errors import LLMError, SchemaValidationError, StructuredOutputError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Provider errors worth a second attempt. Auth and validation errors are not
# retried -- they will fail identically and just burn time.
_RETRYABLE_MARKERS = (
    "rate limit",
    "ratelimit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
    "timeout",
    "timed out",
    "connection",
    "temporarily unavailable",
)


def _is_retryable(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    if "auth" in name or "permission" in name or "notfound" in name:
        return False
    text = f"{name} {exc}".lower()
    if any(marker in text for marker in ("api key", "unauthorized", "401", "403")):
        return False
    return any(marker in text for marker in _RETRYABLE_MARKERS)


class LLMClient(Protocol):
    """Minimal surface the agents need from a chat model."""

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the assistant's raw text response."""
        ...

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Return a validated instance of `schema`.

        Raises `StructuredOutputError` when the model cannot produce output
        matching the schema; callers are expected to have a fallback.
        """
        ...


def normalize_model_name(model: str, provider: str) -> str:
    """Give LiteLLM an unambiguous model id.

    LiteLLM accepts bare names for well-known models but resolves
    `provider/model` reliably across versions, so we prefix when we can infer
    the provider and the caller has not already done so.
    """
    name = model.strip()
    if "/" in name:
        return name
    if provider in ("anthropic", "openai"):
        return f"{provider}/{name}"
    return name


def _extract_json_object(text: str) -> str:
    """Locate the first complete top-level JSON object in `text`.

    Brace-matching that respects string literals and escapes. This tolerates
    models that wrap JSON in prose or fences without depending on the fence
    itself being present or well-formed.
    """
    start = text.find("{")
    if start == -1:
        raise StructuredOutputError("model response contained no JSON object")

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise StructuredOutputError("model response contained an unterminated JSON object")


def summarize_validation_error(exc: ValidationError) -> str:
    """Compress a Pydantic error into one readable line.

    The raw form runs to many lines and ends with a docs URL, which is noise in
    a terminal that is already showing a fallback warning.
    """
    parts = []
    for err in exc.errors()[:3]:
        loc = ".".join(str(x) for x in err.get("loc", ())) or "<root>"
        msg = str(err.get("msg", "")).removeprefix("Value error, ")
        parts.append(f"{loc}: {msg}")
    extra = "" if len(exc.errors()) <= 3 else f" (+{len(exc.errors()) - 3} more)"
    return "; ".join(parts) + extra


def parse_structured(text: str, schema: type[T]) -> T:
    """Validate model text against `schema`, locating the JSON if needed.

    Raises `SchemaValidationError` when the JSON parsed cleanly but its values
    are invalid, and plain `StructuredOutputError` when nothing parseable was
    found. Callers use that distinction to decide whether retrying in a
    different output format could possibly help.
    """
    candidates = [text.strip()]
    # Whole-text validation is tried first; locating an embedded object is a
    # bonus candidate, so its absence is not itself a failure here.
    with contextlib.suppress(StructuredOutputError):
        candidates.append(_extract_json_object(text))

    def is_json_syntax_error(exc: ValidationError) -> bool:
        """Pydantic reports malformed JSON as a ValidationError too.

        Those carry type `json_invalid` and mean the text was never valid JSON,
        which IS worth retrying in another output format -- unlike a genuine
        schema violation.
        """
        errs = exc.errors()
        return bool(errs) and all(e.get("type") == "json_invalid" for e in errs)

    validation_error: ValidationError | None = None
    parse_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return schema.model_validate_json(candidate)
        except ValidationError as exc:
            if is_json_syntax_error(exc):
                parse_error = exc  # not JSON at all -- another format may help
            else:
                # Well-formed JSON, wrong values: a content problem, not a
                # formatting one.
                validation_error = exc
        except Exception as exc:  # not valid JSON at all
            parse_error = exc

    if validation_error is not None:
        raise SchemaValidationError(
            f"{schema.__name__} validation failed -> {summarize_validation_error(validation_error)}"
        ) from validation_error
    raise StructuredOutputError(
        f"no valid {schema.__name__} JSON in model output: {parse_error}"
    ) from parse_error


class LiteLLMClient:
    """Production `LLMClient` backed by LiteLLM's unified completion API."""

    def __init__(self, settings: Settings | None = None, *, max_attempts: int = 3) -> None:
        self._settings = settings or get_settings()
        self._max_attempts = max(1, max_attempts)
        self._model = normalize_model_name(self._settings.default_llm, self._settings.llm_provider)
        self._native_schema_supported = True
        """Flipped off permanently once a provider rejects json_schema mode."""

    @property
    def model(self) -> str:
        return self._model

    def _api_key(self) -> str | None:
        provider = self._settings.llm_provider
        if provider == "anthropic":
            return self._settings.anthropic_api_key or None
        if provider == "openai":
            return self._settings.openai_api_key or None
        return None

    def _call(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        import litellm  # imported lazily: heavy, and not needed for --help

        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                response = litellm.completion(
                    model=self._model,
                    messages=messages,
                    api_key=self._api_key(),
                    timeout=120,
                    **kwargs,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise LLMError("model returned an empty response")
                return str(content)
            except Exception as exc:
                last_exc = exc
                if not _is_retryable(exc) or attempt == self._max_attempts - 1:
                    break
                # Exponential backoff with jitter so parallel callers desynchronize.
                delay = (2**attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self._max_attempts,
                    delay,
                    exc,
                )
                time.sleep(delay)

        raise LLMError(f"LLM call to {self._model!r} failed: {last_exc}") from last_exc

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self._settings.require_llm()
        return self._call(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=(self._settings.temperature if temperature is None else temperature),
            max_tokens=max_tokens or self._settings.llm_max_tokens,
        )

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        self._settings.require_llm()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        shared: dict[str, Any] = {
            "temperature": (self._settings.temperature if temperature is None else temperature),
            "max_tokens": max_tokens or self._settings.llm_max_tokens,
        }

        response_formats: list[dict[str, Any] | None] = []
        if self._native_schema_supported:
            response_formats.append(
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema.model_json_schema(),
                        "strict": False,
                    },
                }
            )
        response_formats.append({"type": "json_object"})
        response_formats.append(None)

        last_error: Exception | None = None
        for response_format in response_formats:
            kwargs = dict(shared)
            if response_format is not None:
                kwargs["response_format"] = response_format
            try:
                text = self._call(messages, **kwargs)
            except LLMError as exc:
                last_error = exc
                if response_format is not None and _unsupported_format(exc):
                    if response_format.get("type") == "json_schema":
                        # Don't pay this probe again for the rest of the run.
                        self._native_schema_supported = False
                    continue
                raise
            try:
                return parse_structured(text, schema)
            except SchemaValidationError as exc:
                # The model produced clean JSON with invalid values. Another
                # response format would not change the values, and re-sending
                # the same prompt reproduces the same answer, so stop here and
                # let the caller's fallback handle it.
                logger.warning("%s; not retrying (a different format cannot fix values)", exc)
                raise
            except StructuredOutputError as exc:
                last_error = exc
                logger.warning("could not locate JSON (%s), trying next output format", exc)

        raise StructuredOutputError(
            f"could not obtain valid {schema.__name__} from {self._model!r}: {last_error}"
        ) from last_error


def _unsupported_format(exc: Exception) -> bool:
    """True when the provider rejected the response_format rather than the prompt."""
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("response_format", "json_schema", "not supported", "unsupported")
    )


class FakeLLMClient:
    """Scripted `LLMClient` for offline tests.

    Responses are consumed in order. A response may be a string (returned/parsed
    as-is), a `BaseModel` (returned directly), or an `Exception` (raised).
    """

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses: list[Any] = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def _next(self, kind: str, system: str, user: str) -> Any:
        self.calls.append({"kind": kind, "system": system, "user": user})
        if not self.responses:
            raise AssertionError(f"FakeLLMClient ran out of responses (call {len(self.calls)})")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return str(self._next("complete", system, user))

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        item = self._next("structured", system, user)
        if isinstance(item, schema):
            return item
        if isinstance(item, BaseModel):
            raise AssertionError(f"expected {schema.__name__}, got {type(item).__name__}")
        if isinstance(item, dict):
            return schema.model_validate(item)
        return parse_structured(str(item), schema)


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    return LiteLLMClient(settings)
