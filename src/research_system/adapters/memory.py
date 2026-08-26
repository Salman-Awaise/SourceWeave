"""Conversational memory behind a protocol.

Memory is strictly optional. mem0's local and cloud APIs have both changed
shape across releases, so `Mem0MemoryStore` normalizes whatever `search`
returns into a plain list of strings, and any initialization failure degrades
to `NullMemoryStore` rather than blocking a query.

Every call is scoped by a non-empty `user_id`; there is no code path that reads
or writes memory without one.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)


class MemoryStore(Protocol):
    """Per-user conversational memory."""

    @property
    def enabled(self) -> bool: ...

    def search(self, query: str, user_id: str, limit: int = 3) -> list[str]: ...

    def add(
        self, messages: list[dict[str, str]], user_id: str, metadata: dict[str, Any] | None = None
    ) -> None: ...

    def get_all(self, user_id: str) -> list[str]: ...


def _require_user_id(user_id: str) -> str:
    text = (user_id or "").strip()
    if not text:
        raise ValueError("memory operations require a non-empty user_id")
    return text


def _is_signature_error(exc: Exception) -> bool:
    """True when a call failed because of the *shape* of the arguments.

    mem0 rejects unknown keyword arguments with either `TypeError` (plain
    Python) or `ValueError` (its own parameter validation, e.g. "Top-level
    entity parameters ... are not supported"). Both mean "wrong API version",
    so the next variant is worth trying. A genuine auth or network failure
    raises something else and must propagate.
    """
    if isinstance(exc, TypeError):
        return True
    if isinstance(exc, ValueError):
        text = str(exc).lower()
        return any(
            marker in text
            for marker in ("not supported", "unexpected", "argument", "parameter", "filters")
        )
    return False


def _call_variants(func: Any, variants: list[dict[str, Any]], *args: Any) -> Any:
    """Call `func` with the first keyword set its installed version accepts.

    mem0's `search`/`add`/`get_all` signatures changed between 1.x and 2.x, and
    the 2.x cloud and local clients still disagree with each other. Rather than
    pinning one release, try the current shape first and fall back to the older
    one, so the adapter keeps working across upgrades.
    """
    last_error: Exception | None = None
    for kwargs in variants:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if not _is_signature_error(exc):
                raise
            last_error = exc
            logger.debug("mem0 call shape %s rejected: %s", sorted(kwargs), exc)
    raise last_error if last_error else RuntimeError("no call variants supplied")


class NullMemoryStore:
    """No-op store used when memory is disabled or unavailable."""

    enabled = False

    def search(self, query: str, user_id: str, limit: int = 3) -> list[str]:
        return []

    def add(
        self, messages: list[dict[str, str]], user_id: str, metadata: dict[str, Any] | None = None
    ) -> None:
        return None

    def get_all(self, user_id: str) -> list[str]:
        return []


def normalize_memory_results(raw: Any) -> list[str]:
    """Flatten any known mem0 response shape into memory strings.

    Handles: `{"results": [...]}`, a bare list, dict entries keyed by `memory`
    or `text`, and plain strings.
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("results", raw.get("memories", []))
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if not isinstance(raw, list):
        return []

    out: list[str] = []
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            value = item.get("memory") or item.get("text") or item.get("content") or ""
            text = str(value).strip()
        else:
            text = ""
        if text:
            out.append(text)
    return out


class Mem0MemoryStore:
    """mem0-backed store. Construction is lazy; failures degrade to no-op."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._memory: Any = None
        self._failed = False

    @property
    def enabled(self) -> bool:
        return not self._failed

    def _get_memory(self) -> Any:
        if self._memory is not None or self._failed:
            return self._memory
        try:
            from mem0 import Memory, MemoryClient

            if self._settings.mem0_api_key.strip():
                self._memory = MemoryClient(api_key=self._settings.mem0_api_key.strip())
            else:
                self._memory = Memory.from_config(
                    {
                        "llm": {
                            "provider": "litellm",
                            "config": {"model": self._settings.default_llm},
                        },
                        "embedder": {
                            "provider": "openai",
                            "config": {"model": self._settings.embedding_model},
                        },
                    }
                )
        except Exception as exc:
            logger.warning("memory unavailable, continuing without it: %s", exc)
            self._failed = True
            self._memory = None
        return self._memory

    def search(self, query: str, user_id: str, limit: int = 3) -> list[str]:
        user = _require_user_id(user_id)
        memory = self._get_memory()
        if memory is None:
            return []
        try:
            raw = _call_variants(
                memory.search,
                [
                    # mem0 2.x: scoping moved into `filters`, `limit` became `top_k`.
                    {"filters": {"user_id": user}, "top_k": limit},
                    # mem0 1.x / older cloud clients.
                    {"user_id": user, "limit": limit},
                ],
                query,
            )
        except Exception as exc:
            logger.warning("memory search failed, continuing without it: %s", exc)
            return []
        return normalize_memory_results(raw)[:limit]

    def add(
        self, messages: list[dict[str, str]], user_id: str, metadata: dict[str, Any] | None = None
    ) -> None:
        user = _require_user_id(user_id)
        memory = self._get_memory()
        if memory is None or not messages:
            return
        try:
            _call_variants(
                memory.add,
                [
                    # `add` is the exception: unlike search/get_all, both the 2.x
                    # cloud and local clients want the entity ID at the top level.
                    # Sending it inside `filters` gets a 400 "At least one entity
                    # ID is required".
                    {"user_id": user, "metadata": metadata or {}},
                    {"filters": {"user_id": user}, "metadata": metadata or {}},
                ],
                messages,
            )
        except Exception as exc:
            logger.warning("memory write failed, continuing: %s", exc)

    def get_all(self, user_id: str) -> list[str]:
        user = _require_user_id(user_id)
        memory = self._get_memory()
        if memory is None:
            return []
        try:
            raw = _call_variants(
                memory.get_all,
                [{"filters": {"user_id": user}}, {"user_id": user}],
            )
        except Exception as exc:
            logger.warning("memory read failed: %s", exc)
            return []
        return normalize_memory_results(raw)


class InMemoryMemoryStore:
    """Offline store for tests. Records are partitioned by user_id."""

    enabled = True

    def __init__(self) -> None:
        self.records: dict[str, list[str]] = {}

    def search(self, query: str, user_id: str, limit: int = 3) -> list[str]:
        user = _require_user_id(user_id)
        terms = {word for word in query.lower().split() if len(word) > 2}
        entries = self.records.get(user, [])
        if not terms:
            return entries[:limit]
        scored = [(sum(term in entry.lower() for term in terms), entry) for entry in entries]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for score, entry in scored if score > 0][:limit]

    def add(
        self, messages: list[dict[str, str]], user_id: str, metadata: dict[str, Any] | None = None
    ) -> None:
        user = _require_user_id(user_id)
        bucket = self.records.setdefault(user, [])
        for message in messages:
            content = str(message.get("content", "")).strip()
            if content:
                bucket.append(content)

    def get_all(self, user_id: str) -> list[str]:
        return list(self.records.get(_require_user_id(user_id), []))


def get_memory_store(settings: Settings | None = None, *, enabled: bool = True) -> MemoryStore:
    """Return a usable memory store, never raising for an optional subsystem."""
    if not enabled:
        return NullMemoryStore()
    try:
        import mem0  # noqa: F401
    except ImportError:
        logger.warning(
            "mem0ai is not installed; running without memory. "
            "Install with: pip install 'research-system[memory]'"
        )
        return NullMemoryStore()
    return Mem0MemoryStore(settings)
