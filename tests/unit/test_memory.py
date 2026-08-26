"""Memory: user isolation, response normalization, and optional-failure behaviour."""

from __future__ import annotations

import pytest

from research_system.adapters.memory import (
    InMemoryMemoryStore,
    Mem0MemoryStore,
    NullMemoryStore,
    _call_variants,
    _is_signature_error,
    normalize_memory_results,
)


# --- user isolation --------------------------------------------------------
def test_records_are_isolated_by_user_id():
    store = InMemoryMemoryStore()
    store.add([{"role": "user", "content": "alice likes python"}], "alice")
    store.add([{"role": "user", "content": "bob likes rust"}], "bob")

    assert store.get_all("alice") == ["alice likes python"]
    assert store.get_all("bob") == ["bob likes rust"]
    assert "rust" not in " ".join(store.search("likes", "alice"))


def test_blank_user_id_is_rejected():
    store = InMemoryMemoryStore()

    for bad in ("", "   "):
        with pytest.raises(ValueError, match="non-empty user_id"):
            store.search("q", bad)
        with pytest.raises(ValueError, match="non-empty user_id"):
            store.add([{"role": "user", "content": "x"}], bad)


def test_search_limit_is_respected():
    store = InMemoryMemoryStore()
    for i in range(10):
        store.add([{"role": "user", "content": f"python fact {i}"}], "u")

    assert len(store.search("python", "u", limit=3)) == 3


# --- response shape normalization -----------------------------------------
@pytest.mark.parametrize(
    "raw",
    [
        {"results": [{"memory": "a"}, {"memory": "b"}]},
        [{"memory": "a"}, {"memory": "b"}],
        [{"text": "a"}, {"content": "b"}],
        ["a", "b"],
        {"memories": ["a", "b"]},
    ],
)
def test_known_mem0_shapes_normalize_to_strings(raw):
    assert normalize_memory_results(raw) == ["a", "b"]


@pytest.mark.parametrize("raw", [None, [], {}, "", 42, {"results": []}])
def test_empty_and_unknown_shapes_normalize_to_empty(raw):
    assert normalize_memory_results(raw) == []


def test_blank_entries_are_dropped():
    assert normalize_memory_results([{"memory": " "}, {"memory": "keep"}]) == ["keep"]


# --- optional failure ------------------------------------------------------
def test_null_store_is_inert():
    store = NullMemoryStore()

    assert store.enabled is False
    assert store.search("q", "u") == []
    assert store.get_all("u") == []
    assert store.add([{"role": "user", "content": "x"}], "u") is None


def test_a_failing_mem0_backend_returns_empty_instead_of_raising(monkeypatch):
    store = Mem0MemoryStore()

    class Broken:
        def search(self, **kwargs):
            raise RuntimeError("mem0 down")

        def add(self, *args, **kwargs):
            raise RuntimeError("mem0 down")

        def get_all(self, **kwargs):
            raise RuntimeError("mem0 down")

    monkeypatch.setattr(store, "_get_memory", lambda: Broken())

    assert store.search("q", "u") == []
    assert store.get_all("u") == []
    store.add([{"role": "user", "content": "x"}], "u")  # must not raise


def test_failed_initialization_disables_the_store(monkeypatch):
    store = Mem0MemoryStore()
    monkeypatch.setattr(
        "research_system.adapters.memory.Mem0MemoryStore._get_memory", lambda self: None
    )

    assert store.search("q", "u") == []


# --- version tolerance (mem0 1.x vs 2.x signatures) ------------------------
def test_signature_errors_are_recognized():
    assert _is_signature_error(TypeError("unexpected keyword argument 'limit'")) is True
    assert (
        _is_signature_error(
            ValueError(
                "Top-level entity parameters frozenset({'user_id'}) are not supported "
                "in get_all(). Use filters={'user_id': '...'} instead."
            )
        )
        is True
    )


def test_real_failures_are_not_mistaken_for_signature_errors():
    assert _is_signature_error(RuntimeError("connection refused")) is False
    assert _is_signature_error(ValueError("invalid api key")) is False


def test_call_variants_uses_the_first_accepted_shape():
    seen = []

    def fn(query, **kwargs):
        seen.append(sorted(kwargs))
        if "filters" not in kwargs:
            raise TypeError("unexpected keyword argument")
        return {"results": [{"memory": "hit"}]}

    out = _call_variants(fn, [{"filters": {"user_id": "u"}}, {"user_id": "u"}], "q")

    assert normalize_memory_results(out) == ["hit"]
    assert seen == [["filters"]]  # modern shape accepted first, no fallback needed


def test_call_variants_falls_back_to_the_legacy_shape():
    seen = []

    def fn(query, **kwargs):
        seen.append(sorted(kwargs))
        if "filters" in kwargs:
            raise ValueError("filters are not supported in this version")
        return {"results": [{"memory": "legacy hit"}]}

    out = _call_variants(fn, [{"filters": {"user_id": "u"}}, {"user_id": "u"}], "q")

    assert normalize_memory_results(out) == ["legacy hit"]
    assert seen == [["filters"], ["user_id"]]


def test_call_variants_does_not_retry_a_genuine_failure():
    calls = []

    def fn(query, **kwargs):
        calls.append(1)
        raise RuntimeError("mem0 service unavailable")

    with pytest.raises(RuntimeError, match="service unavailable"):
        _call_variants(fn, [{"filters": {}}, {"user_id": "u"}], "q")

    assert len(calls) == 1  # no pointless second attempt


def test_search_scopes_by_user_through_filters(monkeypatch):
    """The mem0 2.x shape: filters={'user_id': ...} and top_k, not user_id/limit."""
    captured = {}

    class Modern:
        def search(self, query, **kwargs):
            if "filters" not in kwargs:
                raise ValueError("Top-level entity parameters are not supported")
            captured.update(kwargs, query=query)
            return {"results": [{"memory": "alice likes python"}]}

    store = Mem0MemoryStore()
    monkeypatch.setattr(store, "_get_memory", lambda: Modern())

    assert store.search("what do I like", "alice", limit=5) == ["alice likes python"]
    assert captured["filters"] == {"user_id": "alice"}
    assert captured["top_k"] == 5


def test_get_all_scopes_by_user_through_filters(monkeypatch):
    captured = {}

    class Modern:
        def get_all(self, **kwargs):
            if "filters" not in kwargs:
                raise ValueError("not supported")
            captured.update(kwargs)
            return {"count": 1, "next": None, "previous": None, "results": [{"memory": "m"}]}

    store = Mem0MemoryStore()
    monkeypatch.setattr(store, "_get_memory", lambda: Modern())

    assert store.get_all("bob") == ["m"]
    assert captured["filters"] == {"user_id": "bob"}


def test_add_sends_user_id_at_the_top_level(monkeypatch):
    """`add` differs from search/get_all: the entity ID must NOT be in filters.

    Putting it in `filters` makes the cloud API reject the write with
    400 "At least one entity ID is required", so `user_id` has to be tried first.
    """
    captured = {}

    class Cloud:
        def add(self, messages, **kwargs):
            if "user_id" not in kwargs:
                raise RuntimeError("400: At least one entity ID is required")
            captured.update(kwargs, messages=messages)
            return {"results": []}

    store = Mem0MemoryStore()
    monkeypatch.setattr(store, "_get_memory", lambda: Cloud())

    store.add([{"role": "user", "content": "hi"}], "carol", metadata={"source": "test"})

    assert captured["user_id"] == "carol"
    assert captured["metadata"] == {"source": "test"}
    assert "filters" not in captured


def test_add_does_not_swallow_a_rejected_write_silently(monkeypatch, caplog):
    """A 400 is not a signature error, so it must not be retried into silence."""

    class AlwaysRejects:
        def add(self, messages, **kwargs):
            raise RuntimeError("400: At least one entity ID is required")

    store = Mem0MemoryStore()
    monkeypatch.setattr(store, "_get_memory", lambda: AlwaysRejects())

    with caplog.at_level("WARNING"):
        store.add([{"role": "user", "content": "hi"}], "dave")

    assert any("memory write failed" in r.message for r in caplog.records)


def test_paginated_get_all_response_normalizes():
    """The cloud get_all wraps results in count/next/previous."""
    raw = {
        "count": 2,
        "next": None,
        "previous": None,
        "results": [{"memory": "a"}, {"memory": "b"}],
    }
    assert normalize_memory_results(raw) == ["a", "b"]
