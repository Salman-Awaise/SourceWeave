"""Web search adapter (Tavily).

Tavily returns *snippets*, not full pages. We keep `include_raw_content=False`
and label results as excerpts so nothing downstream treats a snippet as if the
whole source page had been read.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from ..config import Settings, get_settings
from ..core.state import RetrievedDocument
from ..errors import WebSearchError

logger = logging.getLogger(__name__)


class WebSearchProvider(Protocol):
    """Query -> ranked web excerpts."""

    def search(self, query: str, max_results: int = 3) -> list[RetrievedDocument]: ...


class TavilySearch:
    """Production adapter. Client is built lazily on first search."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = None

    def _get_client(self):  # type: ignore[no-untyped-def]
        if self._client is None:
            self._settings.require_web()
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=self._settings.tavily_api_key)
        return self._client

    def search(self, query: str, max_results: int = 3) -> list[RetrievedDocument]:
        if not query.strip():
            return []
        client = self._get_client()
        try:
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_raw_content=False,
            )
        except Exception as exc:
            raise WebSearchError(f"Tavily search failed for {query!r}: {exc}") from exc

        return _map_results(response, query)


def _map_results(response: Any, query: str) -> list[RetrievedDocument]:
    raw_results = (response or {}).get("results", []) if isinstance(response, dict) else []
    documents: list[RetrievedDocument] = []
    for position, item in enumerate(raw_results):
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        url = str(item.get("url") or "").strip()
        if not content or not url:
            continue
        try:
            score = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        documents.append(
            RetrievedDocument(
                content=content,
                source=url,
                score=score,
                metadata={
                    "title": str(item.get("title") or "").strip(),
                    "source_type": "web",
                    "is_excerpt": True,
                    "native_score": score,
                    "rank": position,
                    "query": query,
                },
            )
        )
    return documents


class FakeWebSearch:
    """Offline provider returning canned results per query."""

    def __init__(self, results: dict[str, list[RetrievedDocument]] | None = None) -> None:
        self.results = results or {}
        self.queries: list[str] = []

    def search(self, query: str, max_results: int = 3) -> list[RetrievedDocument]:
        self.queries.append(query)
        return list(self.results.get(query, []))[:max_results]


def get_web_search_provider(settings: Settings | None = None) -> WebSearchProvider:
    return TavilySearch(settings)
