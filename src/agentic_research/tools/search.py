"""Search tool: web-style search behind an injected, swappable backend.

Purpose
-------
Gives agents a "look things up" capability, structured so the *tool*
(`WebSearchTool`) is completely decoupled from *where results come from*
(`SearchBackend`). This is the tool in Module 3 that is honestly split
across the project's honesty categories: the tool wiring and the offline
backend are genuinely Verified; a real internet-backed backend is
explicitly Design-only / Requires real API and network, and is not
implemented here.

Responsibilities
-----------------
* `SearchResultItem` — one search result, provider-agnostic shape.
* `SearchBackend` — the `Protocol` any search provider must implement.
* `InMemorySearchBackend` — a deterministic, offline backend driven by
  caller-supplied fixture data. Used for tests and for local development
  without an API key. **This is a stand-in, not a real search engine.**
* `WebSearchTool` — the `BaseTool` subclass agents call; it delegates to
  whatever `SearchBackend` was injected at construction time.

Design decisions
-----------------
* **`SearchBackend` is a `typing.Protocol`, not an ABC.** Any object with a
  matching `search(query, max_results)` method satisfies it structurally —
  no explicit inheritance required. This keeps a future real backend
  (e.g. wrapping `tavily-python` or `requests` against a search API) free
  to be a plain class with no dependency on this project's class hierarchy,
  which is the more idiomatic Python approach to Dependency Inversion than
  forcing an ABC subclass relationship for something this simple.
* **The backend is injected via `WebSearchTool.__init__`, not looked up
  globally.** This is the Dependency Injection this project has used
  consistently since Module 1 (`Settings` construction, not global state):
  the *caller* decides whether a given `WebSearchTool` instance is backed
  by `InMemorySearchBackend` (tests/offline) or a real provider (production),
  and `WebSearchTool` itself never imports or knows about any concrete
  backend.
* **No real internet-backed backend ships in this module.** This sandbox
  has no network egress (confirmed in Module 1); implementing a backend
  that would silently fail here (or worse, appear to "work" against
  nothing) would violate this project's honesty policy. The `SearchBackend`
  Protocol is the complete, real interface such a backend would implement;
  only its concrete implementation is deferred, and is explicitly listed
  as a Known Risk / dependency for a future module in the Module 3 report.

Tradeoffs
---------
* `InMemorySearchBackend` requires fixture data to return anything
  meaningful. It is not a "smart" mock (no fuzzy matching); tests must
  register the exact query string they intend to search for. This is
  intentional — a fuzzy mock would risk masking real query-formatting bugs
  behind apparent search results.

Limitations
------------
* No pagination, ranking, or freshness semantics — `max_results` simply
  truncates whatever fixture list was registered for a query.

Future improvements
--------------------
* Implement a real `SearchBackend` (e.g. `TavilySearchBackend`) once
  network access and an API key are available; it plugs in with zero
  changes to `WebSearchTool` or `ToolRegistry`, which is the entire point
  of this design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from agentic_research.tools.base import BaseTool, ToolParameter


@dataclass(frozen=True, slots=True)
class SearchResultItem:
    """A single search result, in a provider-agnostic shape.

    Attributes:
        title: The result's title/headline.
        snippet: A short excerpt or summary of the result's content.
        url: The source URL.
    """

    title: str
    snippet: str
    url: str


class SearchBackend(Protocol):
    """Structural interface any search provider must satisfy.

    Any object exposing a matching `search` method satisfies this Protocol
    automatically — no explicit subclassing required (see module docstring,
    "Design decisions").
    """

    def search(self, query: str, max_results: int) -> list[SearchResultItem]:
        """Return up to ``max_results`` results for ``query``."""
        ...


class InMemorySearchBackend:
    """A deterministic, offline `SearchBackend` driven by fixture data.

    This is explicitly **not** a real search engine — see module docstring.
    It exists so `WebSearchTool` and everything built on top of it
    (agents, in later modules) can be developed and tested without network
    access or an API key.
    """

    def __init__(
        self, fixture_data: dict[str, list[SearchResultItem]] | None = None
    ) -> None:
        """Initialize the backend with a lookup table of canned results.

        Args:
            fixture_data: Maps a lowercased, exact query string to the list
                of `SearchResultItem` that should be "found" for it.
                Queries not present in this mapping return an empty list.
                Defaults to an empty mapping (every query returns nothing).
        """
        self._fixture_data: dict[str, list[SearchResultItem]] = dict(
            fixture_data or {}
        )

    def search(self, query: str, max_results: int) -> list[SearchResultItem]:
        """Return the fixture results registered for ``query``, if any.

        Args:
            query: The search query. Matched case-insensitively against
                registered fixture keys.
            max_results: Maximum number of results to return.

        Returns:
            Up to ``max_results`` `SearchResultItem` entries, or an empty
            list if no fixture data was registered for this query.
        """
        results = self._fixture_data.get(query.lower(), [])
        return results[:max_results]


class WebSearchTool(BaseTool):
    """A tool that searches for information via an injected `SearchBackend`."""

    name = "web_search"
    description = (
        "Search for information on a topic and return a list of results "
        "(title, snippet, url)."
    )
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="The search query.",
            required=True,
        ),
        ToolParameter(
            name="max_results",
            type="integer",
            description="Maximum number of results to return (default 5).",
            required=False,
        ),
    ]

    def __init__(self, backend: SearchBackend) -> None:
        """Construct the tool with an injected search backend.

        Args:
            backend: Any object satisfying `SearchBackend`. Production code
                should inject a real, network-backed implementation; tests
                and offline development should inject
                `InMemorySearchBackend`.
        """
        self._backend = backend

    def _run(self, **kwargs: Any) -> list[dict[str, str]]:
        """Delegate to the injected backend and return plain-dict results.

        Args:
            **kwargs: ``query`` (required) and ``max_results`` (optional,
                defaults to 5).

        Returns:
            A list of plain dicts (``{"title", "snippet", "url"}``) rather
            than `SearchResultItem` instances, so this tool's output is
            trivially JSON-serializable for logging and for returning to an
            LLM as tool-result content.
        """
        query: str = kwargs["query"]
        max_results: int = kwargs.get("max_results", 5)
        results = self._backend.search(query, max_results)
        return [
            {"title": item.title, "snippet": item.snippet, "url": item.url}
            for item in results
        ]
