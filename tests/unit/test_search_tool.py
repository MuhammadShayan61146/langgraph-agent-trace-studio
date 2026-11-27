"""Unit tests for ``agentic_research.tools.search``.


"""

from __future__ import annotations

from agentic_research.tools.search import (
    InMemorySearchBackend,
    SearchResultItem,
    WebSearchTool,
)

_FIXTURES = {
    "langgraph": [
        SearchResultItem(
            title="LangGraph Docs",
            snippet="LangGraph is a library for building stateful agents.",
            url="https://example.com/langgraph",
        ),
        SearchResultItem(
            title="LangGraph Tutorial",
            snippet="A walkthrough of building a multi-agent graph.",
            url="https://example.com/langgraph-tutorial",
        ),
    ]
}


# --------------------------------------------------------------------------- #
# InMemorySearchBackend
# --------------------------------------------------------------------------- #
def test_in_memory_backend_returns_registered_results() -> None:
    backend = InMemorySearchBackend(_FIXTURES)
    results = backend.search("langgraph", max_results=5)
    assert len(results) == 2
    assert results[0].title == "LangGraph Docs"


def test_in_memory_backend_is_case_insensitive() -> None:
    backend = InMemorySearchBackend(_FIXTURES)
    assert len(backend.search("LangGraph", max_results=5)) == 2


def test_in_memory_backend_respects_max_results() -> None:
    backend = InMemorySearchBackend(_FIXTURES)
    results = backend.search("langgraph", max_results=1)
    assert len(results) == 1


def test_in_memory_backend_returns_empty_for_unregistered_query() -> None:
    backend = InMemorySearchBackend(_FIXTURES)
    assert backend.search("unregistered topic", max_results=5) == []


def test_in_memory_backend_defaults_to_empty_fixture_data() -> None:
    backend = InMemorySearchBackend()
    assert backend.search("anything", max_results=5) == []


# --------------------------------------------------------------------------- #
# WebSearchTool
# --------------------------------------------------------------------------- #
def test_web_search_tool_returns_plain_dicts() -> None:
    tool = WebSearchTool(backend=InMemorySearchBackend(_FIXTURES))
    result = tool.execute(query="langgraph")
    assert result.success is True
    assert isinstance(result.output, list)
    assert result.output[0] == {
        "title": "LangGraph Docs",
        "snippet": "LangGraph is a library for building stateful agents.",
        "url": "https://example.com/langgraph",
    }


def test_web_search_tool_default_max_results_is_five() -> None:
    many_results = [
        SearchResultItem(title=f"Result {i}", snippet="...", url=f"https://x/{i}")
        for i in range(10)
    ]
    tool = WebSearchTool(backend=InMemorySearchBackend({"broad topic": many_results}))
    result = tool.execute(query="broad topic")
    assert len(result.output) == 5


def test_web_search_tool_honors_explicit_max_results() -> None:
    tool = WebSearchTool(backend=InMemorySearchBackend(_FIXTURES))
    result = tool.execute(query="langgraph", max_results=1)
    assert len(result.output) == 1


def test_web_search_tool_missing_query_reported_as_failure() -> None:
    tool = WebSearchTool(backend=InMemorySearchBackend(_FIXTURES))
    result = tool.execute()
    assert result.success is False


def test_web_search_tool_schema_marks_query_required_and_max_results_optional() -> None:
    tool = WebSearchTool(backend=InMemorySearchBackend())
    schema = tool.to_anthropic_schema()
    assert schema["input_schema"]["required"] == ["query"]
    assert "max_results" in schema["input_schema"]["properties"]


def test_web_search_tool_accepts_any_object_satisfying_the_protocol() -> None:
    """Confirm structural typing: a class with no shared base still works."""

    class _CustomBackend:
        def search(self, query: str, max_results: int) -> list[SearchResultItem]:
            return [SearchResultItem(title="custom", snippet="s", url="u")]

    tool = WebSearchTool(backend=_CustomBackend())
    result = tool.execute(query="anything")
    assert result.success is True
    assert result.output[0]["title"] == "custom"
