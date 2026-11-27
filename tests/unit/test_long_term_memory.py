"""Unit tests for ``agentic_research.memory.long_term``."""

from __future__ import annotations

import tempfile

from agentic_research.core.contracts import ResearchFinding
from agentic_research.core.enums import AgentRole
from agentic_research.memory.long_term import (
    FileLongTermMemoryStore,
    InMemoryLongTermMemoryStore,
    save_findings,
)


# --------------------------------------------------------------------------- #
# InMemoryLongTermMemoryStore
# --------------------------------------------------------------------------- #
def test_in_memory_put_then_get_round_trips() -> None:
    store = InMemoryLongTermMemoryStore()
    store.put("topic-a", "fact-1", {"content": "some fact"})

    record = store.get("topic-a", "fact-1")
    assert record.value == {"content": "some fact"}
    assert record.namespace == "topic-a"
    assert record.key == "fact-1"


def test_in_memory_get_missing_returns_none() -> None:
    store = InMemoryLongTermMemoryStore()
    assert store.get("topic-a", "does-not-exist") is None


def test_in_memory_search_returns_only_matching_namespace_sorted_by_key() -> None:
    store = InMemoryLongTermMemoryStore()
    store.put("topic-a", "z", {"n": 1})
    store.put("topic-a", "a", {"n": 2})
    store.put("topic-b", "m", {"n": 3})

    results = store.search("topic-a")
    assert [r.key for r in results] == ["a", "z"]


def test_in_memory_delete() -> None:
    store = InMemoryLongTermMemoryStore()
    store.put("topic-a", "fact-1", {})
    assert store.delete("topic-a", "fact-1") is True
    assert store.get("topic-a", "fact-1") is None
    assert store.delete("topic-a", "fact-1") is False


def test_in_memory_put_overwrites_existing_key() -> None:
    store = InMemoryLongTermMemoryStore()
    store.put("topic-a", "fact-1", {"n": 1})
    store.put("topic-a", "fact-1", {"n": 2})
    assert store.get("topic-a", "fact-1").value == {"n": 2}


# --------------------------------------------------------------------------- #
# FileLongTermMemoryStore
# --------------------------------------------------------------------------- #
def test_file_store_put_then_get_round_trips() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = FileLongTermMemoryStore(tmp_dir)
        store.put("topic-a", "fact-1", {"content": "persisted fact"})

        record = store.get("topic-a", "fact-1")
        assert record.value == {"content": "persisted fact"}


def test_file_store_search_scans_payloads_not_filenames() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = FileLongTermMemoryStore(tmp_dir)
        store.put("topic-a", "1", {"n": 1})
        store.put("topic-a", "2", {"n": 2})
        store.put("topic-b", "1", {"n": 3})

        results = store.search("topic-a")
        assert len(results) == 2
        assert {r.key for r in results} == {"1", "2"}


def test_file_store_survives_a_fresh_instance_pointed_at_same_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        writer = FileLongTermMemoryStore(tmp_dir)
        writer.put("topic-a", "fact-1", {"content": "durable"})

        reader = FileLongTermMemoryStore(tmp_dir)
        record = reader.get("topic-a", "fact-1")
        assert record is not None
        assert record.value == {"content": "durable"}


def test_file_store_delete() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = FileLongTermMemoryStore(tmp_dir)
        store.put("topic-a", "fact-1", {})
        assert store.delete("topic-a", "fact-1") is True
        assert store.get("topic-a", "fact-1") is None


# --------------------------------------------------------------------------- #
# save_findings
# --------------------------------------------------------------------------- #
def test_save_findings_persists_each_finding_and_is_searchable() -> None:
    store = InMemoryLongTermMemoryStore()
    findings = [
        ResearchFinding(
            source="https://a.example.com",
            content="Finding A",
            collected_by=AgentRole.RESEARCHER,
            confidence=0.8,
        ),
        ResearchFinding(
            source="https://b.example.com",
            content="Finding B",
            collected_by=AgentRole.RESEARCHER,
            confidence=0.6,
        ),
    ]
    save_findings(store, "task-123", findings)

    results = store.search("task-123")
    assert len(results) == 2
    assert {r.value["source"] for r in results} == {
        "https://a.example.com",
        "https://b.example.com",
    }


def test_save_findings_with_empty_list_stores_nothing() -> None:
    store = InMemoryLongTermMemoryStore()
    save_findings(store, "task-123", [])
    assert store.search("task-123") == []
