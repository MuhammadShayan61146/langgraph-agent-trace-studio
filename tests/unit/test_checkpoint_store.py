"""Unit tests for ``agentic_research.memory.checkpoint``."""

from __future__ import annotations

import tempfile

from agentic_research.core.enums import TaskStatus
from agentic_research.core.state import create_initial_state
from agentic_research.memory.checkpoint import (
    FileCheckpointStore,
    InMemoryCheckpointStore,
)


# --------------------------------------------------------------------------- #
# InMemoryCheckpointStore
# --------------------------------------------------------------------------- #
def test_in_memory_save_then_load_round_trips() -> None:
    store = InMemoryCheckpointStore()
    state = create_initial_state("query")
    store.save("thread-1", state)

    loaded = store.load("thread-1")
    assert loaded["original_query"] == "query"


def test_in_memory_load_returns_a_copy_not_the_stored_reference() -> None:
    store = InMemoryCheckpointStore()
    state = create_initial_state("query")
    store.save("thread-1", state)

    loaded = store.load("thread-1")
    loaded["task_status"] = TaskStatus.COMPLETED  # mutate the returned copy

    reloaded = store.load("thread-1")
    assert reloaded["task_status"] == TaskStatus.PENDING  # stored value untouched


def test_in_memory_load_missing_thread_returns_none() -> None:
    store = InMemoryCheckpointStore()
    assert store.load("does-not-exist") is None


def test_in_memory_delete_and_list_threads() -> None:
    store = InMemoryCheckpointStore()
    store.save("a", create_initial_state("q1"))
    store.save("b", create_initial_state("q2"))
    assert store.list_threads() == ["a", "b"]

    assert store.delete("a") is True
    assert store.list_threads() == ["b"]
    assert store.delete("a") is False


# --------------------------------------------------------------------------- #
# FileCheckpointStore
# --------------------------------------------------------------------------- #
def test_file_store_save_then_load_round_trips() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = FileCheckpointStore(tmp_dir)
        state = create_initial_state("What is LangGraph?")
        store.save("thread-1", state)

        loaded = store.load("thread-1")
        assert loaded["original_query"] == "What is LangGraph?"
        assert loaded["task_status"] == TaskStatus.PENDING


def test_file_store_survives_a_fresh_store_instance_pointed_at_same_dir() -> None:
    """The real point of FileCheckpointStore: persistence beyond one
    in-process object's lifetime -- simulated here by constructing a brand
    new store instance against the same directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        writer = FileCheckpointStore(tmp_dir)
        state = create_initial_state("persisted query")
        writer.save("thread-1", state)

        reader = FileCheckpointStore(tmp_dir)  # fresh instance, same directory
        loaded = reader.load("thread-1")
        assert loaded is not None
        assert loaded["original_query"] == "persisted query"


def test_file_store_load_missing_thread_returns_none() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = FileCheckpointStore(tmp_dir)
        assert store.load("does-not-exist") is None


def test_file_store_delete_and_list_threads() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = FileCheckpointStore(tmp_dir)
        store.save("thread-a", create_initial_state("q1"))
        store.save("thread-b", create_initial_state("q2"))
        assert store.list_threads() == ["thread-a", "thread-b"]

        assert store.delete("thread-a") is True
        assert store.list_threads() == ["thread-b"]
