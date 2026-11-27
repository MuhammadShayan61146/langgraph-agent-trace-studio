"""Unit tests for ``agentic_research.memory.file_store.JSONFileStore``.

Uses real temporary directories (`tempfile`) — this class does genuine
disk I/O, so these tests exercise real file reads/writes, not mocks.

Note: uses a plain helper (`_store()`) rather than a `pytest.fixture`, so
these tests run identically under real pytest and under this project's
stdlib-only `scripts/run_tests.py`, which does not implement fixture
injection.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agentic_research.core.exceptions import StateSerializationError
from agentic_research.memory.file_store import JSONFileStore


def _store() -> tuple[JSONFileStore, tempfile.TemporaryDirectory]:
    """Build a JSONFileStore backed by a fresh temp directory.

    Returns the store and the TemporaryDirectory object; callers must keep
    a reference to the latter so it isn't garbage-collected (and cleaned
    up) mid-test.
    """
    tmp = tempfile.TemporaryDirectory()
    return JSONFileStore(tmp.name), tmp


def test_write_then_read_round_trips() -> None:
    store, _tmp = _store()
    store.write("alpha", {"value": 1})
    assert store.read("alpha") == {"value": 1}


def test_read_missing_key_returns_none() -> None:
    store, _tmp = _store()
    assert store.read("does_not_exist") is None


def test_write_overwrites_existing_value() -> None:
    store, _tmp = _store()
    store.write("alpha", {"value": 1})
    store.write("alpha", {"value": 2})
    assert store.read("alpha") == {"value": 2}


def test_delete_removes_entry_and_reports_existence() -> None:
    store, _tmp = _store()
    store.write("alpha", {"value": 1})
    assert store.delete("alpha") is True
    assert store.read("alpha") is None
    assert store.delete("alpha") is False  # already gone


def test_list_keys_returns_sorted_sanitized_keys() -> None:
    store, _tmp = _store()
    store.write("zeta", {})
    store.write("alpha", {})
    assert store.list_keys() == ["alpha", "zeta"]


def test_keys_with_special_characters_are_sanitized_for_filesystem_safety() -> None:
    store, _tmp = _store()
    store.write("../../etc/passwd", {"value": "should not escape base_dir"})
    # Must not have written outside base_dir; must still round-trip via the
    # same (sanitized) key.
    assert store.read("../../etc/passwd") == {"value": "should not escape base_dir"}


def test_empty_key_is_rejected() -> None:
    store, _tmp = _store()
    with pytest.raises(ValueError):
        store.write("", {"value": 1})


def test_corrupted_file_raises_state_serialization_error() -> None:
    store, tmp = _store()
    store.write("alpha", {"value": 1})
    # Corrupt the file directly on disk.
    (Path(tmp.name) / "alpha.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(StateSerializationError):
        store.read("alpha")


def test_base_dir_is_created_if_missing() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        nested = Path(tmp_dir) / "does" / "not" / "exist" / "yet"
        assert not nested.exists()
        JSONFileStore(nested)
        assert nested.exists()
