"""Short-term memory: thread-scoped `GraphState` checkpointing.

Purpose
-------
Lets a research task's state be saved and resumed across separate
`build_graph(...).invoke(...)` calls (or, in this sandbox, separate
process runs) — the "Memory" requirement's short-term half. This is
conceptually what LangGraph's own checkpointer (`MemorySaver`/
`SqliteSaver`, part of `langgraph.checkpoint`) provides; `CheckpointStore`
here is a framework-free equivalent, following the same DI pattern as
`SearchBackend` (Module 3) and `LLMClient` (Module 4).

Design decisions
-----------------
* **`CheckpointStore` is a `Protocol`, keyed by an opaque ``thread_id``**,
  matching LangGraph's own checkpointer key shape
  (`configurable.thread_id`) so that swapping this for a real LangGraph
  checkpointer later (Module 5's `build.py`, once `langgraph` is
  installed) is a drop-in concept, not a redesign.
* **Two concrete implementations, both real** (unlike Module 3/4's
  `InMemory*` mocks, which stood in for something unbuildable here):
  `InMemoryCheckpointStore` (process-local, for tests and short-lived
  runs) and `FileCheckpointStore` (persists to disk via `JSONFileStore` +
  `serialization.py`, genuinely surviving process restarts). Both reach
  full **Verified** status — no network or unavailable package involved.
* **`load()` returns a *copy*, never the stored object itself**, in both
  implementations. `GraphState` is a `TypedDict` (a plain mutable dict at
  runtime); returning the stored reference directly would let a caller
  mutate it and corrupt the "saved" checkpoint without ever calling
  `save()` again. This is the same "state entries are facts, not
  mutable-in-place objects" discipline established for the data contracts
  in Module 2.

Limitations
------------
* No checkpoint history/versioning — `save()` overwrites the previous
  checkpoint for a `thread_id` rather than keeping a log of snapshots.
  LangGraph's real checkpointer supports time-travel (multiple checkpoints
  per thread); that's out of scope here.
"""

from __future__ import annotations

from typing import Protocol

from agentic_research.core.state import GraphState
from agentic_research.memory.file_store import JSONFileStore
from agentic_research.memory.serialization import state_from_dict, state_to_dict


class CheckpointStore(Protocol):
    """Structural interface for saving/loading `GraphState` by thread ID."""

    def save(self, thread_id: str, state: GraphState) -> None:
        """Persist ``state`` as the current checkpoint for ``thread_id``."""
        ...

    def load(self, thread_id: str) -> GraphState | None:
        """Return the checkpointed state for ``thread_id``, or ``None``."""
        ...

    def delete(self, thread_id: str) -> bool:
        """Delete the checkpoint for ``thread_id``; return whether one existed."""
        ...

    def list_threads(self) -> list[str]:
        """Return every ``thread_id`` with a saved checkpoint."""
        ...


class InMemoryCheckpointStore:
    """A process-local `CheckpointStore` backed by a plain dict.

    Real (not a mock/stand-in) for the duration of the process; does not
    survive a process restart. Suitable for tests and single-run usage.
    """

    def __init__(self) -> None:
        """Initialize an empty store."""
        self._checkpoints: dict[str, GraphState] = {}

    def save(self, thread_id: str, state: GraphState) -> None:
        """Store a shallow copy of ``state`` under ``thread_id``."""
        self._checkpoints[thread_id] = dict(state)  # type: ignore[assignment]

    def load(self, thread_id: str) -> GraphState | None:
        """Return a shallow copy of the stored state, or ``None`` if absent."""
        stored = self._checkpoints.get(thread_id)
        return dict(stored) if stored is not None else None  # type: ignore[return-value]

    def delete(self, thread_id: str) -> bool:
        """Remove the checkpoint for ``thread_id`` if present."""
        return self._checkpoints.pop(thread_id, None) is not None

    def list_threads(self) -> list[str]:
        """Return every stored thread ID, sorted."""
        return sorted(self._checkpoints.keys())


class FileCheckpointStore:
    """A `CheckpointStore` that persists checkpoints to disk as JSON files.

    Genuinely survives process restarts, unlike `InMemoryCheckpointStore`
    — verified in this module's tests by writing with one store instance
    and reading with a fresh one pointed at the same directory.
    """

    def __init__(self, base_dir: str) -> None:
        """Initialize the store, creating ``base_dir`` if needed.

        Args:
            base_dir: Directory checkpoint files are written to.
        """
        self._files = JSONFileStore(base_dir)

    def save(self, thread_id: str, state: GraphState) -> None:
        """Serialize and write ``state`` to disk under ``thread_id``."""
        self._files.write(thread_id, state_to_dict(state))

    def load(self, thread_id: str) -> GraphState | None:
        """Read and deserialize the checkpoint for ``thread_id``, if any."""
        data = self._files.read(thread_id)
        return state_from_dict(data) if data is not None else None

    def delete(self, thread_id: str) -> bool:
        """Delete the checkpoint file for ``thread_id`` if present."""
        return self._files.delete(thread_id)

    def list_threads(self) -> list[str]:
        """Return every thread ID with a checkpoint file on disk."""
        return self._files.list_keys()
