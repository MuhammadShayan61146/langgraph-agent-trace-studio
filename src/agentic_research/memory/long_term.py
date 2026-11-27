"""Long-term memory: namespaced facts that persist across threads/sessions.

Purpose
-------
Short-term memory (`checkpoint.py`) resumes *one* research task's full
state. Long-term memory is different: durable, smaller facts meant to be
reused *across* different tasks/threads — e.g. a research finding worth
remembering the next time a related query comes in. This mirrors
LangGraph's own `Store` interface (namespace + key addressed, cross-thread)
conceptually, without depending on `langgraph` to define it.

Responsibilities
-----------------
* `MemoryRecord` — one stored fact: a namespace, a key, a JSON-serializable
  value, and a timestamp.
* `LongTermMemoryStore` — the `Protocol` (put/get/search/delete).
* `InMemoryLongTermMemoryStore` / `FileLongTermMemoryStore` — real
  implementations, same DI pattern as `checkpoint.py`.
* `save_findings` — a small convenience function turning a research run's
  `ResearchFinding` list into stored `MemoryRecord`s, so a future task
  under the same namespace can look them up instead of re-researching.

Design decisions
-----------------
* **`search(namespace)` scans stored payloads for a matching
  ``"namespace"`` field, rather than relying on filename prefixes.**
  `JSONFileStore` sanitizes keys for filesystem safety (see that module's
  docstring), which makes prefix-matching on *filenames* unreliable once
  namespaces contain characters that get sanitized differently at
  different lengths. Storing the namespace inside the payload itself and
  filtering on read is slower (an O(n) scan) but correct regardless of
  what characters a namespace contains — a deliberate simplicity-over-
  performance tradeoff appropriate at this project's scale (see
  Limitations).
* **`save_findings` lives in this module, not in `agents/researcher.py`.**
  Deciding *whether/when* to persist findings long-term is an orchestration
  concern (a future module's job), not something `ResearcherAgent` itself
  should know about — keeping this function here means adding long-term
  persistence later requires wiring one call, not modifying Module 4.

Limitations
------------
* `FileLongTermMemoryStore.search` is O(number of stored records) — reads
  every file to check its namespace. Fine at the scale this project
  operates at (a research assistant's own findings, not a multi-tenant
  store); would need an actual index if namespace cardinality grew large.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from agentic_research.core.contracts import ResearchFinding
from agentic_research.memory.file_store import JSONFileStore


def _utc_now() -> datetime:
    """Return the current UTC time (same seam as `core.contracts._utc_now`)."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """One durable, namespaced fact stored in long-term memory.

    Attributes:
        namespace: A grouping key (e.g. a topic or task ID) records can be
            searched by.
        key: A unique identifier for this record within its namespace.
        value: The record's payload. Must be JSON-serializable.
        timestamp: When the record was stored (UTC). Defaults to now.
    """

    namespace: str
    key: str
    value: dict[str, Any]
    timestamp: datetime = field(default_factory=_utc_now)


class LongTermMemoryStore(Protocol):
    """Structural interface for namespaced, cross-thread fact storage."""

    def put(self, namespace: str, key: str, value: dict[str, Any]) -> None:
        """Store (or overwrite) a record under ``namespace``/``key``."""
        ...

    def get(self, namespace: str, key: str) -> MemoryRecord | None:
        """Return the record at ``namespace``/``key``, or ``None``."""
        ...

    def search(self, namespace: str) -> list[MemoryRecord]:
        """Return every record stored under ``namespace``."""
        ...

    def delete(self, namespace: str, key: str) -> bool:
        """Delete the record at ``namespace``/``key``; return whether it existed."""
        ...


def _composite_key(namespace: str, key: str) -> str:
    """Build the combined storage key for a namespace/key pair."""
    return f"{namespace}::{key}"


class InMemoryLongTermMemoryStore:
    """A process-local `LongTermMemoryStore` backed by a plain dict."""

    def __init__(self) -> None:
        """Initialize an empty store."""
        self._records: dict[str, MemoryRecord] = {}

    def put(self, namespace: str, key: str, value: dict[str, Any]) -> None:
        """Store a `MemoryRecord` for ``namespace``/``key`` with a fresh timestamp."""
        record = MemoryRecord(namespace=namespace, key=key, value=value)
        self._records[_composite_key(namespace, key)] = record

    def get(self, namespace: str, key: str) -> MemoryRecord | None:
        """Return the stored record, or ``None`` if not present."""
        return self._records.get(_composite_key(namespace, key))

    def search(self, namespace: str) -> list[MemoryRecord]:
        """Return every record whose namespace matches, sorted by key."""
        matches = [r for r in self._records.values() if r.namespace == namespace]
        return sorted(matches, key=lambda r: r.key)

    def delete(self, namespace: str, key: str) -> bool:
        """Remove the record for ``namespace``/``key`` if present."""
        return self._records.pop(_composite_key(namespace, key), None) is not None


class FileLongTermMemoryStore:
    """A `LongTermMemoryStore` that persists records to disk as JSON files."""

    def __init__(self, base_dir: str) -> None:
        """Initialize the store, creating ``base_dir`` if needed.

        Args:
            base_dir: Directory record files are written to.
        """
        self._files = JSONFileStore(base_dir)

    def put(self, namespace: str, key: str, value: dict[str, Any]) -> None:
        """Write a record for ``namespace``/``key`` to disk."""
        record = MemoryRecord(namespace=namespace, key=key, value=value)
        self._files.write(
            _composite_key(namespace, key),
            {
                "namespace": record.namespace,
                "key": record.key,
                "value": record.value,
                "timestamp": record.timestamp.isoformat(),
            },
        )

    def get(self, namespace: str, key: str) -> MemoryRecord | None:
        """Read and reconstruct the record for ``namespace``/``key``, if any."""
        data = self._files.read(_composite_key(namespace, key))
        if data is None:
            return None
        return _record_from_dict(data)

    def search(self, namespace: str) -> list[MemoryRecord]:
        """Scan every stored record and return those matching ``namespace``.

        See module docstring "Design decisions" for why this scans payload
        contents rather than filenames, and "Limitations" for its O(n) cost.
        """
        matches: list[MemoryRecord] = []
        for sanitized_key in self._files.list_keys():
            data = self._files.read(sanitized_key)
            if data is not None and data.get("namespace") == namespace:
                matches.append(_record_from_dict(data))
        return sorted(matches, key=lambda r: r.key)

    def delete(self, namespace: str, key: str) -> bool:
        """Delete the record file for ``namespace``/``key`` if present."""
        return self._files.delete(_composite_key(namespace, key))


def _record_from_dict(data: dict[str, Any]) -> MemoryRecord:
    """Reconstruct a `MemoryRecord` from its stored dict form."""
    return MemoryRecord(
        namespace=data["namespace"],
        key=data["key"],
        value=data["value"],
        timestamp=datetime.fromisoformat(data["timestamp"]),
    )


def save_findings(
    store: LongTermMemoryStore, namespace: str, findings: list[ResearchFinding]
) -> None:
    """Persist a list of research findings into long-term memory.

    Each finding is stored under a key derived from its position in the
    list combined with its source, so repeated calls with overlapping
    findings overwrite rather than duplicate entries.

    Args:
        store: The `LongTermMemoryStore` to write into.
        namespace: The namespace to store findings under (e.g. a task ID
            or topic).
        findings: The findings to persist.
    """
    for index, finding in enumerate(findings):
        key = f"{index}:{finding.source}"
        store.put(
            namespace,
            key,
            {
                "source": finding.source,
                "content": finding.content,
                "collected_by": finding.collected_by.value,
                "confidence": finding.confidence,
            },
        )
