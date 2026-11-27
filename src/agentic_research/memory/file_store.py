"""A generic, offline JSON-file-backed key-value store.

Purpose
-------
Both short-term memory (`checkpoint.py`) and long-term memory
(`long_term.py`) need to persist JSON-serializable dicts to disk, keyed by
a string. Rather than each writing its own file I/O (which would violate
DRY and double the surface area needing an audit), both depend on this one
class.

Design decisions
-----------------
* **One file per key**, named after a sanitized version of the key, rather
  than one big JSON file for the whole store. This keeps writes cheap
  (write one small file, not rewrite an ever-growing single file) and
  makes the store trivially inspectable on disk during development (one
  file = one entry).
* **Keys are sanitized for the filesystem** (non-alphanumeric characters
  replaced), not used verbatim as filenames — protects against path
  traversal (a key like ``"../../etc/passwd"``) and against characters
  invalid on some filesystems. This is a real, if narrow, security
  consideration for anything that turns caller-supplied strings into file
  paths.
* **Genuinely real and offline**, unlike Module 3/4's `InMemory*` mocks:
  this class needs no network and no unavailable package — it's stdlib
  `json` + `pathlib`, so it reaches full **Verified** status including
  actual file I/O, tested against a real temporary directory.

Limitations
------------
* Key sanitization means two distinct keys that sanitize to the same
  string (e.g. keys differing only in punctuation) would collide. Not a
  concern for this project's key shapes (thread IDs, namespace/key pairs
  built internally), but worth flagging for any future caller passing
  arbitrary user-supplied strings as keys directly.
* No file locking — concurrent writes to the same key from multiple
  processes could race. Acceptable for this project's current
  single-process usage; would need addressing before any concurrent
  production use.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agentic_research.core.exceptions import StateSerializationError

_SAFE_KEY_PATTERN = re.compile(r"[^A-Za-z0-9_.-]")


def _sanitize_key(key: str) -> str:
    """Convert an arbitrary string key into a safe filename stem.

    Args:
        key: The caller-supplied key.

    Returns:
        A filesystem-safe string: every character outside
        ``[A-Za-z0-9_.-]`` is replaced with ``"_"``.

    Raises:
        ValueError: If ``key`` is empty.
    """
    if not key:
        raise ValueError("key must be a non-empty string.")
    return _SAFE_KEY_PATTERN.sub("_", key)


class JSONFileStore:
    """Persists JSON-serializable dicts to disk, one file per key."""

    def __init__(self, base_dir: str | Path) -> None:
        """Initialize the store, creating ``base_dir`` if it doesn't exist.

        Args:
            base_dir: Directory where entry files are written. Created
                (including parents) if missing.
        """
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        """Return the file path a given key is stored under."""
        return self._base_dir / f"{_sanitize_key(key)}.json"

    def write(self, key: str, data: dict[str, Any]) -> None:
        """Write ``data`` to disk under ``key``, overwriting any prior value.

        Args:
            key: The entry's key.
            data: A JSON-serializable dict.
        """
        path = self._path_for(key)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def read(self, key: str) -> dict[str, Any] | None:
        """Read the entry stored under ``key``, if any.

        Args:
            key: The entry's key.

        Returns:
            The stored dict, or ``None`` if no entry exists for this key.

        Raises:
            StateSerializationError: If the file exists but contains
                invalid JSON (e.g. corrupted on disk).
        """
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StateSerializationError(
                f"Corrupted JSON file for key {key!r}: {exc}",
                details={"key": key, "path": str(path)},
            ) from exc

    def delete(self, key: str) -> bool:
        """Delete the entry stored under ``key``, if any.

        Args:
            key: The entry's key.

        Returns:
            ``True`` if an entry existed and was deleted, ``False`` if
            there was nothing to delete.
        """
        path = self._path_for(key)
        if not path.exists():
            return False
        path.unlink()
        return True

    def list_keys(self) -> list[str]:
        """List every sanitized key currently stored.

        Returns:
            A sorted list of sanitized key strings (filenames without the
            ``.json`` extension). Note this returns the *sanitized* form,
            which may differ from the original key if it contained
            characters outside ``[A-Za-z0-9_.-]`` — callers that need the
            original key back should track it themselves (as
            `long_term.py`'s `MemoryRecord.key` does, by storing the
            original key inside the JSON payload itself).
        """
        return sorted(p.stem for p in self._base_dir.glob("*.json"))
