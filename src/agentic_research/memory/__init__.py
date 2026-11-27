"""Memory layer: state serialization, short-term checkpointing, long-term storage.

Public API re-exported here, matching the pattern established by every
prior module's `__init__.py`.
"""

from agentic_research.memory.checkpoint import (
    CheckpointStore,
    FileCheckpointStore,
    InMemoryCheckpointStore,
)
from agentic_research.memory.file_store import JSONFileStore
from agentic_research.memory.long_term import (
    FileLongTermMemoryStore,
    InMemoryLongTermMemoryStore,
    LongTermMemoryStore,
    MemoryRecord,
    save_findings,
)
from agentic_research.memory.serialization import (
    state_from_dict,
    state_from_json,
    state_to_dict,
    state_to_json,
)

__all__ = [
    "JSONFileStore",
    "state_to_dict",
    "state_from_dict",
    "state_to_json",
    "state_from_json",
    "CheckpointStore",
    "InMemoryCheckpointStore",
    "FileCheckpointStore",
    "LongTermMemoryStore",
    "MemoryRecord",
    "InMemoryLongTermMemoryStore",
    "FileLongTermMemoryStore",
    "save_findings",
]
