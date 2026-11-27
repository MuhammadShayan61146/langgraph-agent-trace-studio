"""Configuration layer: settings loading and logging setup.

Public API re-exported here so callers write::

    from agentic_research.config import load_settings, configure_logging

instead of reaching into submodules directly.
"""

from agentic_research.config.logging_config import configure_logging, get_logger
from agentic_research.config.settings import Settings, load_settings

__all__ = ["Settings", "load_settings", "configure_logging", "get_logger"]
