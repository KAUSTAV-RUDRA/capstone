"""Logging configuration for research scripts.

Provides a consistent logger so ``docs/progress.md`` evidence and experiment
runs are traceable (Rubric #7).

Serves: cross-cutting, all phases of docs/master-execution-plan.md.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging


def configure_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure the root logger's format, level, and optional file handler.

    Args:
        level: Logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
        log_file: Optional path to also write logs to.
    """
    # TODO(phase-1 step-1.2): configure stdlib logging.
    raise NotImplementedError


def get_logger(name: str) -> "logging.Logger":
    """Return a module-scoped logger.

    Args:
        name: Usually ``__name__`` of the caller.

    Returns:
        A configured :class:`logging.Logger`.
    """
    # TODO(phase-1 step-1.2): return a named logger.
    raise NotImplementedError
