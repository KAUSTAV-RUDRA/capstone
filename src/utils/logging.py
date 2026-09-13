"""Logging configuration for research scripts.

Provides a consistent logger so ``docs/progress.md`` evidence and experiment
runs are traceable (Rubric #7).

Serves: cross-cutting, all phases of docs/master-execution-plan.md.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure the root logger's format, level, and optional file handler.

    Args:
        level: Logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
        log_file: Optional path to also write logs to (UTF-8).
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_FORMAT,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Args:
        name: Usually ``__name__`` of the caller.
    """
    return logging.getLogger(name)
