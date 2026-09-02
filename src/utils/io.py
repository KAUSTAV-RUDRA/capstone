"""I/O helpers for corpora, results, and JSON/CSV artefacts.

Centralises reading/writing so experiments write ``results/*.csv`` uniformly.

Serves: cross-cutting, all phases of docs/master-execution-plan.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file into a list of dicts."""
    # TODO(phase-2 step-2.1.2): implement JSONL read.
    raise NotImplementedError


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write a list of dicts to a JSON-lines file."""
    # TODO(phase-2 step-2.1.2): implement JSONL write.
    raise NotImplementedError


def read_csv(path: str | Path) -> "pd.DataFrame":
    """Read a CSV file into a DataFrame."""
    # TODO(phase-2 step-2.1.2): implement CSV read.
    raise NotImplementedError


def write_csv(df: "pd.DataFrame", path: str | Path) -> None:
    """Write a DataFrame to CSV, creating parent dirs as needed."""
    # TODO(phase-2 step-2.1.2): implement CSV write.
    raise NotImplementedError


def ensure_dir(path: str | Path) -> Path:
    """Create the directory (and parents) if missing; return it as a Path."""
    # TODO(phase-2 step-2.1.2): implement mkdir -p.
    raise NotImplementedError
