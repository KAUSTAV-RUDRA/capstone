"""I/O helpers for corpora, results, and JSON/CSV artefacts.

Centralises reading/writing so experiments write ``results/*.csv`` uniformly
and every corpus file is UTF-8 JSON-lines.

Serves: cross-cutting, all phases of docs/master-execution-plan.md.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)


def read_jsonl(path: str | Path, skip_bad_lines: bool = False) -> list[dict[str, Any]]:
    """Read a JSON-lines file into a list of dicts.

    Args:
        path: File to read.
        skip_bad_lines: If True, malformed lines (e.g. a partial trailing line
            left by a killed writer) are logged and skipped instead of raising.
    """
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                if not skip_bad_lines:
                    raise ValueError(f"{path}:{lineno}: malformed JSON line") from exc
                log.warning("%s:%d: skipping malformed JSON line (%s)", path, lineno, exc)
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write dicts to a JSON-lines file (overwrites), creating parent dirs."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Append dicts to a JSON-lines file, creating it (and parent dirs) if needed."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv(path: str | Path) -> "pd.DataFrame":
    """Read a CSV file into a DataFrame."""
    import pandas as pd  # lazy

    return pd.read_csv(path)


def write_csv(df: "pd.DataFrame", path: str | Path) -> None:
    """Write a DataFrame to CSV (no index), creating parent dirs as needed."""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8")


def ensure_dir(path: str | Path) -> Path:
    """Create the directory (and parents) if missing; return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
