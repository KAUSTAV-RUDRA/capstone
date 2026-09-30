"""Scoring-time text transforms. The frozen corpus is never rewritten; these are
applied to rows in memory, identically to human and machine text."""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

_WS_RE = re.compile(r"\s+")


def collapse_whitespace(text: str) -> str:
    """Newlines and runs of any Unicode whitespace -> one space, ends stripped."""
    return _WS_RE.sub(" ", text).strip()


def normalise_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copies of ``rows`` with ``text`` whitespace-collapsed; all other fields unchanged."""
    return [{**r, "text": collapse_whitespace(r["text"])} for r in rows]
