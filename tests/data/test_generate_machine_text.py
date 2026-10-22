"""Held-out generators must never appear outside test (non-negotiable #4)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import load_config
from src.utils.io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "processed" / "corpus.jsonl"


def test_heldout_generators_are_test_only() -> None:
    if not CORPUS.exists():
        pytest.skip(f"{CORPUS} not built (data/ is gitignored)")
    heldout = set(load_config(ROOT / "configs" / "data.yaml")["heldout_generators"])
    rows = [r for r in read_jsonl(CORPUS) if r.get("generator") in heldout]
    assert rows, "no held-out generator rows found; the check would be vacuous"
    assert {r["split"] for r in rows} == {"test"}
