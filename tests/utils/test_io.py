"""Tests for src.utils.io."""
from __future__ import annotations

import pandas as pd

from src.utils.io import append_jsonl, read_csv, read_jsonl, write_csv, write_jsonl


def test_jsonl_and_csv_roundtrip(tmp_path) -> None:
    rows = [{"id": "a", "text": "नमस्ते", "label": 0}, {"id": "b", "text": "hello", "label": 1}]
    path = tmp_path / "sub" / "rows.jsonl"
    write_jsonl(rows, path)
    assert read_jsonl(path) == rows
    append_jsonl([{"id": "c", "text": "x", "label": 1}], path)
    assert [r["id"] for r in read_jsonl(path)] == ["a", "b", "c"]

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "out" / "t.csv"
    write_csv(df, csv_path)
    pd.testing.assert_frame_equal(read_csv(csv_path), df)
