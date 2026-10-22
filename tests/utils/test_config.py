"""Tests for src.utils.config."""
from __future__ import annotations

from src.utils.config import get, load_config, merge_configs


def test_load_and_merge_config(tmp_path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("a: 1\nb: {x: 1, y: 2}\n", encoding="utf-8")
    base = load_config(path)
    assert base == {"a": 1, "b": {"x": 1, "y": 2}}
    merged = merge_configs(base, {"b": {"y": 3, "z": 4}, "c": 5})
    assert merged == {"a": 1, "b": {"x": 1, "y": 3, "z": 4}, "c": 5}
    assert base["b"]["y"] == 2                      # inputs are not mutated
    assert get(merged, "b.y") == 3
    assert get(merged, "b.missing", "d") == "d"


def test_empty_file_is_empty_dict(tmp_path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path) == {}
