"""Tests for src.data.freeze_splits. Serves Phase 2 §2.1.1.

Critical: the overwrite guard must fire (non-negotiable #6).
"""
from __future__ import annotations

import random

import pytest

from src.data.freeze_splits import assign_splits, freeze_splits, length_match, load_frozen_splits


def _config(cal: int = 4) -> dict:
    return {
        "corpus": {"seed": 1, "length_bins": 2, "cal_human_per_bucket": cal, "train_share": 0.5},
        "min_human_calibration_per_bucket": cal,
        "heldout_generators": ["llama"],
        "human_corpus": {"buckets": {"cm": {"bands": {"native": {"sources": [
            {"name": "chat"}, {"name": "comments", "calibration_eligible": False}]}}}}},
    }


def _rows() -> list[dict]:
    human = [{"id": f"h{i}", "label": 0, "language": "cm", "length_words": 10 + i,
              "source": "chat" if i < 6 else "comments"} for i in range(12)]
    machine = [{"id": f"m{i}", "label": 1, "language": "cm", "length_words": 10 + i,
                "generator": "qwen7b", "prompt_id": f"h{i}"} for i in range(0, 12, 2)]
    machine.append({"id": "x0", "label": 1, "language": "cm", "length_words": 12,
                    "generator": "llama", "prompt_id": "h1"})
    return human + machine


def test_freeze_guard_refuses_overwrite(tmp_path) -> None:
    path = tmp_path / "splits.json"
    freeze_splits(_rows(), path, _config())
    assert sum(len(v) for v in load_frozen_splits(path).values()) == 19
    with pytest.raises(FileExistsError):
        freeze_splits(_rows(), path, _config())


def test_cal_only_from_eligible_sources_and_heldout_to_test() -> None:
    rows = _rows()
    human = [r for r in rows if r["label"] == 0]
    machine = [r for r in rows if r["label"] == 1]
    split, group = assign_splits(human, machine, config=_config(), rng=random.Random(0))
    cal = [r for r in human if split[r["id"]] == "cal"]
    assert len(cal) == 4 and all(r["source"] == "chat" for r in cal)
    assert split["x0"] == "test"
    for m in machine:                      # a machine row follows its prompt's group
        if m["generator"] == "qwen7b":
            assert split[m["id"]] == group[m["prompt_id"]]


def test_cal_floor_fails_loudly() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="calibration-eligible"):
        assign_splits([r for r in rows if r["label"] == 0], [], config=_config(cal=7),
                      rng=random.Random(0))


def test_length_match_follows_reference_bins_either_way() -> None:
    human = [{"id": f"h{i}", "length_words": w} for i, w in enumerate([10] * 5 + [100] * 5)]
    machine = [{"id": f"m{i}", "length_words": w} for i, w in enumerate([10] * 8 + [100] * 2)]
    edges = [50]
    kept, excluded, _ = length_match(human, machine, edges, random.Random(0))   # trim machine
    assert sorted(m["length_words"] for m in kept) == [10, 10, 100, 100]
    assert len(excluded) == 6
    kept, excluded, _ = length_match(machine, human, edges, random.Random(0))   # trim human
    assert sorted(h["length_words"] for h in kept) == [10] * 5 + [100]
    assert len(excluded) == 4
