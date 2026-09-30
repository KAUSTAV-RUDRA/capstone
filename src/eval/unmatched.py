"""Scores the length-match-trimmed surplus human rows (T5 robustness check).

``splits.json`` records, under ``excluded.length_match``, the human rows that
Part 13's length matching trimmed out of the corpus (en/hi/te trim the human
side; cm trims machine, so cm has none). They are not in ``corpus.jsonl`` and so
not in the scores table. The check asks: does the adopted detector flag these
human texts more than the matched test humans? They are human writing from the
same sources, just at lengths the matching discarded.

Head A and Head B are run exactly as at scoring time (whitespace-collapsed text,
the same fitted Head A, the same mGPT scorer), cached apart from the corpus
caches so the corpus feature files are never touched.

Serves docs/master-execution-plan.md Phase 3 §3.6 (T5).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import read_jsonl
from src.utils.text import normalise_rows

SPLITS_PATH = "data/processed/splits.json"
CACHE_PATH = "results/norm/unmatched_scores.parquet"


def unmatched_human_rows(config: dict[str, Any], splits_path: str = SPLITS_PATH) -> list[dict[str, Any]]:
    """Raw human rows whose ids ``splits.json`` lists as length-match-excluded."""
    excluded = set(json.loads(Path(splits_path).read_text(encoding="utf-8"))["excluded"]["length_match"])
    human_dir = Path(config["human_corpus"]["output_dir"])
    rows = [r for f in sorted(human_dir.glob("*.jsonl")) for r in read_jsonl(f) if r["id"] in excluded]
    return normalise_rows(rows)


def score_unmatched(data_config: dict[str, Any], models_config: dict[str, Any],
                    cache_path: str = CACHE_PATH) -> pd.DataFrame:
    """``id, language, length_words, headA, headB`` for every unmatched human row, cached."""
    from src.eval.score import curvature_features, stylometric_features
    from src.features.stylometric import HeadA

    rows = unmatched_human_rows(data_config)
    cache = Path(cache_path)
    if cache.exists():
        table = pd.read_parquet(cache)
        if set(table["id"]) == {r["id"] for r in rows}:
            return table

    a_cfg = {**models_config["head_a"], "features_cache": str(cache.with_name("stylometric_unmatched.parquet"))}
    b_cfg = {**models_config["head_b"], "stats_cache": str(cache.with_name("curvature_unmatched.parquet"))}
    feats = stylometric_features(rows, a_cfg)
    head_a = HeadA.load(models_config["head_a"]["model_path"])
    X = np.asarray([feats[r["id"]] for r in rows], dtype=float)
    head_a_score = head_a.predict(X, [r["language"] for r in rows])
    curv = curvature_features(rows, b_cfg)
    table = pd.DataFrame({
        "id": [r["id"] for r in rows],
        "language": [r["language"] for r in rows],
        "length_words": [r.get("length_words") for r in rows],
        "headA": head_a_score,
        "headB": [curv[r["id"]][0] for r in rows],
    })
    cache.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cache, index=False)
    return table
