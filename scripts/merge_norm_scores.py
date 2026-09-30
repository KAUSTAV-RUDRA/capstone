"""Join the normalised per-column score tables into the one wide table the pipeline reads.

results/norm/scores_a.parquet (headA) + results/norm/scores_gpu.parquet (headB, headB_word,
headC, fastdetectgpt_en, ppl, binoculars) -> ``pipeline.scores_path`` of the given config.
Every column must be scored on the same normalised text (decisions.md 2026-09-30).

Usage: python -m scripts.merge_norm_scores --config configs/models_norm.yaml
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.utils.config import load_config

EXPECTED = ["headA", "headB", "headB_word", "headC", "fastdetectgpt_en", "ppl", "binoculars"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--parts", nargs="+", default=["results/norm/scores_a.parquet", "results/norm/scores_gpu.parquet"])
    args = ap.parse_args()
    out = load_config(args.config)["pipeline"]["scores_path"]
    merged = None
    for p in args.parts:
        t = pd.read_parquet(p).set_index("id")
        merged = t if merged is None else merged.join(t, how="outer", rsuffix="_dup")
    missing = [c for c in EXPECTED if c not in merged.columns or merged[c].isna().any()]
    if missing:
        raise SystemExit(f"columns missing or with NaN: {missing}")
    merged.reset_index().to_parquet(out, index=False)
    print(f"{out}: {merged.shape[0]} rows, columns {list(merged.columns)}")


if __name__ == "__main__":
    main()
