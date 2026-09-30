"""Per-bucket test AUROC for every scored column on the v2 corpus.

Two tables, both computed on ``split == "test"``:

* SEEN   - human rows + machine rows from seen generators, per bucket.
* HELD-OUT - human rows + machine rows from held-out generators (llama), per
  bucket that has any. Human test rows are shared with the SEEN table; the
  held-out generator only replaces the machine side (non-negotiable #4).

Roles come from ``machine_corpus.generators`` in the data config, not from
hardcoded names.

Usage::

    python scripts/report_v2_auroc.py --config configs/data.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config  # noqa: E402
from src.utils.io import read_jsonl  # noqa: E402

COLUMNS = ["headA", "headB", "headB_word", "fastdetectgpt_en", "ppl", "binoculars", "headC"]
BUCKETS = ["en", "hi", "te", "cm"]


def auroc(y: np.ndarray, s: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    h, m = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    point = float(roc_auc_score(y, s))
    boots = [roc_auc_score(y[i], s[i]) for i in
             (np.concatenate([rng.choice(h, len(h)), rng.choice(m, len(m))]) for _ in range(n_boot))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def table(rows: pd.DataFrame, scores: pd.DataFrame, cols: list[str], title: str, n_boot: int) -> str:
    rng = np.random.default_rng(0)
    lines = [f"\n== {title} ==", f"{'bucket':6} {'h/m':>9}  " + "  ".join(f"{c:>21}" for c in cols)]
    for b in BUCKETS:
        sub = rows[rows["language"] == b]
        y = sub["label"].to_numpy()
        if len(set(y)) < 2:
            continue
        cells = []
        for c in cols:
            s = scores.loc[sub["id"], c].to_numpy(dtype=float)
            if np.isnan(s).any():
                cells.append("nan")
                continue
            p, lo, hi = auroc(y, s, n_boot, rng)
            cells.append(f"{p:.3f} [{lo:.3f},{hi:.3f}]")
        lines.append(f"{b:6} {int((y == 0).sum()):>4}/{int((y == 1).sum()):<4}  " + "  ".join(f"{x:>21}" for x in cells))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/data.yaml (generator roles)")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--scores", default="results/scores.parquet")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    gens = load_config(args.config)["machine_corpus"]["generators"]
    seen = {g for g, v in gens.items() if v["role"] == "seen"}
    held = {g for g, v in gens.items() if v["role"] == "heldout"}

    scores = pd.read_parquet(args.scores).set_index("id")
    cols = [c for c in COLUMNS if c in scores.columns]
    missing = [c for c in COLUMNS if c not in scores.columns]
    if missing:
        print(f"not scored yet: {', '.join(missing)}")
    corpus = pd.DataFrame(read_jsonl(args.corpus))
    test = corpus[(corpus["split"] == "test") & corpus["id"].isin(scores.index)]
    human = test["label"] == 0
    print(table(test[human | test["generator"].isin(seen)], scores, cols,
                f"TEST, SEEN generators ({', '.join(sorted(seen))})", args.n_boot))
    print(table(test[human | test["generator"].isin(held)], scores, cols,
                f"TEST, HELD-OUT generators ({', '.join(sorted(held))})", args.n_boot))


if __name__ == "__main__":
    main()
