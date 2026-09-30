"""Raw-text vs whitespace-normalised scoring, side by side.

Reads the raw-text scores (results/scores.parquet) and the normalised scores
(one or more parquet files from ``--config configs/models_norm.yaml`` runs), and
prints per-bucket test AUROC for each column on SEEN generators and on
HELD-OUT generators, raw -> normalised. Then re-runs the nine-number
surface-format probe on raw and on normalised text: if the newline artefact was
what it was using, it should fall to about 0.5.

Usage::

    python scripts/report_norm_compare.py --config configs/data.yaml \
        --norm-scores results/norm/scores_a.parquet results/norm/scores_gpu.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_headc_v2 import fmt_features, lr  # noqa: E402

from src.utils.config import load_config  # noqa: E402
from src.utils.io import read_jsonl  # noqa: E402
from src.utils.text import collapse_whitespace  # noqa: E402

BUCKETS = ["en", "hi", "te", "cm"]
COLUMNS = ["headA", "headB", "headB_word", "headC", "fastdetectgpt_en", "ppl", "binoculars"]


def auc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--raw-scores", default="results/scores.parquet")
    ap.add_argument("--norm-scores", nargs="+", required=True)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    gens = load_config(args.config)["machine_corpus"]["generators"]
    seen = {g for g, v in gens.items() if v["role"] == "seen"}
    held = {g for g, v in gens.items() if v["role"] == "heldout"}
    raw = pd.read_parquet(args.raw_scores).set_index("id")
    norm = pd.concat([pd.read_parquet(p).set_index("id") for p in args.norm_scores], axis=1)
    corpus = pd.DataFrame(read_jsonl(args.corpus))
    test = corpus[corpus.split == "test"]
    human = test.label == 0

    for title, roles in (("SEEN generators", seen), ("HELD-OUT generators (llama)", held)):
        sub_all = test[human | test.generator.isin(roles)]
        print(f"\n== TEST AUROC, {title}: raw -> normalised (delta) ==")
        print(f"{'bucket':6} {'h/m':>9}  " + "  ".join(f"{c:>26}" for c in COLUMNS))
        for b in BUCKETS:
            s = sub_all[sub_all.language == b]
            y = s.label.to_numpy()
            if len(set(y)) < 2:
                continue
            cells = []
            for c in COLUMNS:
                if c not in norm.columns:
                    cells.append("not scored")
                    continue
                r = auc(y, raw.loc[s.id, c]); n = auc(y, norm.loc[s.id, c])
                cells.append(f"{r:.3f} -> {n:.3f} ({n - r:+.3f})")
            print(f"{b:6} {int((y == 0).sum()):>4}/{int((y == 1).sum()):<4}  " + "  ".join(f"{x:>26}" for x in cells))

    print("\n== SURFACE-FORMAT PROBE (9 numbers, LR fit on train): raw -> normalised ==")
    print("features: newline count, digit share, punct share, ascii share, ends-with-terminal-punct,")
    print("starts-uppercase, quote share, symbol share, mean word length")
    print(f"{'bucket':6}{'SEEN raw':>10}{'SEEN norm':>11}{'LLAMA raw':>11}{'LLAMA norm':>12}")
    for b in BUCKETS:
        tr = corpus[(corpus.language == b) & (corpus.split == "train")]
        te = corpus[(corpus.language == b) & (corpus.split == "test")]
        out = {}
        for tag, f in (("raw", lambda x: x), ("norm", collapse_whitespace)):
            m = lr().fit(fmt_features([f(t) for t in tr.text]), tr.label)
            for name, roles in (("seen", seen), ("held", held)):
                s = te[(te.label == 0) | te.generator.isin(roles)]
                out[(name, tag)] = (auc(s.label, m.predict_proba(fmt_features([f(t) for t in s.text]))[:, 1])
                                    if s.label.nunique() == 2 else float("nan"))
        print(f"{b:6}{out[('seen', 'raw')]:>10.3f}{out[('seen', 'norm')]:>11.3f}"
              f"{out[('held', 'raw')]:>11.3f}{out[('held', 'norm')]:>12.3f}")

    print("\n== NEWLINES PER TEXT after normalisation (must be 0 everywhere) ==")
    print(corpus.text.map(collapse_whitespace).str.count("\n").sum(), "newlines remain in normalised corpus")


if __name__ == "__main__":
    main()
