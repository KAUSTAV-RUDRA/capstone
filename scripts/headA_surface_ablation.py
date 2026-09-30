"""Is Head A's signal reducible to the surface-format probe?

Per bucket, on SEEN test generators and on held-out llama (hi), all on
whitespace-normalised text (the cached normalised Head A features):

  (a) the 9-number surface-format probe alone (scripts/diagnose_headc_v2.fmt_features)
  (b) Head A alone: all 40 features, per-bucket standardise -> logistic (= HeadA)
  (c) Head A with the surface-format-adjacent features removed
        c1  drop features the probe reads or derives from the same characters:
            digit_ratio, uppercase_ratio, punct_ratio, comma_per_word,
            period_per_sent, question_ratio, exclaim_ratio, quote_ratio,
            punct_diversity, mean_word_len, std_word_len, word_len_burstiness,
            mean_clauses_per_sent, max_clauses_per_sent (clauses = punctuation marks)
        c2  c1, plus sentence-length features (segmented on punctuation):
            mean_sent_len, std_sent_len, sent_len_burstiness
        syntax-only  the 15 parser features (UPOS shares, POS n-gram, tree depth)

If (c) collapses toward (a), Head A is largely surface artefact; if (c) holds,
the stylometric signal is real beyond it.

Usage::

    python scripts/headA_surface_ablation.py --config configs/data.yaml
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

from src.features.stylometric import _FEATURE_NAMES, SYNTAX_FEATURE_NAMES  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import read_jsonl  # noqa: E402
from src.utils.text import collapse_whitespace  # noqa: E402

BUCKETS = ["en", "hi", "te", "cm"]
NAMES = list(_FEATURE_NAMES) + list(SYNTAX_FEATURE_NAMES)
FORMAT_ADJACENT = ["digit_ratio", "uppercase_ratio", "punct_ratio", "comma_per_word", "period_per_sent",
                   "question_ratio", "exclaim_ratio", "quote_ratio", "punct_diversity", "mean_word_len",
                   "std_word_len", "word_len_burstiness", "mean_clauses_per_sent", "max_clauses_per_sent"]
SENT_LEN = ["mean_sent_len", "std_sent_len", "sent_len_burstiness"]
VARIANTS = {
    "(b) Head A, all 40": NAMES,
    "(c1) minus format-adjacent": [n for n in NAMES if n not in FORMAT_ADJACENT],
    "(c2) c1 minus sent-length": [n for n in NAMES if n not in FORMAT_ADJACENT + SENT_LEN],
    "syntax-only (15 parser)": list(SYNTAX_FEATURE_NAMES),
}


def auc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/data.yaml (generator roles)")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--features", default="results/norm/features/stylometric.parquet",
                    help="cached Head A features computed on NORMALISED text")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    gens = load_config(args.config)["machine_corpus"]["generators"]
    seen = {g for g, v in gens.items() if v["role"] == "seen"}
    held = {g for g, v in gens.items() if v["role"] == "heldout"}
    corpus = pd.DataFrame(read_jsonl(args.corpus))
    feats = pd.read_parquet(args.features).set_index("id")["features"]
    X = pd.DataFrame(np.stack(feats.loc[corpus.id].to_numpy()), index=corpus.id, columns=NAMES)
    assert X.shape[1] == 40, X.shape

    rows = []
    for b in BUCKETS:
        tr = corpus[(corpus.language == b) & (corpus.split == "train")]
        te = corpus[(corpus.language == b) & (corpus.split == "test")]
        sets = {"SEEN": te[(te.label == 0) | te.generator.isin(seen)],
                "LLAMA": te[(te.label == 0) | te.generator.isin(held)]}
        norm_text = lambda d: [collapse_whitespace(t) for t in d.text]  # noqa: E731
        probe = lr().fit(fmt_features(norm_text(tr)), tr.label)
        rows.append((b, "(a) 9-number surface probe", {
            k: (auc(s.label, probe.predict_proba(fmt_features(norm_text(s)))[:, 1]) if s.label.nunique() == 2 else np.nan)
            for k, s in sets.items()}))
        for name, cols in VARIANTS.items():
            m = lr().fit(X.loc[tr.id, cols].to_numpy(), tr.label)
            rows.append((b, name, {k: (auc(s.label, m.predict_proba(X.loc[s.id, cols].to_numpy())[:, 1])
                                       if s.label.nunique() == 2 else np.nan) for k, s in sets.items()}))

    print(f"features per variant: " + ", ".join(f"{k.split(' ')[0]}={len(v)}" for k, v in VARIANTS.items()))
    print(f"\n{'bucket':6}{'variant':32}{'SEEN test':>11}{'held-out llama':>16}")
    for b, name, r in rows:
        ll = "" if np.isnan(r["LLAMA"]) else f"{r['LLAMA']:.3f}"
        print(f"{b:6}{name:32}{r['SEEN']:>11.3f}{ll:>16}")
        if name.startswith("syntax"):
            print()


if __name__ == "__main__":
    main()
