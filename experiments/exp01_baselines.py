"""exp01 - all baselines, AUROC + F1 per bucket (Table T1; T3 on held-out).

Table T1: AUROC + F1 per bucket for the eight methods that exist today (headB_word added) --
three raw-statistic baselines (ppl, binoculars, fastdetectgpt_en), the three
heads (headA, headB, headC) and the adopted fusion (fused_ab = headA + headB
+ bucket one-hot; headC is excluded from fusion per
docs/results/headc_diagnosis.md). xlmr and DetectGPT are not in this run (see
score.py's registry notes).

F1's decision threshold is fit per bucket per method on the TRAIN split (the
best-F1 threshold search, :func:`src.eval.metrics.best_f1_threshold`) and
applied unchanged to TEST -- the frozen ``cal`` split (locked §5) is
human-only and carries no F1 signal to tune against.

Writes: results/exp01_baselines.csv, results/T1_main_results.csv
Serves: docs/master-execution-plan.md Phase 2 §2.2.1 (and §2.2.7 for T3).
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

RESULTS_CSV = "results/exp01_baselines.csv"
T1_CSV = "results/T1_main_results.csv"

METHODS = ("ppl", "binoculars", "fastdetectgpt_en", "headA", "headB", "headB_word", "headC", "fused_ab")
METHOD_LABELS = {"fused_ab": "fused (A+B)"}


def run(config_path: str) -> "pd.DataFrame":
    """Evaluate every baseline + head + fusion per bucket and return the T1 table."""
    import pandas as pd

    from src.data.schema import LANGUAGE_BUCKETS
    from src.eval.metrics import auroc, best_f1_threshold, f1_at_threshold
    from src.eval.pipeline import fit_full_pipeline, generator_roles

    artifacts = fit_full_pipeline(config_path)
    _, heldout = generator_roles()
    corpus_by_bucket_split: dict[tuple[str, str], list[dict]] = {}
    for r in artifacts.corpus:
        corpus_by_bucket_split.setdefault((r["language"], r["split"]), []).append(r)

    rows: list[dict] = []
    for bucket in LANGUAGE_BUCKETS:
        train = corpus_by_bucket_split.get((bucket, "train"), [])
        test = corpus_by_bucket_split.get((bucket, "test"), [])
        y_train = [r["label"] for r in train]
        y_test = [r["label"] for r in test]
        for method in METHODS:
            s_train = artifacts.scores.loc[[r["id"] for r in train], method].to_numpy(dtype=float)
            s_test = artifacts.scores.loc[[r["id"] for r in test], method].to_numpy(dtype=float)
            threshold = best_f1_threshold(y_train, s_train)
            # seen-only / held-out-only AUROC: human test rows are shared, only the machine side differs.
            seen_rows = [r for r in test if r["label"] == 0 or r["generator"] not in heldout]
            held_rows = [r for r in test if r["label"] == 0 or r["generator"] in heldout]
            auroc_seen = auroc([r["label"] for r in seen_rows],
                               artifacts.scores.loc[[r["id"] for r in seen_rows], method].to_numpy(dtype=float))
            has_held = any(r["label"] == 1 for r in held_rows)
            auroc_held = (auroc([r["label"] for r in held_rows],
                                artifacts.scores.loc[[r["id"] for r in held_rows], method].to_numpy(dtype=float))
                          if has_held else float("nan"))
            rows.append({
                "bucket": bucket,
                "method": METHOD_LABELS.get(method, method),
                "n_human": int(sum(1 for l in y_test if l == 0)),
                "n_machine": int(sum(1 for l in y_test if l == 1)),
                "auroc": auroc(y_test, s_test),
                "auroc_seen": auroc_seen,
                "auroc_heldout": auroc_held,
                "f1": f1_at_threshold(y_test, s_test, threshold),
                "f1_threshold": threshold,
            })
    df = pd.DataFrame(rows)

    from src.utils.io import write_csv
    write_csv(df, RESULTS_CSV)
    write_csv(df, T1_CSV)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    args = parser.parse_args()
    df = run(args.config)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
