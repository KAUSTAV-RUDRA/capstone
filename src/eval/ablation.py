"""Ablation sweep: A / B / C / A+B / A+B+C / +temperature / +conformal / +abstain (T2).

Isolates the contribution of each component, demonstrating that calibration and
abstention (non-negotiable #3) add measurable value. See
``experiments/exp08_ablation.py`` module docstring for the full per-row
rationale (why AUROC/F1 are identical for A+B vs +temperature by construction,
why A+B+C is reference-only, etc).

Serves docs/master-execution-plan.md Phase 3 §3.7.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# Ablation configurations, in reporting order (T2 rows). Head C is scored
# standalone (C) and in the reference-only A+B+C fusion, but excluded from the
# adopted A+B pipeline that +temperature/+conformal/+abstain build on
# (docs/results/headc_diagnosis.md).
ABLATION_CONFIGS: tuple[str, ...] = (
    "A", "B", "C", "A+B", "A+B+C", "+temperature", "+conformal", "+abstain",
)


def run_ablation(config_path: str = "configs/default.yaml") -> "pd.DataFrame":
    """Run every ablation configuration and return the T2 table per bucket.

    Columns: bucket, config, auroc, f1, fpr, coverage, accuracy. AUROC/F1 are
    None for ``+abstain`` (a 3-way decision, not a ranked score); fpr/accuracy
    are None for the ranked-score rows (A/B/C/A+B/A+B+C/+temperature), where
    coverage is trivially 1.0 (no abstention band yet).
    """
    import numpy as np
    import pandas as pd

    from src.data.schema import LANGUAGE_BUCKETS
    from src.eval.metrics import auroc, best_f1_threshold, f1_at_threshold, false_positive_rate
    from src.eval.pipeline import fit_full_pipeline

    artifacts = fit_full_pipeline(config_path)
    by_split_bucket = {
        (split, bucket): [r for r in artifacts.corpus if r["split"] == split and r["language"] == bucket]
        for split in ("train", "test") for bucket in LANGUAGE_BUCKETS
    }

    rows: list[dict] = []
    for bucket in LANGUAGE_BUCKETS:
        train, test = by_split_bucket[("train", bucket)], by_split_bucket[("test", bucket)]
        y_tr = np.array([r["label"] for r in train])
        y_te = np.array([r["label"] for r in test])
        ids_te = [r["id"] for r in test]

        for cfg, col in (("A", "headA"), ("B", "headB"), ("C", "headC")):
            s_tr = artifacts.scores.loc[[r["id"] for r in train], col].to_numpy(dtype=float)
            s_te = artifacts.scores.loc[ids_te, col].to_numpy(dtype=float)
            t = best_f1_threshold(y_tr, s_tr)
            rows.append({"bucket": bucket, "config": cfg, "auroc": auroc(y_te, s_te),
                        "f1": f1_at_threshold(y_te, s_te, t), "fpr": None,
                        "coverage": 1.0, "accuracy": None})

        for cfg, col in (("A+B", "fused_ab"), ("A+B+C", "fused_abc")):
            s_te = artifacts.scores.loc[ids_te, col].to_numpy(dtype=float)
            rows.append({"bucket": bucket, "config": cfg, "auroc": auroc(y_te, s_te),
                        "f1": f1_at_threshold(y_te, s_te, 0.5), "fpr": None,
                        "coverage": 1.0, "accuracy": None})

        s_cal = artifacts.scores.loc[ids_te, "fused_ab_calibrated"].to_numpy(dtype=float)
        rows.append({"bucket": bucket, "config": "+temperature", "auroc": auroc(y_te, s_cal),
                    "f1": f1_at_threshold(y_te, s_cal, 0.5), "fpr": None,
                    "coverage": 1.0, "accuracy": None})

        tau_005 = artifacts.conformal[0.05].threshold(bucket)
        preds_conf = (s_cal > tau_005).astype(int)
        rows.append({"bucket": bucket, "config": "+conformal", "auroc": auroc(y_te, s_cal),
                    "f1": f1_at_threshold(y_te, s_cal, tau_005), "fpr": false_positive_rate(y_te, preds_conf),
                    "coverage": 1.0, "accuracy": float((preds_conf == y_te).mean())})

        verdicts = artifacts.abstention.decide_batch(s_cal, [bucket] * len(s_cal))
        verdict_labels = np.array([v for v, _ in verdicts])
        covered = verdict_labels != "ABSTAIN"
        preds_abstain = (verdict_labels == "MACHINE").astype(int)
        if covered.any():
            acc = float((preds_abstain[covered] == y_te[covered]).mean())
            human_covered = covered & (y_te == 0)
            fpr = float((preds_abstain[human_covered] == 1).mean()) if human_covered.any() else float("nan")
        else:
            acc, fpr = float("nan"), float("nan")
        rows.append({"bucket": bucket, "config": "+abstain", "auroc": None, "f1": None,
                    "fpr": fpr, "coverage": float(covered.mean()), "accuracy": acc})

    df = pd.DataFrame(rows)
    df["config"] = pd.Categorical(df["config"], categories=ABLATION_CONFIGS, ordered=True)
    return df.sort_values(["bucket", "config"]).reset_index(drop=True)
