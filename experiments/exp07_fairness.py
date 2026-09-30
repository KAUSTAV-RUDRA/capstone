"""exp07 - fairness audit: FPR by bucket and by L1/L2 band, ours vs baselines (T5).

Quantifies the reduction in native/non-native false-positive disparity (RQ4)
using the fused_ab (A+B) system's calibrated probability against both the
PER-BUCKET conformal threshold (the adopted policy, non-negotiable #3) and a
GLOBAL pooled threshold (shown only as the comparison the per-bucket policy
argues against -- never adopted).

The L1/L2 split only exists for en (writer_L1_band in {general, indian});
hi/te/cm calibration data is all `native` (see src/data/schema.py's
WRITER_L1_BANDS comment) so no split is possible there yet.

The unmatched-human robustness check ("do the length_match-trimmed surplus
human rows in splits.json's `excluded` block get flagged more?") scores those
rows with headA + mGPT (GPU, cached in results/norm/unmatched_scores.parquet)
and applies the same fuser, temperature and tau; see src/eval/unmatched.py.

Writes: results/exp07_fairness.csv, results/T5_fairness.csv
Serves: docs/master-execution-plan.md Phase 3 §3.6.
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

RESULTS_CSV = "results/exp07_fairness.csv"
T5_CSV = "results/T5_fairness.csv"

ALPHAS = (0.01, 0.05)


def run(config_path: str) -> "pd.DataFrame":
    """Compute FPR by bucket and L1 band, per-bucket vs global tau; write the T5 table."""
    import numpy as np
    import pandas as pd

    from src.data.schema import LANGUAGE_BUCKETS
    from src.eval.fairness_audit import fpr_by_bucket, fpr_by_l1_band
    from src.eval.pipeline import fit_full_pipeline
    from src.utils.io import write_csv

    artifacts = fit_full_pipeline(config_path)
    test_human = [r for r in artifacts.corpus if r["split"] == "test" and r["label"] == 0]
    ids = [r["id"] for r in test_human]
    buckets = [r["language"] for r in test_human]
    labels = np.zeros(len(test_human), dtype=int)  # human-only by construction

    rows: list[dict] = []
    for alpha in ALPHAS:
        calibrator = artifacts.conformal[alpha]
        probs = artifacts.scores.loc[ids, "fused_ab_calibrated"].to_numpy(dtype=float)
        tau_by_bucket = {b: calibrator.threshold(b) for b in LANGUAGE_BUCKETS}
        tau_global = calibrator.threshold("_global")

        preds_bucket_tau = np.array([1 if p > tau_by_bucket[b] else 0 for p, b in zip(probs, buckets)])
        preds_global_tau = (probs > tau_global).astype(int)
        fpr_bucket = fpr_by_bucket(labels, preds_bucket_tau, buckets)
        fpr_global = fpr_by_bucket(labels, preds_global_tau, buckets)
        n_by_bucket = {b: buckets.count(b) for b in LANGUAGE_BUCKETS}
        for bucket in LANGUAGE_BUCKETS:
            if n_by_bucket[bucket] == 0:
                continue
            rows.append({
                "alpha": alpha, "bucket": bucket, "group": "all",
                "n_human": n_by_bucket[bucket],
                "fpr_per_bucket_tau": fpr_bucket[bucket],
                "fpr_global_tau": fpr_global[bucket],
                "tau_per_bucket": tau_by_bucket[bucket], "tau_global": tau_global,
            })

        en_mask = [b == "en" for b in buckets]
        en_l1_bands = [r["writer_L1_band"] for r, m in zip(test_human, en_mask) if m]
        en_labels = labels[en_mask] if any(en_mask) else labels[:0]
        l1_bucket = fpr_by_l1_band(en_labels, preds_bucket_tau[en_mask], en_l1_bands)
        l1_global = fpr_by_l1_band(en_labels, preds_global_tau[en_mask], en_l1_bands)
        for band in ("general", "indian"):
            if band not in l1_bucket:
                continue
            n_band = sum(1 for b in en_l1_bands if b == band)
            rows.append({
                "alpha": alpha, "bucket": "en", "group": f"writer_L1_band={band}",
                "n_human": n_band,
                "fpr_per_bucket_tau": l1_bucket[band], "fpr_global_tau": l1_global[band],
                "tau_per_bucket": tau_by_bucket["en"], "tau_global": tau_global,
            })

    # en L1/L2 disparity (per-bucket tau), the RQ4 headline number.
    en_alpha_rows = {r["group"]: r for r in rows if r["bucket"] == "en"}
    for alpha in ALPHAS:
        general = next((r for r in rows if r["alpha"] == alpha and r["bucket"] == "en"
                        and r["group"] == "writer_L1_band=general"), None)
        indian = next((r for r in rows if r["alpha"] == alpha and r["bucket"] == "en"
                       and r["group"] == "writer_L1_band=indian"), None)
        if general and indian:
            rows.append({
                "alpha": alpha, "bucket": "en", "group": "L1_L2_disparity (indian - general)",
                "n_human": None,
                "fpr_per_bucket_tau": indian["fpr_per_bucket_tau"] - general["fpr_per_bucket_tau"],
                "fpr_global_tau": indian["fpr_global_tau"] - general["fpr_global_tau"],
                "tau_per_bucket": None, "tau_global": None,
            })

    # Unmatched-human robustness check: human rows the length matching trimmed out of the
    # corpus (en/hi/te; cm trims machine). Scored once by headA + mGPT, then pushed through
    # the same fuser / per-bucket temperature / tau as every other row (src/eval/unmatched.py).
    from src.eval.unmatched import score_unmatched
    from src.utils.config import load_config

    un = score_unmatched(load_config("configs/data.yaml"), load_config(config_path))
    un_buckets = un["language"].tolist()
    un_logit = artifacts.fuser_ab.decision_function(un[["headA", "headB"]].to_numpy(dtype=float), un_buckets)
    un_prob = np.array([artifacts.temperature.transform(np.array([lg]), bucket=b)[0]
                        for lg, b in zip(un_logit, un_buckets)])
    matched_len = {b: float(np.median([r["length_words"] for r in test_human if r["language"] == b]))
                   for b in LANGUAGE_BUCKETS if any(r["language"] == b for r in test_human)}
    for alpha in ALPHAS:
        calibrator = artifacts.conformal[alpha]
        tau_global = calibrator.threshold("_global")
        for bucket in LANGUAGE_BUCKETS:
            m = np.array([b == bucket for b in un_buckets])
            if not m.any():
                continue
            tau_b = calibrator.threshold(bucket)
            rows.append({
                "alpha": alpha, "bucket": bucket, "group": "unmatched_human_robustness_check",
                "n_human": int(m.sum()),
                "fpr_per_bucket_tau": float((un_prob[m] > tau_b).mean()),
                "fpr_global_tau": float((un_prob[m] > tau_global).mean()),
                "tau_per_bucket": tau_b, "tau_global": tau_global,
                "median_length_words": float(un.loc[m, "length_words"].median()),
                "matched_median_length_words": matched_len.get(bucket),
            })

    df = pd.DataFrame(rows)
    write_csv(df, RESULTS_CSV)
    write_csv(df, T5_CSV)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    args = parser.parse_args()
    df = run(args.config)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
