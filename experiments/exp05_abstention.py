"""exp05 - calibration + abstention: risk-coverage curve and T6.

Uses the fusion, temperature and conformal artefacts from
:func:`src.eval.pipeline.fit_full_pipeline` (fused_ab = headA + headB; headC
excluded, docs/results/headc_diagnosis.md). All numbers below are on the TEST
split; temperature is fit on TRAIN and conformal/the abstention median on the
human-only CAL split, so TEST never leaks into either fit.

Writes:
  results/T6_calibration.csv          ECE before/after temperature, accuracy at 50/70/90% coverage
  results/abstention_decisions.csv    per-test-row HUMAN/ABSTAIN/MACHINE verdict
  results/abstention_coverage_sweep.csv   the 30-100% coverage sweep per bucket
  results/F1_risk_coverage.csv        risk-coverage curve points per bucket
  results/F1_risk_coverage.png        the four buckets overlaid
  results/exp05_abstention.csv        alias of T6_calibration.csv (repo convention)
Serves: docs/master-execution-plan.md Phase 3 §3.1-§3.3.
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

RESULTS_CSV = "results/exp05_abstention.csv"
T6_CSV = "results/T6_calibration.csv"
DECISIONS_CSV = "results/abstention_decisions.csv"
SWEEP_CSV = "results/abstention_coverage_sweep.csv"
F1_CSV = "results/F1_risk_coverage.csv"
F1_PNG = "results/F1_risk_coverage.png"

BUCKET_COLORS = {"en": "#4C72B0", "hi": "#DD8452", "te": "#55A868", "cm": "#C44E52"}


def run(config_path: str) -> "pd.DataFrame":
    """Calibrate per bucket, sweep coverage, verify FPR <= alpha; write T6 + F1."""
    import numpy as np
    import pandas as pd

    from src.calibration.abstention import coverage_sweep
    from src.data.schema import LANGUAGE_BUCKETS
    from src.eval.metrics import expected_calibration_error
    from src.eval.pipeline import fit_full_pipeline
    from src.eval.risk_coverage import metrics_at_coverage, risk_coverage_curve
    from src.utils.io import write_csv

    artifacts = fit_full_pipeline(config_path)
    test_by_bucket = {
        b: [r for r in artifacts.corpus if r["split"] == "test" and r["language"] == b]
        for b in LANGUAGE_BUCKETS
    }

    t6_rows, decision_rows, sweep_rows, curve_rows = [], [], [], []
    for bucket in LANGUAGE_BUCKETS:
        rows = test_by_bucket[bucket]
        ids = [r["id"] for r in rows]
        y = np.array([r["label"] for r in rows])
        prob_before = artifacts.scores.loc[ids, "fused_ab"].to_numpy(dtype=float)
        prob_after = artifacts.scores.loc[ids, "fused_ab_calibrated"].to_numpy(dtype=float)
        conf_after = np.maximum(prob_after, 1 - prob_after)

        ece_before = expected_calibration_error(y, prob_before)
        ece_after = expected_calibration_error(y, prob_after)
        row = {
            "bucket": bucket, "temperature": artifacts.temperature.temperature(bucket),
            "ece_before": ece_before, "ece_after": ece_after,
        }
        for cov in (0.5, 0.7, 0.9):
            m = metrics_at_coverage(y, prob_after, conf_after, cov)
            row[f"accuracy_at_{int(cov*100)}pct"] = m["accuracy"]
            row[f"fpr_at_{int(cov*100)}pct"] = m["fpr"]
        t6_rows.append(row)

        # Abstention decisions on this bucket's test rows.
        verdicts = artifacts.abstention.decide_batch(prob_after, [bucket] * len(prob_after))
        for r, p, (verdict, confidence) in zip(rows, prob_after, verdicts):
            decision_rows.append({"id": r["id"], "bucket": bucket, "label": r["label"],
                                  "prob_calibrated": float(p), "verdict": verdict,
                                  "confidence": confidence})

        # Coverage sweep 30% -> 100%.
        tau_machine = artifacts.conformal[0.01].threshold(bucket)
        for sweep_row in coverage_sweep(prob_after, y, tau_machine):
            sweep_rows.append({"bucket": bucket, **sweep_row})

        # Risk-coverage curve (F1), subsampled to <=200 points per bucket for a readable CSV/plot.
        coverage, risk = risk_coverage_curve(y, prob_after, conf_after)
        if len(coverage) > 200:
            idx = np.linspace(0, len(coverage) - 1, 200).astype(int)
            coverage, risk = coverage[idx], risk[idx]
        for c, r_ in zip(coverage, risk):
            curve_rows.append({"bucket": bucket, "coverage": float(c), "risk": float(r_)})

    t6_df = pd.DataFrame(t6_rows)
    write_csv(t6_df, T6_CSV)
    write_csv(t6_df, RESULTS_CSV)
    write_csv(pd.DataFrame(decision_rows), DECISIONS_CSV)
    write_csv(pd.DataFrame(sweep_rows), SWEEP_CSV)
    curve_df = pd.DataFrame(curve_rows)
    write_csv(curve_df, F1_CSV)
    _plot_risk_coverage(curve_df, F1_PNG)

    return t6_df


def _plot_risk_coverage(curve_df: "pd.DataFrame", out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    for bucket, color in BUCKET_COLORS.items():
        sub = curve_df[curve_df["bucket"] == bucket].sort_values("coverage")
        if sub.empty:
            continue
        ax.plot(sub["coverage"], sub["risk"], label=bucket, color=color, linewidth=2)
    ax.set_xlabel("Coverage (fraction not abstained)")
    ax.set_ylabel("Risk (error rate among non-abstained)")
    ax.set_title("Risk-coverage curve, all four buckets")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(title="bucket")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    args = parser.parse_args()
    df = run(args.config)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
