"""exp04 - fusion (logistic) over head scores; must beat every baseline.

Fits the adopted fusion (fused_ab = headA + headB + bucket one-hot; headC
EXCLUDED per docs/results/headc_diagnosis.md) via
:func:`src.eval.pipeline.fit_full_pipeline`, plus fused_abc kept for
reference only, and reports both against every baseline per bucket, plus the
standardised fusion coefficients (interpretability, locked §5).

Writes: results/exp04_fusion.csv, results/fusion_coefficients.csv
Serves: docs/master-execution-plan.md Phase 2 §2.2.6 (Phase 2 exit criterion).
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

RESULTS_CSV = "results/exp04_fusion.csv"
COEF_CSV = "results/fusion_coefficients.csv"

COMPARE_METHODS = ("ppl", "binoculars", "fastdetectgpt_en", "headA", "headB", "headC",
                   "fused_ab", "fused_abc")


def run(config_path: str) -> "pd.DataFrame":
    """Fit fusion, compare to baselines per bucket, write CSVs."""
    import pandas as pd

    from src.data.schema import LANGUAGE_BUCKETS
    from src.eval.metrics import auroc
    from src.eval.pipeline import fit_full_pipeline
    from src.utils.io import write_csv

    artifacts = fit_full_pipeline(config_path)
    test = [r for r in artifacts.corpus if r["split"] == "test"]

    rows = []
    for bucket in LANGUAGE_BUCKETS:
        sub = [r for r in test if r["language"] == bucket]
        y = [r["label"] for r in sub]
        for method in COMPARE_METHODS:
            s = artifacts.scores.loc[[r["id"] for r in sub], method].to_numpy(dtype=float)
            rows.append({"bucket": bucket, "method": method, "auroc": auroc(y, s)})
    df = pd.DataFrame(rows)
    write_csv(df, RESULTS_CSV)

    coef_rows = []
    for feature, weight in artifacts.fuser_ab.coefficients().items():
        coef_rows.append({"fusion": "A+B (adopted)", "feature": feature, "weight_standardised": weight})
    for feature, weight in artifacts.fuser_abc.coefficients().items():
        coef_rows.append({"fusion": "A+B+C (reference, excluded)", "feature": feature,
                          "weight_standardised": weight})
    write_csv(pd.DataFrame(coef_rows), COEF_CSV)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    args = parser.parse_args()
    df = run(args.config)
    print(df.pivot(index="bucket", columns="method", values="auroc").to_string())


if __name__ == "__main__":
    main()
