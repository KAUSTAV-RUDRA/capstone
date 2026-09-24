"""exp08 - full ablation sweep: A / B / C / A+B / A+B+C / +temperature / +conformal / +abstain (T2).

Thin driver over :func:`src.eval.ablation.run_ablation` -- see that module and
its docstring for the per-row rationale (why AUROC/F1 are identical for A+B
vs +temperature by construction, why A+B+C is reference-only, etc).

Writes: results/exp08_ablation.csv, results/T2_ablation.csv
Serves: docs/master-execution-plan.md Phase 3 §3.7.
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

RESULTS_CSV = "results/exp08_ablation.csv"
T2_CSV = "results/T2_ablation.csv"


def run(config_path: str) -> "pd.DataFrame":
    """Run every ablation configuration and write the T2 table."""
    from src.eval.ablation import run_ablation
    from src.utils.io import write_csv

    df = run_ablation(config_path)
    write_csv(df, RESULTS_CSV)
    write_csv(df, T2_CSV)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    args = parser.parse_args()
    df = run(args.config)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
