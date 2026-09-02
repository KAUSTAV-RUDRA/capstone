"""exp07 - fairness audit: FPR by bucket and by L1/L2 band, ours vs baselines (T5).

Quantifies the reduction in native/non-native false-positive disparity (RQ4).

Writes: results/exp07_fairness.csv
Serves: docs/master-execution-plan.md Phase 3 §3.6.
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

RESULTS_CSV = "results/exp07_fairness.csv"


def run(config_path: str) -> "pd.DataFrame":
    """Compute FPR by bucket and L1 band for all systems; write the T5 table."""
    # TODO(phase-3 step-3.6): fairness audit ours vs baselines; write CSV.
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
