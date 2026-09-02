"""exp05 - calibration + abstention: risk-coverage curve and T6.

Fits temperature scaling + per-bucket conformal thresholds, runs the coverage
sweep, and emits the risk-coverage curve (F1) and calibration table (T6:
ECE + accuracy at 50/70/90% coverage). Verifies empirical FPR <= alpha per bucket.

Writes: results/exp05_abstention.csv
Serves: docs/master-execution-plan.md Phase 3 §3.1-§3.3.
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

RESULTS_CSV = "results/exp05_abstention.csv"


def run(config_path: str) -> "pd.DataFrame":
    """Calibrate per bucket, sweep coverage, verify FPR <= alpha; write T6 + F1."""
    # TODO(phase-3 step-3.3): temperature + conformal + abstention sweep; write CSV.
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
