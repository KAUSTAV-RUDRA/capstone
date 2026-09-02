"""exp00 - Fast-DetectGPT smoke test on HC3 English.

Runs the locked scorer on ~200 HC3 samples (100 human / 100 machine) and
reports AUROC. This number is the Rubric-7 evidence for Review-1.

Writes: results/exp00_smoke.csv
Serves: docs/master-execution-plan.md Phase 1, Day 4 (P3).
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

RESULTS_CSV = "results/exp00_smoke.csv"


def run(config_path: str) -> "pd.DataFrame":
    """Run the smoke test and return the results table (also written to CSV)."""
    # TODO(phase-1 day-4): run Fast-DetectGPT on HC3; compute AUROC; write CSV.
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
