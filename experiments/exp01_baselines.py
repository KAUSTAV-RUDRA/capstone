"""exp01 - all baselines, AUROC + F1 per bucket (Table T1; T3 on held-out).

Runs perplexity threshold, DetectGPT, Fast-DetectGPT, Binoculars, and XLM-R
supervised via the shared ``score()`` contract, on each language bucket.

Writes: results/exp01_baselines.csv
Serves: docs/master-execution-plan.md Phase 2 §2.2.1 (and §2.2.7 for T3).
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

RESULTS_CSV = "results/exp01_baselines.csv"


def run(config_path: str) -> "pd.DataFrame":
    """Evaluate every baseline per bucket and return the T1 table."""
    # TODO(phase-2 step-2.2.1): score baselines per bucket; write T1 CSV.
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
