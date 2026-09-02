"""exp04 - fusion (logistic vs GBM) over head scores; must beat every baseline.

Writes: results/exp04_fusion.csv
Serves: docs/master-execution-plan.md Phase 2 §2.2.6 (Phase 2 exit criterion).
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

RESULTS_CSV = "results/exp04_fusion.csv"


def run(config_path: str) -> "pd.DataFrame":
    """Fit fusion, compare to baselines per bucket, pick logistic vs GBM."""
    # TODO(phase-2 step-2.2.6): fit fusion; compare vs baselines; write CSV.
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
