"""exp08 - full ablation sweep: A / B / A+B / +cal / +abstain (Table T2).

Writes: results/exp08_ablation.csv
Serves: docs/master-execution-plan.md Phase 3 §3.7.
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

RESULTS_CSV = "results/exp08_ablation.csv"


def run(config_path: str) -> "pd.DataFrame":
    """Run every ablation configuration per bucket and write the T2 table."""
    # TODO(phase-3 step-3.7): sweep ablation configs per bucket; write CSV.
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
