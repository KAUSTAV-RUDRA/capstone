"""exp02 - Head A (stylometric) evaluation, AUROC per bucket + feature importance.

Writes: results/exp02_stylometric.csv
Serves: docs/master-execution-plan.md Phase 2 §2.2.2.
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

RESULTS_CSV = "results/exp02_stylometric.csv"


def run(config_path: str) -> "pd.DataFrame":
    """Evaluate Head A per bucket and log per-bucket feature importance."""
    # TODO(phase-2 step-2.2.2): fit/score Head A per bucket; write CSV.
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
