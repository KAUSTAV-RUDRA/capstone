"""exp06 - adversarial robustness: clean vs paraphrase vs back-translation vs hybrid (T4).

Evaluates every system across attack types to show fusion holds under
paraphrase where either head alone fails (RQ2).

Writes: results/exp06_adversarial.csv
Serves: docs/master-execution-plan.md Phase 3 §3.5.
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

RESULTS_CSV = "results/exp06_adversarial.csv"


def run(config_path: str) -> "pd.DataFrame":
    """Evaluate all systems across attack types per bucket; write the T4 table."""
    # TODO(phase-3 step-3.5): run clean/paraphrase/back-translation/hybrid; write CSV.
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
