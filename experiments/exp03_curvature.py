"""exp03 - Head B (curvature) evaluation + Telugu/code-mix degradation diagnostic.

Reports curvature-only AUROC per bucket and quantifies whether curvature
degrades on te / cm (the §2.2.4 diagnostic - publishable either way).

Writes: results/exp03_curvature.csv
Serves: docs/master-execution-plan.md Phase 2 §2.2.3-§2.2.4.
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

RESULTS_CSV = "results/exp03_curvature.csv"


def run(config_path: str) -> "pd.DataFrame":
    """Evaluate Head B per bucket and record the degradation diagnostic."""
    # TODO(phase-2 step-2.2.3): score Head B per bucket; write CSV + diagnostic.
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
