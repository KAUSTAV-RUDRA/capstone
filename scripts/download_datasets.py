"""Download public benchmarks and Indic/code-mixed human sources.

Fetches the datasets named in ``configs/data.yaml`` (M4GT-Bench multilingual,
RAID, HC3; IndicCorp, AI4Bharat catalogue, HinGE, COMI-LINGUA, L3Cube, etc.)
into ``data/raw/`` (gitignored).

Takes ``--config``. Serves docs/master-execution-plan.md Phase 1 (Day 1, P2)
and Phase 2 §2.1.2.
"""
from __future__ import annotations

import argparse


def download(config_path: str) -> None:
    """Download every configured dataset into data/raw/."""
    # TODO(phase-2 step-2.1.2): fetch benchmarks + Indic sources per config.
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/data.yaml")
    args = parser.parse_args()
    download(args.config)


if __name__ == "__main__":
    main()
