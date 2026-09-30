"""Export the fitted fusion/temperature/conformal artefacts for the webapp.

The webapp (``webapp/detector/services.py``) must not import
``src.eval.pipeline`` at request time (it re-derives everything from
``results/scores.parquet``, the corpus, and re-fits sklearn objects -- fine
for offline table-building, wasteful and slow to import per Django worker
start). Instead it loads a small, static JSON snapshot of exactly the numbers
inference needs: per-bucket temperature, conformal thresholds (alpha 0.01 for
the MACHINE cutoff, plus 0.05 for reference) and the calibration-split human
median (the HUMAN cutoff). Head C is excluded (docs/results/headc_diagnosis.md).

Usage: python -m scripts.export_calibration --config configs/models_norm.yaml
Writes: results/calibration.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_VERSION = "v2.0-norm"
OUT_PATH = "results/calibration.json"


def run(config_path: str) -> dict:
    from src.data.schema import LANGUAGE_BUCKETS
    from src.eval.pipeline import fit_full_pipeline

    artifacts = fit_full_pipeline(config_path)

    payload = {
        "model_version": MODEL_VERSION,
        "fusion_method": "logistic",
        "heads_used": ["headA", "headB"],
        "head_c_excluded": True,
        "head_c_exclusion_reason": (
            "Its near-perfect AUROC (0.961 on held-out llama hi) is matched by a plain "
            "TF-IDF classifier (0.911-0.921), so it cannot be told apart from a "
            "human-scraped-vs-prompted-generation provenance difference "
            "(docs/decisions.md 2026-09-30; reported as a diagnostic only)."
        ),
        "buckets": {},
    }
    for bucket in LANGUAGE_BUCKETS:
        payload["buckets"][bucket] = {
            "temperature": artifacts.temperature.temperature(bucket),
            "tau_machine": artifacts.conformal[0.01].threshold(bucket),  # alpha=0.01, the MACHINE cutoff
            "tau_alpha_0.05": artifacts.conformal[0.05].threshold(bucket),
            "human_median": artifacts.cal_median[bucket],  # the HUMAN cutoff (abstention.py's "lower")
        }

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_PATH).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models_norm.yaml")
    args = parser.parse_args()
    payload = run(args.config)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
