"""exp12 - abstention at target coverages 70 / 80 / 90% (only the HUMAN cutoff moves).

The gate is MACHINE if p > tau (per-bucket conformal, alpha 0.01, UNCHANGED), HUMAN if p < t,
ABSTAIN between. The shipped t is the human calibration median. Here t is swept so that
coverage (share of rows not abstained) hits each target, per bucket, on calibrated fused_ab
scores. Two ways to pick t:

  oracle   t chosen on the TEST rows to hit the target exactly (an upper bound: t leaks the test mix)
  train    t chosen on the TRAIN rows (deployable; train fuser scores are in-sample, so the
           coverage actually achieved on test is reported and can miss the target)

Because tau is fixed, FPR (humans called MACHINE) and machine recall (machines called MACHINE)
do not move with t; what moves is how many humans are cleared and how many machine texts are
wrongly called HUMAN (false_clear). Coverage depends on the class mix (en test is 73% machine,
te 36%), so class-wise rates are reported alongside it.

Writes: results/exp12_coverage_targets.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.data.schema import LANGUAGE_BUCKETS
from src.eval.pipeline import fit_full_pipeline, generator_roles

TARGETS = (0.7, 0.8, 0.9)


def pick_t(p: np.ndarray, tau: float, target: float) -> float:
    """Smallest t in [0, tau] with coverage(p > tau or p < t) >= target."""
    machine = p > tau
    for t in np.unique(np.concatenate([[0.0], np.clip(p, 0, tau)])):
        if (machine | (p < t)).mean() >= target:
            return float(t)
    return float(tau)


def rates(p, y, tau, t, sel=None):
    sel = np.ones(len(p), bool) if sel is None else sel
    p, y = p[sel], y[sel]
    mach, hum = p > tau, p < t
    H, M = y == 0, y == 1
    cov = mach | hum
    return {"coverage": float(cov.mean()),
            "fpr": float(mach[H].mean()), "human_cleared": float(hum[H].mean()),
            "machine_recall": float(mach[M].mean()) if M.any() else np.nan,
            "false_clear": float(hum[M].mean()) if M.any() else np.nan,
            "machine_abstained": float((~cov)[M].mean()) if M.any() else np.nan,
            "accuracy_covered": float((mach[cov] == y[cov]).mean()),
            "n": int(len(p))}


def run(config_path: str) -> pd.DataFrame:
    a = fit_full_pipeline(config_path)
    _, heldout = generator_roles()
    rows = []
    for b in LANGUAGE_BUCKETS:
        tau = a.conformal[0.01].threshold(b)
        tr = [r for r in a.corpus if r["split"] == "train" and r["language"] == b]
        te = [r for r in a.corpus if r["split"] == "test" and r["language"] == b]
        col = lambda rs: a.scores.loc[[r["id"] for r in rs], "fused_ab_calibrated"].to_numpy(float)  # noqa: E731
        ptr, pte = col(tr), col(te)
        yte = np.array([r["label"] for r in te])
        isheld = np.array([r["generator"] in heldout for r in te])
        plans = [("shipped (cal median)", "-", a.cal_median[b])]
        for tg in TARGETS:
            plans.append(("oracle", tg, pick_t(pte, tau, tg)))
            plans.append(("train", tg, pick_t(ptr, tau, tg)))
        for mode, tg, t in plans:
            rows.append({"bucket": b, "scope": "all test", "mode": mode, "target": tg, "t": t, "tau": tau,
                         **rates(pte, yte, tau, t)})
            if isheld.any():
                for name, sel in (("seen only", (yte == 0) | ~isheld), ("llama only", (yte == 0) | isheld)):
                    rows.append({"bucket": b, "scope": name, "mode": mode, "target": tg, "t": t, "tau": tau,
                                 **rates(pte, yte, tau, t, sel)})
    df = pd.DataFrame(rows)
    df.to_csv("results/exp12_coverage_targets.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    df = run(ap.parse_args().config)
    pd.set_option("display.width", 250)
    print(df[df.scope == "all test"].drop(columns=["scope", "tau"]).round(3).to_string(index=False))
    print(df[df.scope != "all test"].drop(columns=["tau"]).query("mode != 'train'").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
