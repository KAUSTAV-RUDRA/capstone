"""exp13 - both conformal operating points (alpha 0.01 and 0.05): recall, FPR, risk-coverage.

For each alpha, per bucket (all test rows, and on hi: seen-only and held-out llama), with the
per-bucket conformal tau for that alpha and the shipped HUMAN cutoff (human calibration median):

  recall      machine rows with p > tau
  fpr         human rows with p > tau, with a 95% Wilson interval, and whether the conformal
              bound (fpr <= alpha) holds empirically: exact one-sided binomial p-value for
              "true FPR > alpha" given the observed flags
  gate        three-way coverage, false-clear rate (machine called HUMAN), accuracy on covered

Risk-coverage per alpha: MACHINE at tau_alpha fixed, the HUMAN cutoff t swept over the observed
scores; coverage = share not abstained, risk = error rate among covered rows. One curve per
(bucket, alpha); the confidence-ranked curve F1 does not depend on alpha.

Writes: results/exp13_alpha_operating_points.csv, results/exp13_risk_coverage_alpha.csv,
        results/exp13_risk_coverage_alpha.png
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest

from src.data.schema import LANGUAGE_BUCKETS
from src.eval.pipeline import ALPHAS, fit_full_pipeline, generator_roles


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def run(config_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = fit_full_pipeline(config_path)
    _, heldout = generator_roles()
    rows, curves = [], []
    for b in LANGUAGE_BUCKETS:
        te = [r for r in a.corpus if r["split"] == "test" and r["language"] == b]
        p = a.scores.loc[[r["id"] for r in te], "fused_ab_calibrated"].to_numpy(float)
        y = np.array([r["label"] for r in te])
        held = np.array([r["generator"] in heldout for r in te])
        scopes = {"all test": np.ones(len(te), bool)}
        if held.any():
            scopes["seen only"] = (y == 0) | ~held
            scopes["llama only"] = (y == 0) | held
        t_med = a.cal_median[b]
        for alpha in ALPHAS:
            tau = a.conformal[alpha].threshold(b)
            for scope, sel in scopes.items():
                ps, ys = p[sel], y[sel]
                mach, hum = ps > tau, ps < t_med
                H, M = ys == 0, ys == 1
                k, n = int(mach[H].sum()), int(H.sum())
                lo, hi = wilson(k, n)
                cov = mach | hum
                rows.append({
                    "bucket": b, "scope": scope, "alpha": alpha, "tau": tau,
                    "n_human": n, "n_machine": int(M.sum()),
                    "fpr": k / n, "fpr_lo": lo, "fpr_hi": hi,
                    "bound_p_value": float(binomtest(k, n, alpha, alternative="greater").pvalue),
                    "bound_holds": bool(k / n <= alpha),
                    "recall": float(mach[M].mean()),
                    "gate_coverage": float(cov.mean()),
                    "gate_false_clear": float(hum[M].mean()),
                    "gate_machine_abstained": float((~cov)[M].mean()),
                    "gate_accuracy_covered": float((mach[cov] == ys[cov]).mean()),
                })
                # risk-coverage: sweep the HUMAN cutoff t in [0, tau]
                for t in np.unique(np.concatenate([[0.0], np.clip(ps, 0, tau)])):
                    hm = ps < t
                    cv = mach | hm
                    if cv.sum() == 0:
                        continue
                    curves.append({"bucket": b, "scope": scope, "alpha": alpha, "t": float(t),
                                   "coverage": float(cv.mean()),
                                   "risk": float((mach[cv] != ys[cv].astype(bool)).mean())})
    df, cv = pd.DataFrame(rows), pd.DataFrame(curves)
    df.to_csv("results/exp13_alpha_operating_points.csv", index=False)
    # thin curves to <= 300 points per (bucket, scope, alpha)
    cv = cv.groupby(["bucket", "scope", "alpha"], group_keys=False).apply(
        lambda g: g.iloc[np.linspace(0, len(g) - 1, min(len(g), 300)).astype(int)])
    cv.to_csv("results/exp13_risk_coverage_alpha.csv", index=False)
    _plot(cv)
    return df, cv


def _plot(cv: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"en": "#4C72B0", "hi": "#DD8452", "te": "#55A868", "cm": "#C44E52"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150, sharey=True)
    for k, alpha in enumerate(ALPHAS):
        for b, c in colors.items():
            g = cv[(cv.bucket == b) & (cv.scope == "all test") & (cv.alpha == alpha)].sort_values("coverage")
            ax[k].plot(g.coverage, g.risk, color=c, label=b, lw=2)
        g = cv[(cv.bucket == "hi") & (cv.scope == "llama only") & (cv.alpha == alpha)].sort_values("coverage")
        ax[k].plot(g.coverage, g.risk, color=colors["hi"], ls="--", label="hi, llama only")
        ax[k].set_title(f"alpha = {alpha} (HUMAN cutoff swept)")
        ax[k].set_xlabel("Coverage (not abstained)")
        ax[k].grid(alpha=0.3)
        ax[k].set_xlim(0, 1)
    ax[0].set_ylabel("Risk (error rate among covered)")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig("results/exp13_risk_coverage_alpha.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    df, _ = run(ap.parse_args().config)
    pd.set_option("display.width", 250)
    print(df.drop(columns=["tau", "fpr_lo", "fpr_hi"]).round(3).to_string(index=False))
    print(df[["bucket", "scope", "alpha", "fpr", "fpr_lo", "fpr_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
