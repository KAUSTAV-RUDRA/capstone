"""exp10 - T3: held-out generator generalisation (non-negotiable #4).

llama is the sole held-out generator and only generates hi (it failed its own
en/te/cm gates, decisions.md 2026-09-28), so T3 is a hi-bucket table. For every
method it reports test AUROC two ways on the SAME human test rows:

  seen     human test rows vs machine test rows from seen generators (qwen7b/gemma/mistral)
  heldout  human test rows vs machine test rows from llama

with 95% bootstrap CIs and the seen -> held-out drop. Two supporting tables:

  by generator   the same AUROC for each generator separately (hi)
  operating      what the shipped gate does on llama vs seen machine text: detection
                 rate at the per-bucket conformal tau (alpha 0.01 / 0.05), and the
                 HUMAN / ABSTAIN / MACHINE split of the three-way gate

Head C is included as a diagnostic only (excluded from fusion, decisions.md).
Roles come from configs/data.yaml; nothing is fit here.

Writes: results/T3_heldout.csv, results/T3_heldout_by_generator.csv,
        results/T3_heldout_operating.csv
Serves: docs/project-context-master.md §7 (T3).
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

T3_CSV = "results/T3_heldout.csv"
BY_GEN_CSV = "results/T3_heldout_by_generator.csv"
OPERATING_CSV = "results/T3_heldout_operating.csv"

METHODS = ("ppl", "binoculars", "fastdetectgpt_en", "headA", "headB", "headB_word", "headC", "fused_ab")
METHOD_LABELS = {"fused_ab": "fused (A+B)", "headC": "headC (diagnostic)"}
N_BOOT = 1000


def _auroc_ci(y, s, rng) -> tuple[float, float, float]:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    h, m = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    point = float(roc_auc_score(y, s))
    boots = [roc_auc_score(y[i], s[i]) for i in
             (np.concatenate([rng.choice(h, len(h)), rng.choice(m, len(m))]) for _ in range(N_BOOT))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def run(config_path: str) -> "pd.DataFrame":
    import numpy as np
    import pandas as pd

    from src.data.schema import LANGUAGE_BUCKETS
    from src.eval.pipeline import fit_full_pipeline, generator_roles
    from src.utils.io import write_csv

    artifacts = fit_full_pipeline(config_path)
    seen, heldout = generator_roles()
    rng = np.random.default_rng(0)
    test = [r for r in artifacts.corpus if r["split"] == "test"]

    main_rows, gen_rows, op_rows = [], [], []
    for bucket in LANGUAGE_BUCKETS:
        b_test = [r for r in test if r["language"] == bucket]
        human = [r for r in b_test if r["label"] == 0]
        held = [r for r in b_test if r["label"] == 1 and r["generator"] in heldout]
        if not held:
            continue  # no held-out generator covers this bucket
        seen_m = [r for r in b_test if r["label"] == 1 and r["generator"] in seen]
        sets = {"seen": human + seen_m, "heldout": human + held}

        def col(rows, name):
            return artifacts.scores.loc[[r["id"] for r in rows], name].to_numpy(dtype=float)

        for method in METHODS:
            cells = {}
            for k, rows in sets.items():
                y = np.array([r["label"] for r in rows])
                cells[k] = _auroc_ci(y, col(rows, method), rng)
            main_rows.append({
                "bucket": bucket, "method": METHOD_LABELS.get(method, method),
                "n_human": len(human), "n_machine_seen": len(seen_m), "n_machine_heldout": len(held),
                "auroc_seen": cells["seen"][0], "seen_lo": cells["seen"][1], "seen_hi": cells["seen"][2],
                "auroc_heldout": cells["heldout"][0], "heldout_lo": cells["heldout"][1],
                "heldout_hi": cells["heldout"][2],
                "drop": cells["seen"][0] - cells["heldout"][0],
            })
            for g in sorted({r["generator"] for r in b_test if r["label"] == 1}):
                rows = human + [r for r in b_test if r["label"] == 1 and r["generator"] == g]
                y = np.array([r["label"] for r in rows])
                gen_rows.append({"bucket": bucket, "method": METHOD_LABELS.get(method, method), "generator": g,
                                 "role": "heldout" if g in heldout else "seen",
                                 "n_machine": int(y.sum()), "auroc": _auroc_ci(y, col(rows, method), rng)[0]})

        # Operating behaviour of the shipped fused gate.
        for alpha in (0.01, 0.05):
            tau = artifacts.conformal[alpha].threshold(bucket)
            for label, rows in (("human", human), ("machine_seen", seen_m), ("machine_heldout", held)):
                p = col(rows, "fused_ab_calibrated")
                op_rows.append({"bucket": bucket, "alpha": alpha, "group": label, "n": len(rows),
                                "tau": tau, "flag_rate_at_tau": float((p > tau).mean())})
        gate_verdicts = {}
        for label, rows in (("human", human), ("machine_seen", seen_m), ("machine_heldout", held)):
            v = [x for x, _ in artifacts.abstention.decide_batch(col(rows, "fused_ab_calibrated"),
                                                                 [bucket] * len(rows))]
            gate_verdicts[label] = {k: v.count(k) / len(v) for k in ("HUMAN", "ABSTAIN", "MACHINE")}
        for label, d in gate_verdicts.items():
            op_rows.append({"bucket": bucket, "alpha": 0.01, "group": f"{label} (3-way gate)",
                            "n": None, "tau": None, "flag_rate_at_tau": None,
                            "frac_human": d["HUMAN"], "frac_abstain": d["ABSTAIN"], "frac_machine": d["MACHINE"]})

    df = pd.DataFrame(main_rows)
    write_csv(df, T3_CSV)
    write_csv(pd.DataFrame(gen_rows), BY_GEN_CSV)
    write_csv(pd.DataFrame(op_rows), OPERATING_CSV)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    args = parser.parse_args()
    df = run(args.config)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
