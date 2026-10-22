"""exp14 - T7: error analysis of the frozen gate on the CLEAN test split.

Per bucket and per alpha (0.05, 0.01): the shipped three-way gate (MACHINE if p > tau_alpha,
HUMAN if p < the human calibration median, else ABSTAIN) over the fused calibrated
machine-probability. Nothing is fit; artefacts are loaded exactly as shipped.

  T7_error_analysis.csv   confusion counts with rates + 95% Wilson intervals:
                          machine/human rows, correct MACHINE, correct HUMAN, false flag
                          (human -> MACHINE), false clear (machine -> HUMAN), abstain
  T7_breakdown.csv        false-flag and false-clear rates by length bin and by generator
                          (machine rows) / source corpus (human rows)
  T7_examples.md          2 seeded-random real examples per bucket of each error type
  T7_error_analysis.md    the tables, plus the hand-written per-bucket notes in
                          docs/results/T7_bucket_notes.md if that file exists

False clear does not depend on alpha (the HUMAN cutoff is shared), so those rows repeat.
Examples: false flags are drawn from alpha-0.05 false flags (the column ``also_flagged_a01``
says whether the 0.01 gate flagged it too); false clears from the shipped gate.

Writes: docs/results/T7_*. Serves: docs/project-context-master.md §7 (error analysis for T1/T5).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

OUT = Path("docs/results")
LENGTH_EDGES = (0, 25, 50, 100, 150, 200, float("inf"))
LENGTH_LABELS = ("<25", "25-49", "50-99", "100-149", "150-199", "200+")
ALPHAS = (0.05, 0.01)
N_EXAMPLES = 2
MIN_BREAKDOWN_N = 1


def length_bin(n_words: int) -> str:
    for lo, hi, lab in zip(LENGTH_EDGES, LENGTH_EDGES[1:], LENGTH_LABELS):
        if lo <= n_words < hi:
            return lab
    return LENGTH_LABELS[-1]


def build_frame(config_path: str):
    import pandas as pd

    from src.eval.pipeline import fit_full_pipeline

    a = fit_full_pipeline(config_path)               # loads cached/frozen artefacts
    test = [r for r in a.corpus if r["split"] == "test"]
    df = pd.DataFrame(test).set_index("id")
    df = df.join(a.scores[["headA", "headB", "fused_ab_calibrated"]].rename(columns={"fused_ab_calibrated": "p"}))
    df["group"] = df["generator"].where(df["label"] == 1, df["source"])
    df["length_bin"] = df["length_words"].map(length_bin)
    return a, df


def decisions(df, a, bucket: str, alpha: float):
    """(flag, clear, abstain) boolean arrays for one bucket's rows."""
    p = df["p"].to_numpy(dtype=float)
    flag = p > a.conformal[alpha].threshold(bucket)
    clear = p < a.cal_median[bucket]
    return flag, clear, ~flag & ~clear


def confusion_rows(df, a) -> list[dict[str, Any]]:
    from src.data.schema import LANGUAGE_BUCKETS
    from src.eval.metrics import wilson

    rows = []
    for bucket in LANGUAGE_BUCKETS:
        sub = df[df["language"] == bucket]
        y = sub["label"].to_numpy()
        hum, mac = y == 0, y == 1
        for alpha in ALPHAS:
            flag, clear, ab = decisions(sub, a, bucket, alpha)
            cells = {
                "true_machine": (int(mac.sum()), len(sub)),
                "true_human": (int(hum.sum()), len(sub)),
                "correct_machine (machine->MACHINE)": (int((flag & mac).sum()), int(mac.sum())),
                "correct_human (human->HUMAN)": (int((clear & hum).sum()), int(hum.sum())),
                "false_flag (human->MACHINE)": (int((flag & hum).sum()), int(hum.sum())),
                "false_clear (machine->HUMAN)": (int((clear & mac).sum()), int(mac.sum())),
                "abstain (all)": (int(ab.sum()), len(sub)),
                "abstain (human rows)": (int((ab & hum).sum()), int(hum.sum())),
                "abstain (machine rows)": (int((ab & mac).sum()), int(mac.sum())),
            }
            for name, (k, n) in cells.items():
                lo, hi = wilson(k, n)
                rows.append({"bucket": bucket, "alpha": alpha, "cell": name, "count": k, "denominator": n,
                             "rate": k / n if n else float("nan"), "ci_lo": lo, "ci_hi": hi})
    return rows


def breakdown_rows(df, a) -> list[dict[str, Any]]:
    from src.data.schema import LANGUAGE_BUCKETS
    from src.eval.metrics import wilson

    rows = []
    for bucket in LANGUAGE_BUCKETS:
        sub = df[df["language"] == bucket].copy()
        for alpha in ALPHAS:
            flag, clear, _ = decisions(sub, a, bucket, alpha)
            sub["flag"], sub["clear"] = flag, clear
            for error, cls, col in (("false_flag", 0, "flag"), ("false_clear", 1, "clear")):
                of_class = sub[sub["label"] == cls]
                total_err = int(of_class[col].sum())
                for dim in ("length_bin", "group"):
                    order = LENGTH_LABELS if dim == "length_bin" else sorted(of_class[dim].dropna().unique())
                    for level in order:
                        g = of_class[of_class[dim] == level]
                        if len(g) < MIN_BREAKDOWN_N:
                            continue
                        k, n = int(g[col].sum()), len(g)
                        lo, hi = wilson(k, n)
                        rows.append({"bucket": bucket, "alpha": alpha, "error": error,
                                     "dimension": "length_bin" if dim == "length_bin" else
                                     ("source corpus" if cls == 0 else "generator"),
                                     "level": level, "errors": k, "n": n, "rate": k / n,
                                     "ci_lo": lo, "ci_hi": hi,
                                     "share_of_errors": k / total_err if total_err else float("nan")})
    return rows


def examples_md(df, a) -> str:
    import numpy as np

    from src.data.schema import LANGUAGE_BUCKETS

    rng = np.random.default_rng(0)
    out = ["# T7 examples: real errors of the frozen gate on the clean test split", "",
           "Two seeded-random (seed 0) examples per bucket per error type. Texts are the corpus text, "
           "first 200 characters (whitespace collapsed). `p` = fused calibrated machine-probability; "
           "headA / headB = the two raw head scores (higher = more machine-like, headB is inverted on te/hi, "
           "see decisions.md). Decision support only: these are the detector's mistakes, not verdicts on authors.", ""]
    for bucket in LANGUAGE_BUCKETS:
        sub = df[df["language"] == bucket]
        tau05, tau01, low = a.conformal[0.05].threshold(bucket), a.conformal[0.01].threshold(bucket), a.cal_median[bucket]
        out += [f"## {bucket}  (tau 0.05 = {tau05:.3f}, tau 0.01 = {tau01:.3f}, HUMAN cutoff = {low:.3f})", ""]
        pools = (("False flag (human text flagged MACHINE at alpha 0.05)", sub[(sub["label"] == 0) & (sub["p"] > tau05)]),
                 ("False clear (machine text called HUMAN)", sub[(sub["label"] == 1) & (sub["p"] < low)]))
        for title, pool in pools:
            out += [f"### {title} — {len(pool)} such rows", ""]
            if pool.empty:
                out += ["(none)", ""]
                continue
            pick = pool.iloc[rng.choice(len(pool), size=min(N_EXAMPLES, len(pool)), replace=False)]
            for rid, r in pick.iterrows():
                text = " ".join(str(r["text"]).split())[:200]
                extra = f", also flagged at alpha 0.01: {'yes' if r['p'] > tau01 else 'no'}" if r["label"] == 0 else ""
                out += [f"- `{rid}` — {r['group']}, {r['length_words']} words, p = {r['p']:.3f}, "
                        f"headA = {r['headA']:.3f}, headB = {r['headB']:.3f}{extra}", f"  > {text}", ""]
    return "\n".join(out) + "\n"


def _pct(r) -> str:
    return f"{r['rate'] * 100:.1f}% [{r['ci_lo'] * 100:.1f}, {r['ci_hi'] * 100:.1f}]"


def render(conf, brk) -> str:
    import pandas as pd

    lines = ["# T7: error analysis (clean test split, frozen gate)", "",
             "Reproduce: `python -m experiments.exp14_error_analysis --config configs/models_norm.yaml`. "
             "Rates are share of the stated denominator, with 95% Wilson intervals in percent. Examples: `T7_examples.md`. "
             "Gate: MACHINE if p > tau_alpha, HUMAN if p < human calibration median, else ABSTAIN.", "",
             "## Confusion counts", ""]
    c = pd.DataFrame(conf)
    for alpha in ALPHAS:
        lines += [f"### alpha = {alpha}", "",
                  "| bucket | human / machine | correct MACHINE | correct HUMAN | false flag (human→MACHINE) | "
                  "false clear (machine→HUMAN) | abstain (all) | abstain human / machine rows |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for bucket in c["bucket"].unique():
            s = c[(c["bucket"] == bucket) & (c["alpha"] == alpha)].set_index("cell")

            def f(name):
                r = s.loc[name]
                return f"{int(r['count'])}/{int(r['denominator'])} = {_pct(r)}"
            lines.append(f"| {bucket} | {int(s.loc['true_human', 'count'])} / {int(s.loc['true_machine', 'count'])} | "
                         f"{f('correct_machine (machine->MACHINE)')} | {f('correct_human (human->HUMAN)')} | "
                         f"{f('false_flag (human->MACHINE)')} | {f('false_clear (machine->HUMAN)')} | "
                         f"{f('abstain (all)')} | {_pct(s.loc['abstain (human rows)'])} / {_pct(s.loc['abstain (machine rows)'])} |")
        lines.append("")
    lines += ["## Errors by length bin and by generator / source corpus", "",
              "`errors/n`: false flags are over human rows of that level, false clears over machine rows. "
              "`share` = this level's share of all errors of that type in the bucket. Levels with no errors are "
              "omitted from the markdown (all levels are in `T7_breakdown.csv`).", ""]
    b = pd.DataFrame(brk)
    for error, alphas in (("false_flag", ALPHAS), ("false_clear", (0.01,))):
        for alpha in alphas:
            title = f"False flags (human→MACHINE), alpha {alpha}" if error == "false_flag" else \
                "False clears (machine→HUMAN) — independent of alpha"
            lines += [f"### {title}", "", "| bucket | dimension | level | errors/n | rate [95% CI] | share |",
                      "| --- | --- | --- | --- | --- | --- |"]
            for _, r in b[(b["error"] == error) & (b["alpha"] == alpha) & (b["errors"] > 0)].iterrows():
                lines.append(f"| {r['bucket']} | {r['dimension']} | {r['level']} | {r['errors']}/{r['n']} | "
                             f"{_pct(r)} | {r['share_of_errors'] * 100:.0f}% |")
            lines.append("")
    notes = OUT / "T7_bucket_notes.md"
    if notes.exists():
        lines += [notes.read_text(encoding="utf-8").strip(), ""]
    return "\n".join(lines)


def run(config_path: str) -> None:
    import pandas as pd

    from src.utils.io import write_csv

    a, df = build_frame(config_path)
    conf, brk = confusion_rows(df, a), breakdown_rows(df, a)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(pd.DataFrame(conf), OUT / "T7_error_analysis.csv")
    write_csv(pd.DataFrame(brk), OUT / "T7_breakdown.csv")
    (OUT / "T7_examples.md").write_text(examples_md(df, a), encoding="utf-8")
    (OUT / "T7_error_analysis.md").write_text(render(conf, brk), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
