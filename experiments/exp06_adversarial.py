"""exp06 - T4: adversarial robustness, clean vs hybrid vs paraphrase vs back-translation.

Applies the FROZEN detector to attacked rows. Nothing is refit: the fuser
(results/models/fuser_ab.joblib), per-bucket temperature (results/temperature.csv)
and per-bucket conformal thresholds + HUMAN cutoff (results/conformal_thresholds.csv)
are loaded exactly as shipped, then run over the attacked rows' headA / headB scores
(scored by ``python -m src.eval.score --only-attacked``, into the two attacked_scores
parquets named in the config's ``pipeline.attacked_scores`` block). The script asserts
the loaded thresholds equal the frozen file and that re-applying the frozen fuser to the
clean test rows reproduces their stored fused scores.

Per bucket x attack it reports, on the fused calibrated machine-probability:

  auroc (+95% stratified bootstrap CI)  also headA / headB alone
  det_a05 / det_a01   machine rows with p > tau_alpha (alpha 0.05 / 0.01)
  ff_a05 / ff_a01     human rows with p > tau_alpha: false-flag rate, i.e. empirical FPR
  Gate (three-way, MACHINE if p > tau_alpha, HUMAN if p < human cal-median, else ABSTAIN):
  abstain_*           all rows / human rows / machine rows that abstain
  false_clear_*       machine rows called HUMAN

for the gate at alpha 0.01 (the shipped gate) and alpha 0.05. "clean" is the whole
clean test split (same rows as T1/T3); ``clean_matched_auroc`` is the clean version of
exactly the source rows each attack used, for a paired comparison. Slices reported as
separate rows: paraphrase of qwen7b sources (self-paraphrase: the paraphraser model also
wrote the text) vs other sources; attack_changed=false; bt_suspect; bt_failed. Slices that
filter on machine-only properties keep every human row of the attack. AUROC is only given
when both classes have >= MIN_AUROC_N rows. cm has no back-translation (n/a).

Writes: results/exp06_adversarial.csv, docs/results/T4_adversarial.csv, docs/results/T4_adversarial.md
Serves: docs/project-context-master.md §7 (T4); docs/master-execution-plan.md Phase 3 §3.5.
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

RESULTS_CSV = "results/exp06_adversarial.csv"
DOCS_CSV = "docs/results/T4_adversarial.csv"
DOCS_MD = "docs/results/T4_adversarial.md"
FROZEN_THRESHOLDS = "docs/results/conformal_thresholds.csv"
ATTACKED = "data/processed/attacked.jsonl"
ATTACKS = ("hybrid", "paraphrase", "back_translation")
N_BOOT = 1000
MIN_AUROC_N = 30
SELF_PARAPHRASER = "qwen7b"
HEADS = ("headA", "headB")


def _auroc_ci(y, s, rng) -> tuple[float, float, float]:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    h, m = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if len(h) < MIN_AUROC_N or len(m) < MIN_AUROC_N:
        return float("nan"), float("nan"), float("nan")
    point = float(roc_auc_score(y, s))
    boots = [roc_auc_score(y[i], s[i]) for i in
             (np.concatenate([rng.choice(h, len(h)), rng.choice(m, len(m))]) for _ in range(N_BOOT))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def _rate(mask) -> float:
    return float(mask.mean()) if len(mask) else float("nan")


def _put(out: dict, key: str, mask) -> None:
    """``out[key]`` = rate of ``mask``, ``out[key_lo/_hi]`` = its 95% Wilson interval."""
    from src.eval.metrics import wilson

    out[key] = _rate(mask)
    out[f"{key}_lo"], out[f"{key}_hi"] = wilson(int(mask.sum()), len(mask))


def load_frozen(config_path: str):
    """Fuser, temperature, conformal taus and HUMAN cutoffs, loaded from disk without fitting."""
    import numpy as np
    import pandas as pd

    from src.eval.pipeline import fit_full_pipeline

    a = fit_full_pipeline(config_path)            # loads cached artefacts; refits only if columns missing
    frozen = pd.read_csv(FROZEN_THRESHOLDS)
    for alpha in (0.01, 0.05):
        for _, r in frozen[np.isclose(frozen["alpha"], alpha)].iterrows():
            assert abs(a.conformal[alpha].threshold(r["bucket"]) - r["tau"]) < 1e-12, \
                f"threshold drifted from the frozen file: alpha={alpha} bucket={r['bucket']}"
    return a


def score_attacked(a, config_path: str, attacked: list[dict[str, Any]]) -> "pd.DataFrame":
    """headA/headB + frozen fused calibrated probability for every attacked row."""
    import numpy as np
    import pandas as pd

    from src.eval.score import DEFAULT_SCORES  # noqa: F401  (documents where raw scores live)
    from src.utils.config import load_config

    block = (load_config(config_path).get("pipeline") or {}).get("attacked_scores")
    if not block:
        raise KeyError(f"{config_path}: pipeline.attacked_scores (list of parquets) is not set")
    parts = [pd.read_parquet(p).set_index("id") for p in block]
    raw = pd.concat(parts, axis=1)
    raw = raw.loc[:, ~raw.columns.duplicated()]
    ids = [r["id"] for r in attacked]
    missing = [i for i in ids if i not in raw.index or raw.loc[i, list(HEADS)].isna().any()]
    if missing:
        raise RuntimeError(f"{len(missing)} attacked rows lack headA/headB scores, e.g. {missing[:3]}")
    X = raw.loc[ids, list(HEADS)].to_numpy(dtype=float)
    buckets = [r["language"] for r in attacked]
    logit = a.fuser_ab.decision_function(X, buckets)
    prob = np.array([a.temperature.transform(np.array([l]), bucket=b)[0] for l, b in zip(logit, buckets)])
    df = pd.DataFrame({"id": ids, "headA": X[:, 0], "headB": X[:, 1], "p": prob})
    return df.set_index("id")


def sanity_check_clean(a) -> None:
    """Re-applying the frozen fuser to stored clean test scores reproduces stored fused scores."""
    import numpy as np

    test = [r for r in a.corpus if r["split"] == "test"]
    ids = [r["id"] for r in test]
    X = a.scores.loc[ids, list(HEADS)].to_numpy(dtype=float)
    b = [r["language"] for r in test]
    got = np.array([a.temperature.transform(np.array([l]), bucket=k)[0]
                    for l, k in zip(a.fuser_ab.decision_function(X, b), b)])
    want = a.scores.loc[ids, "fused_ab_calibrated"].to_numpy(dtype=float)
    assert np.allclose(got, want, atol=1e-9), "frozen fuser does not reproduce stored clean fused scores"


def evaluate(group: "pd.DataFrame", a, bucket: str, rng) -> dict[str, Any]:
    """All T4 metrics for one set of rows (columns: label, p, headA, headB) in one bucket."""
    import numpy as np

    y = group["label"].to_numpy()
    p = group["p"].to_numpy(dtype=float)
    hum, mac = y == 0, y == 1
    out: dict[str, Any] = {"n_human": int(hum.sum()), "n_machine": int(mac.sum())}
    out["auroc"], out["auroc_lo"], out["auroc_hi"] = _auroc_ci(y, p, rng)
    for head in HEADS:
        out[f"auroc_{head}"] = _auroc_ci(y, group[head].to_numpy(dtype=float), rng)[0]
    lower = a.cal_median[bucket]
    for alpha, tag in ((0.05, "a05"), (0.01, "a01")):
        tau = a.conformal[alpha].threshold(bucket)
        flag, clear = p > tau, p < lower
        abstain = ~flag & ~clear
        _put(out, f"det_{tag}", flag[mac])
        _put(out, f"ff_{tag}", flag[hum])
        _put(out, f"abstain_{tag}", abstain)
        _put(out, f"abstain_human_{tag}", abstain[hum])
        _put(out, f"abstain_machine_{tag}", abstain[mac])
        _put(out, f"false_clear_{tag}", clear[mac])
    if "length_ratio" in group:
        out["mean_length_ratio"] = float(np.nanmean(group["length_ratio"].to_numpy(dtype=float))) \
            if len(group) else float("nan")
    return out


def run(config_path: str) -> "pd.DataFrame":
    import numpy as np
    import pandas as pd

    from src.data.schema import LANGUAGE_BUCKETS
    from src.utils.io import read_jsonl, write_csv

    a = load_frozen(config_path)
    sanity_check_clean(a)
    attacked = read_jsonl(ATTACKED)
    clean_by_id = {r["id"]: r for r in a.corpus}
    scored = score_attacked(a, config_path, attacked)

    meta = pd.DataFrame(attacked).set_index("id")
    att = meta[["label", "language", "attack_type", "source", "source_id", "attack_changed", "length_words"]].copy()
    for c in ("bt_suspect", "bt_failed"):
        att[c] = meta[c] if c in meta else None
    att = att.join(scored)
    att["length_ratio"] = [r["length_words"] / clean_by_id[r["source_id"]]["length_words"]
                           for r in attacked]

    test = [r for r in a.corpus if r["split"] == "test"]
    clean = pd.DataFrame({"id": [r["id"] for r in test], "label": [r["label"] for r in test],
                          "language": [r["language"] for r in test]}).set_index("id")
    clean = clean.join(a.scores[["headA", "headB"]]).join(a.scores[["fused_ab_calibrated"]].rename(
        columns={"fused_ab_calibrated": "p"}))
    clean["length_ratio"] = 1.0

    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []

    def add(bucket, attack, slice_, group, matched_auroc=float("nan")):
        rec = {"bucket": bucket, "attack": attack, "slice": slice_}
        if len(group) == 0:
            rec["note"] = "n/a (no rows)"
        else:
            rec.update(evaluate(group, a, bucket, rng))
        rec["clean_matched_auroc"] = matched_auroc
        rows.append(rec)

    for bucket in LANGUAGE_BUCKETS:
        add(bucket, "clean", "all test", clean[clean["language"] == bucket])
        for attack in ATTACKS:
            sub = att[(att["language"] == bucket) & (att["attack_type"] == attack)]
            if attack == "back_translation" and bucket == "cm":
                rows.append({"bucket": bucket, "attack": attack, "slice": "all", "note": "n/a (cm not back-translated)"})
                continue
            src = clean.loc[sub["source_id"].unique()] if len(sub) else clean.iloc[0:0]
            matched = _auroc_ci(src["label"].to_numpy(), src["p"].to_numpy(dtype=float), rng)[0] if len(src) else float("nan")
            add(bucket, attack, "all", sub, matched)
            if attack == "paraphrase":
                selfp = (sub["label"] == 0) | (sub["source"] == SELF_PARAPHRASER)
                other = (sub["label"] == 0) | (sub["source"] != SELF_PARAPHRASER)
                add(bucket, attack, f"machine sources = {SELF_PARAPHRASER} (self-paraphrase)", sub[selfp])
                add(bucket, attack, f"machine sources != {SELF_PARAPHRASER}", sub[other])
            add(bucket, attack, "attack_changed=false", sub[sub["attack_changed"] == False])  # noqa: E712
            if attack == "back_translation":
                add(bucket, attack, "bt_suspect=true", sub[sub["bt_suspect"] == True])  # noqa: E712
                add(bucket, attack, "bt_failed=true", sub[sub["bt_failed"] == True])  # noqa: E712

    df = pd.DataFrame(rows)
    for c in ("n_human", "n_machine"):
        df[c] = df[c].astype("Int64")
    write_csv(df, RESULTS_CSV)
    write_csv(df, DOCS_CSV)
    with open(DOCS_MD, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(df, a))
    return df


def _fmt(x, pct=False, nd=3) -> str:
    import math

    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x * 100:.1f}%" if pct else f"{x:.{nd}f}"


def render_markdown(df: "pd.DataFrame", a) -> str:
    lines = ["# T4: adversarial robustness (frozen detector)", "",
             "Frozen fuser + per-bucket temperature + per-bucket conformal thresholds, fit on clean data only; "
             "attacked rows are scored, never refit. Reproduce: `python -m experiments.exp06_adversarial "
             "--config configs/models_norm.yaml`. Source: `T4_adversarial.csv`.", "",
             "Columns: AUROC (95% CI); det = machine rows flagged MACHINE at alpha 0.05 / 0.01; "
             "ff = human rows flagged MACHINE (false-flag) at alpha 0.05 / 0.01; "
             "abstain and false-clear (machine called HUMAN) are for the shipped three-way gate (alpha 0.01; the CSV also has 0.05); "
             "[lo, hi] = 95% Wilson interval in percent; len = mean attacked/source word-count ratio. "
             "AUROC needs >= 30 rows per class, else n/a.", ""]
    head = ("| bucket | attack | slice | n h/m | AUROC [95% CI] | clean-matched AUROC | det 5% | det 1% | ff 5% | ff 1% "
            "| abstain 1% | false-clear 1% | len |")
    lines += [head, "|" + " --- |" * 13]

    def cell(r, key):
        if r[key] != r[key]:
            return "n/a"
        return f"{r[key] * 100:.1f}% [{r[key + '_lo'] * 100:.0f}, {r[key + '_hi'] * 100:.0f}]"

    for _, r in df.iterrows():
        if isinstance(r.get("note"), str) and r["note"].startswith("n/a"):
            lines.append(f"| {r['bucket']} | {r['attack']} | {r['slice']} | | {r['note']} |" + " |" * 8)
            continue
        ci = f"{_fmt(r['auroc'])} [{_fmt(r['auroc_lo'])}, {_fmt(r['auroc_hi'])}]" if r["auroc"] == r["auroc"] else "n/a"
        lines.append(
            f"| {r['bucket']} | {r['attack']} | {r['slice']} | {int(r['n_human'])}/{int(r['n_machine'])} | {ci} | "
            f"{_fmt(r['clean_matched_auroc'])} | " + " | ".join(cell(r, k) for k in
            ("det_a05", "det_a01", "ff_a05", "ff_a01", "abstain_a01", "false_clear_a01"))
            + f" | {_fmt(r['mean_length_ratio'], nd=2)} |")
    lines += ["", "## Heads alone vs fused (AUROC, all rows of each attack)", "",
              "| bucket | attack | headA | headB | fused |", "| --- | --- | --- | --- | --- |"]
    main = df[(df["slice"].isin(["all test", "all"])) & df["auroc"].notna()]
    for _, r in main.iterrows():
        lines.append(f"| {r['bucket']} | {r['attack']} | {_fmt(r['auroc_headA'])} | {_fmt(r['auroc_headB'])} | {_fmt(r['auroc'])} |")
    lines += ["", "## Caveats", "",
              "- AUROC intervals are stratified bootstrap (1000 resamples); rates use Wilson intervals, which treat "
              "rows as independent (several attacked rows can share a prompt).",
              "- Under `paraphrase` and `back_translation` the label-0 rows are human text that an LLM / MT model "
              "then rewrote. Their `ff` is a false flag against the original author, but the text is no longer "
              "purely human-written.",
              "- Slices of `attack_changed=false`, `bt_suspect` and `bt_failed` have 0-30 rows (the bt flags 4 and 2 "
              "rows in total); their rates are anecdotes, not estimates.",
              "- The attacked rows are a capped, stratified subset of the test split (see src/data/attack.py); "
              "`clean_matched_auroc` is the clean score of exactly those source rows."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    args = parser.parse_args()
    import pandas as pd

    pd.set_option("display.width", 250)
    df = run(args.config)
    print(df.drop(columns=[c for c in df.columns if c.startswith(("abstain_human", "abstain_machine"))]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
