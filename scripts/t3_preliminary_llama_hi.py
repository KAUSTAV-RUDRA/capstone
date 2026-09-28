"""Preliminary T3 on llama's hi bucket: does NOT touch the frozen corpus.

2026-09-28: llama is the sole held-out generator (phi dropped entirely; llama
failed its own cm and te gates). Its 500 hi rows cleared the hi gate and are
on disk at data/raw/machine/llama.jsonl. This script gives a preliminary read
on held-out generalisation (non-negotiable #4) for hi WITHOUT re-freezing
data/processed/splits.json (frozen, non-negotiable #6): it cleans and
length-matches llama's hi rows against the EXISTING frozen hi human TEST
split only, using the same clean_rows/length_match machinery Part 13 used
(src/data/clean_artifacts.py, src/data/freeze_splits.py), and scores them
with the ALREADY-FITTED headA/headB/fusion/conformal artifacts in
results/models/ and results/conformal_thresholds.csv -- nothing is fit here.

human-vs-qwen numbers are read straight from results/scores.parquet (the
existing frozen hi test split); only human-vs-llama needs fresh headA/headB
inference, on the length-matched llama rows only.

Run::

    python -m scripts.t3_preliminary_llama_hi [--config configs/data.yaml]
        [--models-config configs/models.yaml] [--out docs/results/t3_preliminary.md]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BUCKET = "hi"
GENERATOR = "llama"
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--out", default="docs/results/t3_preliminary.md")
    args = parser.parse_args()

    from sklearn.metrics import roc_auc_score

    from src.data.clean_artifacts import clean_rows
    from src.data.freeze_splits import length_edges, length_match
    from src.eval.pipeline import fit_full_pipeline
    from src.features.curvature import CurvatureScorer
    from src.features.stylometric import HeadA, StylometricExtractor
    from src.utils.config import load_config
    from src.utils.io import read_jsonl

    config = load_config(str(REPO / args.config))
    models_config = load_config(str(REPO / args.models_config))

    # --- 1. llama's raw hi rows, cleaned exactly as Part 13 cleaned qwen's ----
    machine_dir = REPO / config["machine_corpus"]["output_dir"]
    raw = read_jsonl(machine_dir / f"{GENERATOR}.jsonl")
    assert all(r["language"] == BUCKET for r in raw), "expected llama.jsonl to hold hi rows only"
    kept, dropped, report = clean_rows(raw, enforce_guard=False)
    print(f"cleaned: {len(kept)} kept, {len(dropped)} dropped of {len(raw)} "
          f"({100 * len(dropped) / len(raw):.1f}%)")
    if dropped:
        from collections import Counter
        reasons = Counter(r for d in dropped for r in d["drop_reasons"])
        print(f"  drop reasons: {dict(reasons)}")

    # --- 2. length-match against the FROZEN hi human test split (read-only) --
    corpus = read_jsonl(REPO / "data/processed/corpus.jsonl")
    hi_test_human = [r for r in corpus if r["language"] == BUCKET and r["split"] == "test" and r["label"] == 0]
    hi_test_qwen = [r for r in corpus if r["language"] == BUCKET and r["split"] == "test" and r["label"] == 1
                    and r["generator"] == "qwen7b"]
    assert hi_test_human and hi_test_qwen, "frozen hi test split is missing human or qwen rows"

    n_bins = int(config["corpus"]["length_bins"])
    edges = length_edges([r["length_words"] for r in hi_test_human], n_bins)
    rng = random.Random(SEED)
    matched, excluded, bins = length_match(hi_test_human, kept, edges, rng)
    print(f"length-matched: {len(matched)} of {len(kept)} llama rows kept "
          f"(reference: {len(hi_test_human)} frozen hi human test rows)")
    for b in bins:
        print(f"  bin {b['bin']:>10}: reference {b['reference']:>3}, pool_in {b['pool_in']:>3}, "
              f"pool_kept {b['pool_kept']:>3}")

    texts = [r["text"] for r in matched]
    langs = [BUCKET] * len(texts)

    # --- 3. headA -- LOAD the fitted model, do not fit -----------------------
    head_a_cfg = models_config["head_a"]
    head_a = HeadA.load(REPO / head_a_cfg["model_path"])
    extractor = StylometricExtractor(config=head_a_cfg, syntax=True)
    X_a = extractor.raw_features(texts, langs)
    llama_headA = head_a.predict(X_a, langs)
    print(f"headA scored on {len(texts)} llama hi rows (model loaded from {head_a_cfg['model_path']}, not refit)")

    # --- 4. headB -- mGPT forward pass, nothing to fit -----------------------
    head_b_cfg = models_config["head_b"]
    scorer = CurvatureScorer(head_b_cfg["scorer"], device=head_b_cfg.get("device", "cuda"),
                             load_in_8bit=bool(head_b_cfg.get("load_in_8bit", False)), config=head_b_cfg)
    llama_headB = scorer.stats(texts)[:, 0]
    print(f"headB scored on {len(texts)} llama hi rows ({head_b_cfg['scorer']})")

    # --- 5. fusion + conformal -- LOAD, do not fit ----------------------------
    artifacts = fit_full_pipeline(str(REPO / "configs/default.yaml"))
    llama_logit = artifacts.fuser_ab.decision_function(np.column_stack([llama_headA, llama_headB]), langs)
    llama_fused = 1.0 / (1.0 + np.exp(-llama_logit))
    llama_calibrated = artifacts.temperature.transform(llama_logit, bucket=BUCKET)

    # --- 6. human / qwen reference scores -- read straight from scores.parquet
    scores = artifacts.scores  # indexed by id, already has headA/headB/fused_ab_calibrated for the frozen corpus
    human_ids = [r["id"] for r in hi_test_human]
    qwen_ids = [r["id"] for r in hi_test_qwen]
    human_headA = scores.loc[human_ids, "headA"].to_numpy(dtype=float)
    human_headB = scores.loc[human_ids, "headB"].to_numpy(dtype=float)
    human_calibrated = scores.loc[human_ids, "fused_ab_calibrated"].to_numpy(dtype=float)
    qwen_headA = scores.loc[qwen_ids, "headA"].to_numpy(dtype=float)
    qwen_headB = scores.loc[qwen_ids, "headB"].to_numpy(dtype=float)
    qwen_calibrated = scores.loc[qwen_ids, "fused_ab_calibrated"].to_numpy(dtype=float)

    def auroc(human: np.ndarray, machine: np.ndarray) -> float:
        y = np.concatenate([np.zeros(len(human)), np.ones(len(machine))])
        s = np.concatenate([human, machine])
        return float(roc_auc_score(y, s))

    results = {
        "llama": {
            "n": len(matched),
            "headA": auroc(human_headA, llama_headA),
            "headB": auroc(human_headB, llama_headB),
            "fused": auroc(human_calibrated, llama_calibrated),
        },
        "qwen7b": {
            "n": len(hi_test_qwen),
            "headA": auroc(human_headA, qwen_headA),
            "headB": auroc(human_headB, qwen_headB),
            "fused": auroc(human_calibrated, qwen_calibrated),
        },
    }

    tau_hi = artifacts.conformal[0.01].threshold(BUCKET)
    fpr_at_tau = float(np.mean(human_calibrated > tau_hi))
    print(f"\ntau_0.01[hi] = {tau_hi:.4f}; empirical FPR on frozen hi human test "
          f"({len(human_ids)} rows) = {fpr_at_tau:.4f}")

    for name, r in results.items():
        print(f"{name}: n={r['n']}  headA={r['headA']:.3f}  headB={r['headB']:.3f}  fused={r['fused']:.3f}")

    # --- 7. write the report ---------------------------------------------------
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# T3 preliminary — llama on hi",
        "",
        "**Preliminary, not a frozen-corpus result.** `data/processed/splits.json` "
        "was NOT touched. llama's 500 hi rows (the only bucket that cleared its "
        "24-row gate — cm and te both failed theirs, decisions.md 2026-09-28) were "
        "cleaned with `src.data.clean_artifacts.clean_rows` (Part 13's cleaning "
        "rules, unchanged) and length-matched against the EXISTING frozen hi human "
        "test split (392 rows) with `src.data.freeze_splits.length_match` — the "
        "same mechanism Part 13 used, applied read-only against already-frozen "
        "reference data. headA and the fusion/conformal artifacts were LOADED from "
        f"`results/models/` and `results/conformal_thresholds.csv`, not refit. "
        "headB is mGPT inference, which has no fitted state to reuse.",
        "",
        f"Cleaning: {len(kept)} of {len(raw)} llama hi rows kept "
        f"({100 * len(dropped) / len(raw):.1f}% dropped). Length-matching against "
        f"the {len(hi_test_human)}-row frozen hi human test set kept "
        f"**{len(matched)}** of those.",
        "",
        "## AUROC (human hi test vs. machine), hi bucket only",
        "",
        "| generator | n (machine) | headA | headB (mGPT) | fused (headA+headB, calibrated) |",
        "|---|---|---|---|---|",
        f"| llama (held-out) | {results['llama']['n']} | {results['llama']['headA']:.3f} | "
        f"{results['llama']['headB']:.3f} | {results['llama']['fused']:.3f} |",
        f"| qwen7b (seen) | {results['qwen7b']['n']} | {results['qwen7b']['headA']:.3f} | "
        f"{results['qwen7b']['headB']:.3f} | {results['qwen7b']['fused']:.3f} |",
        "",
        f"llama's row count above ({results['llama']['n']}) is the length-matched count, "
        f"not the full 500 -- length-matching trims the pool to the reference "
        f"bin distribution, same as any other generator's rows would be trimmed "
        f"against the human side.",
        "",
        "## FPR at tau_0.01 (hi)",
        "",
        f"`tau_0.01[hi]` = **{tau_hi:.4f}** (from `results/conformal_thresholds.csv`, "
        f"fit on the 1000-row hi calibration split, unchanged). Empirical false-"
        f"positive rate on the frozen hi human **test** split ({len(human_ids)} rows, "
        f"calibrated fused score) = **{fpr_at_tau:.4f}**. This is a property of the "
        f"human side only (same reference set for both llama and qwen comparisons "
        f"above), not generator-specific.",
        "",
        "## Caveats",
        "",
        "- Preliminary: one held-out generator, one bucket (hi only — cm and te "
        "  both failed llama's gate). Not a substitute for the frozen-corpus T3 "
        "  once the corpus is formally extended.",
        "- llama's hi rows are length-matched against frozen human test only; they "
        "  were never length-matched against each other the way Part 13's "
        "  cal/train/test split balances prompt groups, so there is no llama "
        "  train/cal split here — this is test-only, read-only scoring.",
        "- qwen's numbers are the frozen corpus's existing hi test AUROC, included "
        "  for a same-bucket seen-vs-held-out comparison, not recomputed.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
