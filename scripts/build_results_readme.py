"""Assemble docs/results/README.md: one paragraph + rendered table per table.

Review-2 sprint Day 4, final step. Reads the CSVs already written by
experiments/exp01, exp04, exp05, exp07, exp08, exp09 (via
src.eval.pipeline.fit_full_pipeline, run once, cached) and the headC
shortcut diagnosis, and writes one markdown doc with the key numbers spelled
out in prose, per table, plus the rendered table itself (no `tabulate`
dependency in requirements.txt, so tables are hand-formatted).

Usage: python -m scripts.build_results_readme
Writes: docs/results/README.md
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def md_table(df: pd.DataFrame, float_cols: tuple[str, ...] = (), precision: int = 3) -> str:
    df = df.copy()
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{precision}f}")
    df = df.fillna("")
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def main() -> None:
    t1 = pd.read_csv("results/T1_main_results.csv")
    t2 = pd.read_csv("results/T2_ablation.csv")
    t5 = pd.read_csv("results/T5_fairness.csv")
    t6 = pd.read_csv("results/T6_calibration.csv")
    f2 = pd.read_csv("results/F2_fertility_vs_auroc.csv")
    coef = pd.read_csv("results/fusion_coefficients.csv")

    t1_fused = t1[t1["method"] == "fused (A+B)"].set_index("bucket")
    t1_headc = t1[t1["method"] == "headC"].set_index("bucket")
    t1_ppl = t1[t1["method"] == "ppl"].set_index("bucket")

    t5_alpha01 = t5[(t5["alpha"] == 0.01) & (t5["group"] == "all")].set_index("bucket")
    t5_alpha05 = t5[(t5["alpha"] == 0.05) & (t5["group"] == "all")].set_index("bucket")
    t5_l1 = t5[(t5["bucket"] == "en") & (t5["group"].str.startswith("writer_L1_band"))]

    ab_coef = coef[coef["fusion"] == "A+B (adopted)"].set_index("feature")["weight_standardised"]

    parts: list[str] = []
    parts.append("# Results — review-2 sprint, Day 4\n")
    parts.append(
        "Fusion, calibration, abstention and the T1/T2/T5/T6/F1/F2 tables, all built from "
        "`results/scores.parquet` (no GPU). Everything here is reproducible with:\n\n"
        "```\npython -m experiments.exp01_baselines --config configs/default.yaml   # T1\n"
        "python -m experiments.exp04_fusion --config configs/default.yaml       # fusion + coefficients\n"
        "python -m experiments.exp05_abstention --config configs/default.yaml   # T6, F1\n"
        "python -m experiments.exp07_fairness --config configs/default.yaml     # T5\n"
        "python -m experiments.exp08_ablation --config configs/default.yaml     # T2\n"
        "python -m experiments.exp09_fertility_curvature --config configs/default.yaml  # F2\n"
        "python -m scripts.diagnose_headc --config configs/default.yaml         # headC diagnosis\n"
        "```\n\n"
        "All five downstream scripts share one fit "
        "(`src.eval.pipeline.fit_full_pipeline`, cached in `results/scores.parquet` and "
        "`results/conformal_thresholds.csv`/`results/temperature.csv`), so re-running any one of "
        "them alone is consistent with the others; pass `force=True` to `fit_full_pipeline` to refit.\n"
    )

    parts.append("## Head C: excluded from fusion\n")
    parts.append(
        "Head C (MuRIL embeddings + per-bucket logistic head) scores 1.000 AUROC on en/hi/te "
        f"and {t1_headc.loc['cm','auroc']:.3f} on cm — implausibly higher than Head A "
        "(0.91-0.99) and far higher than Head B's curvature (0.12-0.86 on this corpus). "
        "`scripts/diagnose_headc.py` rules out length (single-feature AUROC 0.50-0.57) and "
        "domain (0.50-0.60) as the shortcut — the corpus's length/domain matching held. But a "
        "plain TF-IDF unigram+bigram bag-of-words classifier, with no embeddings at all, tracks "
        "Head C almost exactly (0.99-1.00 on en/hi/te, 0.90 on cm), and the embedding's first "
        "principal component correlates with the label directly (|r| 0.75-0.90). Only one "
        "generator (qwen7b) is scored so far. **Head C is separating on that generator's "
        "lexical fingerprint, not a generalisable AI-vs-human signal — non-negotiable #4's "
        "exact warning about same-generator evaluation.** Full numbers: "
        "[headc_diagnosis.md](headc_diagnosis.md). **Decision: Head C is EXCLUDED from the "
        "adopted fusion.** It is still scored standalone in T1/T2 to document the exclusion "
        "quantitatively, and `fused_abc` (A+B+C) is computed for reference only, never used "
        "downstream of fusion. Re-check once gemma/mistral (seen) and especially llama/phi "
        "(held-out) are scored.\n"
    )

    parts.append("## T1 — main results, seven methods, per bucket\n")
    parts.append(
        "AUROC + F1 for the three raw-statistic baselines (ppl, binoculars, fastdetectgpt_en), "
        "the three heads (headA, headB, headC) and the adopted fusion (fused A+B). F1's "
        "threshold is train-tuned per bucket per method (the frozen `cal` split is human-only, "
        "so it can't supply an F1 threshold) and applied unchanged to test.\n\n"
        f"Fusion (A+B) beats every baseline in every bucket: AUROC "
        f"en {t1_fused.loc['en','auroc']:.3f}, hi {t1_fused.loc['hi','auroc']:.3f}, "
        f"te {t1_fused.loc['te','auroc']:.3f}, cm {t1_fused.loc['cm','auroc']:.3f} — against "
        f"the best baseline (perplexity) at "
        f"en {t1_ppl.loc['en','auroc']:.3f}, hi {t1_ppl.loc['hi','auroc']:.3f}, "
        f"te {t1_ppl.loc['te','auroc']:.3f}, cm {t1_ppl.loc['cm','auroc']:.3f}. Head B alone "
        "(curvature, the paper's headline mechanism) is far weaker on Indic — 0.187 (hi) and "
        "0.120 (te), BELOW chance, an inversion rather than a fragmentation-driven decay (see "
        "F2) — and fusion recovers a strong score there because Head A carries the bucket. cm "
        "is the weakest bucket everywhere (fusion 0.904, F1 0.533), consistent with cm's known "
        "corpus limitations (chat-register calibration vs comment-register test, docs/data/"
        "corpus_card.md).\n\n"
        f"{md_table(t1, float_cols=('auroc','f1','f1_threshold'))}\n"
    )

    parts.append("## T2 — ablation: A / B / C / A+B / A+B+C / +temperature / +conformal / +abstain\n")
    parts.append(
        "`+temperature`'s AUROC and F1 are IDENTICAL to `A+B` in every bucket, by construction: "
        "temperature scaling is `sigmoid(logit / T)`, a monotonic transform for any T > 0, so it "
        "cannot change score ranking (AUROC) or the best-achievable F1 (found by sweeping every "
        "threshold) — its payoff is calibration quality (ECE), reported in T6, not "
        "discrimination. `A+B+C` is shown for reference only (excluded per the headC diagnosis "
        "above) — note how much closer to 1.000 it pulls every bucket, which is itself further "
        "evidence the lift is the shortcut, not real signal. `+conformal` (the per-bucket "
        "alpha=0.05 threshold, no abstention band) trades F1 down a little for a bounded FPR. "
        "`+abstain` (the full three-way gate) reports accuracy and FPR on the covered subset "
        "instead of AUROC/F1 — coverage lands 58-65% with accuracy 98.7-99.2% and FPR under 3% "
        "in every bucket, the headline abstention result.\n\n"
        f"{md_table(t2, float_cols=('auroc','f1','fpr','coverage','accuracy'))}\n"
    )

    parts.append("## T5 — fairness: FPR at alpha 0.01/0.05, per-bucket vs global tau\n")
    global_breaks = []
    for alpha, sub in ((0.01, t5_alpha01), (0.05, t5_alpha05)):
        for bucket, row in sub.iterrows():
            if row["fpr_global_tau"] > alpha:
                global_breaks.append(f"{bucket} at alpha={alpha} (global FPR {row['fpr_global_tau']:.3f})")
    parts.append(
        "The per-bucket conformal threshold is the adopted policy (non-negotiable #3); a global "
        "pooled threshold is computed only for comparison. **The global threshold breaks its own "
        f"FPR guarantee in {len(global_breaks)} case(s): {', '.join(global_breaks) if global_breaks else 'none'}** "
        "— exactly the failure mode per-language calibration exists to prevent. The per-bucket "
        "threshold stays at or under its target alpha everywhere it was fit to.\n\n"
        "en's L1/L2 split (writer_L1_band: general = L1 proxy, indian = Indian-English L2 "
        "proxy) is the fairness headline the whole framework is motivated by (§2, the Stanford "
        "TOEFL finding). "
    )
    l1_note = "no en L1/L2 rows found"
    if not t5_l1.empty:
        g = t5_l1[(t5_l1["alpha"] == 0.05) & (t5_l1["group"] == "writer_L1_band=general")]
        i = t5_l1[(t5_l1["alpha"] == 0.05) & (t5_l1["group"] == "writer_L1_band=indian")]
        if not g.empty and not i.empty:
            gf, iv = float(g["fpr_per_bucket_tau"].iloc[0]), float(i["fpr_per_bucket_tau"].iloc[0])
            direction = "LOWER" if iv < gf else "higher"
            l1_note = (
                f"At alpha=0.05, per-bucket tau: general (L1) FPR {gf:.3f} vs indian (L2 proxy) "
                f"FPR {iv:.3f} — indian is {direction} than general, the OPPOSITE direction from "
                "commercial detectors' native/non-native bias this project is motivated by "
                "(§2). This is a single-generator (qwen7b), single-calibration-split result and "
                "needs re-checking on more generators before it is claimed anywhere; it should "
                "not be read as 'fairness solved'."
            )
    parts.append(l1_note + "\n\n")
    parts.append(
        "**Not computable this sitting:** the unmatched-human robustness check (does FPR hold "
        "on the human rows `data/processed/splits.json`'s `excluded` block trimmed during "
        "length-matching — 1,976 `length_match` + 259 `clean` ids?) needs Head B (mGPT, GPU) "
        "scored on text that has never been scored, and this sprint is explicitly no-GPU. "
        "Written as `fpr=NaN` with a note in the CSV rather than a fabricated number — needs a "
        "GPU sitting.\n\n"
        f"{md_table(t5, float_cols=('fpr_per_bucket_tau','fpr_global_tau','tau_per_bucket','tau_global'))}\n"
    )

    parts.append("## T6 — calibration: ECE before/after temperature; accuracy at 50/70/90% coverage\n")
    ece_lines = ", ".join(f"{r.bucket} {r.ece_before:.3f}→{r.ece_after:.3f}" for r in t6.itertuples())
    parts.append(
        f"ECE before→after temperature scaling: {ece_lines}. Temperature helps in "
        "en/te/cm and is roughly neutral-to-slightly-worse in hi — all four are small in "
        "absolute terms (<0.08), since the fused model was already fairly well-separated. "
        "Accuracy at 50% coverage is near-ceiling everywhere (0.98-1.00) but en's FPR at 50% "
        "coverage is 1.000 on only 5 covered human samples — those 5 are humans the model was "
        "*confidently wrong* about (calibrated probability ~0.99), not a computation error; a "
        "real thing to flag in the paper's limitations, not a bug.\n\n"
        f"{md_table(t6, float_cols=('temperature','ece_before','ece_after','accuracy_at_50pct','fpr_at_50pct','accuracy_at_70pct','fpr_at_70pct','accuracy_at_90pct','fpr_at_90pct'))}\n"
    )

    parts.append("## F1 — risk-coverage curves, four buckets overlaid\n")
    parts.append(
        "`results/F1_risk_coverage.png` (data: `results/F1_risk_coverage.csv`). Risk stays "
        "near-zero out to roughly 80-90% coverage in en/hi/te and rises earlier in cm, matching "
        "cm's weaker standalone fusion AUROC (0.904) and its known calibration/test "
        "register mismatch.\n\n"
        "![F1 risk-coverage curves](../../results/F1_risk_coverage.png)\n"
    )

    parts.append("## F2 — tokenizer fertility vs curvature AUROC, per bucket\n")
    parts.append(
        "`results/F2_fertility_vs_auroc.png` (data: `results/F2_fertility_vs_auroc.csv`). This "
        "argues AGAINST a simple fragmentation story for Head B's hi/te collapse: "
        "fastdetectgpt_en's scorer (Qwen2.5-0.5B) has WORSE fertility than mGPT on hi/te (4.80 "
        "vs 3.56 tokens/word on hi, 11.92 vs 6.32 on te) yet stays clearly above chance there "
        "(0.728, 0.555), while mGPT — with better relative fertility — drops BELOW chance "
        "(0.187, 0.120). Worse fragmentation does not track worse curvature AUROC across "
        "scorers; whatever breaks Head B on hi/te (docs/progress.md 2026-09-24: \"the inversion "
        "is mGPT-specific\") is a property of mGPT's own training distribution on Indic text, "
        "not a fertility artefact. This is worth a sentence in the paper's Head B section.\n\n"
        f"{md_table(f2, float_cols=('tokens_per_word','auroc'))}\n"
    )

    parts.append("## Fusion coefficients (standardised, interpretability)\n")
    coef_lines = ", ".join(f"{k}={v:+.2f}" for k, v in ab_coef.items())
    parts.append(
        f"Adopted fusion (A+B): {coef_lines}. Head A dominates (standardised coefficient "
        f"{ab_coef.get('headA', float('nan')):+.2f}); Head B's coefficient is small and "
        f"NEGATIVE ({ab_coef.get('headB', float('nan')):+.2f}) — the fusion model learns to "
        "partially invert Head B's raw score, consistent with Head B's own below-chance AUROC "
        "on hi/te. Bucket one-hot weights are all modest, meaning the fusion is not just "
        "reproducing a per-bucket base rate.\n\n"
        f"{md_table(coef, float_cols=('weight_standardised',))}\n"
    )

    parts.append(
        "## Known limitations of this sitting\n\n"
        "- Everything above is on **qwen7b only** (the single generator fully scored). "
        "gemma/mistral (seen) and llama/phi (held-out, non-negotiable #4) are not in these "
        "numbers yet — T1/T2's AUROCs and Head C's exclusion both need re-checking once they "
        "land.\n"
        "- The unmatched-human FPR robustness check (T5) needs a GPU sitting to score Head B on "
        "the length-matching-excluded rows.\n"
        "- Temperature is fit on the TRAIN split (the only split besides test with both "
        "classes; the frozen `cal` split is human-only by design) — optimistic by construction, "
        "documented in `src/calibration/temperature.py`.\n"
    )

    Path("docs/results").mkdir(parents=True, exist_ok=True)
    Path("docs/results/README.md").write_text("\n".join(parts), encoding="utf-8")
    print("wrote docs/results/README.md")


if __name__ == "__main__":
    main()
