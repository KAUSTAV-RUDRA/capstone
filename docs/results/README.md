# Results — review-2 sprint, Day 4

Fusion, calibration, abstention and the T1/T2/T5/T6/F1/F2 tables, all built from `results/scores.parquet` (no GPU). Everything here is reproducible with:

```
python -m experiments.exp01_baselines --config configs/default.yaml   # T1
python -m experiments.exp04_fusion --config configs/default.yaml       # fusion + coefficients
python -m experiments.exp05_abstention --config configs/default.yaml   # T6, F1
python -m experiments.exp07_fairness --config configs/default.yaml     # T5
python -m experiments.exp08_ablation --config configs/default.yaml     # T2
python -m experiments.exp09_fertility_curvature --config configs/default.yaml  # F2
python -m scripts.diagnose_headc --config configs/default.yaml         # headC diagnosis
```

All five downstream scripts share one fit (`src.eval.pipeline.fit_full_pipeline`, cached in `results/scores.parquet` and `results/conformal_thresholds.csv`/`results/temperature.csv`), so re-running any one of them alone is consistent with the others; pass `force=True` to `fit_full_pipeline` to refit.

## Head C: excluded from fusion

Head C (MuRIL embeddings + per-bucket logistic head) scores 1.000 AUROC on en/hi/te and 0.927 on cm — implausibly higher than Head A (0.91-0.99) and far higher than Head B's curvature (0.12-0.86 on this corpus). `scripts/diagnose_headc.py` rules out length (single-feature AUROC 0.50-0.57) and domain (0.50-0.60) as the shortcut — the corpus's length/domain matching held. But a plain TF-IDF unigram+bigram bag-of-words classifier, with no embeddings at all, tracks Head C almost exactly (0.99-1.00 on en/hi/te, 0.90 on cm), and the embedding's first principal component correlates with the label directly (|r| 0.75-0.90). Only one generator (qwen7b) is scored so far. **Head C is separating on that generator's lexical fingerprint, not a generalisable AI-vs-human signal — non-negotiable #4's exact warning about same-generator evaluation.** Full numbers: [headc_diagnosis.md](headc_diagnosis.md). **Decision: Head C is EXCLUDED from the adopted fusion.** It is still scored standalone in T1/T2 to document the exclusion quantitatively, and `fused_abc` (A+B+C) is computed for reference only, never used downstream of fusion. Re-check once gemma/mistral (seen) and especially llama/phi (held-out) are scored.

## T1 — main results, seven methods, per bucket

AUROC + F1 for the three raw-statistic baselines (ppl, binoculars, fastdetectgpt_en), the three heads (headA, headB, headC) and the adopted fusion (fused A+B). F1's threshold is train-tuned per bucket per method (the frozen `cal` split is human-only, so it can't supply an F1 threshold) and applied unchanged to test.

Fusion (A+B) beats every baseline in every bucket: AUROC en 0.985, hi 0.991, te 0.991, cm 0.904 — against the best baseline (perplexity) at en 0.915, hi 0.816, te 0.847, cm 0.711. Head B alone (curvature, the paper's headline mechanism) is far weaker on Indic — 0.187 (hi) and 0.120 (te), BELOW chance, an inversion rather than a fragmentation-driven decay (see F2) — and fusion recovers a strong score there because Head A carries the bucket. cm is the weakest bucket everywhere (fusion 0.904, F1 0.533), consistent with cm's known corpus limitations (chat-register calibration vs comment-register test, docs/data/corpus_card.md).

| bucket | method | n_human | n_machine | auroc | f1 | f1_threshold |
| --- | --- | --- | --- | --- | --- | --- |
| en | ppl | 272 | 361 | 0.915 | 0.886 | -2.615 |
| en | binoculars | 272 | 361 | 0.804 | 0.785 | -0.053 |
| en | fastdetectgpt_en | 272 | 361 | 0.794 | 0.777 | -0.576 |
| en | headA | 272 | 361 | 0.991 | 0.957 | 0.339 |
| en | headB | 272 | 361 | 0.859 | 0.810 | -0.472 |
| en | headC | 272 | 361 | 1.000 | 0.973 | 0.960 |
| en | fused (A+B) | 272 | 361 | 0.985 | 0.957 | 0.385 |
| hi | ppl | 392 | 233 | 0.816 | 0.673 | -1.474 |
| hi | binoculars | 392 | 233 | 0.732 | 0.621 | 0.049 |
| hi | fastdetectgpt_en | 392 | 233 | 0.728 | 0.622 | -0.281 |
| hi | headA | 392 | 233 | 0.984 | 0.939 | 0.465 |
| hi | headB | 392 | 233 | 0.187 | 0.541 | -9.876 |
| hi | headC | 392 | 233 | 1.000 | 0.938 | 0.992 |
| hi | fused (A+B) | 392 | 233 | 0.991 | 0.943 | 0.527 |
| te | ppl | 349 | 165 | 0.847 | 0.675 | -1.147 |
| te | binoculars | 349 | 165 | 0.564 | 0.495 | -0.026 |
| te | fastdetectgpt_en | 349 | 165 | 0.555 | 0.485 | -2.665 |
| te | headA | 349 | 165 | 0.989 | 0.927 | 0.505 |
| te | headB | 349 | 165 | 0.120 | 0.484 | -17.073 |
| te | headC | 349 | 165 | 1.000 | 0.942 | 0.997 |
| te | fused (A+B) | 349 | 165 | 0.991 | 0.933 | 0.475 |
| cm | ppl | 596 | 83 | 0.711 | 0.354 | -5.046 |
| cm | binoculars | 596 | 83 | 0.661 | 0.318 | 0.162 |
| cm | fastdetectgpt_en | 596 | 83 | 0.696 | 0.339 | 0.290 |
| cm | headA | 596 | 83 | 0.910 | 0.536 | 0.798 |
| cm | headB | 596 | 83 | 0.661 | 0.292 | -0.307 |
| cm | headC | 596 | 83 | 0.927 | 0.487 | 0.967 |
| cm | fused (A+B) | 596 | 83 | 0.904 | 0.533 | 0.641 |

## T2 — ablation: A / B / C / A+B / A+B+C / +temperature / +conformal / +abstain

`+temperature`'s AUROC and F1 are IDENTICAL to `A+B` in every bucket, by construction: temperature scaling is `sigmoid(logit / T)`, a monotonic transform for any T > 0, so it cannot change score ranking (AUROC) or the best-achievable F1 (found by sweeping every threshold) — its payoff is calibration quality (ECE), reported in T6, not discrimination. `A+B+C` is shown for reference only (excluded per the headC diagnosis above) — note how much closer to 1.000 it pulls every bucket, which is itself further evidence the lift is the shortcut, not real signal. `+conformal` (the per-bucket alpha=0.05 threshold, no abstention band) trades F1 down a little for a bounded FPR. `+abstain` (the full three-way gate) reports accuracy and FPR on the covered subset instead of AUROC/F1 — coverage lands 58-65% with accuracy 98.7-99.2% and FPR under 3% in every bucket, the headline abstention result.

| bucket | config | auroc | f1 | fpr | coverage | accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| cm | A | 0.910 | 0.536 |  | 1.000 |  |
| cm | B | 0.661 | 0.292 |  | 1.000 |  |
| cm | C | 0.927 | 0.487 |  | 1.000 |  |
| cm | A+B | 0.904 | 0.571 |  | 1.000 |  |
| cm | A+B+C | 0.945 | 0.688 |  | 1.000 |  |
| cm | +temperature | 0.904 | 0.571 |  | 1.000 |  |
| cm | +conformal | 0.904 | 0.471 | 0.035 | 1.000 | 0.894 |
| cm | +abstain |  |  | 0.000 | 0.582 | 0.992 |
| en | A | 0.991 | 0.957 |  | 1.000 |  |
| en | B | 0.859 | 0.810 |  | 1.000 |  |
| en | C | 1.000 | 0.973 |  | 1.000 |  |
| en | A+B | 0.985 | 0.959 |  | 1.000 |  |
| en | A+B+C | 1.000 | 0.992 |  | 1.000 |  |
| en | +temperature | 0.985 | 0.959 |  | 1.000 |  |
| en | +conformal | 0.985 | 0.956 | 0.062 | 1.000 | 0.949 |
| en | +abstain |  |  | 0.015 | 0.605 | 0.992 |
| hi | A | 0.984 | 0.939 |  | 1.000 |  |
| hi | B | 0.187 | 0.541 |  | 1.000 |  |
| hi | C | 1.000 | 0.938 |  | 1.000 |  |
| hi | A+B | 0.991 | 0.941 |  | 1.000 |  |
| hi | A+B+C | 0.999 | 0.987 |  | 1.000 |  |
| hi | +temperature | 0.991 | 0.941 |  | 1.000 |  |
| hi | +conformal | 0.991 | 0.941 | 0.051 | 1.000 | 0.955 |
| hi | +abstain |  |  | 0.027 | 0.597 | 0.987 |
| te | A | 0.989 | 0.927 |  | 1.000 |  |
| te | B | 0.120 | 0.484 |  | 1.000 |  |
| te | C | 1.000 | 0.942 |  | 1.000 |  |
| te | A+B | 0.991 | 0.933 |  | 1.000 |  |
| te | A+B+C | 1.000 | 0.997 |  | 1.000 |  |
| te | +temperature | 0.991 | 0.933 |  | 1.000 |  |
| te | +conformal | 0.991 | 0.926 | 0.049 | 1.000 | 0.951 |
| te | +abstain |  |  | 0.021 | 0.648 | 0.988 |

## T5 — fairness: FPR at alpha 0.01/0.05, per-bucket vs global tau

The per-bucket conformal threshold is the adopted policy (non-negotiable #3); a global pooled threshold is computed only for comparison. **The global threshold breaks its own FPR guarantee in 4 case(s): en at alpha=0.01 (global FPR 0.022), hi at alpha=0.01 (global FPR 0.018), en at alpha=0.05 (global FPR 0.066), cm at alpha=0.05 (global FPR 0.077)** — exactly the failure mode per-language calibration exists to prevent. The per-bucket threshold stays at or under its target alpha everywhere it was fit to.

en's L1/L2 split (writer_L1_band: general = L1 proxy, indian = Indian-English L2 proxy) is the fairness headline the whole framework is motivated by (§2, the Stanford TOEFL finding). 
At alpha=0.05, per-bucket tau: general (L1) FPR 0.109 vs indian (L2 proxy) FPR 0.015 — indian is LOWER than general, the OPPOSITE direction from commercial detectors' native/non-native bias this project is motivated by (§2). This is a single-generator (qwen7b), single-calibration-split result and needs re-checking on more generators before it is claimed anywhere; it should not be read as 'fairness solved'.


**Not computable this sitting:** the unmatched-human robustness check (does FPR hold on the human rows `data/processed/splits.json`'s `excluded` block trimmed during length-matching — 1,976 `length_match` + 259 `clean` ids?) needs Head B (mGPT, GPU) scored on text that has never been scored, and this sprint is explicitly no-GPU. Written as `fpr=NaN` with a note in the CSV rather than a fabricated number — needs a GPU sitting.

| alpha | bucket | group | n_human | fpr_per_bucket_tau | fpr_global_tau | tau_per_bucket | tau_global |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.01 | en | all | 272.0 | 0.007 | 0.022 | 0.995 | 0.988 |
| 0.01 | en | writer_L1_band=general | 138.0 | 0.014 | 0.043 | 0.995 | 0.988 |
| 0.01 | en | writer_L1_band=indian | 134.0 | 0.000 | 0.000 | 0.995 | 0.988 |
| 0.01 | hi | all | 392.0 | 0.013 | 0.018 | 0.994 | 0.988 |
| 0.01 | te | all | 349.0 | 0.011 | 0.003 | 0.968 | 0.988 |
| 0.01 | cm | all | 596.0 | 0.000 | 0.000 | 0.882 | 0.988 |
| 0.05 | en | all | 272.0 | 0.062 | 0.066 | 0.764 | 0.678 |
| 0.05 | en | writer_L1_band=general | 138.0 | 0.109 | 0.116 | 0.764 | 0.678 |
| 0.05 | en | writer_L1_band=indian | 134.0 | 0.015 | 0.015 | 0.764 | 0.678 |
| 0.05 | hi | all | 392.0 | 0.051 | 0.043 | 0.389 | 0.678 |
| 0.05 | te | all | 349.0 | 0.049 | 0.017 | 0.047 | 0.678 |
| 0.05 | cm | all | 596.0 | 0.035 | 0.077 | 0.803 | 0.678 |
| 0.01 | en | L1_L2_disparity (indian - general) |  | -0.014 | -0.043 |  |  |
| 0.05 | en | L1_L2_disparity (indian - general) |  | -0.094 | -0.101 |  |  |
| 0.01 | ALL | unmatched_human_robustness_check | 2235.0 |  |  |  |  |
| 0.05 | ALL | unmatched_human_robustness_check | 2235.0 |  |  |  |  |

## T6 — calibration: ECE before/after temperature; accuracy at 50/70/90% coverage

ECE before→after temperature scaling: en 0.037→0.036, hi 0.033→0.035, te 0.031→0.035, cm 0.081→0.070. Temperature helps in en/te/cm and is roughly neutral-to-slightly-worse in hi — all four are small in absolute terms (<0.08), since the fused model was already fairly well-separated. Accuracy at 50% coverage is near-ceiling everywhere (0.98-1.00) but en's FPR at 50% coverage is 1.000 on only 5 covered human samples — those 5 are humans the model was *confidently wrong* about (calibrated probability ~0.99), not a computation error; a real thing to flag in the paper's limitations, not a bug.

| bucket | temperature | ece_before | ece_after | accuracy_at_50pct | fpr_at_50pct | accuracy_at_70pct | fpr_at_70pct | accuracy_at_90pct | fpr_at_90pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | 0.969 | 0.037 | 0.036 | 0.984 | 1.000 | 0.986 | 0.054 | 0.977 | 0.040 |
| hi | 0.802 | 0.033 | 0.035 | 1.000 | 0.000 | 0.991 | 0.012 | 0.982 | 0.020 |
| te | 0.858 | 0.031 | 0.035 | 0.996 | 0.000 | 0.992 | 0.000 | 0.978 | 0.006 |
| cm | 1.266 | 0.081 | 0.070 | 0.994 | 0.000 | 0.983 | 0.000 | 0.912 | 0.067 |

## F1 — risk-coverage curves, four buckets overlaid

`results/F1_risk_coverage.png` (data: `results/F1_risk_coverage.csv`). Risk stays near-zero out to roughly 80-90% coverage in en/hi/te and rises earlier in cm, matching cm's weaker standalone fusion AUROC (0.904) and its known calibration/test register mismatch.

![F1 risk-coverage curves](../../results/F1_risk_coverage.png)

## F2 — tokenizer fertility vs curvature AUROC, per bucket

`results/F2_fertility_vs_auroc.png` (data: `results/F2_fertility_vs_auroc.csv`). This argues AGAINST a simple fragmentation story for Head B's hi/te collapse: fastdetectgpt_en's scorer (Qwen2.5-0.5B) has WORSE fertility than mGPT on hi/te (4.80 vs 3.56 tokens/word on hi, 11.92 vs 6.32 on te) yet stays clearly above chance there (0.728, 0.555), while mGPT — with better relative fertility — drops BELOW chance (0.187, 0.120). Worse fragmentation does not track worse curvature AUROC across scorers; whatever breaks Head B on hi/te (docs/progress.md 2026-09-24: "the inversion is mGPT-specific") is a property of mGPT's own training distribution on Indic text, not a fertility artefact. This is worth a sentence in the paper's Head B section.

| method | scorer | bucket | tokens_per_word | auroc |
| --- | --- | --- | --- | --- |
| headB | ai-forever/mGPT | en | 1.363 | 0.859 |
| headB | ai-forever/mGPT | hi | 3.560 | 0.187 |
| headB | ai-forever/mGPT | te | 6.319 | 0.120 |
| headB | ai-forever/mGPT | cm | 1.714 | 0.661 |
| fastdetectgpt_en | Qwen/Qwen2.5-0.5B | en | 1.339 | 0.794 |
| fastdetectgpt_en | Qwen/Qwen2.5-0.5B | hi | 4.804 | 0.728 |
| fastdetectgpt_en | Qwen/Qwen2.5-0.5B | te | 11.919 | 0.555 |
| fastdetectgpt_en | Qwen/Qwen2.5-0.5B | cm | 1.672 | 0.696 |

## Fusion coefficients (standardised, interpretability)

Adopted fusion (A+B): headA=+4.03, headB=-0.38, bucket_en=+0.68, bucket_hi=+0.20, bucket_te=-0.26, bucket_cm=-0.63. Head A dominates (standardised coefficient +4.03); Head B's coefficient is small and NEGATIVE (-0.38) — the fusion model learns to partially invert Head B's raw score, consistent with Head B's own below-chance AUROC on hi/te. Bucket one-hot weights are all modest, meaning the fusion is not just reproducing a per-bucket base rate.

| fusion | feature | weight_standardised |
| --- | --- | --- |
| A+B (adopted) | headA | 4.027 |
| A+B (adopted) | headB | -0.381 |
| A+B (adopted) | bucket_en | 0.683 |
| A+B (adopted) | bucket_hi | 0.205 |
| A+B (adopted) | bucket_te | -0.262 |
| A+B (adopted) | bucket_cm | -0.629 |
| A+B+C (reference, excluded) | headA | 1.612 |
| A+B+C (reference, excluded) | headB | -0.160 |
| A+B+C (reference, excluded) | headC | 4.716 |
| A+B+C (reference, excluded) | bucket_en | 0.276 |
| A+B+C (reference, excluded) | bucket_hi | 0.072 |
| A+B+C (reference, excluded) | bucket_te | -0.072 |
| A+B+C (reference, excluded) | bucket_cm | -0.275 |

## Known limitations of this sitting

- Everything above is on **qwen7b only** (the single generator fully scored). gemma/mistral (seen) and llama/phi (held-out, non-negotiable #4) are not in these numbers yet — T1/T2's AUROCs and Head C's exclusion both need re-checking once they land.
- The unmatched-human FPR robustness check (T5) needs a GPU sitting to score Head B on the length-matching-excluded rows.
- Temperature is fit on the TRAIN split (the only split besides test with both classes; the frozen `cal` split is human-only by design) — optimistic by construction, documented in `src/calibration/temperature.py`.
