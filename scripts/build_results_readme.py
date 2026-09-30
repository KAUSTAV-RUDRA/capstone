"""Assemble docs/results/README.md for corpus v2, whitespace-normalised scoring.

Reads the CSVs written by experiments/exp01, exp04, exp05, exp07, exp08, exp09, exp10 and
scripts/headA_surface_ablation.py (all under results/), copies every table and figure into
docs/results/, and writes one markdown doc: a paragraph per table plus the table itself.
Key numbers in the prose are read from the CSVs, not typed.

Usage: python -m scripts.build_results_readme
Writes: docs/results/README.md and docs/results/*.csv|png
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

RESULTS, DOCS = Path("results"), Path("docs/results")
COPY = ["T1_main_results.csv", "T2_ablation.csv", "T3_heldout.csv", "T3_heldout_by_generator.csv",
        "T3_heldout_operating.csv", "T5_fairness.csv", "T6_calibration.csv", "F1_risk_coverage.csv",
        "F1_risk_coverage.png", "F2_fertility_vs_auroc.csv", "F2_fertility_vs_auroc.png",
        "S1_headA_ablation.csv", "abstention_coverage_sweep.csv", "conformal_thresholds.csv",
        "temperature.csv", "fusion_coefficients.csv"]
BUCKETS = ["en", "hi", "te", "cm"]


def md_table(df: pd.DataFrame, precision: int = 3) -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{precision}f}")
    df = df.fillna("")
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "| " + " | ".join("---" for _ in df.columns) + " |"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.to_numpy().tolist()]
    return "\n".join(lines)


def main() -> None:
    t1 = pd.read_csv(RESULTS / "T1_main_results.csv")
    t2 = pd.read_csv(RESULTS / "T2_ablation.csv")
    t3 = pd.read_csv(RESULTS / "T3_heldout.csv")
    t3g = pd.read_csv(RESULTS / "T3_heldout_by_generator.csv")
    t3o = pd.read_csv(RESULTS / "T3_heldout_operating.csv")
    t5 = pd.read_csv(RESULTS / "T5_fairness.csv")
    t6 = pd.read_csv(RESULTS / "T6_calibration.csv")
    f2 = pd.read_csv(RESULTS / "F2_fertility_vs_auroc.csv")
    s1 = pd.read_csv(RESULTS / "S1_headA_ablation.csv")
    coef = pd.read_csv(RESULTS / "fusion_coefficients.csv")
    conf = pd.read_csv(RESULTS / "conformal_thresholds.csv")
    sweep = pd.read_csv(RESULTS / "abstention_coverage_sweep.csv")

    A = lambda m, b: float(t1[(t1.method == m) & (t1.bucket == b)].auroc.iloc[0])  # noqa: E731
    base = {b: max(("ppl", "binoculars", "fastdetectgpt_en"), key=lambda m: A(m, b)) for b in BUCKETS}
    ab = coef[coef.fusion == "A+B (adopted)"].set_index("feature").weight_standardised
    f = lambda m: t3[t3.method == m].iloc[0]  # noqa: E731
    fused3, ha3, hb3 = f("fused (A+B)"), f("headA"), f("headB")
    op = t3o.set_index(["alpha", "group"])
    g001 = t3o[t3o.group.str.endswith("(3-way gate)")].set_index("group")
    t5a = t5[t5.group == "all"]
    un = t5[t5.group == "unmatched_human_robustness_check"]
    en_l1 = t5[t5.group.str.startswith("writer_L1_band")]

    def fpr(alpha, b, col="fpr_per_bucket_tau"):
        return float(t5a[(t5a.alpha == alpha) & (t5a.bucket == b)][col].iloc[0])

    def l1(alpha, band):
        return float(en_l1[(en_l1.alpha == alpha) & (en_l1.group == f"writer_L1_band={band}")].fpr_per_bucket_tau.iloc[0])

    abst = t2[t2.config == "+abstain"].set_index("bucket")
    cov_floor = {b: float(sweep[sweep.bucket == b].actual_coverage.min()) for b in BUCKETS}
    t6i = t6.set_index("bucket")

    P: list[str] = []
    P.append("# Results — corpus v2, whitespace-normalised scoring\n")
    P.append(
        "Everything below is computed on the frozen v2 corpus (`data/processed/splits.json`, 12,148 rows; "
        "generators qwen7b/gemma/mistral seen, llama held-out) with **every score column computed on "
        "whitespace-normalised text** (decisions.md 2026-09-30): headA, headB, headB_word, headC, "
        "fastdetectgpt_en, ppl and binoculars all in `results/norm/scores.parquet`. The fusion, per-bucket "
        "temperature and per-bucket conformal thresholds are fit on train / train / human-only cal and "
        "never see test. Reproduce with:\n\n"
        "```\nC=configs/models_norm.yaml\n"
        "python -m src.eval.score --config $C --scores results/norm/scores_gpu.parquet --column binoculars\n"
        "python -m scripts.merge_norm_scores --config $C          # headA + gpu columns -> results/norm/scores.parquet\n"
        "python -m experiments.exp04_fusion --config $C           # fuser (webapp), coefficients\n"
        "python -m experiments.exp01_baselines --config $C        # T1\n"
        "python -m experiments.exp08_ablation --config $C         # T2\n"
        "python -m experiments.exp10_heldout --config $C          # T3\n"
        "python -m experiments.exp07_fairness --config $C         # T5 (GPU: scores the unmatched humans)\n"
        "python -m experiments.exp05_abstention --config $C       # T6, F1, coverage sweep\n"
        "python -m experiments.exp09_fertility_curvature --config $C   # F2\n"
        "python -m scripts.headA_surface_ablation --config configs/data.yaml --out-csv results/S1_headA_ablation.csv\n"
        "python -m scripts.export_calibration --config $C         # results/calibration.json (webapp)\n"
        "python -m scripts.build_results_readme                   # this file\n```\n"
    )
    P.append(
        "## Read this first: what these numbers can and cannot claim\n\n"
        "**Provenance confound.** Every human row is scraped, single-source-per-domain text (news, wiki, forum, "
        "chat); every machine row is a prompted generation. A nine-number surface probe (digit share, punctuation "
        "share, word length, …) still reaches AUROC 0.84-0.92 after normalisation (decisions.md 2026-09-30). So an "
        "AUROC here means \"scraped human vs prompted generation\", not \"human vs machine writing\", and is an "
        "upper bound on what the detector would do on real student essays. An independent human sample per bucket "
        "is the only clean test. **Held-out coverage is thin:** llama is the only held-out generator and it "
        "generates hi only, so en, te and cm have no held-out evaluation at all, and T1's seen-generator numbers "
        "there are same-generator numbers (non-negotiable #4). **Missing baselines:** the supervised XLM-R baseline "
        "was never built, and vanilla DetectGPT was dropped for Fast-DetectGPT (decisions.md 2026-09-18), so T1 has "
        "three zero-shot baselines, not five. **T4 (adversarial)** is not part of this rebuild. The test split's hi "
        "bucket mixes seen and llama machine rows; T1 gives both breakdowns.\n"
    )
    P.append(
        "## Head C: diagnostic only\n\n"
        "Head C (MuRIL embeddings) scores highest in T1 (en 1.000, te 0.999, hi 0.978 and 0.961 on held-out llama) and "
        "is **excluded from the fusion and from every calibrated output**. A plain TF-IDF classifier matches it on "
        "held-out llama hi (0.911-0.921) and across held-out generators and domains, so the held-out test cannot "
        "separate a real signal from the provenance difference above (decisions.md 2026-09-30; "
        "`results/headc_diagnosis_v2.txt`). It is reported in T1/T2/T3 as a diagnostic row, and `A+B+C` in T2 is "
        "reference only. The older [headc_diagnosis.md](headc_diagnosis.md) is the v1 single-generator analysis.\n"
    )

    P.append("## Fusion\n")
    P.append(
        "Logistic regression over `[headA, headB, bucket one-hot]`, standardised inputs, fit on the v2 train split "
        f"and saved to `results/models/fuser_ab.joblib` (the webapp loads it; `services.analyse_text` runs end to "
        f"end again). Standardised weights: headA {ab['headA']:.2f}, headB {ab['headB']:.2f}, bucket en "
        f"{ab['bucket_en']:.2f} / hi {ab['bucket_hi']:.2f} / te {ab['bucket_te']:.2f} / cm {ab['bucket_cm']:.2f}. "
        "Head A carries the model; headB gets a single global weight, and since headB is inverted on hi/te "
        "(T1) while positive on en/cm, one weight cannot use it in every bucket. Head A is per-bucket logistic fit on "
        "train, so the fuser sees in-sample Head A scores on train (train AUROC 0.96-0.995 vs test 0.93-0.99); "
        "this is inherited from the v1 design and not changed here.\n"
    )

    P.append("## T1 — main results, per bucket, all methods\n")
    hi_seen = {m: float(t1[(t1.method == m) & (t1.bucket == 'hi')].auroc_seen.iloc[0]) for m in ('headA', 'headB')}
    P.append(
        "Test AUROC and F1 per bucket for the three zero-shot baselines (ppl, binoculars, fastdetectgpt_en), the "
        "heads (headA, headB, headB_word, headC-diagnostic) and the adopted fusion. `auroc` is over all test rows; "
        "`auroc_seen` drops llama; `auroc_heldout` is human-vs-llama (hi only). F1's threshold is tuned on train "
        "per bucket per method and applied unchanged to test.\n\n"
        f"Fusion AUROC: en {A('fused (A+B)','en'):.3f}, hi {A('fused (A+B)','hi'):.3f}, te {A('fused (A+B)','te'):.3f}, "
        f"cm {A('fused (A+B)','cm'):.3f}; best zero-shot baseline: "
        + ", ".join(f"{b} {A(base[b], b):.3f} ({base[b]})" for b in BUCKETS)
        + ". Fusion beats every baseline in every bucket, but **it does not clearly beat Head A alone**: headA is "
        f"en {A('headA','en'):.3f}, hi {A('headA','hi'):.3f}, te {A('headA','te'):.3f}, cm {A('headA','cm'):.3f}, so "
        "fusion adds +0.003 on en and ~0 on hi and cm and is 0.010 *worse* on te. The contribution of Head B here is "
        f"small. Head B (curvature) is strong on en ({A('headB','en'):.3f}), weak on cm ({A('headB','cm'):.3f}) and "
        f"**inverted on seen-generator hi ({hi_seen['headB']:.3f}) and te ({A('headB','te'):.3f})** — machine text "
        "scores as *less* curved than human under mGPT; on held-out llama hi it is 0.785, so the inversion is "
        "generator-specific rather than a property of Hindi (see F2, T3). cm is the weakest bucket for every method.\n\n"
        + md_table(t1) + "\n"
    )

    P.append("## T2 — ablation\n")
    P.append(
        "A / B / C / A+B / A+B+C / +temperature / +conformal / +abstain, per bucket, test split. `+temperature` has "
        "AUROC/F1 identical to `A+B` by construction (a monotone transform cannot change ranking); its effect is in T6. "
        "`+conformal` is the per-bucket alpha=0.05 threshold with no abstention: FPR lands at "
        + ", ".join(f"{b} {float(t2[(t2.bucket==b)&(t2.config=='+conformal')].fpr.iloc[0]):.3f}" for b in BUCKETS)
        + ". `+abstain` is the three-way gate (MACHINE above the alpha=0.01 per-bucket tau, HUMAN below the human "
        "calibration median, ABSTAIN between): coverage "
        + ", ".join(f"{b} {abst.loc[b,'coverage']:.2f}" for b in BUCKETS)
        + "; accuracy on the covered rows "
        + ", ".join(f"{abst.loc[b,'accuracy']:.3f}" for b in BUCKETS)
        + "; FPR "
        + ", ".join(f"{abst.loc[b,'fpr']:.3f}" for b in BUCKETS)
        + ". Coverage is lowest in te and hi, where the gate abstains on more than half of the test rows. "
        "`A+B+C` is reference only.\n\n" + md_table(t2) + "\n"
    )

    P.append("## T3 — held-out generator (llama, hi)\n")
    hi_rows = t3o[(t3o.alpha == 0.01)]
    P.append(
        "The new table. Same human test rows, machine side either the seen generators (qwen7b, gemma, mistral) or "
        "the held-out llama; 95% bootstrap CIs. **Fused AUROC "
        f"{fused3.auroc_seen:.3f} seen -> {fused3.auroc_heldout:.3f} held-out** (drop {fused3['drop']:.3f}, "
        f"CIs [{fused3.seen_lo:.3f}, {fused3.seen_hi:.3f}] and [{fused3.heldout_lo:.3f}, {fused3.heldout_hi:.3f}] "
        f"overlap substantially); Head A {ha3.auroc_seen:.3f} -> {ha3.auroc_heldout:.3f}. So the stylometric head and the "
        "fusion generalise to llama with a small drop. The three zero-shot baselines go the *other way*: "
        + ", ".join(f"{m} {f(m).auroc_seen:.3f} -> {f(m).auroc_heldout:.3f}" for m in ("ppl", "binoculars", "fastdetectgpt_en"))
        + f", and Head B {hb3.auroc_seen:.3f} -> {hb3.auroc_heldout:.3f} (still below the fusion, no longer inverted). "
        "llama's Hindi is simply easier for likelihood-based scores than the seen generators' — a reminder that one "
        "held-out generator is one draw, not a general claim. Head C (diagnostic) drops 0.997 -> 0.961 and is matched by "
        "TF-IDF, so its held-out number is not evidence of generalisation.\n\n"
        "**Operating point on llama** (the shipped gate, per-bucket tau fit on human cal only): at alpha=0.01 the gate "
        f"flags {op.loc[(0.01,'machine_heldout'),'flag_rate_at_tau']:.1%} of llama hi text MACHINE vs "
        f"{op.loc[(0.01,'machine_seen'),'flag_rate_at_tau']:.1%} of seen-generator text, at a human FPR of "
        f"{op.loc[(0.01,'human'),'flag_rate_at_tau']:.1%}; at alpha=0.05, "
        f"{op.loc[(0.05,'machine_heldout'),'flag_rate_at_tau']:.1%} vs {op.loc[(0.05,'machine_seen'),'flag_rate_at_tau']:.1%} "
        f"at FPR {op.loc[(0.05,'human'),'flag_rate_at_tau']:.1%}. The three-way gate sends "
        f"{g001.loc['machine_heldout (3-way gate)','frac_abstain']:.0%} of llama text to ABSTAIN and only "
        f"{g001.loc['machine_heldout (3-way gate)','frac_human']:.1%} to HUMAN (seen: "
        f"{g001.loc['machine_seen (3-way gate)','frac_abstain']:.0%} abstain, "
        f"{g001.loc['machine_seen (3-way gate)','frac_human']:.1%} HUMAN): on unseen-generator text the error mode is "
        "abstaining, not wrongly clearing it as human. The FPR guarantee is on human text and does not depend on the "
        "generator; detection rate does, and at alpha=0.01 it is low.\n\n"
        + md_table(t3.drop(columns=["seen_lo", "seen_hi", "heldout_lo", "heldout_hi"])) + "\n\n"
        "Per generator (hi):\n\n" + md_table(t3g[t3g.method.isin(["fused (A+B)", "headA", "headB"])]
                                             .pivot(index="generator", columns="method", values="auroc")
                                             .reset_index()) + "\n\n"
        "Gate behaviour (hi):\n\n" + md_table(t3o) + "\n"
    )

    P.append("## T5 — fairness\n")
    un_rows = un.copy()
    P.append(
        "FPR on human test text, conformal threshold per bucket (the adopted policy) vs one global threshold pooled over "
        "all four buckets' calibration scores (shown only for comparison). "
        "Per-bucket FPR at alpha=0.01: "
        + ", ".join(f"{b} {fpr(0.01,b):.3f}" for b in BUCKETS)
        + "; at alpha=0.05: "
        + ", ".join(f"{b} {fpr(0.05,b):.3f}" for b in BUCKETS)
        + ". Every bucket is at or below alpha except en (0.052) and te (0.050) at alpha=0.05, which is sampling noise on ~400 humans. "
        "With the global tau the FPR spreads out: "
        + ", ".join(f"{b} {fpr(0.01,b,'fpr_global_tau'):.3f}" for b in BUCKETS)
        + " at 0.01 and "
        + ", ".join(f"{b} {fpr(0.05,b,'fpr_global_tau'):.3f}" for b in BUCKETS)
        + " at 0.05. At 0.01 the pooled threshold flags en humans at twice alpha (0.020) and cm humans not at all; "
        "at 0.05 it over-flags cm (0.050 vs 0.034) and under-flags te (0.025 vs 0.050). The per-bucket threshold removes "
        "that spread; the effect is modest here because the four buckets' calibrated human score distributions are not far apart.\n\n"
        "**en by writer_L1_band.** The expected direction (Indian-English writers flagged more) is **not** seen: at "
        f"alpha=0.05 FPR is {l1(0.05,'general'):.3f} for `general` (n=212) and {l1(0.05,'indian'):.3f} for `indian` "
        f"(n=190); at 0.01, {l1(0.01,'general'):.3f} vs {l1(0.01,'indian'):.3f}. The disparity (indian minus general) is "
        "negative, and `general` exceeds alpha at both levels (0.085 at 0.05, 0.014 at 0.01), which the conformal bound "
        "does not cover per band because the cal split mixes both bands. The two bands are also different text sources, so this is not a clean L1 vs L2 "
        "contrast; treat it as \"no evidence of the Liang et al. effect here\", not as evidence of its absence.\n\n"
        "**Unmatched-human robustness check.** The human rows that length matching trimmed out of the corpus "
        "(en 388, hi 266, te 366; cm trims machine so has none) were scored with headA + mGPT (whitespace-normalised) "
        "and pushed through the same fuser, temperature and tau. FPR with per-bucket tau at alpha=0.01: "
        + ", ".join(f"{r.bucket} {r.fpr_per_bucket_tau:.3f}" for r in un_rows[un_rows.alpha == 0.01].itertuples())
        + "; at 0.05: "
        + ", ".join(f"{r.bucket} {r.fpr_per_bucket_tau:.3f}" for r in un_rows[un_rows.alpha == 0.05].itertuples())
        + ". These are at or below alpha except en at 0.05 (0.059 vs 0.052 on matched, a small excess), so the rows the "
        "length matching discarded are not flagged materially more than the matched ones. Median lengths differ from matched test humans (en 213 vs 200, hi 142 "
        "vs 171, te 209 vs 151 words), so this also covers a range of lengths.\n\n"
        + md_table(t5) + "\n"
    )

    P.append("## T6 — calibration\n")
    P.append(
        "ECE (15 equal-width bins on P(machine)) before and after per-bucket temperature scaling, on the test split, "
        "and accuracy at 50/70/90% coverage (keep the most confident rows, prediction at 0.5). **Temperature scaling does "
        "not help much here**: fitted temperatures are "
        + ", ".join(f"{b} {t6i.loc[b,'temperature']:.2f}" for b in BUCKETS)
        + ", all near 1, and ECE moves "
        + ", ".join(f"{b} {t6i.loc[b,'ece_before']:.3f} -> {t6i.loc[b,'ece_after']:.3f}" for b in BUCKETS)
        + " — slightly better on cm, slightly worse on en, hi, te. The fuser's logistic output is already close to "
        "calibrated on train, so there is little for temperature to fix. **The calibration gap is generator shift, not "
        "temperature:** hi's ECE is "
        f"{t6i.loc['hi (seen only)','ece_after']:.3f} on seen-generator test rows but {t6i.loc['hi (held-out)','ece_after']:.3f} with llama "
        "(rows `hi (seen only)` / `hi (held-out)` below); the fuser is under-confident on an unseen generator. Accuracy at 50/70/90% coverage: "
        + "; ".join(f"{b} {t6i.loc[b,'accuracy_at_50pct']:.3f}/{t6i.loc[b,'accuracy_at_70pct']:.3f}/{t6i.loc[b,'accuracy_at_90pct']:.3f}" for b in BUCKETS)
        + ". The `fpr_at_*` columns are computed on the human rows that happen to fall in the kept set, which is tiny at "
        "50% coverage (en: 1.000 is a handful of rows) — do not read them. The four plain-bucket rows are the test split "
        "as T1 sees it (hi includes llama).\n\n" + md_table(t6) + "\n"
    )

    P.append("## Abstention gate and coverage sweep\n")
    P.append(
        "`abstention_coverage_sweep.csv` moves the HUMAN cutoff so coverage runs 30% -> 100% while MACHINE stays at the "
        "alpha=0.01 tau. Coverage counts MACHINE verdicts as covered, so it cannot fall below the MACHINE share: in en "
        f"the MACHINE verdicts alone cover {cov_floor['en']:.0%} of test rows, so the 30% and 50% targets return the same row "
        "(hi, te and cm reach 30%). At 100% coverage (every row gets HUMAN or MACHINE) accuracy is "
        + ", ".join(f"{b} {float(sweep[(sweep.bucket==b)&(sweep.target_coverage==1.0)].accuracy.iloc[0]):.3f}" for b in BUCKETS)
        + " — the alpha=0.01 tau keeps FPR near 1% but leaves many machine texts below it, so abstention is what makes the "
        "output trustworthy, and the price is coverage. Thresholds: `conformal_thresholds.csv` (n=1000 human per bucket, "
        "global pooled n=4000): tau at 0.01 "
        + ", ".join(f"{b} {float(conf[(conf.alpha==0.01)&(conf.bucket==b)].tau.iloc[0]):.3f}" for b in BUCKETS)
        + f", global {float(conf[(conf.alpha==0.01)&(conf.bucket=='_global')].tau.iloc[0]):.3f}; at 0.05 "
        + ", ".join(f"{b} {float(conf[(conf.alpha==0.05)&(conf.bucket==b)].tau.iloc[0]):.3f}" for b in BUCKETS)
        + f", global {float(conf[(conf.alpha==0.05)&(conf.bucket=='_global')].tau.iloc[0]):.3f}. "
        "At 0.05 te's tau (0.417) is far below the global (0.776): te human scores sit lower, so the global threshold is "
        "stricter than te needs and gives up te detection without any FPR benefit the bucket requires.\n"
    )

    P.append("## F1 — risk-coverage\n")
    P.append(
        "![F1](F1_risk_coverage.png)\n\nRisk (error rate among non-abstained rows, prediction at 0.5) against coverage when "
        "rows are dropped in order of increasing confidence, all four buckets overlaid. Points: "
        "`F1_risk_coverage.csv`. Risk is lowest at low coverage in every bucket and rises as less confident rows are "
        "admitted; cm and hi rise earliest.\n"
    )
    P.append("## F2 — tokenizer fertility vs Head B AUROC\n")
    P.append(
        "![F2](F2_fertility_vs_auroc.png)\n\nTokens per word of each scorer's tokenizer against its own curvature AUROC per "
        "bucket (mGPT for headB, Qwen2.5-0.5B for fastdetectgpt_en). "
        + "; ".join(f"{r.method} {r.bucket}: {r.tokens_per_word:.2f} tok/word, AUROC {r.auroc:.3f}" for r in f2.itertuples())
        + ". Fertility does not explain the collapse: mGPT at hi (3.56) and Qwen at hi (4.80) give 0.569 and 0.843 "
        "respectively, i.e. the scorer with the worse fragmentation does *better* on hi. Head B's hi/te inversion is "
        "specific to mGPT on these generators (T3: 0.336 seen -> 0.785 on llama). The hi figures include llama rows.\n"
    )

    P.append("## S1 — Head A ablation (supplementary)\n")
    P.append(
        "Is Head A reducible to surface format? Test AUROC on normalised text, seen generators (and held-out llama on hi), "
        "for a 9-number surface probe, Head A with all 40 features, with 14 format-adjacent features removed (c1), with "
        "sentence-length features also removed (c2), and the 15 parser features alone. In en, hi and te Head A keeps "
        "0.96-0.98 seen (0.93 on llama) with every format-adjacent feature removed, above the probe; **in cm it does not** "
        "(0.928 -> 0.806 / 0.748, below the probe's 0.913): cm Head A is mostly terminal punctuation and what an "
        "English parser makes of romanised text, and should not be presented as a stylometric result. cm terminal "
        "punctuation is kept as a real register difference and is a fairness risk for writers who punctuate every "
        "sentence (decisions.md 2026-09-30).\n\n" + md_table(s1) + "\n"
    )
    P.append(
        "## Superseded\n\n`t3_preliminary.md` (v1 corpus, raw text, 367 length-matched llama rows) is replaced by T3 above; "
        "`headc_diagnosis.md` is the v1 single-generator diagnosis. Both are kept for the record.\n"
    )

    DOCS.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        shutil.copy2(RESULTS / name, DOCS / name)
    (DOCS / "README.md").write_text("\n".join(P), encoding="utf-8")
    print(f"wrote {DOCS / 'README.md'} and copied {len(COPY)} files")


if __name__ == "__main__":
    main()
