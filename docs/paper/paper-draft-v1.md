# Stylometry, Curvature and Per-Language Conformal Abstention for Detecting Machine-Generated Text in Multilingual Student Settings

*Approved project title: "A Stylometric and Perplexity-Curvature Framework for Detecting Machine-Generated Text in Multilingual Student Submissions with Calibrated Abstention". The paper title above is shorter and matches what the results show; use either.*

**Authors:** Kaustav Harinarayana Rudra, «teammate 2», «teammate 3»
**Affiliation:** Department of Artificial Intelligence and Data Science (Honours & Experiential Track), KL University, Vijayawada, India
**Guide:** Mr. Bhavanam Venkata Suresh Reddy
**Target venue:** «to decide with guide: ICON 2026 / IEEE or Springer conference / journal»

> **v1 status (22 Oct 2026).** Every result blank from v0 is filled from `docs/results/` at commit `4b7467e` (T1–T7, F1, F2, S1, exp11–exp14). Only three blanks remain, and only you can fill them: the two teammate names and the venue. Section numbers and table numbers match v0 so reviewers' notes still line up. Claims that v0 made and the results did not support have been rewritten, not softened; the change log at the end lists each one.

---

## Abstract

Detectors of machine-generated text (MGT) are increasingly used in academic-integrity workflows, where a false positive is an accusation of misconduct. Most detectors are calibrated on English, and none offers a false-positive bound that holds separately for each language a student population writes in. We study detection in English (en), Hindi (hi), Telugu (te) and Romanised Hindi-English code-mixed text (cm). We build IndicStudentMGT, 12,148 length-matched passages (7,689 human, 4,459 machine) from three seen open-weight generators and one held-out generator, and evaluate a 40-feature stylometric head, conditional probability curvature under the multilingual scorer mGPT-1.3B, and their logistic fusion, wrapped in **per-language split-conformal calibration with a three-way HUMAN / ABSTAIN / MACHINE output**. Four findings stand out. (1) The stylometric head carries detection: fused test AUROC is 0.989 / 0.961 / 0.963 / 0.928 (en / hi / te / cm), against 0.901 / 0.843 / 0.832 / 0.748 for the best zero-shot baseline, but fusion adds at most 0.003 over the stylometric head alone, with confidence intervals that include zero. (2) Curvature under mGPT is *inverted* on seen-generator Hindi (AUROC 0.336) and Telugu (0.235); the same statistic under Qwen2.5-0.5B is not (0.752, 0.589), mGPT itself is not inverted on held-out Llama Hindi (0.785), and tokenizer fertility does not explain the pattern. (3) Per-language calibration holds the empirical human false-positive rate at 0.008–0.010 at α = 0.01 and 0.034–0.052 at α = 0.05 in every bucket, with no evidence of violation (exact binomial p ≥ 0.45), whereas one pooled threshold spreads it from 0.000 to 0.020 at α = 0.01. The guarantee is expensive: at α = 0.01 only 16% of Telugu and 39% of Hindi machine text is flagged, rising to 90% and 82% at α = 0.05. (4) Against a frozen detector, light hybrid edits barely matter outside cm; back-translation is the effective attack, cutting recall at α = 0.05 to 54–68% and raising machine text wrongly cleared as human from ≤1.3% to 4–7%; LLM paraphrasing of *human* text raises false flags to 30–49%. All AUROCs are bounded by a provenance confound between scraped human text and prompted generations, which we measure and report. Code, frozen splits and a corpus card accompany the paper.

**Keywords:** machine-generated text detection, conformal prediction, selective prediction, stylometry, multilingual NLP, code-mixing, academic integrity, fairness

---

## 1 Introduction

Large language models now produce fluent text in dozens of languages, and universities face submissions that may be partly or wholly machine-written. Detectors exist, but deploying one in a multilingual student population raises a problem that is not only about accuracy: **the cost of a false positive is an accusation of misconduct**, and that cost can fall unevenly across the languages students write in.

The most widely used detectors rely, directly or indirectly, on how predictable a passage is under a reference language model. Machine text is predictable because the generator picks high-probability continuations, but predictability has other causes. Liang et al. (2023) found that seven detectors flagged, on average, 61.3% of TOEFL essays by non-native English writers while being near-perfect on native essays. In India the problem compounds: students write Indian English, write in Hindi, Telugu and other Indic scripts, and code-mix in Roman script. Multilingual benchmarks (Wang et al., 2024; Macko et al., 2023, 2025) show sharp drops outside English but include no Indic student-register or Romanised code-mixed text, and no detector we know of bounds its false-positive rate separately per language.

This paper asks three questions. **RQ1:** does conditional probability curvature (Bao et al., 2024), which measures whether a passage sits at a local maximum of a scorer's likelihood rather than how likely it is, transfer to Hindi, Telugu and code-mixed text when computed under a multilingual scorer? **RQ2:** does fusing curvature with a language-agnostic stylometric head beat either head alone? **RQ3:** can per-language conformal calibration with abstention hold a stated false-positive rate in every language, and what does that cost in detection?

Our contributions, stated as the results support them:

1. **A negative result for curvature on Indic text, with a diagnosis.** Under mGPT-1.3B, curvature is inverted on seen-generator Hindi and Telugu. The inversion is specific to the scorer–generator pair: a different scorer is not inverted on the same text, and mGPT is not inverted on a held-out generator's Hindi. Tokenizer fertility, the obvious explanation, does not account for it (§6.7).
2. **A null result for fusion.** Stylometry alone reaches 0.928–0.986 AUROC. Adding curvature changes AUROC by −0.009 to +0.003 with a global fusion weight and by at most +0.003 with per-language weights, with every 95% interval spanning zero except a +0.003 gain on held-out Llama Hindi (§6.2).
3. **Per-language conformal abstention that holds its bound, and a measurement of its price.** Empirical FPR is at or below α in every bucket at α = 0.01 and consistent with α at α = 0.05; a pooled threshold does not equalise rates. The trade-off between the bound and recall is steep and language-dependent, and we report both operating points (§6.4–6.5).
4. **A robustness and error profile of the frozen system.** We apply paraphrase, back-translation and hybrid edits to a detector that is never refit, and analyse every false flag and false clear on the clean test set by length, source and generator (§6.6, §6.8).

We also release **IndicStudentMGT** with a corpus card, frozen splits and hashes, and a reference implementation in which the research pipeline has no dependency on the reviewer-facing web tool.

---

## 2 Related Work

**Supervised detectors.** Fine-tuned encoders such as RoBERTa-based classifiers (Solaiman et al., 2019; Guo et al., 2023) achieve high in-distribution accuracy but degrade on unseen generators and domains (Li et al., 2024), and need labelled machine text from the generators one expects to meet.

**Zero-shot statistical detectors.** GLTR (Gehrmann et al., 2019) visualised token-rank statistics. DetectGPT (Mitchell et al., 2023) introduced the probability-curvature hypothesis: machine text lies near a local maximum of the model's log-probability. Fast-DetectGPT (Bao et al., 2024) replaced the perturbation loop with an analytic conditional curvature from the scorer's own next-token distributions. Binoculars (Hans et al., 2024) scores the ratio of perplexity to cross-perplexity between two models. These were validated mainly on English and with English-centric scorers.

**Robustness.** RAID (Dugan et al., 2024) evaluated detectors under eleven attacks and found broad degradation under paraphrase. RADAR (Hu et al., 2023) trained detectors adversarially against a paraphraser. DetectRL (Wu et al., 2024) stress-tested detectors in realistic writing scenarios and showed that generation artefacts give detectors shortcuts; we strip such artefacts explicitly (§4.2) and additionally find and remove a newline artefact (§4.3).

**Multilingual benchmarks.** M4GT-Bench (Wang et al., 2024) spans nine languages, MULTITuDE (Macko et al., 2023) eleven, MultiSocial (Macko et al., 2025) twenty-two. Performance falls sharply outside English. None includes Telugu or Romanised code-mixed student-register text, and none stratifies calibration by language.

**Fairness and conformal calibration.** Liang et al. (2023) established the non-native false-positive problem. Zhu et al. (2025) bound the FPR of zero-shot MGT detectors with *multiscaled* conformal prediction, computing one conformal quantile per text-length bin on English and Chinese data with a binary output. Group-conditional (Mondrian) conformal prediction is classical (Vovk et al., 2005). What has not been done, to our knowledge, is to condition the bound on **language and script, including Romanised code-mixed text**, to pair it with an abstention band, and to measure whether the per-language bound holds and what it costs on Indic text. Selective prediction and risk–coverage analysis (Geifman & El-Yaniv, 2017) frame the abstention.

---

## 3 Method

### 3.1 Problem statement
Given a passage *x* and its language bucket *b* ∈ {en, hi, te, cm}, output one of {HUMAN, ABSTAIN, MACHINE} with a calibrated score and an explanation, subject to: for human-written *x* in bucket *b*, P(output = MACHINE) ≤ α.

### 3.2 Preprocessing and routing
Whitespace runs, including newlines, are collapsed to a single space before any feature extraction or tokenisation, identically for human and machine text (§4.3 explains why). Script is detected by Unicode block. Code-mixing is estimated as the share of tokens matching a curated list of Hindi function words, restricted to forms that do not collide with English. Routing: Devanagari → hi; Telugu script → te; Latin script with a code-mix ratio ≥ 0.3 → cm; otherwise en. The same 0.3 threshold gates which source rows enter the cm bucket, so routing and corpus construction agree. Routing accuracy on out-of-corpus text was not measured separately; misrouting is a route by which the per-bucket guarantee can fail (§8).

### 3.3 Head A: stylometric features
40 features in two groups. **Surface (25),** computed without any model: lexical diversity (TTR, root TTR, MTLD, hapax ratio, bigram repetition), word- and sentence-length statistics and their burstiness, function-word ratio and function/content-word transition proxies, clauses per sentence, eight punctuation statistics, and digit and uppercase shares. **Syntax (15),** from a per-bucket parser (Stanza for hi and te, spaCy `en_core_web_sm` for en and cm): ten Universal POS ratios, three POS n-gram statistics, mean dependency-tree depth and mean dependency distance. No parser exists for Romanised Hinglish, so cm's syntax features come from an English parser and are weaker evidence. A logistic regression per bucket maps the standardised features to a score.

### 3.4 Head B: conditional probability curvature
For a scorer *p_θ* and tokens *x = (x_1,…,x_n)*, Fast-DetectGPT defines

  d(x) = [ log p_θ(x) − E_{x̃∼q}[log p_θ(x̃)] ] / sqrt( Var_{x̃∼q}[log p_θ(x̃)] ),

with *q* the scorer's own conditional distribution at each position, so mean and variance are analytic from one forward pass. Large *d(x)* indicates a local maximum of *p_θ*. We use mGPT-1.3B (Shliazhko et al., 2022), a 61-language causal model, in fp16. Its fertility is 1.36 tokens/word on en, 3.56 on hi, 6.32 on te and 1.71 on cm, against 1.34, 4.80, 11.92 and 1.67 for the English-default Qwen2.5-0.5B. We also compute a **word-level** variant (`headB_word`) that sums per-token log-probabilities, means and variances within each source word before computing *d*, so the statistic is invariant to how finely a word is split. The Qwen2.5-0.5B version of the same statistic is reported as the baseline `fastdetectgpt_en`.

### 3.5 Head C: semantic embeddings (diagnostic only)
MuRIL-base (Khanuja et al., 2021) mean-pooled embeddings with a per-bucket logistic layer. Head C scores highest of all methods, but a plain TF-IDF classifier matches it on held-out data and across generators and domains, so on this corpus its score cannot be separated from the provenance difference between human and machine rows (§4.3). **Head C is excluded from the fusion and from every calibrated output**, and is reported as a diagnostic row only.

### 3.6 Fusion
Logistic regression over standardised [Head A, Head B, bucket one-hot], fit on the training split. Standardised weights: Head A 3.63, Head B 0.33; bucket offsets en 0.43, hi 0.04, te 0.03, cm −0.58. Because Head B is positively oriented on en and cm but inverted on hi and te (§6.1), one global Head B weight cannot use it in every bucket; §6.2 tests per-bucket weights.

### 3.7 Per-language conformal calibration
For each bucket we hold out a calibration set *C_b* of *n_b* = 1,000 human passages used for nothing else. After per-bucket temperature scaling, we compute fused scores {s_i} on *C_b* and set

  τ_b(α) = the ⌈(n_b + 1)(1 − α)⌉-th smallest value of {s_i}.

**Proposition.** If a new human passage from bucket *b* is exchangeable with *C_b*, then P(s_new > τ_b(α)) ≤ α. *Proof.* Standard split-conformal argument (Vovk et al., 2005; Lei et al., 2018). ∎

The bound holds for any scorer and fusion model, but only for human text exchangeable with *C_b*. For cm, calibration draws only from the two sources with pre-ChatGPT provenance (CMU Hinglish DoG, 894; HinGE, 106), which are chat-register; cm's train and test humans are 98% social-media comments. **cm's test FPR is therefore reported as measured, not as bounded** (§4.4). Fitted thresholds:

| α | en | hi | te | cm | pooled (n = 4,000) |
|---|---|---|---|---|---|
| 0.01 | 0.995 | 0.974 | 0.983 | 0.871 | 0.976 |
| 0.05 | 0.707 | 0.737 | 0.417 | 0.819 | 0.776 |

**Why stratify.** A pooled threshold bounds only the pooled FPR. If human score distributions differ by bucket, a bucket's FPR can exceed α while the pooled rate stays below it. At α = 0.05 the te threshold (0.417) sits far below the pooled one (0.776): te human scores are lower, so a pooled threshold is stricter than te needs and gives up te detection without buying any FPR the bucket requires.

### 3.8 Abstention
MACHINE if *s* > τ_b(α); HUMAN if *s* < λ_b; ABSTAIN otherwise. The shipped λ_b is the median fused score of the bucket's human calibration set (en 0.029, hi 0.023, te 0.005, cm 0.133). λ_b never affects the FPR, which τ_b alone sets; lowering or raising it only trades ABSTAIN against HUMAN, and for machine text a HUMAN verdict is a **false clear**. We sweep λ_b to trace risk–coverage curves (F1) and report the shipped operating point.

### 3.9 Explanation
Each decision reports the bucket and its τ_b, both head scores, the head with the larger fusion contribution, and the five stylometric features with the largest standardised deviation. Every interface string frames the output as decision support for a human reviewer, never as an accusation.

---

## 4 The IndicStudentMGT Benchmark

### 4.1 Human text
All human text predates or is independent of LLM availability except where noted. **en:** HC3 human answers (Guo et al., 2023) and Wikipedia (low page id as a pre-2020 proxy) for *general* English; the English side of Samanantar (Ramesh et al., 2022) for *Indian* English, giving a writer-band split used in §6.4. **hi, te:** IndicCorpV2 news (Doddapaneni et al., 2023), 75%, blended with Indic Wikipedia, 25%. **cm:** COMI-LINGUA's Roman-script comments (Sheth et al., 2025), CMU Hinglish DoG chat (Zhou et al., 2018) and HinGE (Srivastava & Singh, 2021). COMI-LINGUA has no stated collection date and is therefore excluded from calibration.

### 4.2 Machine text
For each human passage, one seen generator writes a prompt-matched passage from the passage's first sentence, in its language and to its length bin (for cm: casual Roman-script Hinglish). Generators run through Ollama at 4-bit (Q4_K_M), temperature 0.8, top-p 0.95, with a token budget scaled by each bucket's measured tokenizer fertility. **Seen:** Qwen2.5-7B-Instruct (en, hi, te, cm), Gemma-2-9B-it (en, hi, te, cm), Mistral-7B-Instruct (en, hi). **Held-out, test only:** Llama-3.1-8B-Instruct (hi). Coverage is uneven, and this is itself a finding: Mistral did not produce usable te or cm text; Phi-3.5-mini failed instruction compliance on cm and length control on te and was dropped; Llama produced near-fixed-length Telugu regardless of prompt and failed cm's short-passage variance check. **Only hi has a held-out generator.**

Cleaning, applied identically to both classes, strips preambles ("Certainly! Here is…"), trailing assistant chatter and markdown, and drops rows with emoji, instruction echo, truncation or the wrong script: 455 rows in all. Two Indic cells lose many rows to stray script (Mistral hi 31.6%, Gemma te 39.5%), so the surviving machine text in those cells is biased toward passages the generator wrote cleanly.

### 4.3 Two artefacts found and handled
**Newlines.** In the frozen corpus every human row has zero newlines (the sources are flattened passages) while generator rows average 3.7–9.8. Newline count alone separates the classes at test AUROC 0.922 (en), 0.947 (hi) and 0.938 (te). Real essays have paragraph breaks, so this is an artefact of corpus construction. We collapse whitespace before scoring for every method (§3.2); Head A's AUROC is essentially unchanged by this (en 0.989 → 0.986, te 0.977 → 0.973), so it was not relying on newlines.

**Provenance.** After normalisation, a nine-number surface probe (digit share, punctuation share, word length and similar) still reaches 0.918 (en), 0.836 (hi), 0.871 (te) and 0.913 (cm). Every human row is scraped single-source text and every machine row is a prompted generation, so register differences between those two pipelines are learnable and cannot be normalised away. **Every AUROC in this paper therefore measures "scraped human versus prompted generation"**, and is an upper bound on what the detector would achieve on real student essays (§8).

### 4.4 Length matching and splits
Train/test passages are length-matched in five quantile bins: in en, hi and te the surplus human side is trimmed to the machine distribution; in cm the machine side is trimmed to the human one. Per bucket, 1,000 human passages go to calibration and are used for nothing else; remaining rows are split 50/50 train/test by prompt group, so a human passage and its machine counterpart never straddle splits, and each seen generator is halved within one row. Every prompt with a held-out row is forced to test. Splits are frozen and hashed in the corpus card.

**Table S-splits. Final counts.**

| bucket | class | cal | train | test | total |
|---|---|---|---|---|---|
| en | human | 1,000 | 403 | 402 | 1,805 |
| en | machine (qwen / gemma / mistral) | – | 1,090 | 1,091 | 2,181 |
| hi | human | 1,000 | 452 | 449 | 1,901 |
| hi | machine, seen (qwen / gemma / mistral) | – | 442 | 444 | 886 |
| hi | machine, held-out (llama) | – | 0 | 479 | 479 |
| te | human | 1,000 | 394 | 397 | 1,791 |
| te | machine (qwen / gemma) | – | 224 | 225 | 449 |
| cm | human | 1,000 | 595 | 597 | 2,192 |
| cm | machine (qwen / gemma) | – | 232 | 232 | 464 |
| **all** | | **4,000** | | | **12,148** |

Median test length (words, human / machine): en 200 / 206, hi 171 / 169, te 151 / 151, cm 20 / 20. cm calibration humans are much longer (median 85 words, chat turns) than cm test humans (20 words, comments), which is the register mismatch noted in §3.7. te and cm have fewer than 500 machine rows.

### 4.5 Adversarial variants
Attacks are applied to a capped, stratified sample of the test split; the same source rows are used for every attack so attacks are compared on identical text. **Hybrid** (400 machine + 150 human per bucket where available): 20% of sentences receive one programmatic edit, chosen from synonym swap (WordNet, POS-gated and first-sense-only for en; hand tables for hi and te), clause reorder, or sentence split/merge. **Paraphrase** (250 machine + 100 human per bucket; te 225, cm 232 machine): Qwen2.5-7B-Instruct, "rewrite in your own words, same language, same length", with an explicit instruction to keep cm in Roman script. **Back-translation** (same sample, en/hi/te only): NLLB-200-distilled-600M (NLLB Team, 2022), en→hi→en, hi→en→hi and te→en→te, sentence by sentence with a per-sentence length budget; outputs failing a script check are retried once and otherwise kept unchanged and flagged (2 rows), and outputs outside a 0.5–1.6 length ratio are flagged (4 rows). Back-translation is not applied to cm, which has no Roman-script Hinglish MT path. IndicTrans2 was planned but its weights are gated and its toolkit did not build on the project hardware.

Paraphrase shortens text (mean length ratio en 0.85, hi 0.80, te 0.67, cm 0.97); back-translation preserves length (1.00–1.05). Qwen2.5-7B is both a generator and the paraphraser, so some paraphrased rows are Qwen rewriting its own text; T4 splits these out.

---

## 5 Experimental Setup

**Baselines.** Perplexity (negated mean log-perplexity under Qwen2.5-0.5B, `ppl`); Binoculars (observer Qwen2.5-0.5B, performer Qwen2.5-0.5B-Instruct); Fast-DetectGPT with Qwen2.5-0.5B as scorer (`fastdetectgpt_en`). Vanilla DetectGPT was dropped for compute (it needs ~100 mask-fill perturbations per passage) since Fast-DetectGPT approximates it analytically; the planned supervised XLM-R baseline was not built.

**Metrics.** AUROC with stratified bootstrap 95% intervals (1,000 resamples); F1 at a per-bucket threshold tuned on train; empirical FPR and recall at τ_b(α) with Wilson 95% intervals and an exact one-sided binomial test of FPR > α; expected calibration error (15 equal-width bins); coverage, risk and false-clear rate for the three-way gate. Fusion, temperature and thresholds are fit on train / train / human-only calibration and never see test. Attacked rows are scored by the frozen system and nothing is refit.

**Hardware and runtime.** Everything ran on one RTX 4060 Laptop GPU (8 GB). Generation used Ollama with four parallel workers at roughly 100 tokens/s, in resumable sittings of at most two hours. Paraphrasing ran at about 0.12 rows/s; back-translation of 1,025 rows took 41 minutes; scoring the 4,239 attacked rows took 28 minutes (Head A, CPU) and 6 minutes (Head B, GPU). Every result table is produced by one script per table from a single frozen score file.

---

## 6 Results

### 6.1 Main results (Table T1)

**Table T1. Test AUROC per bucket.** hi includes held-out Llama rows; the hi seen-only and held-out columns are in T3.

| method | en | hi | te | cm |
|---|---|---|---|---|
| ppl (Qwen2.5-0.5B) | 0.901 | 0.838 | 0.832 | 0.748 |
| Binoculars | 0.840 | 0.839 | 0.630 | 0.668 |
| fastdetectgpt_en (Qwen2.5-0.5B) | 0.822 | 0.843 | 0.589 | 0.714 |
| Head B, curvature (mGPT) | 0.900 | 0.569 | **0.235** | 0.647 |
| Head B, word-level (mGPT) | 0.886 | 0.573 | 0.229 | 0.625 |
| **Head A, stylometric** | 0.986 | 0.960 | **0.973** | 0.928 |
| **Fused (A + B)** | **0.989** | **0.961** | 0.963 | 0.928 |
| *Head C, MuRIL (diagnostic)* | *1.000* | *0.978* | *0.999* | *0.953* |

Three readings. **(a)** Fused beats every zero-shot baseline in every bucket, by 0.088 (en) to 0.180 (cm), but it does not clearly beat Head A alone: +0.003 on en, +0.001 on hi, 0.000 on cm and −0.010 on te. **(b)** Head B is strong on en (0.900), weak on cm (0.647) and **below 0.5 on hi and te**: machine text in those buckets is *less* curved under mGPT than human text. On seen-generator hi it is 0.336 (T3). Word-level aggregation changes nothing (within 0.025 everywhere). **(c)** cm is the weakest bucket for every method, and its F1 at the train-tuned threshold is 0.766 against 0.897–0.966 elsewhere.

### 6.2 Ablation and per-bucket fusion (Table T2)

**Table T2. Ablation, test split.** `+conformal` is the per-bucket α = 0.05 threshold with no abstention; `+abstain` is the three-way gate at α = 0.01 with the shipped HUMAN cutoff.

| bucket | A | B | A+B | +temperature | +conformal: FPR / acc. | +abstain: coverage / acc. on covered |
|---|---|---|---|---|---|---|
| en | 0.986 | 0.900 | 0.989 | 0.989 | 0.052 / 0.946 | 0.730 / 0.994 |
| hi | 0.960 | 0.569 | 0.961 | 0.961 | 0.045 / 0.861 | 0.447 / 0.982 |
| te | 0.973 | 0.235 | 0.963 | 0.963 | 0.050 / 0.932 | 0.371 / 0.974 |
| cm | 0.928 | 0.647 | 0.928 | 0.928 | 0.034 / 0.829 | 0.689 / 0.953 |

Temperature scaling cannot change AUROC by construction; its effect is in T6. Abstention lifts accuracy on the rows it decides to 0.953–0.994, at a coverage of 37–73%; te and hi abstain on more than half their test rows.

**Does a per-bucket Head B weight rescue fusion?** We refit the fuser three ways: one global Head B weight (adopted), interaction terms, and a separate fuser per bucket, and repeated the separate fit with out-of-fold Head A scores on train.

**Table T2b. Fused minus Head A, test AUROC, 95% paired-bootstrap interval.**

| bucket | global weight (adopted) | separate per bucket | fitted Head B weight, separate |
|---|---|---|---|
| en | +0.003 [−0.000, +0.006] | +0.003 [−0.002, +0.009] | +1.96 |
| hi | +0.001 [−0.002, +0.004] | +0.001 [−0.001, +0.002] | +0.05 |
| hi, held-out Llama | +0.007 [+0.003, +0.011] | +0.003 [+0.001, +0.005] | — |
| te | −0.009 [−0.018, −0.002] | +0.003 [−0.004, +0.011] | −0.48 |
| cm | +0.001 [−0.002, +0.003] | +0.002 [−0.003, +0.008] | +0.44 |

Per-bucket weights let Head B take its own sign (negative on te) and remove the global fuser's te loss, but the gain over Head A is at most +0.003 and the interval includes zero everywhere except held-out Llama hi, where the gain is real but tiny. Out-of-fold Head A changes no conclusion. **RQ2 has a null answer on this corpus: curvature adds nothing measurable beyond stylometry.**

**Is Head A just surface format?** (Table S1.) Removing all 14 format-adjacent features (digit, uppercase and punctuation shares, word-length statistics, clause counts) leaves Head A at 0.982 (en), 0.964 (hi), 0.962 (te) and 0.932 on held-out Llama hi, above the nine-number surface probe (0.918, 0.836, 0.871, 0.713). Removing sentence-length features as well still leaves 0.942–0.970. **In cm it does not hold:** Head A falls from 0.928 to 0.806 (0.748 without sentence length), below the probe's 0.913. cm's Head A is mostly terminal punctuation (human cm ends a sentence with punctuation 38% of the time, generators 97–100%) plus what an English parser makes of Romanised text, and we do not present it as a stylometric result.

### 6.3 Generalisation to an unseen generator (Table T3)

**Table T3. hi only: same 449 human test rows; machine side seen generators (444) or held-out Llama (479).**

| method | seen | held-out Llama | drop |
|---|---|---|---|
| ppl | 0.786 | 0.887 | −0.101 |
| Binoculars | 0.755 | 0.917 | −0.162 |
| fastdetectgpt_en | 0.752 | 0.927 | −0.175 |
| Head B (mGPT) | 0.336 | 0.785 | −0.450 |
| Head A | 0.973 | 0.947 | +0.027 |
| **Fused (A + B)** | **0.969 [0.957, 0.980]** | **0.953 [0.942, 0.965]** | +0.015 |
| *Head C (diagnostic)* | *0.997* | *0.961* | *+0.036* |

Stylometry and the fusion generalise to Llama with a small, overlapping-interval drop. The likelihood-based scores move the other way: every one of them is *better* on Llama, and Head B is no longer inverted. Llama's Hindi is simply easier for likelihood scores than the seen generators'. One held-out generator is one draw, and en, te and cm have none, so this is not a general generalisation claim.

Per generator in hi, Head B is 0.814 (Gemma), 0.785 (Llama), 0.159 (Qwen) and 0.119 (Mistral), while fused stays at 0.953–0.975. **The inversion belongs to two specific generators under one specific scorer.**

At the deployed gate (α = 0.01) the system flags 46.1% of Llama hi text as MACHINE against 31.1% of seen-generator text, at a human FPR of 0.9%, and sends only 0.6% of Llama text to HUMAN: on an unseen generator the system's error mode is abstaining, not wrongly clearing.

### 6.4 Fairness: per-language false-positive rates (Table T5)

**Table T5. Human test FPR, per-bucket thresholds (adopted) vs one pooled threshold (comparison only).** Brackets: Wilson 95%; p: exact binomial test of FPR > α.

| bucket (n human) | α = 0.01 per-bucket | α = 0.01 pooled | α = 0.05 per-bucket | α = 0.05 pooled |
|---|---|---|---|---|
| en (402) | 0.010 [0.004, 0.025], p = 0.57 | 0.020 | 0.052 [0.034, 0.079], p = 0.45 | 0.047 |
| hi (449) | 0.009 [0.003, 0.023], p = 0.66 | 0.007 | 0.045 [0.029, 0.068], p = 0.73 | 0.038 |
| te (397) | 0.008 [0.003, 0.022], p = 0.76 | 0.013 | 0.050 [0.033, 0.077], p = 0.52 | 0.025 |
| cm (597) | 0.008 [0.004, 0.019], p = 0.71 | 0.000 | 0.034 [0.022, 0.051], p = 0.98 | 0.050 |
| **spread (max − min)** | **0.002** | **0.020** | **0.018** | **0.025** |

**The bound holds empirically.** At α = 0.01 every bucket is at or below 0.010. At α = 0.05, en (21/402) and te (20/397) sit a hair above 0.05, every interval contains 0.05, and no test finds a violation; with about 400 human rows per bucket we cannot distinguish 0.05 from 0.06, so we say "consistent with the bound", not "below it". The pooled threshold flags en humans at twice α and cm humans not at all at α = 0.01, and at α = 0.05 over-flags cm and under-flags te. The effect is modest because the four buckets' calibrated human score distributions are not far apart on this corpus, and cm's per-bucket number carries the register caveat of §3.7.

**Indian vs general English.** The Liang et al. direction is not seen: at α = 0.05 FPR is 0.085 for general English (n = 212) and 0.016 for Indian English (n = 190); at α = 0.01, 0.014 vs 0.005. General English exceeds α at both levels, which the bound does not cover because calibration pools both bands. The two bands are also different sources (HC3/Wikipedia vs Samanantar news), so we read this as "no evidence of the effect here", not as evidence of its absence.

**Rows removed by length matching.** Human rows trimmed out by length matching (en 388, hi 266, te 366) were scored through the same pipeline: FPR at α = 0.01 is 0.003, 0.008, 0.003, and at α = 0.05 0.059, 0.041, 0.044. Only en at α = 0.05 exceeds α, slightly; matching did not hide a population the detector over-flags.

### 6.5 Calibration, abstention and the price of the guarantee (Table T6, Figure F1)

**Temperature does little.** Fitted temperatures are 0.89 (en), 1.03 (hi), 0.82 (te) and 1.20 (cm), and ECE moves 0.026 → 0.027, 0.081 → 0.081, 0.028 → 0.037 and 0.057 → 0.051. **The calibration gap is generator shift, not temperature:** hi's ECE is 0.025 on seen-generator rows and 0.076 with Llama.

**Table T6b. What α buys (shipped gate).**

| bucket | recall α = 0.01 | recall α = 0.05 | machine abstained 0.01 → 0.05 | coverage 0.01 → 0.05 | false clear (either α) |
|---|---|---|---|---|---|
| en | 0.798 | 0.946 | 20% → 5% | 0.73 → 0.85 | 0.2% |
| hi | 0.389 | 0.815 | 60% → 18% | 0.45 → 0.75 | 0.8% |
| te | **0.160** | **0.902** | 83% → 8% | 0.37 → 0.67 | 1.3% |
| cm | 0.315 | 0.474 | 59% → 43% | 0.69 → 0.75 | **9.5%** |

At α = 0.01 the system is safe but in te practically useless: five of six machine passages go to ABSTAIN. Relaxing to α = 0.05 lifts te recall from 0.16 to 0.90 and hi from 0.39 to 0.82 with the bound still empirically holding. Accuracy on covered rows falls only modestly (en 0.994 → 0.982, te 0.974 → 0.945). **The guarantee is cheap to state and expensive to keep at strict α, and the price differs by language.**

**Lowering the HUMAN cutoff is not the fix.** With τ fixed, moving λ_b only converts ABSTAIN into HUMAN, and for machine text that is a false clear. Forcing 70% coverage at α = 0.01 costs false clears of 0.2% (en), 18% (hi), 9% (te) and 10% (cm); 90% coverage costs 7.5%, 47%, 57% and 44%. The shipped median cutoff keeps false clears at or below 1.3% outside cm. In risk–coverage terms (F1) risk at α = 0.01 is flat at 1–3% up to about 50–60% coverage, then climbs steeply in hi, te and cm; at α = 0.05 the curves are flatter but carry a risk floor near α, and the two cross at roughly 60–70% coverage. A usable hi or te deployment needs α near 0.05 or a higher-recall score, not a lower HUMAN cutoff.

### 6.6 Robustness under attack (Table T4)

The detector, fuser, temperatures and thresholds are frozen; attacked rows are only scored.

**Table T4. Frozen detector under attack.** det and ff: machine and human rows flagged MACHINE at α = 0.05; false clear and abstain: the shipped α = 0.01 gate. AUROC brackets: bootstrap 95%; "matched" is the clean AUROC of the same source rows.

| bucket | attack | n h / m | AUROC [95% CI] | matched | det | ff | false clear | abstain |
|---|---|---|---|---|---|---|---|---|
| en | clean | 402 / 1091 | 0.989 [0.984, 0.993] | – | 94.6% | 5.2% | 0.2% | 27.0% |
| en | hybrid | 150 / 400 | 0.985 [0.976, 0.994] | 0.990 | 93.2% | 4.0% | 0.2% | 30.2% |
| en | paraphrase | 100 / 250 | 0.920 [0.890, 0.949] | 0.985 | 96.8% | **49.0%** | 0.4% | 43.7% |
| en | back-translation | 100 / 250 | 0.937 [0.911, 0.960] | 0.985 | **68.4%** | 3.0% | **4.0%** | 58.6% |
| hi | clean | 449 / 923 | 0.961 [0.949, 0.971] | – | 81.5% | 4.5% | 0.8% | 55.3% |
| hi | hybrid | 150 / 400 | 0.964 [0.946, 0.979] | 0.970 | 80.5% | 2.7% | 2.2% | 61.1% |
| hi | paraphrase | 100 / 250 | 0.873 [0.831, 0.912] | 0.971 | 91.6% | **38.0%** | 1.2% | 66.3% |
| hi | back-translation | 100 / 250 | 0.895 [0.860, 0.929] | 0.971 | **58.4%** | 4.0% | **6.0%** | 63.4% |
| te | clean | 397 / 225 | 0.963 [0.944, 0.980] | – | 90.2% | 5.0% | 1.3% | 62.9% |
| te | hybrid | 150 / 225 | 0.965 [0.942, 0.985] | 0.961 | 86.7% | 4.0% | 1.3% | 70.7% |
| te | paraphrase | 100 / 225 | 0.872 [0.821, 0.917] | 0.951 | 90.2% | **30.0%** | 1.3% | 67.4% |
| te | back-translation | 100 / 225 | 0.840 [0.794, 0.886] | 0.951 | **53.8%** | 7.0% | **7.1%** | 73.2% |
| cm | clean | 597 / 232 | 0.928 [0.911, 0.945] | – | 47.4% | 3.4% | 9.5% | 31.1% |
| cm | hybrid | 150 / 232 | 0.844 [0.804, 0.879] | 0.925 | 40.1% | 4.0% | **15.1%** | 52.4% |
| cm | paraphrase | 100 / 232 | 0.883 [0.841, 0.919] | 0.924 | 46.1% | 4.0% | 9.5% | 54.5% |
| cm | back-translation | – | n/a | – | – | – | – | – |

**Hybrid edits barely matter**, except in cm, where AUROC falls from 0.925 to 0.844 and false clears rise from 9.5% to 15.1%: twenty percent of sentences edited is a large change to a 20-word passage.

**Back-translation is the attack that works.** It leaves human text alone (false flags 3–7%) but cuts recall at α = 0.05 from 82–95% to 54–68% and at α = 0.01 to 12–34%, and raises false clears from 0.2–1.3% to 4.0–7.1%. AUROC on the matched source rows drops by 0.048 (en), 0.076 (hi) and 0.111 (te). The gate mostly responds by abstaining more (abstention 59–73%) rather than clearing, but the rise in false clears is real.

**Paraphrase mostly moves human text toward MACHINE.** Machine text is still caught (det at or above clean in en, hi and te). The AUROC drop comes from the human side: after an LLM rewrites them, 49% (en), 38% (hi) and 30% (te) of human passages are flagged at α = 0.05, and 5–7% even at α = 0.01. These rows are no longer purely human-written and are outside the exchangeability assumption, so this is not a violation of the guarantee, but it has a direct policy meaning: **the detector responds to LLM involvement, not to who wrote the first draft**, and a student who runs their own essay through a paraphraser will often be flagged. cm is the exception (false flags unchanged at 4.0%). Paraphrase also shortens hi and te text by 20% and 33%, so length is a confound there. Self-paraphrase (Qwen rewriting Qwen's own text) shows no consistent effect: AUROC 0.905 vs 0.928 (en), 0.881 vs 0.870 (hi), 0.884 vs 0.839 (te), 0.860 vs 0.900 (cm) for Qwen-source vs other-source rows.

**Fusion does not rescue the heads under attack either.** Fused vs Head A alone: en paraphrase 0.920 vs 0.906 and back-translation 0.937 vs 0.934; hi 0.873 vs 0.882 and 0.895 vs 0.884; te 0.872 vs 0.885 and 0.840 vs 0.853. Head B alone falls under every attack (en 0.900 → 0.769 paraphrase, 0.721 back-translation). The v0 hypothesis that "the fused detector holds where either head alone fails" is not supported.

### 6.7 Tokenizer fertility and the inversion (Figure F2)

The plan predicted that curvature would degrade as fertility rises. It degrades, but fertility is not why. mGPT on hi (3.56 tokens/word) gives 0.569 while Qwen on hi (4.80, *worse* fragmentation) gives 0.843; mGPT on te (6.32) gives 0.235 while Qwen on te (11.92) gives 0.589. Word-level aggregation, which removes the dependence on how finely a word is split, leaves every bucket within 0.025 of the token-level statistic. And mGPT's hi inversion disappears on held-out Llama text (0.785) but is extreme on Qwen (0.159) and Mistral (0.119) text.

**What the evidence supports:** the orientation of a curvature score is a property of the (scorer, language, generator) combination and has to be measured, not assumed. On the v1 corpus, mean mGPT curvature on IndicCorp Hindi human text (−1.57) sat *above* Qwen-generated Hindi (−3.89), while under Qwen2.5-0.5B the human mean (−0.36) sat correctly below the machine mean (+0.48). One explanation consistent with this is overlap between mGPT's pretraining data and the public Indic corpora our human text comes from, which would place human text near mGPT's likelihood maxima. We did not test that explanation directly.

### 6.8 Error analysis (Table T7)

**Table T7. Clean test split, shipped gate.** Rates with Wilson 95% intervals.

| bucket | false flag α = 0.05 | false flag α = 0.01 | false clear (either α) | correct HUMAN | abstain α = 0.01, human / machine rows |
|---|---|---|---|---|---|
| en | 21/402 = 5.2% [3.4, 7.9] | 4/402 = 1.0% | 2/1091 = 0.2% [0.1, 0.7] | 53.0% | 46.0% / 20.0% |
| hi | 20/449 = 4.5% [2.9, 6.8] | 4/449 = 0.9% | 7/923 = 0.8% [0.4, 1.6] | 54.1% | 45.0% / 60.3% |
| te | 20/397 = 5.0% [3.3, 7.7] | 3/397 = 0.8% | 3/225 = 1.3% [0.5, 3.8] | 47.6% | 51.6% / 82.7% |
| cm | 20/597 = 3.4% [2.2, 5.1] | 5/597 = 0.8% | **22/232 = 9.5% [6.3, 13.9]** | 78.9% | 20.3% / 59.1% |

About half the humans in en, hi and te receive a confident HUMAN; nearly all the rest abstain rather than being flagged. What the errors have in common (correlations in small groups, not causes):

- **Errors track Head A, almost never Head B.** False-flagged humans have mean Head A of 0.73–0.94, against 0.03–0.16 for correctly handled humans; falsely cleared machine text has Head A near 0.01–0.03 (0.22 in cm).
- **Formal, encyclopedic human text is flagged.** In en, 12 of 21 false flags come from HC3 (a 9.8% rate in that source) and 6 from Wikipedia (6.7%), against 1.6% for the Indian-news source; all are 100+ words. In hi and te the false flags are ordinary news and Wikipedia prose with no length pattern; 85% of te's are IndicCorp.
- **Fewer digits.** False-flagged humans have about 0.2–0.4% digits by character, against 1.0–1.2% for other humans, consistent with Head A partly reading the surface register of scraped text (§4.3).
- **cm's errors are a short-text problem.** All 22 cm false clears are under 50 words (rate 10.3% under 25 words, 12.5% at 25–49), Head A has little to work with, and both generators are affected (Qwen 13.3%, Gemma 6.7%).
- **Outside cm, wrongly clearing machine text is rare** (2, 7 and 3 rows), so the main cost of the guarantee is abstention, not error.

---

## 7 Discussion

**Stylometry, not curvature, is what transfers.** Across four languages, three seen generators and one held-out one, a 40-feature stylometric head with a per-language linear classifier matches or beats every likelihood-based method and every fusion with one. Its signal in en, hi and te survives removing all format-adjacent features. Curvature works on English (0.900) and fails, in one direction or the other, depending on which scorer meets which generator. A detector that leans on curvature for Indic text would inherit that instability silently.

**The guarantee works, and its price is the real result.** Per-language conformal thresholds did what the proposition says: empirical FPR at or below α at 0.01 in every bucket, and consistent with α at 0.05. The cost is recall, and it is not shared equally: at α = 0.01 Telugu machine text is caught 16% of the time and English 80%. An institution choosing α is choosing how much of that asymmetry to accept. On this corpus most of the useful operating range is near α = 0.05.

**What a single threshold would have done.** Here the pooled threshold's damage is modest: it doubles the en false-flag rate at α = 0.01 and, at α = 0.05, applies a threshold to te human text that is stricter than te's own calibration requires. The per-language threshold removes that spread at no cost to the bound. We expect the gap to be larger with real student writing, where human score distributions are likely to differ more between languages than our scraped sources do; that is untested.

**Which output to trust.** A MACHINE verdict carries the bound. HUMAN is reliable in en, hi and te (false clears ≤ 1.3%) and unreliable on short cm text. ABSTAIN is common and is the correct answer for a large share of text at strict α.

**Deployment policy we recommend.** (1) Act on MACHINE only as a trigger for human review, never as a finding. (2) Default α = 0.05, reporting the bound as "consistent with" rather than guaranteed below. (3) Route ABSTAIN to review or to no action, never to suspicion. (4) Do not issue HUMAN verdicts on cm text under 50 words; MACHINE verdicts there remain bounded. (5) Treat a paraphrased or machine-translated essay as AI-assisted text: the detector will often flag it, and a policy should decide in advance whether that is misconduct. (6) Recalibrate each language on about 1,000 of the institution's own human texts before use.

---

## 8 Limitations

1. **Provenance confound.** All human text is scraped and all machine text is prompted; a surface probe separates them at 0.84–0.92 AUROC after normalisation. Every AUROC is an upper bound for real essays, and only an independent human sample per language can give a clean test.
2. **No real student submissions.** Ethics approval for collecting student writing was not obtained in the project window. Deployment-time human text may not be exchangeable with our calibration sources, which would void the bound; per-institution recalibration is required.
3. **One held-out generator, Hindi only.** en, te and cm have no held-out evaluation, so their T1 numbers are seen-generator numbers.
4. **cm calibration register.** cm calibration is chat (median 85 words) while cm test is comments (median 20). cm's FPR is measured, not bounded, and cm's Head A is largely terminal punctuation, a fairness risk for writers who punctuate every sentence.
5. **Small and uneven machine sets.** te has 449 and cm 464 machine rows; heavy stray-script drops in Mistral hi and Gemma te bias those cells toward cleanly-written passages.
6. **Indian-English proxy.** Samanantar's English side is sentence-shuffled; passages are runs of sentences and topically incoherent, and the band comparison in §6.4 compares different sources, not L1 vs L2 writers.
7. **Attacks.** Back-translation used NLLB-200-distilled-600M, not IndicTrans2, and was not applied to cm. Attacks cover a capped sample of 100–150 humans and 225–400 machine passages per bucket, so per-cell rates carry 4–10 point intervals. Paraphrase shortens text, confounding length in hi and te.
8. **Scorer.** mGPT-1.3B was the largest multilingual causal model that fit the hardware. The inversion's cause is not established.
9. **Missing baselines.** No supervised baseline (XLM-R) and no vanilla DetectGPT.
10. **Routing.** Language routing on out-of-corpus text was not separately evaluated; misrouted text receives another language's threshold.

---

## 9 Ethics Statement

The system is designed and presented as decision support: it does not accuse, its outputs are meant for human review, and it abstains when unsure. The per-language bound exists to prevent disproportionate flagging of particular language groups, and we report where it does not hold (cm) and what it costs. Our paraphrase results show that the detector also flags human writing an LLM has rewritten, which institutions must handle as a policy question before deployment. All human text is from public corpora with licences recorded in the corpus card; no student data was collected. NLLB-200 (CC-BY-NC) was used only to generate evaluation attacks and is not part of the deployed system.

---

## 10 Conclusion

On a four-language benchmark of Indian student-relevant text, a stylometric detector with per-language conformal abstention reaches 0.93–0.99 AUROC, holds its stated false-positive rate in every language, and generalises to an unseen generator with a small drop. Probability curvature, the method we expected to carry the multilingual signal, does not: under a multilingual scorer it inverts on Hindi and Telugu for some generators, adds nothing measurable when fused, and its failure is not explained by tokenizer fertility. The bound is real but expensive at strict α, especially for Telugu, and back-translation remains an effective way to hide machine text.

Three directions follow. First, calibrate and test on real student essays in each language, which is the only way past the provenance confound. Second, extend held-out evaluation to Telugu and code-mixed text and to commercial generators. Third, find a higher-recall score for Indic text, since the per-language bound shows exactly how much detection the current one gives up to keep its promise.

---

## References
Bao, G., Zhao, Y., Teng, Z., Yang, L., & Zhang, Y. (2024). Fast-DetectGPT: Efficient zero-shot detection of machine-generated text via conditional probability curvature. *ICLR*.
Doddapaneni, S., et al. (2023). Towards leaving no Indic language behind: Building monolingual corpora, benchmark and models for Indic languages (IndicCorpV2). *ACL*.
Dugan, L., et al. (2024). RAID: A shared benchmark for robust evaluation of machine-generated text detectors. *ACL*.
Gehrmann, S., Strobelt, H., & Rush, A. (2019). GLTR: Statistical detection and visualization of generated text. *ACL (demo)*.
Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS*.
Guo, B., et al. (2023). How close is ChatGPT to human experts? Comparison corpus, evaluation, and detection (HC3). *arXiv:2301.07597*.
Hans, A., et al. (2024). Spotting LLMs with Binoculars: Zero-shot detection of machine-generated text. *ICML*.
Hu, X., Chen, P.-Y., & Ho, T.-Y. (2023). RADAR: Robust AI-text detection via adversarial learning. *NeurIPS*.
Khanuja, S., et al. (2021). MuRIL: Multilingual representations for Indian languages. *arXiv:2103.10730*.
Lei, J., et al. (2018). Distribution-free predictive inference for regression. *JASA*.
Li, Y., et al. (2024). MAGE: Machine-generated text detection in the wild. *ACL*.
Liang, W., Yuksekgonul, M., Mao, Y., Wu, E., & Zou, J. (2023). GPT detectors are biased against non-native English writers. *Patterns, 4*(7).
Macko, D., et al. (2023). MULTITuDE: Large-scale multilingual machine-generated text detection benchmark. *EMNLP*.
Macko, D., et al. (2025). MultiSocial: Multilingual benchmark of machine-generated text detection of social-media texts.
Mitchell, E., Lee, Y., Khazatsky, A., Manning, C., & Finn, C. (2023). DetectGPT: Zero-shot machine-generated text detection using probability curvature. *ICML*.
NLLB Team (2022). No language left behind: Scaling human-centered machine translation. *arXiv:2207.04672*.
Ramesh, G., et al. (2022). Samanantar: The largest publicly available parallel corpora collection for 11 Indic languages. *TACL*.
Sheth, R., et al. (2025). COMI-LINGUA: Expert annotated large-scale dataset for multitask NLP in Hindi-English code-mixing. *arXiv:2503.21670*.
Shliazhko, O., et al. (2022). mGPT: Few-shot learners go multilingual. *arXiv:2204.07580*.
Solaiman, I., et al. (2019). Release strategies and the social impacts of language models. *arXiv:1908.09203*.
Srivastava, V., & Singh, M. (2021). HinGE: A dataset for generation and evaluation of code-mixed Hinglish text. *Workshop on Evaluation and Comparison of NLP Systems*.
Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
Wang, Y., et al. (2024). M4GT-Bench: Evaluation benchmark for black-box machine-generated text detection. *ACL*.
Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *JASA, 22*(158).
Wu, J., et al. (2024). DetectRL: Benchmarking LLM-generated text detection in real-world scenarios. *NeurIPS Datasets and Benchmarks*.
Zhou, K., Prabhumoye, S., & Black, A. W. (2018). A dataset for document grounded conversations (CMU DoG). *EMNLP*.
Zhu, X., Ren, Y., Cao, Y., Lin, X., Fang, F., & Li, Y. (2025). Reliably bounding false positives: A zero-shot machine-generated text detection framework via multiscaled conformal prediction. *arXiv:2505.05084*.

*Check before submission:* exact venue strings for HinGE and MultiSocial, and the model citations (Qwen2.5, Gemma 2, Mistral 7B, Llama 3.1, Stanza, spaCy, WordNet) in the venue's required style.

---

## Change log, v0 → v1

| v0 said | v1 says | Why |
|---|---|---|
| Contribution 2: stylometry–curvature fusion with complementary failure modes, robust to paraphrase | Fusion is a null result; Head A alone carries detection, under attack too | T1, T2b, T4 heads-vs-fused |
| Curvature under a multilingual scorer recovers Indic signal | Curvature under mGPT is inverted on seen-generator hi/te; orientation depends on scorer × generator | T1, T3 per-generator, F2 |
| Fertility explains where the signal degrades | Fertility does not explain it; word-level aggregation changes nothing | F2, T1 headB_word |
| Prior conformal MGT work uses a single pooled threshold | Zhu et al. (2025) stratify by text length; we stratify by language and script and add abstention | Corrected after reading the paper |
| Head C as a third fused head | Head C is diagnostic only | Matched by TF-IDF on held-out data (§3.5) |
| Five generators, two held-out | Three seen, one held-out (Llama, hi only); Phi dropped | Corpus card, decisions log |
| Five baselines incl. DetectGPT and XLM-R | Three zero-shot baselines | DetectGPT dropped for compute; XLM-R not built |
| Back-translation with IndicTrans2, cm via transliteration | NLLB-200-distilled-600M; cm not back-translated | IndicTrans2 gated; toolkit would not build |
| cm mean 55 words | cm test median 20 words; calibration 85 | v2 corpus |
| §6.8 length-controlled table | §6.8 error analysis (T7) by length, source and generator | Length table not run; T7 answers the Review-2 panel's question |
| Global threshold expected to fail badly on Telugu / Indian English | Modest spread; Indian-English effect not observed | T5 |
