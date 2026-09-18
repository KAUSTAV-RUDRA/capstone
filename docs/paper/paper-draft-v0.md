# A Stylometric and Perplexity-Curvature Framework for Detecting Machine-Generated Text in Multilingual Student Submissions with Calibrated Abstention

**Authors:** «Author 1», «Author 2», «Author 3»
**Affiliation:** Department of Artificial Intelligence and Data Science (Honours & Experiential Track), KL University, Vijayawada, India
**Guide:** Mr. Bhavanam Venkata Suresh Reddy
**Target venue:** «ICON 2026 / IEEE conference / journal — decide by Part 31»

> Every «…» is a blank to be filled from `docs/results/` once Parts 13–28 complete. Table numbers T1–T6 and figures F1–F2 map directly to `results/*.csv` and `docs/results/*.png`. Text around the blanks is written to survive either outcome (curvature holds on Telugu / curvature degrades on Telugu) — pick the matching sentence when the numbers land.

---

## Abstract

Detectors of machine-generated text (MGT) are increasingly deployed in academic-integrity workflows, yet the dominant family of methods scores text by its predictability under a language model, a quantity that is confounded with writing proficiency. Prior work has shown that such detectors falsely flag a majority of essays written by non-native English speakers. Students in multilingual settings such as India write in second-language English, in Indic scripts, and in Romanised code-mixed varieties, none of which existing detectors are calibrated for. We present a framework that (i) replaces raw perplexity with conditional probability curvature computed under a multilingual scorer, (ii) fuses it with a language-agnostic stylometric head that is robust to paraphrase, and (iii) wraps the fused score in **per-language split-conformal calibration with abstention**, yielding a distribution-free guarantee that the false-positive rate on human-written text does not exceed a chosen α in each language bucket independently. We construct IndicStudentMGT, a benchmark of «N_total» prompt- and length-matched human and machine passages across English, Hindi, Telugu and Hindi-English code-mixed text, with three seen and two held-out generators and three adversarial attacks. On held-out generators the fused detector reaches «AUROC_heldout_overall» AUROC, versus «AUROC_best_baseline» for the strongest zero-shot baseline. Per-language calibration holds the empirical false-positive rate below «FPR_max_ours» in every bucket at α = 0.01, where a single global threshold reaches «FPR_max_global» on «worst_bucket_global». We further quantify a «fertility_te_vs_en»× tokenizer-fertility penalty for Telugu relative to English and show its effect on curvature-based detection. Code, corpus card and a de-identified benchmark release accompany the paper.

**Keywords:** machine-generated text detection, conformal prediction, selective prediction, stylometry, multilingual NLP, code-mixing, academic integrity, fairness

---

## 1 Introduction

Large language models now produce fluent text in dozens of languages, and universities face submissions that may be partially or wholly machine-written. A number of commercial and academic detectors exist, but their deployment in a multilingual student population raises a concern that is not merely one of accuracy: **the cost of a false positive is an accusation of misconduct**, and that cost falls unevenly.

The most widely used detectors rely, directly or indirectly, on *perplexity*: how predictable a passage is under a reference language model. Machine-generated text is predictable because the generator selects high-probability continuations. But predictability has other causes. Liang et al. (2023) evaluated seven detectors on TOEFL essays written by non-native English speakers and found an average false-positive rate of 61.3%, with one detector flagging 97.8% of the essays; the same detectors were near-perfect on native-speaker essays. Second-language writing draws on a smaller vocabulary and simpler constructions, and is therefore predictable for reasons unrelated to machine authorship. Any detector built on predictability inherits this bias.

The problem compounds in India. Students write Indian English, a distinct variety; they write in Hindi, Telugu and other Indic scripts; and in informal contexts they code-mix, typically in Roman script. Each of these lowers perplexity under an English-trained model. Existing multilingual benchmarks (Wang et al., 2024; Macko et al., 2023, 2025) show sharp drops in detector performance outside English but include no Indic-language or code-mixed student writing, and no existing detector offers a per-language bound on its false-positive rate.

We make three contributions.

1. **Curvature under a multilingual scorer, evaluated on Indic and code-mixed text.** We adopt conditional probability curvature (Bao et al., 2024), which measures whether a passage sits at a local maximum of the scorer's likelihood rather than how high its likelihood is, and compute it under a scorer whose tokenizer covers Hindi and Telugu. We provide the first systematic evaluation of curvature-based detection on Hindi, Telugu and Romanised Hindi-English text, together with a tokenizer-fertility analysis that explains where and why the signal degrades («one clause on the direction of the Telugu result»).
2. **Stylometry–curvature fusion robust to paraphrase.** A language-agnostic stylometric head captures sentence-rhythm, lexical-richness and punctuation regularities that survive paraphrasing. We show that the two heads have complementary failure modes: «T2/T4 sentence, e.g. "under paraphrase the curvature head's AUROC falls from X to Y while the fused detector retains Z"».
3. **Per-language conformal abstention.** We fit split-conformal thresholds separately for each language bucket on human-only calibration data, yielding a finite-sample guarantee that the false-positive rate in each bucket is at most α. A three-way output (HUMAN / ABSTAIN / MACHINE) reports risk–coverage trade-offs rather than a forced verdict. Against a single global threshold, per-language calibration «reduces the worst-bucket FPR from X to Y and the Indian-English/general-English FPR gap from A to B».

We additionally release **IndicStudentMGT**, a prompt- and length-matched benchmark with a documented corpus card, and a reference implementation in which the research pipeline is independent of the reviewer-facing web tool.

---

## 2 Related Work

**Supervised detectors.** Fine-tuned encoders such as RoBERTa-based classifiers (Solaiman et al., 2019; Guo et al., 2023) achieve high in-distribution accuracy but degrade on unseen generators and domains (Li et al., 2024). They also require labelled machine text from the generators one expects to encounter, which is not available in practice.

**Zero-shot statistical detectors.** GLTR (Gehrmann et al., 2019) visualised token-rank statistics. DetectGPT (Mitchell et al., 2023) introduced the *probability curvature* hypothesis: machine text lies near a local maximum of the model's log-probability, so random perturbations lower its likelihood more than they lower that of human text. Fast-DetectGPT (Bao et al., 2024) replaced the perturbation loop with an analytic *conditional probability curvature* computed from the scorer's own next-token distributions, achieving a ~340× speed-up. Binoculars (Hans et al., 2024) scores text by the ratio of perplexity under one model to cross-perplexity between two. All of these were validated on English; where multilingual results exist, they use English-centric scorers.

**Robustness.** RAID (Dugan et al., 2024) evaluated detectors under eleven adversarial attacks and found broad degradation under paraphrase. RADAR (Hu et al., 2023) trained detectors adversarially against a paraphraser. DetectRL and DetectRL-X extended stress-testing across languages and attack scenarios, and documented that generation artefacts (preambles such as "Sure! Here is…") appear in the majority of some models' outputs and provide a shortcut that detectors exploit; we strip such artefacts explicitly (§4).

**Multilingual benchmarks.** M4 and M4GT-Bench (Wang et al., 2024) span nine languages and eight generators; MULTITuDE (Macko et al., 2023) eleven languages in news; MultiSocial (Macko et al., 2025) twenty-two languages in social media. Reported performance falls sharply outside English. None includes Hindi, Telugu or Romanised code-mixed text, and none stratifies calibration by language.

**Fairness and reliability.** Liang et al. (2023) established the non-native false-positive problem. Recent work has applied conformal prediction to MGT detection to bound the false-positive rate (arXiv 2505.05084), using a single threshold fitted on pooled calibration data. A pooled threshold guarantees the *average* FPR; it does not guarantee the FPR within any sub-population, and when the calibration pool is dominated by one language the guarantee silently fails for the others. Our per-language stratification addresses exactly this gap. Selective prediction and risk–coverage analysis (Geifman & El-Yaniv, 2017) provide the framing for abstention.

---

## 3 Method

### 3.1 Problem statement
Given a passage *x* and its language bucket *b* ∈ {en, hi, te, cm}, output one of {HUMAN, ABSTAIN, MACHINE} together with a calibrated confidence and an explanation, subject to the constraint that for human-written *x* in bucket *b*, P(output = MACHINE) ≤ α.

### 3.2 Preprocessing and language identification
Unicode normalisation (NFC; Indic-specific normalisation via indic-nlp-library), script detection by Unicode block, and a code-mix ratio computed against a curated Hindi function-word list restricted to forms that do not collide with English. Passages are routed to buckets: Devanagari → hi, Telugu script → te, Latin with code-mix ratio ≥ «τ_cm» → cm, otherwise en. The routing decides which calibration threshold applies; misrouting is therefore a source of guarantee failure and is measured in «Table S-route».

### 3.3 Head A — stylometric features
«N_A» features in six families: sentence rhythm (mean, variance, extrema of sentence length — *burstiness*), lexical richness (TTR, root TTR, MTLD), function-word distribution, punctuation frequencies and diversity, word-shape statistics, and syntactic complexity (POS n-grams and mean dependency depth via Stanza for hi/te and spaCy for en/cm). Features are computed without a language model and are therefore independent of tokenizer fertility. A per-bucket logistic head maps the feature vector to a score. Stylometric features respond to *how* a text is written rather than *what* it says, and sentence-rhythm features in particular are preserved by paraphrasers that rewrite lexis while keeping structure.

### 3.4 Head B — conditional probability curvature
Let *p_θ* be a causal language model. For a token sequence *x = (x_1,…,x_n)*, Fast-DetectGPT defines

  d(x) = [ log p_θ(x) − E_{x̃∼q}[log p_θ(x̃)] ] / sqrt( Var_{x̃∼q}[log p_θ(x̃)] ),

where *q* is the scorer's own conditional distribution at each position, so that the expectation and variance are computed analytically from the next-token distributions in a single forward pass. Large *d(x)* indicates that *x* is a local maximum of *p_θ* — a signature of machine generation — while human text is near zero. Crucially, *d(x)* is a second-order quantity: a passage may have high likelihood (be predictable) without being a maximum, which is what decouples the signal from writing proficiency.

We use mGPT-1.3B (Shliazhko et al., 2022), a 61-language causal model, as *p_θ*. Table «F2» reports tokenizer fertility: mGPT requires 1.36 tokens/word on English, 3.56 on Hindi, 6.32 on Telugu and 1.71 on code-mixed text; the corresponding figures for the English-default Qwen2.5-0.5B scorer are 1.34, 4.80, 11.92 and 1.67. Curvature is computed over tokens, so higher fertility means the statistic aggregates over sub-word fragments whose individual predictability carries less authorship signal. We report Head B under both scorers to isolate the effect of scorer choice.

### 3.5 Head C — semantic embeddings
MuRIL-base (Khanuja et al., 2021), an encoder pretrained on 17 Indian languages, provides mean-pooled 768-d embeddings; a per-bucket logistic layer maps them to a score. This head is supervised and is expected to generalise less well across generators; it is included to test whether meaning-level regularities add information beyond A and B (§6.2).

### 3.6 Fusion
The head scores and a one-hot bucket indicator are combined by «logistic regression / LightGBM — chosen by calibration-split AUROC (Part 21)». We restrict the fusion model to low capacity so that the contribution of each head to a decision remains inspectable (§3.9).

### 3.7 Per-language conformal calibration
For each bucket *b* we hold out a calibration set *C_b* of *n_b* human-written passages that are used for nothing else. After temperature scaling on *C_b*, we compute the fused scores {s_i} for *i* ∈ *C_b* and set

  τ_b(α) = the ⌈(n_b + 1)(1 − α)⌉-th smallest value of {s_i}.

**Proposition.** If a new human-written passage from bucket *b* is exchangeable with *C_b*, then P(s_new > τ_b(α)) ≤ α. *Proof.* Standard split-conformal argument (Vovk et al., 2005; Lei et al., 2018): under exchangeability the rank of *s_new* among the *n_b + 1* scores is uniform. ∎

The guarantee is distribution-free: it holds regardless of the scorer, the fusion model, or the language, provided the calibration passages are human and exchangeable with deployment-time human passages of the same bucket. We use *n_b* = 1,000 for en, hi and te and *n_cm* = «n_cm» for cm, restricted to sources with pre-ChatGPT provenance (§4.3). We report α ∈ {0.01, 0.05}.

**Why stratify.** A threshold τ_global fitted on the pooled calibration set ∪_b C_b bounds the *pooled* FPR. If human scores differ in distribution across buckets — as they do when the scorer's fertility differs by 4.6× — then FPR_b can exceed α for some *b* while the pooled rate stays below α. Section 6.4 measures this directly.

### 3.8 Abstention
We output MACHINE if *s* > τ_b(α), HUMAN if *s* < λ_b, and ABSTAIN otherwise, where λ_b sweeps from the calibration median downwards to trace a risk–coverage curve (Figure F1). Coverage is the fraction of passages receiving a non-abstain output; risk is the error rate on that fraction. Operating points are reported at 50%, 70% and 90% coverage. Abstention never lowers the FPR guarantee, which is set by τ_b alone; it trades coverage for accuracy on the remaining decisions.

### 3.9 Explanation
Each decision reports the bucket and its τ_b, the three head scores, the head with the largest fusion contribution, and the five stylometric features with the largest standardised deviation. The output is framed as decision support for a human reviewer; the system does not produce an accusation.

---

## 4 The IndicStudentMGT Benchmark

### 4.1 Human text
8,800 passages, 2,200 per bucket, 120–300 words for en/hi/te, all with pre-ChatGPT provenance except where noted. Sources: **en** — HC3 human answers (Guo et al., 2023) and Wikipedia (low page-id proxy for pre-2020 creation) for *general* English (1,100); the English side of Samanantar (Ramesh et al., 2022) for *Indian* English (1,100), giving a writer_L1_band stratification used in §6.4. **hi, te** — IndicCorpV2 news (75%) blended with Indic Wikipedia (25%) to avoid a pure-news domain. **cm** — COMI-LINGUA TN (Roman-script Hinglish sentences), CMU Hinglish DoG (2018) and HinGE (2021); mean 55 words, reflecting the naturally short form of code-mixed writing.

Cleaning rejected passages with template holes, PTB-tokenisation artefacts, LaTeX, mixed script above a per-source gate, and near-duplicates (MinHash). Rejection rates and per-source statistics are in the corpus card (`docs/data/corpus_card.md`).

### 4.2 Machine text
For every human passage we generate prompt-matched machine text from its first sentence and a length target equal to the human passage's length bin, in the passage's language (for cm: "casual Hindi-English Hinglish, Roman script"). **Seen generators:** «Qwen2.5-7B-Instruct, Gemma-2-9B-it / fallback, Mistral-7B-Instruct-v0.3 / fallback». **Held-out generators (test only):** «Llama-3.1-8B-Instruct / fallback, Phi-3.5-mini-instruct». Total machine passages: «N_machine». Preambles, refusals and markdown are stripped in all four languages; outputs under 40 words or with off-script content above 40% are dropped («% dropped»). Machine passages are then downsampled per generator to the human length-bin distribution within each bucket, so that length is not a confound.

### 4.3 Splits
Per bucket: 1,000 human passages to calibration (never used elsewhere); remaining human 50/50 train/test; seen-generator machine 50/50 train/test; held-out machine 100% test. For cm, calibration draws only from CMU Hinglish DoG and HinGE, whose collection dates precede LLM availability; COMI-LINGUA, whose collection date is unstated, is confined to train/test. Splits are frozen in `splits.json` and never regenerated. Final counts: «Table S-splits».

### 4.4 Adversarial variants
Machine test passages (and 300 human test passages per bucket) receive three attacks: **paraphrase** (Qwen2.5-7B-Instruct, "rewrite in your own words, same language and length"); **back-translation** (IndicTrans2: en→hi→en, hi→en→hi, te→en→te; for cm via Devanagari normalisation and re-romanisation); **hybrid** (20% of sentences edited programmatically by synonym swap, clause reorder, or split/merge, simulating light human editing).

---

## 5 Experimental Setup

**Baselines.** Perplexity threshold (negated mean log-perplexity, Qwen2.5-0.5B); DetectGPT (T5-large mask-fill, 100 perturbations; test and calibration splits only); Fast-DetectGPT with the English-default Qwen2.5-0.5B scorer; Binoculars (observer Qwen2.5-0.5B, performer Qwen2.5-0.5B-Instruct); XLM-RoBERTa-base fine-tuned two epochs on the pooled training split.

**Metrics.** AUROC (threshold-free); F1 at each method's own threshold; empirical FPR and TPR at τ(α); expected calibration error (ECE); coverage and risk at fixed coverage.

**Hardware.** All experiments on a single RTX 4060 Laptop GPU (8 GB), fp16 for models ≤ 3B and 4-bit quantisation above. Generation: «h» GPU-hours; scoring: «h». Every passage is scored once by every method into a single table from which all results are computed; the pipeline is reproducible with one command per stage.

---

## 6 Results

### 6.1 Main results (Table T1)
«Insert T1: AUROC and F1 per method × bucket, seen generators, test split.»

«Narrative: (a) on en, fused ≈ best baseline; (b) on hi/te, headB (mGPT) vs fastdetectgpt_en (Qwen) gap = the multilingual-scorer effect; (c) cm row with the length footnote; (d) headA alone on te — is it competitive?»

### 6.2 Ablation (Table T2)
«Insert T2: A / B / C / A+B / A+B+C / +temperature / +conformal / +abstain, per bucket.»

«Narrative: is A+B > max(A,B) in every bucket? Does C add anything on held-out generators (if not, say so and drop it from the deployed system)? What does +abstain buy at 70% coverage?»

### 6.3 Generalisation to unseen generators (Table T3)
«Insert T3.» «Narrative: fused AUROC on held-out vs seen; per-generator; does XLM-R collapse here as expected?»

### 6.4 Fairness: per-language and per-band false-positive rates (Table T5)
«Insert T5: FPR at α=0.01 and 0.05 for every method with a global threshold vs ours with per-bucket thresholds, per bucket and for en split by writer_L1_band.»

«Narrative — this is the central table: with a global threshold, the worst-bucket FPR is X on bucket Y (versus α = 0.01); the Indian-English band shows FPR A vs B for general English under baseline Z. With per-bucket calibration every bucket is ≤ «max», and the band gap closes to «gap». State plainly whether the guarantee held empirically.»

### 6.5 Calibration and abstention (Table T6, Figure F1)
«Insert T6 and F1.» «Narrative: ECE before/after temperature; accuracy at 50/70/90% coverage; which buckets abstain most and why (likely te and cm).»

### 6.6 Robustness under attack (Table T4)
«Insert T4: AUROC, TPR@τ, FPR@τ per method, clean vs paraphrase vs back-translation vs hybrid.»

«Narrative: headB drops most under paraphrase; headA holds; fused retains «Z». Under attack, does FPR stay ≤ α? (It should — attacks on human text should not raise s above τ_b if exchangeability roughly holds; report the empirical number.)»

### 6.7 Tokenizer fertility and the Telugu question (Figure F2)
Fertility rises from 1.36 tokens/word (en) to 6.32 (te) under mGPT and to 11.92 under Qwen. «Insert F2: fertility vs headB AUROC vs fastdetectgpt_en AUROC per bucket.» «Choose: (a) "Curvature retains AUROC ≥ X on Telugu despite 4.6× fertility, but the English-scorer variant falls to Y, confirming that scorer coverage rather than the curvature statistic itself is the limiting factor"; or (b) "Curvature under mGPT falls to X on Telugu; Head A retains Y; the fused detector at Z shows the stylometric head compensating exactly where fertility is highest — a negative result for curvature and a positive one for fusion."»

### 6.8 Length-controlled comparison (Table S-length)
«All buckets restricted to 40–120 words.» «Narrative: does the cm gap close when length is controlled?»

---

## 7 Discussion

«After results. Themes to cover: (1) predictability vs optimisation — the curvature framing's empirical support; (2) the cost of the guarantee in coverage; (3) what a global threshold would have done to Telugu writers, concretely; (4) which head to trust in which condition; (5) implications for deployment policy: act only on MACHINE at α = 0.01, route ABSTAIN to human review.»

---

## 8 Limitations

1. **No real student submissions.** Ethics approval for collecting student writing could not be obtained within the project window; human text is drawn from public pre-2022 corpora. Deployment-time human writing may not be exchangeable with these sources, which would weaken the conformal guarantee; we recommend re-calibration on institution-specific human samples before use.
2. **Indian-English proxy.** The Samanantar English side consists of shuffled sentences, not documents; passages are runs of consecutive filtered sentences and are topically incoherent. Discourse-level stylometric features on this band are therefore less reliable; the band comparison in §6.4 should be read with this in mind.
3. **Code-mixed provenance and length.** COMI-LINGUA's collection date is unstated; it is excluded from calibration but present in train/test. The cm bucket averages 55 words and is bimodal (chat vs comments); §6.8 controls for length.
4. **Domain.** hi and te are 75% news; en is a four-domain mix. Some cross-bucket differences are partly domain differences.
5. **Generators.** Five open-weight models; commercial APIs are not evaluated.
6. **Scorer.** mGPT-1.3B is the largest multilingual causal model that fits the target hardware; larger Indic-specialised scorers may change the fertility picture.
7. **Exchangeability.** The guarantee assumes calibration and deployment human text are exchangeable within a bucket; distribution shift (new topics, new writer populations) can void it. Per-institution recalibration is cheap (1,000 human passages) and recommended.

---

## 9 Ethics Statement

The system is designed as decision support and is presented as such in every interface string: it does not accuse, and its output is intended to be reviewed by a human. The per-language false-positive bound exists specifically to prevent disproportionate flagging of non-native and Indic-language writers. All human text is from public corpora with licences recorded in the corpus card; no student data was collected. The released benchmark is de-identified. We caution against deployment without institution-specific calibration and a documented human-review process.

---

## 10 Conclusion

«Two paragraphs after results: what was shown, what the guarantee delivers in practice, and what comes next (real student calibration, more Indic languages, commercial generators).»

---

## References
Bao, G., Zhao, Y., Teng, Z., Yang, L., & Zhang, Y. (2024). Fast-DetectGPT: Efficient zero-shot detection of machine-generated text via conditional probability curvature. *ICLR*.
Dugan, L., et al. (2024). RAID: A shared benchmark for robust evaluation of machine-generated text detectors. *ACL*.
Gehrmann, S., Strobelt, H., & Rush, A. (2019). GLTR: Statistical detection and visualization of generated text. *ACL demo*.
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
Ramesh, G., et al. (2022). Samanantar: The largest publicly available parallel corpora collection for 11 Indic languages. *TACL*.
Shliazhko, O., et al. (2022). mGPT: Few-shot learners go multilingual. *arXiv:2204.07580*.
Solaiman, I., et al. (2019). Release strategies and the social impacts of language models. *arXiv:1908.09203*.
Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
Wang, Y., et al. (2024). M4GT-Bench: Evaluation benchmark for black-box machine-generated text detection. *ACL*.
«Conformal MGT detection, arXiv:2505.05084 — full citation.»
«DetectRL / DetectRL-X — full citations.»
«COMI-LINGUA, HinGE, CMU Hinglish DoG, IndicCorpV2, IndicTrans2 — full citations.»
