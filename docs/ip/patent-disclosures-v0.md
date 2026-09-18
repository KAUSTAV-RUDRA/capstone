# INVENTION DISCLOSURES — Patent 1 and Patent 2
Format follows a standard university IPR-cell Invention Disclosure Form (IDF). Replace with the cell's own template when received; the content maps section-for-section. «…» = fill later.

---

# PATENT 1
## A Method and System for Language-Stratified, Confidence-Gated Detection of Machine-Generated Text with a Bounded False-Positive Rate

### 1. Inventors
«Name, department, contribution %» × «N»; Guide: Mr. Bhavanam Venkata Suresh Reddy (role: «supervisor / co-inventor — decide per IPR-cell policy»)

### 2. Title of invention
A method and system for detecting machine-generated text in which a detection score is thresholded using language-stratified conformal calibration on human-only reference data, producing a three-way decision (human / abstain / machine) with a finite-sample guarantee that the false-positive rate within each language stratum does not exceed a chosen level.

### 3. Field
Natural language processing; automated content authorship attribution; academic-integrity decision support; statistical calibration of machine-learning classifiers.

### 4. Background and problem
Existing detectors of machine-generated text produce a scalar score and apply a single threshold. When the writer population is linguistically heterogeneous — multiple languages, scripts, or proficiency levels — the score distribution for human-written text differs across sub-populations. A single threshold set to achieve a target false-positive rate on average will exceed that rate for some sub-populations and undershoot it for others. In an academic-integrity context this results in disproportionate false accusations against writers of particular languages or non-native writers of the dominant language. Prior work (Liang et al., 2023) documented false-positive rates above 60% for non-native English writers under seven existing detectors. Prior conformal approaches (e.g. arXiv 2505.05084) bound only the pooled false-positive rate and therefore do not solve this problem.

### 5. Summary of the invention
The invention:
(a) routes an input text to one of a plurality of *strata* (e.g. language or script buckets) using a language-identification module;
(b) maintains, for each stratum *b*, a reference set *C_b* of human-written texts used solely for calibration;
(c) computes a detection score *s* for the input using any scoring function (in the preferred embodiment, a fusion of stylometric and probability-curvature scores);
(d) determines, for each stratum and each target false-positive level α, a threshold τ_b(α) equal to the ⌈(n_b + 1)(1 − α)⌉-th smallest score among the reference set *C_b*;
(e) emits MACHINE if *s* > τ_b(α), HUMAN if *s* < λ_b, and ABSTAIN otherwise, where λ_b is a stratum-specific lower threshold selected to achieve a target coverage;
(f) records, with each decision, the stratum, the thresholds applied, the component scores and an explanation.

By construction, under exchangeability of *C_b* with deployment-time human text of stratum *b*, the probability that a human-written text of stratum *b* receives the MACHINE label is at most α, independently for every stratum, regardless of the scoring function.

### 6. Detailed description
**6.1 Stratification module.** Unicode-block script detection; Romanised-code-mixing detection via a curated function-word lexicon excluding forms that collide with the dominant language; assignment to strata {en, hi, te, cm} in the reference embodiment. Strata may be extended to proficiency bands or institutional cohorts.
**6.2 Reference sets.** For each stratum, *n_b* ≥ «500» human-written texts of known provenance predating the availability of large language models, or collected with consent from the deployment population. Reference sets are never used to train or tune the scoring function.
**6.3 Scoring function.** Any scalar score monotone in machine-likelihood. Preferred embodiment: logistic fusion of (i) a stylometric feature vector («N_A» features: sentence-length statistics, lexical richness, function-word distribution, punctuation, syntactic depth), (ii) conditional probability curvature under a multilingual causal language model, and optionally (iii) an encoder-embedding head.
**6.4 Threshold determination.** After optional monotone recalibration (temperature scaling) on *C_b*, τ_b(α) is the empirical quantile defined in 5(d). Thresholds are stored per stratum and per α.
**6.5 Decision and abstention.** As in 5(e). The lower threshold λ_b is swept to produce a risk–coverage curve; operating points are selected by policy.
**6.6 Audit record.** Each decision is persisted with stratum, τ_b, λ_b, component scores, driving component, top contributing features, model version and timestamp, enabling downstream human review.
**6.7 Recalibration.** New reference texts can be appended to *C_b* and thresholds recomputed without retraining the scoring function, allowing institution-specific calibration.

### 7. Novelty over prior art
| Prior art | Limitation | This invention |
|---|---|---|
| Single-threshold detectors (GPTZero-type, DetectGPT, Fast-DetectGPT, Binoculars) | No FPR guarantee; documented bias against non-native writers | Per-stratum finite-sample FPR bound |
| Pooled conformal MGT detection (arXiv 2505.05084) | Guarantee on average only; fails within strata when calibration pool is imbalanced | Guarantee holds in every stratum independently |
| Supervised classifiers with per-language models | Require labelled machine text per language; no guarantee | Human-only reference sets; scorer-agnostic |

### 8. Advantages
Distribution-free guarantee; scorer-agnostic (works with any future detector); recalibration without retraining; auditable three-way output suitable for policy ("act only on MACHINE at α = 0.01").

### 9. Experimental evidence
«From T5/T6 after Part 22: worst-stratum FPR under global threshold = X on stratum Y; under stratified thresholds ≤ α in all strata; band gap reduced from A to B; coverage at α = 0.01 = C%.»

### 10. Independent claim candidates
1. A computer-implemented method for classifying a text as machine-generated or human-written, comprising: assigning the text to one of a plurality of language strata; computing a detection score; retrieving a stratum-specific threshold determined as an empirical quantile of detection scores of a stratum-specific reference set consisting of human-written texts; and outputting a machine-generated label only if the detection score exceeds the stratum-specific threshold, whereby the probability of labelling a human-written text as machine-generated is bounded by a predetermined level within each stratum.
2. The method of claim 1, further comprising outputting an abstain label when the detection score lies between the stratum-specific threshold and a stratum-specific lower threshold selected to achieve a target coverage.
3. The method of claim 1, wherein the detection score is a fusion of a stylometric score computed without a language model and a probability-curvature score computed under a multilingual causal language model.
4. The method of claim 1, wherein at least one stratum corresponds to Romanised code-mixed text identified by a function-word lexicon.
5. A system comprising a processor and memory storing instructions to perform the method of claim 1, and a persistent audit store recording, for each classification, the stratum, thresholds, component scores and an explanation.
6. The method of claim 1, wherein the stratum-specific reference set is augmented with newly collected human-written texts and the threshold recomputed without modification of the detection-score function.

### 11. Prior-art search log
«Google Patents / Lens.org / arXiv queries and top 10 hits with one-line differentiation — Part 33.»

### 12. Drawings
Fig. 1 — system block diagram (stratifier → scorer → per-stratum threshold store → decision gate → audit). Fig. 2 — threshold determination flowchart. Fig. 3 — «F1 risk–coverage curves».

### 13. Publication / disclosure status
Not yet published. Paper submission planned «venue, date». No prior public disclosure other than internal university reviews on «3 Sept 2026, …».

### 14. Commercial applicability
Academic-integrity offices; recruitment screening (SOP / cover-letter authenticity); editorial and publishing workflows; any multilingual population where a bounded false-accusation rate is required.

---

# PATENT 2
## A Method for Script-Aware Probability-Curvature Scoring of Multi-Script and Romanised Code-Mixed Text for Machine-Generated Text Detection

### 1. Inventors
«as above»

### 2. Title
A method for computing a probability-curvature detection statistic on text comprising multiple writing systems or Romanised code-mixed language, using script normalisation, code-mix estimation, fertility-aware token aggregation and a multilingual scorer selected by measured tokenizer fertility.

### 3. Field
Natural language processing; zero-shot detection of machine-generated text; multilingual and code-mixed text processing.

### 4. Background and problem
Zero-shot curvature detectors (DetectGPT, Fast-DetectGPT) compute a statistic over the per-token log-probabilities of a scorer language model. The statistic is meaningful when tokens correspond to linguistically meaningful units. For scripts under-represented in the scorer's tokenizer, text fragments into many sub-word tokens per word (measured: 6.3 tokens/word for Telugu under a 61-language model, 11.9 under an English-default model, versus 1.4 for English). The curvature statistic then aggregates over fragments whose individual predictability carries little authorship information, degrading detection. Romanised code-mixed text (e.g. Hindi in Latin script mixed with English) presents a second failure: an English tokenizer treats it as misspelt English, and no existing detector routes or normalises it.

### 5. Summary of the invention
The invention:
(a) detects the script composition of the input and estimates a code-mix ratio using a lexicon of function words of the embedded language restricted to forms not colliding with the matrix language;
(b) normalises the input (Unicode NFC; Indic-specific normalisation; optional transliteration of Romanised segments to native script or vice versa) into a canonical form for scoring;
(c) selects, from a plurality of candidate scorer models, the model with the lowest measured tokenizer fertility for the detected script, fertility being pre-computed on reference text;
(d) computes per-token conditional probability curvature under the selected scorer;
(e) aggregates the per-token statistics at the **word** level rather than the token level, by grouping tokens into their source words and combining within-word statistics before computing the sequence-level statistic, so that highly fertile words do not dominate the aggregate;
(f) outputs the resulting curvature score together with the fertility measurement and script composition as auxiliary features for downstream fusion or calibration.

### 6. Detailed description
**6.1 Script and code-mix detection.** Per-character Unicode block counting with combining-mark handling (letters only in the numerator and denominator); code-mix ratio = matched embedded-language function words / total tokens.
**6.2 Normalisation.** NFC; indic-nlp-library normalisation for Devanagari and Telugu; whitespace normalisation around punctuation (including danda) so that informal spacing is not a detectable artefact; optional transliteration (ITRANS/ISO) for cross-script consistency.
**6.3 Scorer selection by fertility.** A fertility table (tokens per word per script per candidate model) computed offline on reference human text; at inference the scorer with minimum fertility for the detected script is selected. Reference embodiment: mGPT-1.3B for Devanagari and Telugu; either mGPT or an English scorer for Latin-script text.
**6.4 Word-level aggregation.** Given token log-probabilities ℓ_1..ℓ_n and analytic per-token means μ_i and variances σ_i² under the scorer's conditional distribution, group tokens by source word *w*; compute word-level ℓ_w = Σ ℓ_i, μ_w = Σ μ_i, σ_w² = Σ σ_i²; compute the curvature statistic over words. This makes the statistic invariant to how finely a word is tokenised.
**6.5 Output.** Curvature score; fertility; script composition; code-mix ratio — the latter three serving as stratification and explanatory features.

### 7. Novelty over prior art
| Prior art | Limitation | This invention |
|---|---|---|
| DetectGPT / Fast-DetectGPT | Token-level statistic; English scorer; no script handling | Word-level aggregation; fertility-selected multilingual scorer; script normalisation |
| Multilingual MGT benchmarks (M4GT, MULTITuDE) | Evaluate existing detectors; do not modify the statistic | New statistic computation |
| Code-mixed NLP (LID, POS) | No MGT detection | First curvature detector routing and normalising Romanised code-mixed text |

### 8. Advantages
Recovers curvature signal on high-fertility scripts; extends zero-shot detection to Romanised code-mixed text; provides fertility and code-mix ratio as explanatory features; requires no training data.

### 9. Experimental evidence
«From T1/F2 after Part 22: headB (mGPT) vs fastdetectgpt_en (Qwen) AUROC on hi, te, cm; effect of word-level aggregation if implemented as an ablation (recommended: add a `headB_tok` vs `headB_word` comparison in Part 15 — this is the ablation that supports claim 1(e)).»

### 10. Independent claim candidates
1. A computer-implemented method for computing a machine-generation detection statistic for a text, comprising: detecting the writing script of the text; selecting a causal language model from a plurality of candidates according to a pre-computed tokenizer fertility for the detected script; obtaining per-token log-probabilities and per-token conditional moments under the selected model; grouping tokens according to their source words; aggregating the log-probabilities and moments within each group; and computing a curvature statistic over the aggregated group values.
2. The method of claim 1, further comprising estimating a code-mix ratio of the text using a lexicon of function words of an embedded language restricted to forms absent from the matrix language, and routing the text to a code-mixed processing path when the ratio exceeds a threshold.
3. The method of claim 2, wherein the code-mixed processing path comprises transliteration of Romanised segments to the native script of the embedded language prior to scoring.
4. The method of claim 1, further comprising normalising whitespace around punctuation marks including the Devanagari danda prior to scoring.
5. The method of claim 1, wherein the fertility, script composition and code-mix ratio are output as auxiliary features to a downstream fusion or calibration module.
6. A system configured to perform the method of claim 1.

### 11. Prior-art search log
«Part 33.»

### 12. Drawings
Fig. 1 — pipeline (script detect → normalise → scorer select → token stats → word aggregate → curvature). Fig. 2 — fertility table and selection rule. Fig. 3 — «F2».

### 13–14. Disclosure status; applicability
As Patent 1. Applicability additionally: any multilingual NLP system requiring token-count-invariant likelihood statistics.

---

## ACTION ITEM CREATED BY THIS DRAFT
Patent 2's strongest claim (word-level aggregation, 6.4) is not yet in the code. Add to **Part 15**: implement `headB_word` alongside `headB` (token-level) and report both. If word-level aggregation improves te/hi AUROC, it becomes the headline of Patent 2 and a paragraph in the paper's §3.4. If it doesn't, drop claim 1(e) and lead with fertility-based scorer selection + code-mix routing. Either way the disclosure is filed on measured evidence.
