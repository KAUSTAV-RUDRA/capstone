# INVENTION DISCLOSURES — Patent 1 and Patent 2 (v1, on measured evidence)

Format follows a standard university IPR-cell Invention Disclosure Form (IDF). Replace with the cell's own template when received; the content maps section for section. All evidence is from `docs/results/` at commit `4b7467e`.

**Remaining blanks (only you can fill):** co-inventor names and contribution shares; the guide's role per IPR-cell policy; first public disclosure date if the paper is submitted first.

**Read this first: what changed from v0, and how strong each one is.**

| | Patent 1 | Patent 2 |
|---|---|---|
| v0 core claim | Per-language conformal threshold + abstention | Word-level curvature aggregation + fertility-based scorer selection |
| What the results did | Supported. FPR ≤ α in every language at α = 0.01; consistent with α at 0.05 | v0 claim **failed**: word-level aggregation changed AUROC by < 0.025; fertility did not predict which scorer works |
| v1 core claim | Unchanged core, narrowed against new prior art (Zhu et al. 2025 stratify by length), plus a length-conditioned HUMAN-suppression rule from the error analysis | **Replaced**: per-stratum qualification and orientation of curvature scorers on human reference data, with disqualification of inverted scorers |
| Honest strength | Moderate. Group-conditional conformal prediction is known in general; the claimable part is the language/script + code-mix stratification, the three-way gate built from the same reference set, and the short-text rule | **Weak.** Selecting a model by validation performance is routine; the inventive step is arguable. Consider filing it as dependent claims inside Patent 1, or keep it as a disclosure only if HTE counts disclosures |

Before filing either, ask the HTE coordinator what counts as a "patent" deliverable (IDF submitted, provisional filed, or complete specification). That decides whether Patent 2 is worth a separate filing.

---

# PATENT 1
## A Method and System for Language- and Script-Stratified, Confidence-Gated Detection of Machine-Generated Text with a Per-Stratum Bounded False-Positive Rate

### 1. Inventors
Kaustav Harinarayana Rudra (research lead: calibration and abstention design, evaluation), «co-inventor 2, contribution %», «co-inventor 3, contribution %». Guide: Mr. Bhavanam Venkata Suresh Reddy (role per IPR-cell policy).

### 2. Title of invention
A method and system for detecting machine-generated text in which an input is assigned to a language-and-script stratum (including Romanised code-mixed text), scored, and thresholded against stratum-specific conformal quantiles of human-only reference texts, producing a three-way decision (HUMAN / ABSTAIN / MACHINE) whose false-positive rate on human text is bounded within each stratum, and in which HUMAN verdicts are withheld for texts below a stratum-specific length.

### 3. Field
Natural language processing; authorship attribution; academic-integrity decision support; statistical calibration of classifiers.

### 4. Background and problem
Detectors of machine-generated text output a score and apply one threshold. When writers use several languages and scripts, human-written score distributions differ by language, so one threshold set for a target false-positive rate on average over-flags some languages and under-flags others. In academic integrity this means disproportionate false accusations against some language groups. Liang et al. (2023) documented false-positive rates above 60% for non-native English writers across seven detectors.

The closest prior art, Zhu et al. (2025, arXiv:2505.05084, "multiscaled conformal prediction"), bounds the false-positive rate with one conformal quantile per **text-length bin**, on English and Chinese data, with a **binary** output. It does not stratify by language or script, does not handle Romanised code-mixed text, and has no abstention. Group-conditional ("Mondrian") conformal prediction is known in general (Vovk et al., 2005); this disclosure claims a specific application and gate, not the general statistical technique.

### 5. Summary of the invention
(a) Assign the input to one of a plurality of strata by script detection and a code-mix ratio computed from a lexicon of embedded-language function words restricted to forms that do not collide with the matrix language (reference embodiment: English, Hindi, Telugu, Romanised Hindi-English; routing threshold 0.3).
(b) For each stratum *b*, hold a reference set *C_b* of human-written texts used only for calibration (reference embodiment: 1,000 per stratum, of pre-2022 provenance).
(c) Compute a detection score *s* with any scoring function monotone in machine-likelihood.
(d) For each stratum and target level α, set τ_b(α) to the ⌈(n_b + 1)(1 − α)⌉-th smallest reference score.
(e) Set a stratum-specific lower threshold λ_b from the same reference set (reference embodiment: its median score).
(f) Output MACHINE if *s* > τ_b(α), HUMAN if *s* < λ_b, else ABSTAIN.
(g) **Length-conditioned HUMAN suppression:** for strata in which short texts carry insufficient signal, convert a HUMAN verdict to ABSTAIN when the text is shorter than a stratum-specific minimum length (reference embodiment: code-mixed text under 50 words). MACHINE verdicts are unaffected, so the false-positive bound is preserved.
(h) Record with each decision the stratum, τ_b, λ_b, component scores, the dominant component, the top contributing features, model version and timestamp.

Under exchangeability of *C_b* with deployment-time human text of stratum *b*, P(MACHINE | human text of stratum *b*) ≤ α, separately for every stratum and for any scoring function.

### 6. Detailed description
**6.1 Stratification.** Unicode-block script detection (letters only); Romanised code-mixing by function-word lexicon; Devanagari → hi, Telugu → te, Latin with code-mix ratio ≥ 0.3 → cm, else en. Strata may be extended to further languages, proficiency bands or institutional cohorts.
**6.2 Reference sets.** Human texts of known provenance, never used to train or tune the scorer. Stratum-specific reference sets should match the deployment register: in testing, a code-mixed reference set of chat turns (median 85 words) did not bound false positives on short social-media comments (median 20 words) by construction, so the method prescribes drawing *C_b* from the deployment population when available (6.7).
**6.3 Scoring function.** Any scalar score. Reference embodiment: logistic fusion of a 40-feature stylometric vector (25 model-free surface features and 15 parser features) and conditional probability curvature under a multilingual causal language model, with per-stratum temperature scaling.
**6.4 Thresholds.** As in 5(d), stored per stratum and per α. Reference values: α = 0.01: en 0.995, hi 0.974, te 0.983, cm 0.871; α = 0.05: en 0.707, hi 0.737, te 0.417, cm 0.819 (pooled single threshold for comparison: 0.976 and 0.776).
**6.5 Gate and abstention.** As in 5(e)–(g). λ_b never affects the false-positive rate; it trades ABSTAIN against HUMAN, and for machine text a HUMAN verdict is a false clear. The length rule in 5(g) removes the observed short-text false clears.
**6.6 Audit record.** As in 5(h), enabling human review and appeal.
**6.7 Recalibration.** New human texts are added to *C_b* and τ_b, λ_b recomputed without retraining the scorer, allowing institution-specific calibration with about 1,000 texts per stratum.
**6.8 Operating-point selection.** α may be chosen per stratum from measured recall at each α, since the cost of a strict α differs sharply by language (Section 9).

### 7. Novelty over prior art
| Prior art | What it does | Difference |
|---|---|---|
| Single-threshold detectors (DetectGPT, Fast-DetectGPT, Binoculars, commercial tools) | One threshold, no bound | Per-stratum finite-sample bound |
| Zhu et al. 2025, multiscaled conformal prediction | One quantile per length bin; English/Chinese; binary output | Language- and script-stratified, including Romanised code-mixed routing; three-way gate with λ_b from the same reference set; length rule applied to HUMAN verdicts only, inside a language stratum |
| Group-conditional conformal prediction (Vovk et al. 2005) | General statistical method | Specific stratification, gate and audit for machine-text detection |
| Per-language supervised classifiers | Need labelled machine text per language; no bound | Human-only reference sets; scorer-agnostic |

### 8. Advantages
Distribution-free per-stratum bound; works with any present or future scorer; recalibration without retraining; three-way output suited to policy ("MACHINE triggers review; ABSTAIN never triggers suspicion"); explicit handling of Romanised code-mixed text.

### 9. Experimental evidence (corpus of 12,148 passages; ~400–600 human test passages per stratum)

**False-positive rate on human test text.**

| stratum | α = 0.01, per-stratum | α = 0.01, pooled | α = 0.05, per-stratum | α = 0.05, pooled |
|---|---|---|---|---|
| en | 0.010 | 0.020 | 0.052 | 0.047 |
| hi | 0.009 | 0.007 | 0.045 | 0.038 |
| te | 0.008 | 0.013 | 0.050 | 0.025 |
| cm | 0.008 | 0.000 | 0.034 | 0.050 |
| spread | **0.002** | 0.020 | **0.018** | 0.025 |

Every per-stratum rate is at or below α at 0.01; at 0.05 all Wilson 95% intervals contain 0.05 and exact binomial tests show no violation (p ≥ 0.45). The pooled threshold doubles the English rate at α = 0.01 and leaves code-mixed text unflagged. Human rows excluded by length matching (n = 266–388 per stratum) give 0.003–0.008 at α = 0.01.

**Gate behaviour.** At α = 0.01 the gate wrongly clears machine text as HUMAN in 0.2% (en), 0.8% (hi) and 1.3% (te) of cases, with accuracy 0.953–0.994 on the texts it decides. On an unseen generator (Llama-3.1-8B, Hindi) it sends 0.6% to HUMAN and 53% to ABSTAIN: its failure mode is abstaining, not clearing.

**Length rule (5(g)).** In the code-mixed stratum all 22 false clears (9.5% of machine test texts) are under 50 words (10.3% under 25 words, 12.5% at 25–49 words); no other stratum shows this pattern. Withholding HUMAN below 50 words in that stratum removes all of them without changing the false-positive rate.

**Cost of the bound, per stratum (supports 6.8).** Recall at α = 0.01 vs 0.05: en 0.80 → 0.95, hi 0.39 → 0.82, te 0.16 → 0.90, cm 0.32 → 0.47.

**Under attack (frozen system, nothing refit).** Back-translation raises false clears to 4.0–7.1% and abstention to 59–73%; human false flags stay at 3–7% at α = 0.05. LLM-paraphrased human text is flagged at 30–49% at α = 0.05; such text falls outside the exchangeability condition and is disclosed as a limitation, not a failure of the bound.

### 10. Independent and dependent claim candidates
1. A computer-implemented method for classifying a text as machine-generated or human-written, comprising: assigning the text to one of a plurality of strata according to its writing script and a code-mix ratio computed against a lexicon of embedded-language function words; computing a detection score; retrieving an upper threshold for the assigned stratum determined as an empirical conformal quantile of detection scores of a stratum-specific reference set consisting only of human-written texts; retrieving a lower threshold for the assigned stratum determined from the same reference set; and outputting a machine-generated label when the score exceeds the upper threshold, a human-written label when the score is below the lower threshold, and an abstain label otherwise; whereby the probability of labelling a human-written text as machine-generated is bounded by a predetermined level within each stratum.
2. The method of claim 1, further comprising replacing a human-written label with an abstain label when the text is shorter than a minimum length specific to its stratum, without altering machine-generated labels.
3. The method of claim 1, wherein at least one stratum corresponds to Romanised code-mixed text, and the code-mix ratio counts only embedded-language function words absent from the matrix language.
4. The method of claim 1, wherein the predetermined level is selected separately for each stratum according to measured detection recall at a plurality of candidate levels.
5. The method of claim 1, wherein the lower threshold is the median detection score of the stratum's reference set.
6. The method of claim 1, wherein the reference set of a stratum is replaced or augmented with human-written texts from a deployment population and the thresholds are recomputed without modifying the detection-score function.
7. The method of claim 1, wherein the detection score is a fusion of a stylometric score computed without a language model and a probability-curvature score, the curvature component's weight and sign being determined per stratum (see Patent 2).
8. A system comprising a processor and memory storing instructions to perform the method of claim 1, and a persistent audit store recording, for each classification, the stratum, both thresholds, component scores, the dominant component and the top contributing features.

### 11. Prior-art search log (starting point; extend on Google Patents and Lens.org before filing)
| Query | Closest hit | Differentiation |
|---|---|---|
| "conformal prediction" machine-generated text false positive | Zhu et al. 2025, arXiv:2505.05084 | Length bins, binary output, no language/script strata |
| AI text detector language-specific threshold | none found in this search | — |
| Mondrian conformal group-conditional coverage | Vovk et al. 2005 and follow-ups | General method; no MGT gate |
| non-native writer AI detector bias | Liang et al. 2023 | Documents the problem; no method |
| code-mixed Hinglish AI-generated text detection | none found in this search | — |

### 12. Drawings
Fig. 1: system block diagram (router → scorer → per-stratum threshold store → three-way gate with length rule → audit store). Fig. 2: threshold-determination flowchart. Fig. 3: per-stratum FPR, per-stratum vs pooled threshold (Section 9 table). Fig. 4: risk–coverage curves at α = 0.01 and 0.05 (`docs/results/exp13_risk_coverage_alpha.png`).

### 13. Publication and disclosure status
Not yet published. Internal university reviews on 3 Sept 2026 (Review-1) and Review-2 (Sept/Oct 2026). **File the IDF before submitting the paper**: in India, a paper published first can count against novelty.

### 14. Commercial applicability
Academic-integrity offices; admissions and recruitment screening of statements of purpose and cover letters; editorial workflows; any multilingual population that needs a bounded false-accusation rate.

---

# PATENT 2
## A Method for Per-Stratum Qualification and Orientation of Probability-Curvature Scorers for Machine-Generated Text Detection Using Human Reference Data

### 1. Inventors
As Patent 1.

### 2. Title
A method for selecting, orienting or disabling a probability-curvature language-model scorer separately for each language stratum, by measuring the scorer's separation and direction on that stratum's human reference texts against a machine probe set, so that a scorer whose statistic is inverted or uninformative for a stratum is not used, or is used with the measured sign.

### 3. Field
Zero-shot detection of machine-generated text; multilingual NLP; model selection for detection pipelines.

### 4. Background and problem
Curvature detectors (DetectGPT, Fast-DetectGPT) assume that machine text sits nearer a local maximum of the scorer's likelihood than human text does, and they are used with one scorer and one sign for all input. We found this assumption fails in a way that existing methods do not detect:

- Under the multilingual scorer mGPT-1.3B, curvature is **inverted** on Hindi (AUROC 0.336) and Telugu (0.235): human text looks *more* machine-like than machine text.
- The same statistic under a different scorer, Qwen2.5-0.5B, is correctly oriented on the same text (0.752, 0.589).
- The inversion also depends on the generator: under mGPT, Hindi from Gemma scores 0.814 and from Llama 0.785, but from Qwen 0.159 and from Mistral 0.119.
- The obvious predictor, tokenizer fertility, does not explain it: Qwen fragments Hindi more than mGPT does (4.80 vs 3.56 tokens/word) yet works better; aggregating at the word level to remove fragmentation effects changes AUROC by less than 0.025.

A pipeline that fixes one scorer and one sign therefore silently uses an inverted signal in some languages, and fusing it with a global weight degrades the combined detector (Telugu: −0.009 AUROC versus the stylometric component alone).

### 5. Summary of the invention
For each stratum *b* and each candidate scorer *m*:
(a) compute the curvature statistic on the stratum's human reference set *C_b* (the same set used for conformal calibration, Patent 1) and on a machine probe set *P_b* generated by one or more available generators in that stratum's language;
(b) compute the separation AUROC_{b,m} and its orientation sign_{b,m} = sign(AUROC_{b,m} − 0.5), with a bootstrap interval;
(c) **disqualify** scorer *m* for stratum *b* if the interval for AUROC_{b,m} contains 0.5, and record it as inverted if the interval lies entirely below 0.5;
(d) select, per stratum, the qualified scorer with the largest |AUROC − 0.5|, or none;
(e) pass to the fusion stage a per-stratum curvature weight whose sign is the measured orientation, or zero if no scorer qualifies, so the fused detector falls back to the remaining components;
(f) store a per-stratum, per-generator-family qualification table, and **re-run qualification** when a new generator family is observed in deployment, because orientation was found to depend on the generator;
(g) optionally flag a scorer as **reference-contaminated** for a stratum when the mean curvature of human reference text exceeds that of the machine probe set, indicating that the human reference text sits near the scorer's likelihood maxima (for example through overlap with the scorer's pretraining data).

### 6. Detailed description
**6.1 Statistic.** Conditional probability curvature d(x) per Fast-DetectGPT, from one forward pass per candidate scorer.
**6.2 Probe set.** A few hundred machine texts per stratum from the generators available at qualification time; no labelled data from deployment is needed.
**6.3 Qualification.** Stratified bootstrap (1,000 resamples) for the AUROC interval; margin and interval width are configuration parameters.
**6.4 Orientation-aware fusion.** Per-stratum logistic fusion in which the curvature weight is fitted per stratum and constrained to the measured sign, or fixed at zero.
**6.5 Re-qualification trigger.** When the deployment pipeline encounters text routed to a stratum from a generator family absent from *P_b* (if known, e.g. from institution reports), qualification is repeated with that generator's samples.
**6.6 Output.** Per-stratum scorer choice, sign, weight, qualification interval and contamination flag, recorded in the model version and in each decision's audit record.

### 7. Novelty over prior art
| Prior art | What it does | Difference |
|---|---|---|
| DetectGPT, Fast-DetectGPT, Binoculars | One scorer, fixed sign, all languages | Per-stratum scorer qualification and sign |
| Multilingual MGT benchmarks (M4GT, MULTITuDE) | Report per-language accuracy of fixed detectors | Act on the measurement: disqualify, orient or select per stratum |
| Standard model selection by validation | Pick the best model on validation data | Uses the calibration reference set of a conformal gate, tests orientation separately, ties qualification to generator family with re-qualification, adds a contamination flag |

### 8. Advantages
Prevents silent use of inverted signals; lets a fused detector fall back to stylometry where curvature fails; needs no labelled deployment data; reuses the human reference set already required by Patent 1.

### 9. Experimental evidence
**Orientation depends on scorer (test AUROC, curvature statistic):**

| stratum | mGPT-1.3B | Qwen2.5-0.5B |
|---|---|---|
| en | 0.900 | 0.822 |
| hi | 0.569 (seen generators only: 0.336) | 0.843 (seen only: 0.752) |
| te | **0.235** | 0.589 |
| cm | 0.647 | 0.714 |

**Orientation depends on generator (Hindi, mGPT):** Gemma 0.814, Llama (held-out) 0.785, Qwen 0.159, Mistral 0.119.

**Fertility and word-level aggregation do not fix it** (this is why the v0 claims were withdrawn): mGPT 3.56 tokens/word → 0.569 on hi; Qwen 4.80 → 0.843. Word-level vs token-level mGPT: en 0.886 vs 0.900, hi 0.573 vs 0.569, te 0.229 vs 0.235, cm 0.625 vs 0.647.

**Contamination indicator (5(g)), earlier single-generator corpus:** mean mGPT curvature on IndicCorp Hindi human text −1.57 against −3.89 for machine text (human above machine: inverted); under Qwen −0.36 against +0.48 (correctly oriented).

**Orientation-aware fusion (5(e)):** per-stratum fitted curvature weights en +1.96, hi +0.05, te −0.48, cm +0.44. On Telugu this removes the global fuser's loss (fused minus stylometry −0.009 → +0.003). The overall gain from curvature after orientation is small (≤ +0.003 AUROC, intervals including zero except +0.003 on held-out Hindi), so the practical value of the invention is **preventing harm from an inverted scorer**, not adding accuracy. State it that way in the filing.

### 10. Claim candidates
1. A computer-implemented method for configuring a machine-generated-text detector, comprising, for each of a plurality of language strata: computing a probability-curvature statistic under each of a plurality of candidate language models for a set of human-written reference texts of the stratum and for a set of machine-generated probe texts of the stratum; determining for each candidate a separation measure and an orientation; disqualifying any candidate whose separation is not distinguishable from chance; and configuring the detector to use, for texts of the stratum, the qualified candidate with the greatest separation together with its measured orientation, or no curvature statistic if no candidate qualifies.
2. The method of claim 1, wherein the human-written reference texts are the reference set used to compute stratum-specific conformal thresholds.
3. The method of claim 1, wherein the curvature statistic enters a fused score with a per-stratum weight constrained to the measured orientation or fixed at zero.
4. The method of claim 1, further comprising repeating the determination when texts from a generator family absent from the probe set are encountered.
5. The method of claim 1, further comprising flagging a candidate as contaminated for a stratum when the mean statistic of the human-written reference texts exceeds that of the probe texts.
6. A system configured to perform the method of claim 1.

### 11. Prior-art search log
| Query | Closest hit | Differentiation |
|---|---|---|
| curvature detector scorer selection multilingual | Fast-DetectGPT (Bao et al. 2024): fixed scorer/sampler pair | No per-language qualification or orientation |
| inverted AI text detector non-English | none found in this search | — |
| detector pretraining data contamination human text perplexity | general membership-inference literature | Not applied to orienting a detector per language |

### 12. Drawings
Fig. 1: qualification loop (strata × candidate scorers → AUROC and sign → qualify/orient/disable → fusion weights). Fig. 2: per-stratum, per-scorer AUROC table. Fig. 3: per-generator AUROC under mGPT for Hindi.

### 13–14. Disclosure status; applicability
As Patent 1. Additionally applicable to any detection pipeline that reuses a likelihood-based statistic across languages or domains.
