# Capstone Execution Plan
### Stylometric + Perplexity-Curvature Framework for Multilingual MGT Detection with Calibrated Abstention
**Status:** Guide-approved (Sept 2026) · **Window:** ~16 weeks to final submission

---

# PART 1 — LITERATURE REVIEW

## 1.1 Read in this order (Tier 1 — non-negotiable, read fully)

| # | Paper | Why it matters to you |
|---|---|---|
| 1 | **DetectGPT** — Mitchell et al., ICML 2023 (arXiv 2301.11305) | The curvature hypothesis your entire method B rests on. Machine text sits at a local maximum of the log-prob surface. |
| 2 | **Fast-DetectGPT** — Bao et al., ICLR 2024 (arXiv 2310.05130) | Replaces DetectGPT's expensive perturbation loop with **conditional probability curvature**. ~340× faster. **Use this as your base, not vanilla DetectGPT** — vanilla will not fit your compute budget. |
| 3 | **Binoculars** — Hans et al., ICML 2024 (arXiv 2401.12070) | Zero-shot, two-model perplexity/cross-perplexity ratio. Strongest zero-shot baseline; you must beat or match it. |
| 4 | **GPT Detectors Are Biased Against Non-Native English Writers** — Liang et al., *Patterns* 2023 | Your entire motivation. 61.3% average FPR on TOEFL essays; ~19.8% flagged by all seven detectors at once. Cite in abstract, intro, and fairness chapter. |
| 5 | **M4 / M4GT-Bench** — Wang et al., ACL 2024 (arXiv 2402.11175) | Your primary multilingual dataset. 8 generators, 6 domains, 9 languages. Multilingual split: ~157K train / 42K test. |
| 6 | **RAID** — Dugan et al., ACL 2024 | 600K+ texts, 11 models, 8 domains, expanded to 6.2M via **11 adversarial attack types**. This is your robustness track. |
| 7 | **Multiscaled Conformal Prediction for MGT detection** — arXiv 2505.05084 | Directly does what you're doing on the abstention side: conformal prediction to *bound* false positives with a statistical guarantee. Read this closely — it is your nearest competitor and your best methodological template. |

## 1.2 Tier 2 — skim for framing and baselines

- **GLTR** (Gehrmann et al., 2019) — the original probability-rank visualisation. Cite as the ancestor of all statistical detectors.
- **HC3** (Guo et al., 2023) — the ChatGPT-vs-human comparison corpus everyone benchmarks on.
- **RADAR** (Hu et al., NeurIPS 2023) — adversarial-learning-based robust detection. Your paraphrase-robustness comparison point.
- **MAGE** (Li et al., 2024) — large-scale out-of-domain generalisation benchmark.
- **MULTITuDE** (Macko et al., 2023) — 11 languages, news domain. Note its weakness: only 3 languages have training data.
- **MultiSocial** (Macko et al., 2025) — 22 languages, social media, 7 LLMs with 3× paraphrasing.
- **DetectRL / DetectRL-X** (2024–2026) — real-world stress-testing. DetectRL-X is 3.46M samples, 8 languages, 8 attack scenarios, 4 length granularities. **The newest multilingual robustness benchmark — check if you can get it.**
- **CEAID** (arXiv 2509.26051) — multilingual benchmark for Central European languages; useful template for *how to construct* a regional multilingual benchmark, which is exactly your contribution shape.
- **Survey:** *Human or Machine? A Survey on MGT Detection*, IEEE Access 2026, Vol 14, pp. 34113–34136 — the most current survey. Use its taxonomy for your related-work section structure.
- **Survey:** *AI-Generated Text Detection: A Comprehensive Review of Active and Passive Approaches* (Jan 2026) — the passive/active split is a clean way to organise your related work.

## 1.3 Tier 3 — Indic / code-mixed resources (your novelty zone)

- **AI4Bharat `indicnlp_catalog`** (GitHub) — the master catalogue of Indic NLP resources. Start here for raw Indic corpora.
- **HinGE** (Srivastava & Singh, 2021) — Hinglish generation + evaluation corpus, human and rule-generated.
- **COMI-LINGUA** (arXiv 2503.21670) — expert-annotated large-scale Hindi-English code-mixing dataset, multitask.
- **L3Cube MeCorpus / MeBERT** — code-mixed Marathi-English; the template for extending to another Indian language.
- **MuRIL** (Google) — multilingual representations for Indian languages. **Strong candidate for your scorer model** on Indic text.
- **IndicBERT / IndicNLPSuite** (AI4Bharat) — alternative encoder backbone.

> **The gap you are filling, stated for the paper:** curvature-based zero-shot detection has been validated in English and a handful of European languages. It has *not* been systematically evaluated on Indian-language and Romanised code-mixed student writing, and no existing work couples it with a calibrated abstention layer whose false-positive rate is bounded per-language. That sentence is your contribution claim — build everything to defend it.

## 1.4 How to write the related-work section

Four subsections, in this order:
1. **Supervised detectors** (RoBERTa-family, fine-tuned LLM classifiers) → fail out-of-domain and on unseen generators.
2. **Zero-shot statistical detectors** (GLTR → DetectGPT → Fast-DetectGPT → Binoculars) → generalise better, but calibrated only for English.
3. **Robustness and adversarial work** (RADAR, RAID, DetectRL) → everything degrades under paraphrase.
4. **Fairness and reliability** (Liang et al., conformal-prediction work) → the FPR problem is unsolved, and this is where you plant your flag.

---

# PART 2 — DATASET PLAN

## 2.1 Three-layer corpus strategy

**Layer 1 — Public benchmark (weeks 1–2).** Establishes comparability. Non-negotiable for publication.
- **M4GT-Bench multilingual split** as primary.
- **RAID** for the adversarial track (use the attack-typed subsets directly — don't re-implement attacks you don't have to).
- **HC3** as a sanity-check / smoke-test set because it's small and fast.

**Layer 2 — Indic + code-mixed construction (weeks 2–5). This is your contribution.**

Build `IndicStudentMGT`. Target ~8–10K pairs minimum, ideally 15K.

| Component | Source | Target size |
|---|---|---|
| Human, English (Indian L2) | Real student submissions (see §2.2) + public Indian student essay corpora | 2,500 |
| Human, Hindi/Telugu/Marathi | AI4Bharat corpora, IndicCorp subsets, Wikipedia-derived | 2,000 |
| Human, code-mixed | HinGE, COMI-LINGUA, L3Cube | 1,500 |
| Machine — seen generators | Generate with 3 open models (e.g. Llama-3-8B, Gemma-2-9B, Mistral-7B) via prompts matched to the human prompts | 3,000 |
| Machine — **held-out** generators | 2 different models never seen in training/calibration | 2,000 |
| Adversarial | Paraphrased (back-translation via IndicTrans2, plus one humanizer-style rewrite), and human-edited hybrids | 2,000 |

**Critical construction rules — violating these kills the paper:**
1. **Prompt-match human and machine text.** If your human essays are on "climate change" and your machine essays are on "the French Revolution," your detector learns topic, not authorship. Generate machine text *from the same prompts* that produced the human text.
2. **Match length distributions.** Truncate/bin both sides. Length is the sneakiest confound in this field.
3. **Strip generation artifacts.** DetectRL was shown to contain LLM preamble patterns like "Sure! Here is…" in ~98.5% of its Claude data — detectors trained on that learn a shortcut, not a signal. Regex-strip all preambles, refusals, and markdown scaffolding before saving.
4. **Hold out generators strictly.** Two models must appear in test only. Same for at least one domain.
5. **Version and freeze splits early.** Write `splits.json` in week 3 and never touch it again.

**Layer 3 — Real student submissions (weeks 3–6, runs in parallel).**
This is what makes it a *student submission* detector rather than another benchmark paper. See ethics below.

## 2.2 Ethics and collection protocol (do this properly — it's also a patent/consultancy asset)

- Written consent form; participation voluntary and ungraded.
- Anonymise at ingest: strip names, roll numbers, emails; assign opaque IDs.
- Collect metadata you'll need for the fairness audit: language(s) of the writer, self-reported English proficiency band, course/domain. **Do not** collect anything you won't use.
- Get institutional/departmental sign-off in writing. Reviewers ask, and your guide will need it on record.
- Store the raw corpus offline; release only derived features or a de-identified subset.

## 2.3 What to log per sample

`id | text | label | language | code_mix_ratio | generator | domain | length_tokens | attack_type | writer_L1_band | split`

That schema is what lets you produce every table in Part 3 without recomputing anything.

---

# PART 3 — SYSTEM ARCHITECTURE

```
                    ┌─────────────────────────────┐
   raw submission → │ Preprocess & Language ID    │
                    │ (script normalise, romanised│
                    │  detection, code-mix ratio) │
                    └────────────┬────────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
  ┌───────────────┐   ┌────────────────────┐   ┌──────────────────┐
  │ HEAD A        │   │ HEAD B             │   │ HEAD C           │
  │ Stylometric   │   │ Curvature          │   │ Semantic         │
  │               │   │                    │   │ (optional)       │
  │ • func-word   │   │ Fast-DetectGPT     │   │ XLM-R / MuRIL    │
  │   freqs       │   │ conditional prob   │   │ embedding        │
  │ • burstiness  │   │ curvature          │   │ + light head     │
  │   (sent-len   │   │                    │   │                  │
  │   variance)   │   │ scorer: MuRIL /    │   │ ONLY if compute  │
  │ • punctuation │   │ mGPT / Qwen-0.5B   │   │ allows           │
  │ • TTR, MTLD   │   │                    │   │                  │
  │ • POS n-grams │   │ + Binoculars-style │   │                  │
  │ • syntactic   │   │   cross-perplexity │   │                  │
  │   depth       │   │   ratio            │   │                  │
  └───────┬───────┘   └─────────┬──────────┘   └────────┬─────────┘
          │                     │                       │
          └──────────┬──────────┴───────────────────────┘
                     ▼
        ┌────────────────────────────┐
        │ FUSION                     │
        │ Logistic regression or     │
        │ shallow GBM over head      │
        │ scores. Keep it simple and │
        │ interpretable — it is a    │
        │ feature of the paper, not  │
        │ a limitation.              │
        └────────────┬───────────────┘
                     ▼
        ┌────────────────────────────┐
        │ PER-LANGUAGE CALIBRATION   │
        │ Temperature scaling, then  │
        │ split-conformal thresholds │
        │ fitted SEPARATELY for each │
        │ language / code-mix bucket │
        └────────────┬───────────────┘
                     ▼
        ┌────────────────────────────┐
        │ ABSTENTION GATE            │
        │ HUMAN | ABSTAIN | MACHINE  │
        │ + confidence + which head  │
        │   drove the decision       │
        └────────────┬───────────────┘
                     ▼
              Reviewer report
```

## 3.1 The design decisions that carry the novelty

**Per-language conformal calibration.** This is the single most defensible idea in the project. Existing detectors fit one global threshold; non-native English then lands on the wrong side of it. You fit a separate conformal quantile per language bucket, using only human-written calibration text, which gives a **distribution-free guarantee that the false-positive rate stays under α in every language**. That is a theorem, not a hope — and it is exactly what an academic-integrity office needs.

**Abstention as a first-class output.** Report **risk–coverage curves**. The headline claim is of the form: "at 70% coverage, false-positive rate ≤ 1% in all five language buckets." Never lead with raw accuracy.

**Curvature, not perplexity.** Raw perplexity is precisely the signal that discriminates against L2 writers — non-native writing has low perplexity for reasons unrelated to AI. Curvature is a second-order property and is far less confounded with proficiency. **This argument is your intro's best paragraph and probably your strongest patent claim.**

**Interpretability output.** Return which head fired and which stylometric features were extreme. A black-box score is not usable in a disciplinary process.

## 3.2 Stack

- Python 3.11, PyTorch, HuggingFace `transformers`
- `scikit-learn` (fusion + calibration), `mapie` or hand-rolled split conformal
- spaCy / Stanza for POS and dependency features; `indic-nlp-library` for Indic normalisation
- Scorer models: start with a 0.5B–1.5B multilingual model. **Do not fine-tune anything large.** If a 7B won't fit, use 4-bit via `bitsandbytes`, or fall back to MuRIL-base.
- Ollama locally for text generation during dataset construction (you already have this workflow)
- Serving: FastAPI + a thin React or Streamlit reviewer UI
- Track everything in Weights & Biases or a plain CSV log — but track it

## 3.3 Results tables you must produce

| Table | Content |
|---|---|
| T1 | Main results: your system vs 5 baselines, AUROC + F1, per language |
| T2 | **Ablation**: stylometric-only / curvature-only / fused / fused+calibrated / +abstention |
| T3 | Unseen-generator generalisation |
| T4 | Adversarial robustness: clean vs paraphrase vs back-translation vs human-edited |
| T5 | **Fairness**: FPR broken down by language and L1/L2 band, yours vs baselines |
| T6 | Calibration: ECE + accuracy at fixed coverage (50/70/90%) |
| F1 | Risk–coverage curve, all languages overlaid |

T2 and T5 are the two that get the paper accepted. Build the pipeline so both are one script away.

---

# PART 4 — 16-WEEK TIMELINE

**Weeks 1–2 · Foundation**
- Set up repo, environment, experiment logging. `README` with the frozen problem statement.
- Download M4GT-Bench + HC3. Get RAID access started (it's large — begin early).
- Read Tier 1 papers #1–#4. Write 1-page summary notes per paper as you go.
- Reproduce vanilla Fast-DetectGPT on HC3 English. **Milestone: a working curvature score.**
- Start the consent/ethics paperwork for student data — it has the longest lead time.

**Weeks 3–4 · Baselines + corpus start**
- Implement all baselines: perplexity threshold, RoBERTa supervised, DetectGPT, Fast-DetectGPT, Binoculars.
- Freeze splits (`splits.json`). Build the logging schema.
- Begin Indic/code-mixed corpus assembly: HinGE, COMI-LINGUA, AI4Bharat pulls.
- Read Tier 1 #5–#7 and the two surveys. **Milestone: baseline numbers table on M4GT.**

**Weeks 5–6 · Corpus complete + Head A**
- Machine-text generation via Ollama, prompt-matched and length-matched. Artifact stripping.
- Student submission collection running.
- Implement stylometric head; feature-importance analysis per language.
- **Milestone: `IndicStudentMGT` v1 frozen + stylometric-only results.**

**Weeks 7–8 · Head B multilingual + fusion**
- Swap scorer to MuRIL/mGPT; validate curvature behaviour on Indic and code-mixed text (this *will* need debugging — budget for it).
- Build fusion layer. **Milestone: fused system beats every baseline on your corpus.**
- **Patent 1 disclosure draft** — the per-language calibrated abstention mechanism. Draft while the idea is fresh.

**Weeks 9–10 · Calibration + abstention**
- Temperature scaling, then split-conformal per language bucket.
- Risk–coverage curves; ECE. **Milestone: T6 + F1 complete.** This is the intellectual core — protect this time.

**Weeks 11–12 · Adversarial + fairness**
- Paraphrase, back-translation, hybrid-edit attacks. RAID attack subsets.
- Fairness audit by language and proficiency band. **Milestone: T4 + T5 complete.**
- **Patent 2 disclosure draft** — the multilingual curvature pipeline.

**Weeks 13–14 · Writing + deployment**
- Full ablation sweep → T2. Freeze all results.
- Write the paper: Method → Experiments → Results → Related Work → Intro → Abstract (in that order; abstract last, always).
- FastAPI service + reviewer UI. **Milestone: paper draft + working demo.**
- **Consultancy engagement 1** — pitch to the exam cell / academic integrity office with the demo in hand. The demo is the pitch.

**Weeks 15–16 · Finish**
- Internal review, guide feedback, revise. Target venue selection.
- Consultancy 2. Patent filings. Final report, viva deck, GitHub cleanup with a reproducibility README.

---

## Risk register — the five things most likely to sink this

| Risk | Mitigation |
|---|---|
| Curvature signal is weak on code-mixed text | Detect early (week 7, not week 12). Fallback: report it as a *finding* — "curvature degrades on code-mixed input, stylometry compensates" is a publishable negative result and strengthens the fusion argument. |
| Student data collection stalls on approvals | Start paperwork week 1. Fallback: proficiency-proxy corpora (TOEFL-style public essay sets) instead of local collection. |
| Compute too small for a decent scorer | 4-bit quantisation; MuRIL-base floor. Choose the model in week 2, not week 8. |
| Scope creep from teammates | The six non-negotiables in the handoff brief. Anything else is future work. |
| Everything is left to the last month | Weeks 9–10 are the contribution. If you're behind, cut Head C and the adversarial *breadth* — never cut calibration. |
