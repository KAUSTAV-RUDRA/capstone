# PROJECT CONTEXT — PASTE THIS FIRST
### Multilingual Machine-Generated Text Detection with Calibrated Abstention

> **To the AI reading this:** this document is the complete, current truth about the project. Read all of it before responding. Do not propose anything that contradicts §4. If something you need is not here, ask — do not assume. When I give you a task, scope your response to that task only.

---

## 1. Who I am and what this is

Final-year B.Tech (AI & Data Science) student, KL University, HTE Honours track, Y23 batch, final semester. This is my capstone, guide-approved in Sept 2026. Team of 3. It must independently produce **six deliverables**: the capstone itself, 1 research paper, 2 patents, 2 consultancy engagements — all by mid-December 2026.

**Title:** A Stylometric and Perplexity-Curvature Framework for Detecting Machine-Generated Text in Multilingual Student Submissions with Calibrated Abstention

**One sentence:** a detector that tells human-written from LLM-written student text, works on Hindi, Telugu and code-mixed input, and refuses to answer when unsure — with a statistical guarantee that its false-positive rate stays below a chosen level in every language.

---

## 2. Why this exists (the problem)

Commercial AI-text detectors fail in three ways:
1. They collapse under paraphrasing.
2. They are English-centric and flag non-native writing as AI — a Stanford study found ~61% average false-positive rate on TOEFL essays across seven detectors.
3. They output a confident score with no notion of "I don't know," which causes wrongful accusations.

Raw perplexity is the mechanism behind failure #2: non-native and L2 writing has low perplexity for reasons unrelated to AI. So this project uses **curvature** (a second-order property) instead, adds **stylometry** as a complementary signal, and wraps both in **per-language conformal abstention**.

---

## 3. The method

```
submission → preprocess + language ID (script normalise, romanised detect, code-mix ratio)
           → HEAD A  stylometric features (function words, burstiness, punctuation, TTR/MTLD, POS n-grams, syntactic depth)
           → HEAD B  Fast-DetectGPT conditional-probability curvature, scorer = mGPT-1.3B
           → HEAD C  (optional, only if VRAM ≥ 8GB) MuRIL embeddings + shallow head
           → FUSION  logistic regression or shallow GBM over head scores (interpretable by design)
           → PER-LANGUAGE CALIBRATION  temperature scaling → split-conformal threshold fitted separately per bucket (en / hi / te / code-mixed) on human-only calibration text
           → ABSTENTION GATE  HUMAN | ABSTAIN | MACHINE + confidence + which head drove it + top stylometric features
           → reviewer report (decision support, never an automatic verdict)
```

**The contribution, in three bullets:**
1. First evaluation of curvature-based zero-shot detection on Hindi, Telugu and Romanised code-mixed student text, with tokenizer-fragmentation analysis.
2. A stylometry–curvature fusion that holds under paraphrase where either head alone fails.
3. Per-language conformal abstention with a distribution-free FPR bound, measurably reducing native/non-native false-positive disparity.

---

## 4. NON-NEGOTIABLES — never propose anything that violates these

1. Multilingual and code-mixed input stays in scope. Never scope down to English-only.
2. Curvature (Fast-DetectGPT), not raw perplexity.
3. Abstention + per-language conformal thresholds are the research contribution. Never remove or simplify them. Never collapse to a global threshold.
4. Evaluation uses held-out generators only. Same-generator evaluation is worthless.
5. No fine-tuning of large models. No model above 2B parameters. Hardware floor is 4GB VRAM; use 8-bit if needed.
6. `data/processed/splits.json` is frozen once written. Never modify or regenerate it.
7. `src/` contains zero Django imports and runs with Django uninstalled. Django lives in `webapp/` and calls into `src/` via `webapp/detector/services.py` only.
8. Output is decision support, never an automatic accusation. Every UI string, paper sentence and patent claim reflects this.
9. MuRIL is encoder-only and CANNOT compute perplexity or curvature. It is for Head C only.

---

## 5. Locked decisions

| Decision | Value | Why |
|---|---|---|
| Scorer model (Head B) | mGPT-1.3B; fallback Qwen2.5-0.5B | Only available multilingual causal LM with real Telugu coverage at runnable size |
| Language buckets | en, hi, te, code-mixed (Hinglish primarily) | Four separate calibrations, four fairness rows |
| Min calibration data | 1,000 human texts per bucket, separate from train/test | Conformal quantile is unstable below ~500 |
| Web layer | Django + DRF, SQLite in dev | Admin panel = free audit interface for consultancy; ORM = decision evidence trail |
| Fusion | Logistic regression first; GBM only if it wins on calibration AUROC | Interpretability is a feature |
| Seen generators | 3 open models via local Ollama (e.g. Llama-3-8B, Gemma-2-9B, Mistral-7B) | Free, reproducible |
| Held-out generators | 2 models, test only | Non-negotiable #4 |
| Adversarial attacks | Back-translation (IndicTrans2), LLM paraphrase, 20%-human-edited hybrids | Matches RAID/DetectRL practice |
| Real student data | Undecided; proxy corpora are the primary path; go/no-go at week 6 | Approvals uncertain |

---

## 6. Datasets

**Public benchmarks:** M4GT-Bench multilingual split (primary), RAID (adversarial subsets), HC3 (smoke tests).
**Indic / code-mixed human sources:** AI4Bharat `indicnlp_catalog`, IndicCorp, HinGE, COMI-LINGUA, L3Cube, public Indian student essay sets, TOEFL-style L2 corpora.
**Own corpus:** `IndicStudentMGT` — target 10–15K samples, prompt-matched and length-matched human/machine pairs.

**Construction rules:** prompt-match human and machine text · match length distributions per bucket · strip generation artifacts ("Sure! Here is…", refusals, markdown) · hold out 2 generators + 1 domain · freeze splits early.

**Per-sample schema:**
`id | text | label | language | code_mix_ratio | generator | domain | length_tokens | attack_type | writer_L1_band | split`

---

## 7. Evaluation — the tables that must exist

| Table | Content |
|---|---|
| T1 | Main results vs 5 baselines (perplexity threshold, DetectGPT, Fast-DetectGPT, Binoculars, XLM-R supervised), AUROC + F1, per bucket |
| T2 | **Ablation**: A / B / A+B / +calibration / +abstention |
| T3 | Unseen-generator generalisation |
| T4 | Adversarial: clean vs paraphrase vs back-translation vs hybrid |
| T5 | **Fairness**: FPR by bucket and by L1/L2 band, ours vs baselines |
| T6 | Calibration: ECE + accuracy at 50/70/90% coverage |
| F1 | Risk–coverage curve, all buckets overlaid |

Headline claim shape: "at X% coverage, FPR ≤ Y% in every language bucket." Never lead with raw accuracy.

---

## 8. Repository structure

```
mgt-detect/
├── CLAUDE.md                    ← auto-loaded context for Claude Code
├── README.md
├── requirements.txt
├── manage.py                    ← Django entry
├── docs/
│   ├── handoff-brief.md, execution-plan.md, master-execution-plan.md
│   ├── progress.md              ← dated daily log (review evidence)
│   ├── decisions.md             ← every locked decision with date + reason
│   ├── lit/                     ← paper notes + literature_table.xlsx
│   ├── data/                    ← corpus card
│   ├── ip/                      ← patent disclosures, prior art
│   └── hte/                     ← rubrics, emails, receipts, letters
├── configs/                     ← default.yaml, models.yaml, data.yaml
├── src/                         ← PURE PYTHON. Zero Django.
│   ├── data/       schema, loaders, freeze_splits, generate_machine_text, clean_artifacts, build_indic_corpus
│   ├── features/   stylometric, curvature, semantic, language_id
│   ├── fusion/     fuser
│   ├── calibration/ temperature, conformal, abstention
│   ├── baselines/  perplexity_threshold, detectgpt, fast_detectgpt, binoculars, roberta_supervised
│   ├── eval/       metrics, risk_coverage, fairness_audit, ablation, explain
│   └── utils/      config, logging, io
├── experiments/                 ← exp01_baselines … exp08_ablation, each writes results/*.csv
├── webapp/                      ← Django project
│   ├── settings.py, urls.py, wsgi.py
│   └── detector/   models, views, forms, services (ONLY bridge to src/), urls, admin, templates/, static/
├── scripts/                     ← check_hardware, download_datasets
├── tests/                       ← mirrors src/
├── data/raw/, data/processed/   ← gitignored
└── results/                     ← gitignored
```

**Conventions:** scripts only, no notebooks in main · every script takes `--config` · no hardcoded paths · one component per session · commit after every verified component · ask before installing anything not in requirements.txt.

---

## 9. Team

| | Owns |
|---|---|
| **P1 (me)** | Head B, fusion, calibration/abstention, paper, Patent 1, all presentations |
| **P2** | Corpus, ethics, splits, all eval scripts, fairness audit, Patent 2, Consultancy 2 |
| **P3** | Head A, baselines, Django webapp, demo, deployment, Consultancy 1 |

---

## 10. Deliverables and timeline

| Deliverable | Target | Owner |
|---|---|---|
| Review-1 (rubric: topic 10, lit 15, gap 15, spec 15, architecture 15, methodology 10, implementation 15, presentation 5) | ~16 Sep 2026 | All |
| Fused detector beats baselines on own corpus | week 8 (late Oct) | P1 |
| Calibration + abstention + adversarial + fairness; results frozen | week 12 (20 Nov) | P1, P2 |
| Research paper submitted (venue TBD: ICON / IEEE-Springer conf / journal fallback) | week 15 (mid-Dec) | P1 |
| Patent 1 — per-language calibrated abstention gate | filed week 16 | P1 |
| Patent 2 — script-aware curvature scoring for code-mixed text | filed week 16 | P2 |
| Consultancy 1 — KL University academic integrity / exam cell pilot | completion letter week 15 | P3 |
| Consultancy 2 — T&P cell or external college | completion letter week 16 | P2 |
| Final review + report + demo | ~18 Dec 2026 | All |

Phase detail lives in `docs/master-execution-plan.md`.

---

## 11. Key literature (cite these; don't re-derive them)

DetectGPT (Mitchell 2023) · Fast-DetectGPT (Bao, ICLR 2024) · Binoculars (Hans, ICML 2024) · Liang et al. *GPT Detectors Are Biased Against Non-Native English Writers*, Patterns 2023 · M4GT-Bench (Wang, ACL 2024) · RAID (Dugan, ACL 2024) · Multiscaled Conformal Prediction for MGT detection (arXiv 2505.05084 — nearest competitor; our differentiator is per-language calibration) · RADAR (Hu, NeurIPS 2023) · GLTR (Gehrmann 2019) · HC3 (Guo 2023) · DetectRL-X (2026) · CEAID (arXiv 2509.26051) · IEEE Access 2026 survey vol 14 pp 34113–34136.

Related-work structure: (1) supervised detectors → (2) zero-shot statistical → (3) robustness/adversarial → (4) fairness & reliability.

---

## 12. Known risks and pre-decided responses

- Curvature weak on Telugu → report as a finding; fusion carries the result.
- Corpus late → drop to 3 buckets, never cut calibration.
- Ethics never clears → proxy corpora (already primary).
- Overloaded → cut Head C and adversarial *breadth* first; calibration is last to go.

---

## 13. How to work with me

- Casual tone, no preamble, no lectures. Direct execution.
- One task per session. Tell me how to verify it. Stop when done.
- Before anything structural (fusion, calibration, Django models): give 2–3 options with trade-offs and your pick. I decide.
- If you're about to violate §4, stop and say so.
- If you need information that isn't here, ask one specific question.

---

## 14. Current status (I update this line before every paste)

«e.g. "Review-1 on 16 Sep. Repo scaffolded. Hardware: RTX 3060 6GB. Head A works on 5 samples. Fast-DetectGPT not yet run."»
