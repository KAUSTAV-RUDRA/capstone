# MASTER EXECUTION PLAN
### Stylometric + Perplexity-Curvature MGT Detection with Calibrated Abstention
**Team:** 3 · **Start:** Wed 2 Sep 2026 · **Review-1:** ~Wed 16 Sep 2026 · **Final:** ~mid-Dec 2026

> **How to use:** every step has an owner (P1/P2/P3), an output, and a "done when" test. Tick them in order. Nothing in a later phase depends on something unticked in an earlier one. Update `docs/progress.md` at the end of every working day — it's the evidence trail for Rubric #7 and every review after it.

---

## TEAM ROLES (fixed for the whole project)

| Role | Owns | Why this split |
|---|---|---|
| **P1 — Research lead (Kaustav)** | Head B (curvature), fusion, calibration/abstention, paper, Patent 1, all reviews/presentations | The contribution lives here; it needs one owner with full context |
| **P2 — Data & evaluation** | Corpus construction, ethics/consent, splits, all eval scripts, fairness audit, Patent 2 support, Consultancy 2 | Data is 40% of the work and has the longest lead times |
| **P3 — Engineering & product** | Head A (stylometry), baselines, Django webapp, demo, deployment, Consultancy 1, repo hygiene | Turns research into the thing reviewers and clients actually see |

**Weekly rituals (non-negotiable):**
- Mon 30 min: assign the week's steps from this doc
- Fri 30 min: each person demos what runs; update `docs/progress.md`; commit
- Every review deck is built by P1 from `docs/progress.md`, not from memory

---

## DELIVERABLE TRACKER (the six things you're graded on)

| # | Deliverable | Phase | Owner | Status |
|---|---|---|---|---|
| D1 | Capstone (Review-1 → Final) | 1–7 | All | ☐ |
| D2 | Research paper | 4 | P1 | ☐ |
| D3 | Patent 1 — per-language calibrated abstention gate | 5 | P1 | ☐ |
| D4 | Patent 2 — script-aware curvature scoring for code-mixed text | 5 | P2 | ☐ |
| D5 | Consultancy 1 — KL University academic integrity / exam cell | 6 | P3 | ☐ |
| D6 | Consultancy 2 — external institution or T&P cell | 6 | P2 | ☐ |

---

# PHASE 0 — CONFIRM THE RULES (Wed 2 – Thu 3 Sep)

Do these before anything else. They change the shape of Phases 5 and 6.

| Step | Owner | Output | Done when |
|---|---|---|---|
| 0.1 Ask the HTE coordinator in writing: what counts as a "patent" — drafted disclosure, university IPR cell submission, or IPO provisional filing? Get the answer by email. | P1 | Saved email in `docs/hte/` | Written answer received |
| 0.2 Ask what counts as a "consultancy project" — is an MoU/letter of engagement + deliverable report enough? Is an internal university unit acceptable as a client? | P1 | Saved email | Written answer received |
| 0.3 Ask for the Review-2 / Review-3 / Final rubrics now. | P1 | Rubrics in `docs/hte/` | Files received |
| 0.4 Confirm exact Review-1 date, slot, duration, and expected format (PPT + report? demo?) | P1 | Note in `docs/progress.md` | Confirmed |
| 0.5 Confirm the ethics/consent route for student data: who signs, how long it takes | P2 | Note in progress.md | Named person + timeline |

**Default assumptions if answers are slow** (the plan below satisfies the strictest version): patents = provisional filings via university IPR cell; consultancy = signed engagement letter + delivered report + client completion letter.

---

# PHASE 1 — REVIEW-1 (Thu 3 – Wed 16 Sep) · 10 working days

**Goal:** score 85+/100. The rubric weights lit review, gap/objectives, spec, architecture, and initial implementation at 15 each — that's 75 marks in five items. Build for those five.

## 1.1 Rubric → deliverable map

| Rubric item | Marks | What you hand in | Owner |
|---|---|---|---|
| 1 Topic & problem | 10 | Report §1 + slides 2–3: problem statement, motivation with the 61.3% FPR finding, feasibility | P1 |
| 2 Literature review | 15 | **Literature table: 18+ papers**, grouped by the four themes; report §2; slides 4–6 | P1 + P2 |
| 3 Gap & objectives | 15 | Gap statement (one paragraph), 4 research questions, 5 objectives; report §3; slide 7 | P1 |
| 4 Research spec | 15 | Scope in/out table, FR list, NFR list, expected outcomes with target numbers; report §4; slides 8–9 | P3 |
| 5 Architecture & design | 15 | Architecture diagram, module table, data-flow diagram, tech-stack table with justification; report §5; slides 10–12 | P3 + P1 |
| 6 Methodology | 10 | Method description of Heads A/B, fusion, conformal calibration, abstention; justification vs alternatives; report §6; slide 13 | P1 |
| 7 Initial implementation | 15 | **Live demo** + GitHub link + screenshots + progress log; report §7; slides 14–15 | P3 + P2 |
| 8 Presentation & Q&A | 5 | Rehearsed 12-min talk, Q&A sheet | All |

## 1.2 Day-by-day

**Day 1 — Thu 3 Sep**
- [ ] P1: Phase 0 emails sent. Repo scaffold prompt run in Claude Code; skeleton committed.
- [ ] P2: Download HC3 and M4GT-Bench multilingual split. Start ethics paperwork draft.
- [ ] P3: Environment + `requirements.txt` + `scripts/check_hardware.py`. Post the hardware output in the team chat.

**Day 2 — Fri 4 Sep**
- [ ] P1: Read DetectGPT, Fast-DetectGPT, Liang et al. Write 1-page notes each into `docs/lit/`.
- [ ] P2: Read M4GT-Bench, RAID, conformal-MGT paper. Notes into `docs/lit/`. Start `docs/lit/literature_table.xlsx` (columns: paper, year, venue, method type, languages, dataset, key result, limitation, relevance to us).
- [ ] P3: Implement `src/features/stylometric.py` (Head A). Run on 5 sample strings incl. Telugu and Hinglish. Commit.

**Day 3 — Sat 5 Sep**
- [ ] P1: **Tokenizer fertility test** — mGPT vs Qwen2.5-0.5B on 100 Telugu + 100 Hindi sentences. Record tokens/char. Lock the scorer model. Write result into `docs/decisions.md`.
- [ ] P2: Literature table to 12 rows. Read Binoculars, RADAR, GLTR.
- [ ] P3: Implement `src/baselines/fast_detectgpt.py` with the locked scorer. Get a score out on 10 HC3 English samples.

**Day 4 — Mon 7 Sep**
- [ ] P1: Draft report §1 (topic/problem) and §3 (gap, RQs, objectives). Use the four RQs below.
- [ ] P2: Literature table to 18 rows. Draft report §2 (lit review) from the table — themes, not paper-by-paper.
- [ ] P3: Run Fast-DetectGPT on 200 HC3 samples (100 human / 100 machine). Compute AUROC. **This number is your Rubric-7 evidence.** Save to `results/exp00_smoke.csv`.

**Day 5 — Tue 8 Sep**
- [ ] P1: Draft report §6 (methodology) with justification table: "why curvature not perplexity", "why conformal not temperature-only", "why fusion not single head".
- [ ] P2: Pull 50 Hindi + 50 Telugu human samples from AI4Bharat/IndicCorp; 50 Hinglish from HinGE. Run Head A on all 150 — capture the feature-distribution plot per language. **Second Rubric-7 evidence.**
- [ ] P3: Draft report §4 (spec): scope table, FR-01…FR-12, NFR-01…NFR-08, expected outcomes with target metrics.

**Day 6 — Wed 9 Sep**
- [ ] P1: Architecture diagram (draw.io or Mermaid) — the Part-3 diagram from the execution plan, cleaned up. Data-flow diagram. Module table.
- [ ] P2: Fairness-motivation figure: run a plain perplexity threshold on the 150 Indic samples vs 50 English — show the FPR gap. This is the slide that makes reviewers sit up.
- [ ] P3: Django skeleton: project + `detector` app + one upload form + one results page that calls `services.py` → Fast-DetectGPT + Head A and shows both scores. Ugly is fine. Working is required.

**Day 7 — Thu 10 Sep**
- [ ] P1: Report §5 (architecture) written around P1's diagrams. Assemble full report v1 (§1–§7 + references in IEEE format). Send to guide for feedback.
- [ ] P2: Report §7 (initial implementation) — screenshots, AUROC number, feature plots, GitHub commit log screenshot, progress.md excerpt.
- [ ] P3: Demo hardening: 5 pre-loaded sample texts (en human, en machine, hi human, te human, hinglish machine) selectable from a dropdown so the demo never depends on typing.

**Day 8 — Fri 11 Sep**
- [ ] P1: Slide deck v1 — 15 slides, template below. Incorporate guide feedback on report.
- [ ] P2: Q&A prep sheet — the 25 questions below, with 2-line answers each. Everyone reads it.
- [ ] P3: Record a 90-second screen capture of the demo as backup in case live demo fails.

**Day 9 — Mon 14 Sep**
- [ ] All: Rehearsal 1, timed. P1 presents §1–§6, P3 demos §7, P2 handles data questions. Cut to 12 minutes.
- [ ] P1: Report v2 with any corrections. Freeze.
- [ ] P2/P3: Fix whatever broke in rehearsal.

**Day 10 — Tue 15 Sep**
- [ ] All: Rehearsal 2 with guide if possible. Print report. Export PDF of slides. Push final commit tagged `review-1`.
- [ ] Wed 16 Sep: **Review-1.**

## 1.3 Research questions and objectives (copy into report §3)

**Gap statement:** Zero-shot curvature-based detectors have been validated on English and a small set of European languages. No published work evaluates them on Indian-language or Romanised code-mixed student writing, and no existing detector bounds its false-positive rate per language with a statistical guarantee — the failure mode that causes wrongful accusations of non-native writers.

**RQ1.** Does perplexity-curvature retain discriminative power on Hindi, Telugu, and Hindi-English code-mixed text, and how does tokenizer fragmentation affect it?
**RQ2.** Do stylometric features and curvature features carry complementary signal, such that fusion outperforms either alone under paraphrase attack?
**RQ3.** Can per-language conformal calibration bound the false-positive rate below a chosen α in every language bucket, and at what cost to coverage?
**RQ4.** Does the proposed system reduce the false-positive disparity between native and non-native English writers relative to existing detectors?

**Objectives.** O1 Construct a four-bucket (en/hi/te/code-mixed) benchmark with prompt- and length-matched human/machine pairs and held-out generators. O2 Implement and evaluate stylometric and curvature heads independently. O3 Design a fusion + per-language conformal abstention layer with a distribution-free FPR guarantee. O4 Evaluate robustness under paraphrase, back-translation, and hybrid editing. O5 Deliver a reviewer-facing web tool with interpretable, decision-support output.

## 1.4 Slide template (15 slides, 12 minutes)

1 Title · 2 Problem (with the 61.3% number) · 3 Motivation & feasibility · 4 Lit review: supervised & zero-shot · 5 Lit review: robustness & fairness · 6 Comparison table of 8 key methods · 7 Gap + RQs + objectives · 8 Scope & requirements · 9 Expected outcomes with target numbers · 10 Architecture · 11 Data flow & modules · 12 Tech stack with justification · 13 Methodology & justification · 14 Implementation progress: numbers + plots · 15 Live demo + timeline to Review-2

## 1.5 Q&A prep — the 25 questions you will be asked

1. Why not just use Turnitin/GPTZero? 2. What is curvature and why is it better than perplexity? 3. Why will this not also discriminate against non-native writers? 4. What is conformal prediction in one sentence? 5. What does "abstain" mean for the student — what happens next? 6. How do you get Telugu machine-generated text? 7. How do you know your human text is really human? 8. What if the student paraphrases with another AI? 9. What generators do you hold out and why? 10. How big is the dataset? 11. What hardware does this run on? 12. Why mGPT? 13. Why Django? 14. What is the novelty — someone has done conformal prediction for this already? 15. What is the patentable component? 16. Who is the consultancy client? 17. What is your baseline and what number do you need to beat? 18. What is the risk if curvature fails on Telugu? 19. How is code-mixed text tokenised? 20. What is your target venue? 21. How will you measure fairness? 22. What are the ethical concerns with student data? 23. What is each team member doing? 24. What is done by Review-2? 25. What will you do if you're behind?

Answer to 14 (memorise it): "Existing conformal work fits one global threshold. We fit per-language thresholds on human-only calibration text, which is what bounds FPR *in every language*, not just on average — and we couple it with a stylometric head that compensates where curvature degrades on code-mixed input."

---

# PHASE 2 — CORE RESEARCH BUILD (Thu 17 Sep – Fri 23 Oct) · Weeks 3–8

**Goal:** a fused detector that beats all baselines on your own corpus. Everything in Phase 3 depends on this.

## 2.1 Corpus (P2, weeks 3–6)

| Step | Output | Done when |
|---|---|---|
| 2.1.1 Freeze `data/processed/splits.json` with held-out generators and one held-out domain named in `configs/data.yaml` | splits.json | File exists and the overwrite guard fires |
| 2.1.2 Human text: ≥1,000 per bucket for calibration + ≥1,500 per bucket for train/test. Sources: IndicCorp, AI4Bharat catalogue, HinGE, COMI-LINGUA, public Indian student essays, TOEFL-style L2 corpora | `data/raw/human/` | Counts logged per bucket |
| 2.1.3 Machine text via Ollama: 3 seen generators, prompt-matched to the human texts, length-binned | `data/raw/machine/` | Length KS-test p>0.05 human vs machine per bucket |
| 2.1.4 `clean_artifacts.py`: strip preambles ("Sure! Here…"), refusals, markdown. Report the % stripped | Cleaned corpus | Manual spot-check of 50 samples finds zero artifacts |
| 2.1.5 Held-out: 2 unseen generators, test only | `data/raw/machine_heldout/` | Never appears in train or cal |
| 2.1.6 Adversarial: back-translation via IndicTrans2, LLM paraphrase, 20%-human-edited hybrids | `data/raw/adversarial/` | Each attack type ≥400 samples per bucket |
| 2.1.7 Ethics decision gate — **end of week 6**: real student data go/no-go | Note in decisions.md | Decided; no more time spent if no-go |
| 2.1.8 Corpus card: `docs/data/IndicStudentMGT_card.md` — sources, licences, counts, construction rules | Datasheet | Complete; this becomes a paper appendix |

## 2.2 Heads and fusion (P1 + P3, weeks 3–8)

| Step | Owner | Output | Done when |
|---|---|---|---|
| 2.2.1 All baselines implemented with common `score()` interface: perplexity threshold, DetectGPT, Fast-DetectGPT, Binoculars, XLM-R supervised | P3 | `src/baselines/` | `exp01_baselines.py` produces T1-baseline rows per bucket |
| 2.2.2 Head A finalised: language-aware feature norms, feature importance per bucket | P3 | `exp02_stylometric.py` → results | Stylometric-only AUROC per bucket logged |
| 2.2.3 Head B: Fast-DetectGPT with mGPT; script normalisation; fertility-aware segmentation for Romanised code-mix | P1 | `src/features/curvature.py` | Curvature-only AUROC per bucket logged |
| 2.2.4 **Diagnostic**: does curvature degrade on te/code-mixed? Quantify | P1 | Plot + note in decisions.md | Answer written down either way — both outcomes are publishable |
| 2.2.5 Head C (optional, only if VRAM ≥8GB): MuRIL embeddings + logistic head | P3 | `src/features/semantic.py` | Skip if <8GB; note the decision |
| 2.2.6 Fusion: logistic regression and shallow GBM over head outputs; pick by cal-set AUROC | P1 | `src/fusion/fuser.py`, `exp04_fusion.py` | Fused > every baseline on every bucket |
| 2.2.7 Unseen-generator eval | P2 | `exp01` rerun on held-out → T3 | Table exists |
| 2.2.8 **Patent 2 invention disclosure v0** — script-aware curvature pipeline. Write it the week 2.2.3 works | P2 + P1 | `docs/ip/patent2_disclosure_v0.md` | Draft exists |

**Phase 2 exit criteria:** T1 (partial), T3 done · fused detector wins on all four buckets · corpus card written · decision on Head C and on student data recorded.

---

# PHASE 3 — CALIBRATION, ABSTENTION, ROBUSTNESS, FAIRNESS (Mon 26 Oct – Fri 20 Nov) · Weeks 9–12

**Goal:** the contribution. Protect these four weeks from everything else.

| Step | Owner | Output | Done when |
|---|---|---|---|
| 3.1 Temperature scaling on fusion output, per bucket | P1 | `src/calibration/temperature.py` | ECE per bucket logged |
| 3.2 Split-conformal thresholds per bucket on human-only calibration sets, α ∈ {0.01, 0.05} | P1 | `src/calibration/conformal.py` | Empirical FPR ≤ α on test in every bucket |
| 3.3 Abstention gate: human / abstain / machine; coverage sweep | P1 | `src/calibration/abstention.py`, `exp05` | Risk–coverage curve F1 + T6 |
| 3.4 Headline claim extracted: "at X% coverage, FPR ≤ Y% in all buckets" | P1 | Sentence in decisions.md | One sentence, with numbers |
| 3.5 Adversarial eval: clean vs paraphrase vs back-translation vs hybrid, all systems | P2 | `exp06` → T4 | Table done |
| 3.6 Fairness audit: FPR by bucket and by L1/L2 band, yours vs baselines | P2 | `exp07` → T5 | Table done; disparity reduction quantified |
| 3.7 Full ablation sweep: A / B / A+B / +cal / +abstain | P2 | `exp08` → T2 | Table done |
| 3.8 Interpretability: per-decision report — which head, top-5 stylometric features, curvature score | P3 | `src/eval/explain.py` | Rendered in Django results page |
| 3.9 **Patent 1 invention disclosure v0** — per-language conformal abstention gate | P1 | `docs/ip/patent1_disclosure_v0.md` | Draft exists |
| 3.10 Freeze all results. Tag `results-frozen` | P2 | Git tag | No experiment script changes after this without a logged reason |

**Review-2 likely lands in this phase.** Build its deck from progress.md + T2/T3/T6/F1. Same 15-slide skeleton; replace slides 14–15 with results.

---

# PHASE 4 — RESEARCH PAPER (weeks 10–15, overlapping)

**Owner: P1.** Write in this order — never abstract first.

| Step | Week | Output | Done when |
|---|---|---|---|
| 4.1 Choose venue. Options: ICON (Indian NLP, Dec), a Springer/IEEE conference with Jan–Feb deadline, or a journal (Expert Systems with Applications, IEEE Access) if timeline slips | 10 | Venue + deadline in decisions.md | Chosen |
| 4.2 Method section — from the architecture doc | 10 | `paper/method.tex` | Guide has read it |
| 4.3 Experimental setup — from corpus card + configs | 11 | `paper/setup.tex` | — |
| 4.4 Results — T1–T6, F1, with 2-sentence takeaways each | 12 | `paper/results.tex` | Every table has a claim |
| 4.5 Related work — four subsections from execution plan §1.4, 25+ citations | 13 | `paper/related.tex` | — |
| 4.6 Introduction — problem, 61.3% number, gap, three contribution bullets | 13 | `paper/intro.tex` | — |
| 4.7 Limitations + ethics statement (mandatory for this topic) | 14 | — | — |
| 4.8 Abstract, title, keywords | 14 | — | Written last |
| 4.9 Internal review: P2 and P3 each read cold and list every unclear sentence | 14 | Issues list | All resolved |
| 4.10 Guide review → revise → submit | 15 | Submission receipt in `docs/hte/` | **D2 done** |

**Three contribution bullets (memorise):** (1) first evaluation of curvature-based zero-shot detection on Hindi, Telugu, and Romanised code-mixed student text, with a tokenizer-fragmentation analysis; (2) a stylometry–curvature fusion that holds up under paraphrase where either head alone fails; (3) per-language conformal abstention with a distribution-free FPR bound, reducing native/non-native false-positive disparity by a measured margin.

---

# PHASE 5 — PATENTS (weeks 8–16)

Plan for the strictest interpretation until Phase 0.1 says otherwise: **provisional filing through the university IPR cell.** Two inventions, two files.

## Patent 1 — Per-language calibrated abstention gate (P1)
*"A method and system for confidence-gated classification of machine-generated text using language-stratified conformal thresholds derived from human-only calibration corpora."*

## Patent 2 — Script-aware curvature scoring for code-mixed text (P2 with P1)
*"A method for computing probability-curvature detection scores on multi-script and Romanised code-mixed text using fertility-aware segmentation and script normalisation."*

| Step | Both patents | Done when |
|---|---|---|
| 5.1 Contact the university IPR cell; get their invention disclosure form (IDF) and their process/timeline | Form in `docs/ip/` | Week 8 |
| 5.2 Prior-art search: Google Patents + Lens.org + arXiv, 10+ closest hits each, 1-line differentiation for each | `docs/ip/prior_art_p1.md`, `_p2.md` | Week 9–10 |
| 5.3 IDF filled: title, field, background, problem, summary, detailed description with flowchart, 3–5 independent-claim candidates, advantages, inventors | Completed IDFs | Week 11 |
| 5.4 Flowchart figures — one per patent, boxes and arrows, no code | PNG + source | Week 11 |
| 5.5 Submit to IPR cell; iterate on their feedback | Submission acknowledgement | Week 12–13 |
| 5.6 Provisional specification drafted (with IPR cell / attorney if they provide one) | Provisional spec | Week 14–15 |
| 5.7 Filing receipt / application number | Receipt in `docs/hte/` | **D3, D4 done** — week 16 |

**If HTE says a disclosure document is enough:** stop at 5.3 + 5.4 and submit those. Weeks 14–15 go back to the paper.

---

# PHASE 6 — CONSULTANCY (weeks 8–16)

Plan for: signed engagement letter + scoped deliverable + delivered report + client completion/acknowledgement letter.

## Consultancy 1 — KL University academic integrity / exam cell (P3)
Internal client, shortest path, demo already exists.

| Step | Done when |
|---|---|
| 6.1.1 Identify the right person (Dean Academics / exam cell / CoE office). Ask guide who. | Name + meeting booked, week 8 |
| 6.1.2 One-page proposal: problem, what you'll deliver (pilot deployment + evaluation report + usage guidelines), timeline, what you need from them (nothing sensitive) | Sent, week 9 |
| 6.1.3 Engagement letter signed on university letterhead | Signed, week 10 |
| 6.1.4 Pilot: Django app deployed on a university machine or free-tier cloud; 2-week trial with 3–5 faculty on their own samples | Trial log, weeks 11–13 |
| 6.1.5 Deliverable report: system description, pilot results, recommended workflow (detector → abstain → human review), limitations, policy suggestions | PDF, week 14 |
| 6.1.6 Completion letter from client | Letter in `docs/hte/` — **D5 done**, week 15 |

## Consultancy 2 — external or T&P cell (P2)
Candidates, in order of ease: (a) KL University T&P cell — screening for AI-written SOPs/cover letters; (b) a neighbouring college's exam cell; (c) an ed-tech/coaching institute in Vijayawada.

| Step | Done when |
|---|---|
| 6.2.1 Pick the client by week 9; get an intro through guide or T&P | Meeting booked |
| 6.2.2 Proposal adapted to their use case | Sent, week 10 |
| 6.2.3 Engagement letter | Signed, week 11 |
| 6.2.4 Scoped deliverable — for T&P: a batch-screening report on a de-identified sample set + guidelines; for a college: same pilot as C1 | Delivered, week 14 |
| 6.2.5 Completion letter | **D6 done**, week 15–16 |

---

# PHASE 7 — PRODUCT, FINAL REVIEW, CLOSE-OUT (weeks 13–16)

| Step | Owner | Output | Done when |
|---|---|---|---|
| 7.1 Django app complete: upload (text/PDF/DOCX), language auto-detect, three-way verdict with confidence, explanation panel, batch mode, admin audit log of every decision | P3 | Deployed app | All FRs from Review-1 spec ticked |
| 7.2 Reproducibility README: one command to rerun every experiment from raw data | P2 | README | A teammate reproduces T2 from scratch |
| 7.3 Final report: Review-1 report expanded with all results, IP section, consultancy section, individual contributions | P1 | PDF | Guide-approved |
| 7.4 Final viva deck (20 slides) + 3-minute demo video | P1 + P3 | Deck + MP4 | Rehearsed twice |
| 7.5 Evidence folder `docs/hte/`: paper receipt, 2 patent receipts, 2 engagement letters, 2 completion letters, rubrics, all review feedback | P2 | Folder complete | Every deliverable has a document |
| 7.6 GitHub cleanup, tag `v1.0`, licence file, corpus release (de-identified subset) | P3 | Public repo | Tagged |

---

## MASTER TIMELINE

| Week | Dates | Phase | Milestone |
|---|---|---|---|
| 1–2 | 2–16 Sep | 1 | **Review-1** |
| 3–4 | 17–30 Sep | 2 | Baselines + splits frozen; corpus 50% |
| 5–6 | 1–14 Oct | 2 | Corpus frozen; Head A + Head B results; student-data go/no-go |
| 7–8 | 15–23 Oct | 2 | Fusion beats baselines; Patent 2 v0; IPR cell contacted; C1 client met |
| 9–10 | 26 Oct–6 Nov | 3, 4, 5, 6 | Conformal + abstention; venue chosen; prior-art done; engagement letters |
| 11–12 | 9–20 Nov | 3, 4, 5, 6 | Adversarial + fairness + ablation; results frozen; IDFs submitted; pilots running. **Review-2 (likely)** |
| 13–14 | 23 Nov–4 Dec | 4, 6, 7 | Paper full draft; consultancy reports delivered; app complete |
| 15–16 | 7–18 Dec | 4, 5, 7 | Paper submitted; patents filed; completion letters; **Final review** |

---

## RISK REGISTER

| Risk | Trigger | Action |
|---|---|---|
| Curvature dead on Telugu | 2.2.4 shows AUROC <0.6 | Reframe RQ1 as a negative finding; fusion carries the result; paper still stands |
| Corpus late | Week 6, <1,000 human per bucket | Cut to 3 buckets (drop the weakest) and say so; never cut calibration |
| Ethics never clears | Week 6 gate | Proxy corpora only; already the primary plan |
| Patent definition stricter than assumed | Phase 0.1 answer | Phase 5 already plans for the strictest case |
| No external consultancy client | Week 10, nothing signed | Both consultancies internal (T&P + exam cell) — confirm acceptable in 0.2 |
| P1 overloaded | Week 9 | P3 takes Patent 1 paperwork; P2 takes paper §4.3/4.5 |
| Review-2 rubric weights something unplanned | Phase 0.3 answer | Adjust Phase 3 in week 9 planning |
| Everything slips to December | Week 12, results not frozen | Paper goes to a journal with no deadline; patents stop at IDF; app stays minimal. Deliverables still complete |

---

## WHAT "FULLY DONE" LOOKS LIKE

Every row in the deliverable tracker ticked, `docs/hte/` holds a document for each, the repo is tagged `v1.0`, and any one of the three of you can present the whole thing alone for 20 minutes without notes.
