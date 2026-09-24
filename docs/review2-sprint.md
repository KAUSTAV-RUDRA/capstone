# REVIEW-2 SPRINT — 7 days
**Decision: pause generation. Run the full pipeline end-to-end on qwen7b + the human corpus, and walk in with real numbers.**

Generation resumes after the review. Nothing is lost — the corpus is frozen, resumable, and gemma/mistral ratios are already measured.

---

## Why this pivot

| Rubric item | Marks | Where you stand today | Where you stand if you keep generating | Where you stand after this sprint |
|---|---|---|---|---|
| 1 Implementation progress | 15 | Generation only | Generation only, 45% done | Full pipeline, every module working |
| 2 Code quality | 15 | Strong | Strong | Strong |
| 3 Integration & end-to-end | 15 | **Placeholder Django** | Placeholder Django | Real verdicts flowing src → webapp |
| 4 Testing & validation | 15 | **40+ tests, 5-check gate** | Same | Same + a written test plan |
| 5 Results & performance | 15 | **None** | None | T1, T2, T5, T6, F1, F2 with real numbers |
| 6 Innovation | 10 | **Strong story** | Strong | Strong |
| 7 Documentation & Git | 10 | **Strong** | Strong | Strong |
| 8 Presentation | 5 | — | — | Live demo |

Items 3 and 5 are 30 marks and both are currently zero. This sprint converts them.

**What you give up:** T3 (held-out generators) needs llama/phi, which don't exist yet. That's fine — it becomes "Future Work / next sprint" on slide 15, which the rubric explicitly asks for.

---

## Day 1 — Freeze the corpus (Part 13) · ~1 h, low GPU

```
Part 13 on what exists. Do NOT wait for gemma/mistral.

1. src/data/clean_artifacts.py — implement the spec already in its docstring:
   strip_boilerplate (shared with the gate), then drop rows with emoji,
   instruction echo, truncation, or off-script content above the human
   baseline. Apply to human AND machine rows identically, so cleaning is
   never itself a class signal. Report % dropped per bucket per source.
2. Length-match: 5 word-count bins per bucket, downsample machine to the
   human bin distribution.
3. freeze_splits.py: per bucket — 1,000 human to calibration (cm draws only
   from date-certain sources, per the existing calibration_eligible flag);
   remaining human 50/50 train/test; qwen7b machine 50/50 train/test.
   Overwrite guard on splits.json.
4. Write data/processed/corpus.jsonl with the full 11-field schema.
5. Print bucket x source x split counts. Update docs/data/corpus_card.md.
   Commit "data: corpus v1 frozen (qwen7b)".
```

**Done when:** `splits.json` exists and the count table prints. This is the gate for everything else.

---

## Day 2 — Head A + Head B scored · ~3 h, GPU

```
Score the frozen corpus. Cache to results/scores.parquet after every column.

1. Head A: upgrade the POS/syntax proxies to real taggers — stanza for hi
   and te, spaCy for en and cm. Keep the existing 25 features, add POS
   n-grams and mean dependency depth (~40 total). Fit a per-bucket logistic
   head on the train split -> column "headA".
2. Head B: Fast-DetectGPT with ai-forever/mGPT, fp16 CUDA, batch 8 ->
   column "headB".
3. ALSO implement headB_word: the same statistic aggregated at WORD level
   (group tokens by source word, sum log-probs and conditional moments per
   word, then compute curvature over words). This is Patent 2's core claim
   and needs the headB vs headB_word comparison as evidence.
4. Print per-bucket AUROC for headA, headB, headB_word on the test split.
   Commit.
```

**Done when:** three columns in the parquet and a per-bucket AUROC table. **This is the first real result of the project** — screenshot it.

If time is short: score the calibration and test splits fully, subsample train to 1,000 per bucket. Note it.

---

## Day 3 — Head C + baselines · ~2.5 h, GPU

```
1. headC: google/muril-base-cased mean-pooled embeddings -> per-bucket
   logistic head fit on train.
2. ppl: negated mean log-perplexity, Qwen/Qwen2.5-0.5B.
3. fastdetectgpt_en: Fast-DetectGPT with Qwen2.5-0.5B (the English-default
   scorer) — this is the comparison that isolates the multilingual-scorer
   effect against headB's mGPT.
4. binoculars: observer Qwen2.5-0.5B, performer Qwen2.5-0.5B-Instruct.
5. Per-bucket AUROC for all of them. Commit.
```

Skip XLM-R if time is tight — it's a supervised baseline and the zero-shot ones carry T1.

---

## Day 4 — Fusion, calibration, tables · ~1.5 h, no GPU

```
Everything from results/scores.parquet.

1. fusion/fuser.py: logistic regression on [headA, headB, headC, bucket
   one-hot], fit on train -> "fused".
2. calibration/temperature.py: per-bucket temperature on the calibration
   split.
3. calibration/conformal.py: per bucket, alpha in {0.01, 0.05},
   tau = ceil((n+1)(1-alpha))-th smallest calibration score. ALSO fit a
   GLOBAL tau on pooled calibration — T5 needs both.
4. calibration/abstention.py: MACHINE > tau_0.01; HUMAN below the
   calibration median; else ABSTAIN. Sweep the lower threshold for coverage
   30->100%.
5. Tables, each a CSV in results/ and a markdown file in docs/results/:
   T1  AUROC + F1 for every method, per bucket
   T2  ablation: A / B / C / A+B / A+B+C / +temperature / +conformal / +abstain
   T5  FAIRNESS: FPR at alpha=0.01 and 0.05 per bucket, global threshold vs
       per-bucket tau; and for en split by writer_L1_band (general vs indian)
   T6  ECE per bucket before/after temperature; accuracy at 50/70/90% coverage
   F1  risk-coverage curves, four buckets overlaid (PNG)
   F2  tokenizer fertility vs headB AUROC vs fastdetectgpt_en AUROC per bucket
   Plus headB vs headB_word per bucket (Patent 2 evidence).
6. docs/results/README.md: one paragraph per table with the key numbers.
   Commit "results: T1 T2 T5 T6 F1 F2".
```

**This is the day that matters most.** T5 is your headline: does per-bucket calibration beat a global threshold, and by how much?

**Send me docs/results/ the moment it exists.** I fill in the paper and both patents that evening.

---

## Day 5 — Integration: wire the real pipeline into Django · ~2 h

This is rubric item 3, 15 marks, currently zero.

```
Wire webapp/detector/services.py to the real pipeline. analyse_text() must:
1. Run language_id -> bucket (implement the phase-2 stubs: script detection
   by Unicode block, romanised-Hindi detection via the existing function-word
   lexicon, code-mix ratio).
2. Run headA, headB, headC on the text.
3. Fuse, apply temperature, apply that bucket's conformal tau from
   results/calibration.json.
4. Return verdict HUMAN/ABSTAIN/MACHINE, confidence, driving head, top-5
   stylometric features by standardised deviation, and the tau applied.
5. Load models ONCE at module level and cache — not per request.
6. model_version = "v1.0-qwen7b".
Result page shows real verdicts. Persist every Decision row (audit trail).
Batch page: CSV in -> table out. Dashboard: counts by verdict and bucket
from real Decision rows.
Do not touch src/. Commit.
```

**Done when:** you paste a human paragraph and get HUMAN, paste a ChatGPT paragraph and get MACHINE, with a real confidence and a real threshold. That's your live demo.

---

## Day 6 — Test plan, deck, documentation · no GPU

**Test plan** (rubric 4 — you have the tests, you need the document):

```
Write docs/test-plan.md from the tests that already exist:
- Unit tests: 40+ in tests/, listed by module with what each validates.
- The generation gate as a validation harness: 5 checks (length median,
  stratification, code-mix share, emoji, token cap) with the three real
  defects each caught — cm at 1.98 length, cm at 0.03 Hindi share, gemma hi
  at 1.51 from a cross-generator ratio.
- Defect log: every defect found and its resolution, with commit hashes.
  Pull these from docs/progress.md — cm code-mixing, the bin-local probe
  defect, the strip-order bug, the selection-limit guard.
- Non-functional validation: inference latency per document, VRAM ceiling,
  reproducibility (one command per stage).
Commit.
```

**Then the deck.** Send me `docs/results/README.md` plus screenshots and I build the 17-slide deck to match the required structure.

---

## Day 7 — Rehearse

Full run-through, timed. Live demo from a browser that's already open with the server running. Backup screenshots on slide 10 in case of failure.

---

## The 17 required PPT sections → what fills each

| # | Section | Content |
|---|---|---|
| 1 | Title & team | As Review-1 |
| 2 | Problem & objectives | 61.3% FPR; the four RQs |
| 3 | Gap & proposed solution | No per-language FPR bound exists |
| 4 | Updated literature | The 20-paper table (built, in docs/lit/) |
| 5 | Architecture & design | Updated diagram with the gate in the data pipeline |
| 6 | Technologies | mGPT, MuRIL, Ollama, stanza/spaCy, sklearn, Django |
| 7 | Module-wise progress | Corpus ✓, Head A ✓, Head B ✓, Head C ✓, fusion ✓, calibration ✓, webapp ✓, generation 45% |
| 8 | Integration | src → services.py → Django; the audit-trail DB |
| 9 | Testing strategy | docs/test-plan.md; 40+ tests; the gate as a harness |
| 10 | Results & screenshots | T1, T2, F1, the Django verdict pages |
| 11 | Performance analysis | Per-bucket AUROC; FPR bound holding; latency; coverage cost |
| 12 | Challenges & solutions | **Your strongest slide** — see below |
| 13 | Innovation | Per-language conformal; word-level curvature; the gate |
| 14 | Current status | This table, honestly |
| 15 | Future work | Remaining 4 generators, T3 held-out, adversarial T4, consultancy pilot |
| 16 | Conclusion | The guarantee, demonstrated |
| 17 | References | The 20-paper table |

**Slide 12 (Challenges) writes itself and reviewers love it.** Four real ones with numbers:
1. *32 h → 5 h generation* — HF/bitsandbytes to Ollama with a persistent worker pool.
2. *Code-mixed text wasn't code-mixed* — machine cm scored 0.03 Hindi function-word share vs 0.30 human; caught by a gate check we built, fixed by template, now 0.35.
3. *Compliance ratios don't transfer across models* — qwen needed 0.55 on Hindi, gemma 0.73, mistral 0.82; a config keyed on bucket alone silently applied qwen's to all, producing 1.51× length. Fixed by making it per-generator.
4. *Mistral is structurally unsuited to Telugu* — 13.3 tokens/word pins every request at the token cap; dropped that bucket for that generator with the reason recorded.

Each is a measurement, a diagnosis, and a fix. That is exactly what rubric 6 asks for.

---

## What to say about generation being incomplete

Don't hide it. Say it in this order, on slide 14:

> "We have five generators planned; one is complete and the second is 45% done. We deliberately paused generation to run the full detection pipeline end-to-end on the complete generator, because a working system with real numbers is more valuable than a larger corpus with no results. Generation resumes immediately after this review, and the remaining generators are straight runs — every template and ratio is already measured and gate-validated."

That's a defensible engineering decision, not a shortfall.
