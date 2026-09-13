# TWO-WEEK SPRINT — full scope
**Start:** Thu 4 Sep 2026 · **Target:** Wed 17 Sep · Solo, Claude Code in auto mode, GPU runs overnight

## Scope — nothing cut except one thing
Everything in `docs/execution-plan.md` Part 3 and `docs/master-execution-plan.md` Phases 2–3 is in. The only omission is **real student submissions**, because consent/ethics approval cannot clear in two weeks. The paper's L1/L2 fairness analysis uses public Indian-English human text vs general English human text as the proxy, and the Limitations section says so.

| Component | Full scope, as planned |
|---|---|
| Buckets | en, hi, te, cm |
| Human text | ≥ 1,000 calibration + ≥ 1,000 train/test per bucket; pre-2022 sources; plus an Indian-English sub-bucket for the L1/L2 split |
| Machine text | 3 seen generators (Qwen2.5-7B-Instruct, Gemma-2-9B-it, Mistral-7B-Instruct) + 2 held-out (Llama-3.1-8B-Instruct, Phi-3.5-mini) — all 4-bit on the 4060 |
| Heads | A stylometric (with real spaCy/Stanza POS + dependency depth), B curvature (mGPT), C semantic (MuRIL) |
| Baselines | perplexity threshold, DetectGPT (perturbation, T5-based), Fast-DetectGPT (English scorer), Binoculars, XLM-R supervised |
| Fusion | logistic regression vs LightGBM, chosen on calibration AUROC |
| Calibration | temperature scaling → per-bucket split-conformal, α ∈ {0.01, 0.05} |
| Abstention | risk–coverage sweep |
| Attacks | LLM paraphrase, back-translation (IndicTrans2 en↔hi/te; en via hi for cm), 20%-human-edit hybrid (programmatic synonym/reorder edits) |
| Tables | T1, T2, T3, T4, T5 (bucket × L1/L2), T6, F1, F2 (fertility vs AUROC) |

**The design that makes this fit two weeks:** every text is scored once by every method into `results/scores.parquet`; all tables are pandas over that file. The expensive parts (generation, scoring) run overnight unattended.

## GPU budget (RTX 4060 8GB, realistic)
| Job | Estimate |
|---|---|
| Machine text: ~3,000 prompts × 5 generators × ~250 tokens, 4-bit, batch 8 | 10–14 h |
| Back-translation: ~6,000 texts × 2 passes, IndicTrans2-1B | 2–3 h |
| Paraphrase: ~6,000 texts | 3–4 h |
| Scoring: ~20,000 rows × {mGPT, Qwen, Binoculars pair, MuRIL, XLM-R} | 4–6 h |
| DetectGPT baseline (100 perturbations): test + cal splits only | 3–4 h |
| Total | ~25–30 GPU-hours → three overnight runs |

## Day plan
| Day | Date | Overnight GPU (autonomous) | Daytime |
|---|---|---|---|
| 1 | Thu 4 | **Autopilot A1** — human corpora, fertility, splits; generation (seen 1–3) | Email HTE re patent/consultancy; literature table 18 rows |
| 2 | Fri 5 | **A2** — held-out generators, clean, length-match, freeze | Review A1 counts; corpus card |
| 3 | Sat 6 | **B** — score everything (heads A/B/C + 5 baselines) | — |
| 4 | Sun 7 | **C** — fusion, temperature, conformal, abstention → T1 T2 T3 T5 T6 F1 F2 | Read tables; note surprises |
| 5 | Mon 8 | **D** — three attacks, rescore → T4, refresh all tables | Paper outline; venue chosen |
| 6 | Tue 9 | Re-runs for anything that broke | Wire final pipeline into Django: real verdicts, real abstention |
| 7–8 | Wed 10–Thu 11 | Buffer | I draft Method / Setup / Results from your tables; you review |
| 9–10 | Fri 12–Sat 13 | — | Related Work, Intro, Abstract, Limitations, Ethics → paper v1 |
| 11 | Sun 14 | — | Guide reads paper; Patent 1 & 2 disclosure drafts |
| 12 | Mon 15 | — | Results-led deck (20 slides), demo hardening |
| 13 | Tue 16 | — | Revise paper; rehearse; tag v1.0 |
| 14 | Wed 17 | **Review** | |

## Hard stops (Claude Code pauses and reports; everything else runs through)
1. Gated HF model → skip to the listed fallback, note it, continue.
2. GPU OOM → halve batch, retry once; if still OOM, drop to 4-bit or the next-smaller listed model; note it.
3. Any bucket < 800 human texts after all sources → report and continue.
4. Any single scorer taking > 6 h projected → score test+cal splits fully, train split subsampled to 2,000 per bucket; note it.

---

## AUTOPILOT A1 — human corpora, fertility, generation part 1

```
Read docs/project-context-master.md and docs/sprint-plan.md fully. Full
scope per the sprint plan. Auto mode. Use .\venv\Scripts\python.exe for
everything. Commit after each numbered step. Append counts and timings to
docs/progress.md at each step. Only pause on the hard stops in the sprint
plan; otherwise run to the end.

1. Install into the venv and pin in requirements.txt: datasets, sentencepiece,
   protobuf, pyarrow, lightgbm, stanza, spacy + en_core_web_sm, ctranslate2
   (for IndicTrans2 if available, else transformers path), evaluate, nltk
   (+ wordnet), datasketch (MinHash). Download stanza models for hi and te.

2. HUMAN TEXT. Target 2,200 per bucket (1,000 cal + 1,200 train/test),
   minimum 800. Pre-2022 provenance only. Chunk to 120–300-word passages;
   max one passage per source document; dedupe by MinHash. Record source,
   licence, count in docs/data/corpus_card.md.
   - en_general: HC3 human (reddit_eli5, open_qa, wiki_csai) + wikimedia/wikipedia
     20231101.en restricted to low page ids (pre-2020 creation proxy) — 1,100.
   - en_indian (L1/L2 proxy): English side of ai4bharat/samanantar
     (Indian-origin parallel text) + any pre-2022 Indian news corpus on HF — 1,100.
     Both form bucket "en" with writer_L1_band ∈ {general, indian}.
   - hi: ai4bharat/IndicCorpV2 Hindi; fallback wikimedia/wikipedia 20231101.hi.
   - te: ai4bharat/IndicCorpV2 Telugu; fallback wikimedia/wikipedia 20231101.te.
   - cm: LingoIITGN/COMI-LINGUA human Hinglish; add cmu_hinglish_dog and any
     HinGE mirror on HF; filter ≥ 15 words, Roman-script Hindi content ≥ 30%.

3. TOKENIZER FERTILITY on 200 human sentences per bucket for: ai-forever/mGPT,
   Qwen/Qwen2.5-0.5B, meta-llama/Llama-3.1-8B (tokenizer only; skip if gated),
   google/gemma-2-2b (tokenizer only). Save results/tokenizer_fertility.csv.
   Lock Head B scorer in configs/models.yaml: mGPT; if its hi or te fertility
   > 4.0 tokens/word, flag it in docs/decisions.md but keep mGPT.

4. PROMPT-MATCHED GENERATION, part 1 — the three SEEN generators, each
   for 100% of human passages in every bucket. Prompt = first sentence of the
   human passage + "Write about N words in <language> on this topic,
   continuing naturally." For cm: "in casual Hindi-English Hinglish, Roman
   script, the way students text." Match N to the human passage's length bin.
   4-bit bitsandbytes, batch 8, temperature 0.8, top_p 0.95. Free GPU between
   models. Save raw to data/raw/machine/<generator>.jsonl.
   - Qwen/Qwen2.5-7B-Instruct
   - google/gemma-2-9b-it   (gated → google/gemma-2-2b-it)
   - mistralai/Mistral-7B-Instruct-v0.3   (gated → HuggingFaceH4/zephyr-7b-beta)
   Log tokens/sec and ETA per model at start; if projected total > 14 h,
   reduce to 80% of prompts per generator and note it.

5. Commit "data: human corpora + seen generators". Print the count table and stop.
```

## AUTOPILOT A2 — held-out generators, clean, freeze

```
Read docs/sprint-plan.md. Auto mode. venv python. Commit per step.

1. HELD-OUT GENERATORS, for 60% of human prompts per bucket:
   - meta-llama/Llama-3.1-8B-Instruct 4-bit  (gated → NousResearch/Meta-Llama-3.1-8B-Instruct;
     if that fails → HuggingFaceTB/SmolLM2-1.7B-Instruct. Never a Qwen model —
     same family as a seen generator.)
   - microsoft/Phi-3.5-mini-instruct
2. CLEAN: src/data/clean_artifacts.py — strip preambles in all four
   languages ("Sure", "Here is", "Certainly", "ज़रूर", "यहाँ", "ఖచ్చితంగా",
   etc.), markdown, trailing refusals, meta-commentary; drop < 40 words or
   wrong-script > 40% for the bucket. Report % dropped per generator × bucket.
3. LENGTH-MATCH: 5 word-count bins per bucket; downsample machine per
   generator to the human bin distribution. Report final counts.
4. FREEZE: src/data/freeze_splits.py — per bucket: human → 1,000 calibration
   (human only, never touched again), remaining human 50/50 train/test;
   machine-seen → 50/50 train/test; machine-held-out → 100% test. Overwrite
   guard on splits.json. Write data/processed/corpus.jsonl with the full
   schema. Complete docs/data/corpus_card.md.
5. Commit "data: corpus v1 frozen". Print bucket × {human, gen1..gen5} ×
   {cal, train, test} table and stop.
```

## AUTOPILOT B — score everything

```
Read docs/sprint-plan.md. Auto mode. venv python. Commit per column.
GOAL: results/scores.parquet — one row per corpus text; columns: id, bucket,
label, generator, split, length_words, writer_L1_band, attack_type, plus one
column per method below. Cache to parquet after every column.

1. Head A upgrade: replace POS/syntax proxies with real stanza (hi, te) and
   spaCy (en; cm via en + heuristics) POS n-grams and mean dependency depth.
   Keep the 25 original features, add these → ~40 features. Fit a per-bucket
   logistic head on the train split → column `headA`. Save models to
   results/models/.
2. `headB`: Fast-DetectGPT with the locked mGPT scorer, fp16, batch 8.
3. `headC`: MuRIL-base mean-pooled embeddings → per-bucket logistic head fit
   on train → column `headC`.
4. Baselines, same interface, higher = machine:
   `ppl` negated mean log-perplexity (Qwen2.5-0.5B);
   `fastdetectgpt_en` Fast-DetectGPT with Qwen2.5-0.5B (English-default
   scorer — shows the multilingual-scorer effect vs headB);
   `binoculars` observer Qwen2.5-0.5B / performer Qwen2.5-0.5B-Instruct;
   `xlmr` xlm-roberta-base fine-tuned 2 epochs on the train split, pooled
   across buckets; save the checkpoint;
   `detectgpt` vanilla with T5-large mask-filling, 100 perturbations,
   TEST + CAL splits only (hard stop 4 applies to train).
5. Sanity table: single-column AUROC per bucket on test (seen generators).
   Append to docs/progress.md. Commit "scores: all methods cached". Stop.
```

## AUTOPILOT C — fusion, calibration, tables

```
Read docs/sprint-plan.md. Auto mode. All from results/scores.parquet; no
model runs.

1. src/fusion/fuser.py: features [headA, headB, headC, bucket one-hot].
   Fit logistic regression AND LightGBM (100 trees, depth 3) on train; pick
   by calibration-split AUROC; column `fused`. Record choice in decisions.md.
2. src/calibration/temperature.py: per-bucket temperature on the
   calibration split → `fused_cal`.
3. src/calibration/conformal.py: per bucket, calibration split (human only),
   α ∈ {0.01, 0.05}: tau_α = ceil((n+1)(1−α))-th smallest fused_cal score.
   Also fit a GLOBAL tau on pooled calibration for every baseline and for
   fused (needed for T5). Save results/calibration.json.
4. src/calibration/abstention.py: MACHINE if > tau_0.01; HUMAN if below the
   calibration median; else ABSTAIN. Sweep lower threshold for coverage
   30→100%. Save risk–coverage arrays per bucket.
5. Tables → results/*.csv AND docs/results/*.md, 2-sentence takeaway each:
   T1 main: AUROC + F1@own-tau for every method, per bucket + overall; test,
      seen generators.
   T2 ablation: A / B / C / A+B / A+B+C / +temperature / +conformal (FPR,
      TPR @α=0.01) / +abstain (risk @70% coverage) — per bucket.
   T3 held-out generators: T1 columns on held-out rows only, and per generator.
   T5 fairness: FPR at α=0.01 and 0.05 per bucket AND for en split by
      writer_L1_band (general vs indian), for every method with a global
      threshold vs ours with per-bucket tau. Report the general–indian FPR gap.
   T6 calibration: ECE per bucket before/after temperature; accuracy at
      50/70/90% coverage.
   F1: risk–coverage curves, buckets overlaid (PNG).
   F2: tokenizer fertility vs headB AUROC vs fastdetectgpt_en AUROC per bucket.
6. docs/results/README.md: one paragraph per table with key numbers.
   Commit "results: T1 T2 T3 T5 T6 F1 F2". Stop.
```

## AUTOPILOT D — adversarial

```
Read docs/sprint-plan.md. Auto mode. venv python.

1. On machine TEST rows, all buckets, produce three attacked copies:
   a) paraphrase: Qwen/Qwen2.5-7B-Instruct 4-bit, "Rewrite in your own
      words, same language, same length"; attack_type="paraphrase".
   b) back_translation: ai4bharat/indictrans2-en-indic-1B and
      indictrans2-indic-en-1B (transformers path): en→hi→en for en; hi→en→hi
      for hi; te→en→te for te; for cm: Devanagari-normalise via
      indic-nlp-library transliteration, hi→en→hi, re-romanise.
      attack_type="backtranslation".
   c) hybrid: programmatic 20% edit — for 20% of sentences apply one of
      {synonym swap (WordNet for en; small Indic synonym lists for hi/te),
      clause reorder, sentence split/merge}; attack_type="hybrid".
   Ids suffixed _para / _bt / _hyb. Also attack 300 HUMAN test rows per
   bucket the same way (to measure FPR under attack).
2. Score all new rows with every column (reuse Autopilot B code); append.
3. T4 adversarial: AUROC, TPR@tau_0.01 and FPR@tau_0.01 for every method,
   clean vs each attack, per bucket. Add an "under attack" block to T2.
   Re-run Autopilot C tables so everything is consistent.
4. Update docs/results/README.md. Commit "results: T4 adversarial". Stop.
```

---

## After D — send me
`docs/results/README.md`, all table CSVs, F1/F2 PNGs, `docs/data/corpus_card.md`, `configs/models.yaml`, `docs/decisions.md`. I write the paper from those.
