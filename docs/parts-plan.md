# PART-BY-PART PLAN — every part ≤ 2 hours, every part resumable
Run parts in order. Each has: what it does, how long, the exact Claude Code prompt, and how you know it's done.
Rule for every part: Claude Code in normal (not auto) mode unless the part says otherwise; commit at the end; paste me the "done when" output.

Three commands do most of the heavy lifting. **Part 4 builds them once**; later parts just run them. Each writes progress to disk continuously and stops itself at `--max-minutes`. Re-running the same command resumes.

```
python -m src.data.generate --generator <qwen7b|gemma|mistral|llama|phi> --buckets en,hi,te,cm --max-minutes 110
python -m src.eval.score    --column <name> --max-minutes 110
python -m src.data.attack   --attack <paraphrase|backtranslation|hybrid> --max-minutes 110
```

---

## STAGE 1 — HUMAN CORPORA  (Parts 1–3, ~4 h total, no GPU)

### Part 1 — English human text (1 h)
```
Implement src/data/build_human_corpus.py as a resumable CLI:
  python -m src.data.build_human_corpus --bucket <en|hi|te|cm> --target 2200
It pulls sources for the bucket, chunks to 120–300-word passages (one per
source doc), dedupes with MinHash (datasketch), tags language/domain/
writer_L1_band/source, and appends to data/raw/human/<bucket>.jsonl —
skipping ids already present so it can be re-run. Install datasets,
datasketch, pyarrow; pin them.
Sources for en (pre-2022 only):
  - writer_L1_band=general: HC3 human answers (Hello-SimpleAI/HC3: reddit_eli5,
    open_qa, wiki_csai) + wikimedia/wikipedia 20231101.en low page-ids — 1,100
  - writer_L1_band=indian: English side of ai4bharat/samanantar — 1,100
Run it for en. Print count per writer_L1_band. Commit.
```
Done when: `data/raw/human/en.jsonl` has ≥ 1,800 rows, both bands present.

### Part 2 — Hindi + Telugu human text (1 h)
```
Add hi and te sources to build_human_corpus.py:
  - hi: ai4bharat/IndicCorpV2 config hi (stream, take first 20k docs);
        fallback wikimedia/wikipedia 20231101.hi
  - te: same for Telugu
Run for hi, then te, target 2200 each. Print counts. Commit.
```
Done when: hi.jsonl and te.jsonl each ≥ 1,800 rows.

### Part 3 — Code-mixed human text + corpus card (1 h)
```
Add cm sources: LingoIITGN/COMI-LINGUA (human Hinglish, Roman script);
cmu_hinglish_dog; search HF for a HinGE mirror. Filter ≥ 15 words and
Roman-script Hindi content ≥ 30% (use a Hindi-word list). Run for cm,
target 2200 (accept ≥ 800). Then write docs/data/corpus_card.md: per
bucket — sources, licence, count, band split, mean length. Commit.
```
Done when: cm.jsonl ≥ 800 rows; corpus_card.md written.

---

## STAGE 2 — TOOLS + SCORER LOCK  (Part 4, ~1.5 h)

### Part 4 — Tokenizer fertility + the three resumable commands (1.5 h)
```
1. scripts/tokenizer_fertility.py: for ai-forever/mGPT, Qwen/Qwen2.5-0.5B,
   google/gemma-2-2b, meta-llama/Llama-3.1-8B (tokenizer only, skip if
   gated): mean tokens/word on 200 human sentences per bucket from
   data/raw/human/. Save results/tokenizer_fertility.csv. Lock mGPT as
   scorer in configs/models.yaml; note fertility in docs/decisions.md.
2. Build three RESUMABLE CLIs. Each: loads its input, finds which ids are
   already in its output file, processes only the rest in batches of 8,
   appends to the output after every batch, and exits cleanly when
   --max-minutes is reached, printing "done N of M, resume with same
   command". GPU 4-bit via bitsandbytes where the model is > 3B.
   a) src/data/generate.py — args --generator (HF id), --buckets,
      --fraction, --max-minutes. Prompt-matched: first sentence of the
      human passage + "Write about N words in <language> on this topic,
      continuing naturally"; cm: "casual Hindi-English Hinglish, Roman
      script, the way students text". Output data/raw/machine/<gen>.jsonl.
   b) src/eval/score.py — args --column, --max-minutes. Reads
      data/processed/corpus.jsonl (and attacked rows), writes/updates
      results/scores.parquet one column at a time. Column implementations
      are registered functions; stub them now, fill in Stage 4.
   c) src/data/attack.py — args --attack, --max-minutes. Stub now, fill in
      Stage 6.
3. Dry-run generate.py with Qwen/Qwen2.5-0.5B-Instruct on 8 prompts to
   prove resume works (run, kill, re-run). Commit.
```
Done when: fertility CSV exists; generate dry-run resumes correctly.

---

## STAGE 3 — MACHINE TEXT  (Parts 5–12, each ≤ 2 h, GPU)
Every part is the same command; you run it until it prints "complete". If it stops at 110 minutes, that's one part; run it again next sitting — that's the next part. Generation runs on local **Ollama**, Q4_K_M for all five generators (docs/decisions.md 2026-09-13 "Machine text").

**One-time setup**
```
setx OLLAMA_NUM_PARALLEL 4               # then Quit Ollama from the tray and reopen it
                                         # check: %LOCALAPPDATA%\Ollama\server.log shows OLLAMA_NUM_PARALLEL:4
ollama pull qwen2.5:7b-instruct
ollama pull gemma2:9b-instruct-q4_K_M
ollama pull mistral:7b-instruct
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
python scripts/tokenizer_fertility.py    # fertility of each generator's tokenizer -> its token budget
```

**Who generates what.** Each human passage gets exactly one seen generator (`sha256(prompt_id) mod 3`), so each seen run covers about a third of every bucket. The two held-out generators each take a disjoint 30 % slice from a second hash and are test only. No `--fraction`: the assignment sets the size. Generator aliases, Ollama tags and HF fallbacks live in `configs/data.yaml` under `machine_corpus.generators`.

**Token budget.** `num_predict = requested words x fertility(bucket) x 1.3 + 64`, capped at 2048. Rows that still hit it carry `truncated: true` for Part 13.

| Part | Command | Role | Prompts |
|---|---|---|---|
| 5–6 | `python -m src.data.generate --generator qwen7b --buckets en,hi,te,cm --max-minutes 110` | seen | 2,874 |
| 7–8 | `... --generator gemma ...` | seen | 2,997 |
| 9–10 | `... --generator mistral ...` | seen | 2,929 |
| 11 | `... --generator llama ...` | held-out | 2,727 |
| 12 | `... --generator phi ...` | held-out | 2,605 |

Sittings per generator are re-projected from each first run. Four worker threads keep the server's 4 slots busy, and every 8 completed requests log tokens/sec and effective concurrency: about `x4` is the target; `x1.0` means the server is not running 4 slots, so fix the setup, then resume.

**Fallback.** If Ollama cannot serve a generator: `--backend hf [--model <hf id>]` (transformers + bitsandbytes 4-bit). It replaces that generator's whole file; generate.py refuses to append to a file another backend wrote. Note it in docs/decisions.md.

Tell Claude Code: "Run this command. When it exits, tell me tokens/sec and done-of-total." Commit after each sitting.

Done when: all five `data/raw/machine/<alias>.jsonl` files report "complete".

---

## STAGE 4 — CLEAN, FREEZE, SCORE  (Parts 13–18)

### Part 13 — Clean, length-match, freeze (1 h, no GPU)
```
1. src/data/clean_artifacts.py: strip preambles in en/hi/te ("Sure", "Here
   is", "Certainly", "ज़रूर", "यहाँ है", "ఖచ్చితంగా", "ఇక్కడ"), markdown,
   trailing refusals; drop < 40 words or wrong-script > 40%. Report %
   dropped per generator × bucket.
2. Length-match: 5 word-count bins per bucket; downsample machine per
   generator to the human bin distribution.
3. src/data/freeze_splits.py: per bucket — human: 1,000 calibration, rest
   50/50 train/test; seen-machine 50/50 train/test; held-out 100% test.
   Overwrite guard. Write data/processed/splits.json and corpus.jsonl.
4. Print bucket × generator × split counts. Update corpus_card.md. Commit
   "data: corpus v1 frozen".
```
Done when: counts table printed; splits.json exists.

### Part 14 — Head A upgrade + headA column (2 h)
```
Install stanza (download hi, te models) and spacy en_core_web_sm; pin.
Extend src/features/stylometric.py: real POS n-grams and mean dependency
depth via stanza (hi, te) / spaCy (en, cm) — keep the 25 existing features,
total ~40. Register column "headA" in src/eval/score.py: extract features
for all rows, fit per-bucket logistic regression on train split, score all
rows. Save models to results/models/. Run:
  python -m src.eval.score --column headA --max-minutes 110
Print per-bucket AUROC on test. Commit.
```

### Part 15 — headB (mGPT curvature) (2 h, may need 2 sittings)
```
Register "headB": Fast-DetectGPT with ai-forever/mGPT fp16 CUDA batch 8.
Run: python -m src.eval.score --column headB --max-minutes 110
Print per-bucket AUROC on test when complete. Commit.
```

### Part 16 — headC (MuRIL) (1 h)
```
Register "headC": google/muril-base-cased mean-pooled embeddings →
per-bucket logistic head fit on train. Run score --column headC. AUROC. Commit.
```

### Part 17 — Zero-shot baselines (1.5 h)
```
Register and run three columns (higher = machine):
  "ppl": negated mean log-perplexity, Qwen/Qwen2.5-0.5B
  "fastdetectgpt_en": Fast-DetectGPT with Qwen/Qwen2.5-0.5B (English-default scorer)
  "binoculars": observer Qwen2.5-0.5B, performer Qwen2.5-0.5B-Instruct
Run each with --max-minutes 35. AUROCs. Commit.
```

### Part 18 — Supervised baseline XLM-R (1.5 h)
```
Register "xlmr": fine-tune xlm-roberta-base 2 epochs on the train split
(all buckets pooled), max_len 256, batch 16 fp16; save checkpoint to
results/models/xlmr/; score all rows. Run. AUROC. Commit.
```

Done when Stage 4 ends: `results/scores.parquet` has all 7 method columns for all rows.

> Parts 19–20 (vanilla DetectGPT) were **deleted on 2026-09-18** — Fast-DetectGPT
> is its efficient successor with an identical statistic, so running both cost two
> GPU sittings for no extra evidence. Baselines are ppl, fastdetectgpt_en,
> binoculars, xlmr. See decisions.md.

---

## STAGE 5 — FUSION, CALIBRATION, TABLES  (Parts 21–22, no GPU)

### Part 21 — Fusion + calibration + abstention (1 h)
```
From results/scores.parquet only:
1. src/fusion/fuser.py: [headA, headB, headC, bucket one-hot] → logistic
   regression AND LightGBM (100 trees, depth 3) on train; choose by
   calibration AUROC; column "fused". Record in decisions.md.
2. src/calibration/temperature.py: per-bucket temperature on calibration → "fused_cal".
3. src/calibration/conformal.py: per bucket, α ∈ {0.01, 0.05},
   tau_α = ceil((n+1)(1−α))-th smallest calibration score. Also GLOBAL tau
   on pooled calibration for every baseline and for fused. Save
   results/calibration.json.
4. src/calibration/abstention.py: MACHINE > tau_0.01; HUMAN < calibration
   median; else ABSTAIN; sweep lower threshold coverage 30→100%; save
   risk–coverage arrays. Commit.
```

### Part 22 — All tables except T4 (1 h)
```
Produce results/*.csv and docs/results/*.md (2-sentence takeaway each):
 T1 main (AUROC + F1@own-tau, every method, per bucket + overall, test, seen)
 T2 ablation (A / B / C / A+B / A+B+C / +temp / +conformal FPR,TPR@0.01 / +abstain risk@70%)
 T3 held-out (T1 columns on held-out rows, per generator)
 T5 fairness (FPR@0.01/0.05 per bucket and for en by writer_L1_band, every
    method global-tau vs ours per-bucket-tau; report the general–indian gap)
 T6 calibration (ECE before/after temperature; acc at 50/70/90% coverage)
 F1 risk–coverage PNG; F2 fertility vs headB vs fastdetectgpt_en AUROC PNG
docs/results/README.md: one paragraph per table with key numbers. Commit.
```
**→ Send me docs/results/ at this point. I start the paper while you do Stage 6.**

---

## STAGE 6 — ADVERSARIAL  (Parts 23–26, GPU)

### Part 23 — Attack CLI + hybrid attack (1 h, no GPU)
```
Fill in src/data/attack.py. Input: TEST-split rows only, sampled per bucket
per attack — <=400 machine + 150 human, stratified by length bin and by
generator so every generator stays represented. "hybrid": 20% of sentences get one of {synonym
swap (WordNet for en; small hi/te synonym lists), clause reorder,
split/merge}. Ids suffixed _hyb, attack_type="hybrid", appended to
data/processed/attacked.jsonl. Run it. Commit.
```
### Part 24 — Paraphrase attack (1 × 2 h)
```
python -m src.data.attack --attack paraphrase --max-minutes 110
(Qwen/Qwen2.5-7B-Instruct 4-bit: "Rewrite in your own words, same
language, same length"; suffix _para)
```
### Part 25 — Back-translation attack (1 × 1.5 h)
```
python -m src.data.attack --attack backtranslation --max-minutes 90
(ai4bharat/indictrans2-en-indic-1B + indic-en-1B: en→hi→en, hi→en→hi,
te→en→te; cm: Devanagari-normalise via indic-nlp-library, hi→en→hi,
re-romanise; suffix _bt)
```
### Part 26 — Rescore attacked rows (2 h)
```
python -m src.eval.score --all-columns --only-attacked --max-minutes 110
(run until complete). Then T4: AUROC, TPR@tau_0.01, FPR@tau_0.01 per
method, clean vs each attack, per bucket; add "under attack" block to T2;
re-run Part 22 tables. Update docs/results/README.md. Commit.
```
**→ Send me the refreshed docs/results/.**

---

## STAGE 7 — PRODUCT, PAPER, DECK  (Parts 29–35)

| Part | What | Time | Who |
|---|---|---|---|
| 29 | Wire the real pipeline into Django: services.py runs headA/B/C → fused → per-bucket conformal → verdict with confidence, abstention, explanation; result page shows real verdicts | 1.5 h | you + Claude Code |
| 30 | Batch mode + dashboard from real Decision rows; admin filters | 1 h | you + Claude Code |
| 31 | Paper v1 — I draft Method, Setup, Results, Related Work from your tables; you review | — | me |
| 32 | Paper v1 — Intro, Abstract, Limitations, Ethics; assemble full draft | — | me, you review |
| 33 | Patent 1 + 2 invention disclosure documents (from the mechanisms now built) | 1 h | me draft, you review |
| 34 | Final deck, results-led, 20 slides | — | me |
| 35 | Reproducibility README; tag v1.0; rehearse | 1 h | you |

---

## Total
GPU parts: ~11 sittings (Stages 3, 4, 6) after the 2026-09-18 scope cuts — vanilla
DetectGPT dropped (−2) and the four attack sittings collapsed to two (−2).
Non-GPU parts: ~10 sittings. Paper/deck: mine.
At one sitting a day → ~3 weeks. At two a day → ~1.5 weeks.
