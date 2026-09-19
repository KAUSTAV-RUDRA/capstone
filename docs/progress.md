# Progress log

Dated daily log — the evidence trail for Rubric #7 and every review.
Newest entries at the top.

---

## 2026-09-19 — the length gate moved into the code

The gate that decides whether a bucket's prompt regime is working existed only
as a rule in `decisions.md` and an ad-hoc watcher run by hand in a second
terminal. That is the blind spot that let `cm` reach 470 bad rows: a check that
is a habit rather than code runs only when someone remembers to run it, and the
version that ran tested only the lower bound and waved a 1.72 through.

- **It is now `run_with_gates` in `src/data/generate.py`.** Each bucket is taken
  to its first 24 rows and judged *before* the rest of that bucket is generated:
  median machine/human word ratio inside **0.75–1.25**, at most **20 %** of rows
  at the token cap. A failure aborts the run and prints what failed, what it
  cost, and the command to drop the bucket's rows. Thresholds live in
  `configs/data.yaml` under `machine_corpus.gate`.
- **The statistic is the median, not the mean or a ratio of totals.** cm's
  failure was concentrated in the short passages (21 human words → 84 machine
  words), which a length-weighted ratio of totals dilutes and one runaway row
  drags a mean around: on the 470 discarded rows the median read 1.98 to the
  mean's 1.88.
- **Probe rows are corpus rows,** written to the same file and skipped by the
  bulk phase, so a passing gate costs nothing. Rows already on disk count
  towards the 24, so a resume judges what is there rather than generating a
  fresh 24 — and a bucket whose bad rows were never cleaned up keeps failing,
  which is deliberate. There is no `--skip-gate` flag, for the same reason.
- No change to `src/utils/resumable.py`: the gate is a phase in `generate.py`,
  not a new hook in the spine `score.py` and `attack.py` also depend on.

**Verify:** `venv\Scripts\python.exe tests\data\test_generate.py` → 24 PASS
offline. The regression test drives a 4.0x bucket through `run_with_gates` and
asserts the bulk phase never runs: 24 rows, not 470.

---

## 2026-09-18 — Part 6 (in progress): qwen7b machine text

Three halted runs in one sitting, all for length-matching defects. The failures
are the useful part and some of this belongs in the paper's corpus section.

- **Run 1 halted (39 min).** Telugu per-stream decode fell to 11.7 tok/s, below
  the 15 tok/s floor. Chasing that surfaced the real problem: machine `te` was
  56 words against 153 human, **0.37**, with `done_reason: stop` on 100 % of rows
  and nothing at the cap — *under-production, not truncation*. `hi` was 0.57 on
  the same fault; `en` was fine at 0.90.
- **Fix.** A 4-variant probe showed an explicit "at least N words / N or more
  sentences / do not stop early" framing nearly doubles Indic output with no loss
  of script purity (0.96 → 1.00) and no repetition. Budget raised to
  `human_words x fertility x 1.5 + 128`, cap 3600, `num_ctx` 4608. `hi`/`te` rows
  discarded and regenerated; `en` kept — it needs no fix, so **the corpus now
  runs two prompt templates**, which the paper must state.
- **Scope cuts:** te capped at 400 and hi at 500 per generator, stratified across
  the human length bins with the ids frozen under `data/processed/selection/`;
  vanilla DetectGPT dropped (7 score columns, not 8); attacks sample the test
  split. See decisions.md.
- **Run 2 halted** to reorder buckets so the `cm`/`hi` gates would fire early.
- **Run 3 halted (470 cm rows discarded).** `cm` generated at **1.88** of human
  length — median 1.98, p90 4.94, 91 % above 1.25x. Cause: the `max(8, n/12)`
  sentence floor. Eight sentences implies ~88 words whatever the word target
  says, so a 21-word passage was asked for "29 words / 8 or more sentences" and
  wrote 84 (**4.0x**). The gate missed it because it only tested the *lower*
  bound and passed a 1.72. Fixed: `cm` uses `max(2, n/15)`, hi/te keep
  `max(8, n/12)` (their bins are all >= 100, so their prompts are unchanged and
  the 56 good `te` rows stay valid); gate now fails outside 0.75–1.25.
- **Measured, not assumed:** a cm probe showed the plain template *under*-produces
  at 0.76 — `cm` needs the sentence clause, just not that floor.
- **`hi` was probed before its 500 rows, not after 24, and it paid for itself.**
  12 passages x 3 variants: plain template **0.57**, insistent framing with the
  sentence clause **1.40**, insistent without it **1.21**. The regime `hi` was
  about to run — insistent framing plus 0.85 compensation — would have written
  **~1.59** of human length: a 1.35 over-producer with its ask inflated a further
  1.18x. That is cm's failure repeated, caught before the rows existed.
- **`hi` takes the plain template at ratio 0.55, not the best raw ratio.** On raw
  numbers the no-sentence variant (1.21) is closest to 1.0, but
  `compensated_words` only ever *inflates* the ask — it clamps `ratio <= 1.0` and
  never asks for fewer words than the human passage. A 1.17 in the config would
  be silently clamped to 1.0 and `hi` would generate at 1.21 while the config
  claimed otherwise. Relaxing that clamp was **rejected**: it widens a core
  function's contract to serve one bucket, and the alternative needs no code
  change. Since compensation can only correct *under*-production, the right
  template is the one `hi` under-produces under — the plain one, ask inflated
  x1.8, exactly the mechanism `te` already uses. Two independent measurements
  agree on the value: the probe (0.55) and the real 731-row `hi` corpus
  generated under that template (0.57).
- **The rule that generalises:** pick the template whose bias the existing
  mechanism can correct. Four buckets now run three templates and three ratios,
  and the paper's corpus section must say so — prompt regime is a per-bucket
  empirical choice, not one design applied uniformly.
- State at the end of the sitting: **en 730/730 (0.90), te 56/400 (0.94),
  cm 0/701, hi 0/500**. `data/` is gitignored, so this log is the record.

---

## 2026-09-13 — Part 4: tokenizer fertility + three resumable CLIs

- **Head B scorer LOCKED to `ai-forever/mGPT`** on measured fertility
  (`scripts/tokenizer_fertility.py`, 200 passages/bucket, tokenizers only,
  `results/tokenizer_fertility.csv`). tokens/word — mGPT 1.363 / 3.560 / 6.319 /
  1.714 across en/hi/te/cm; Qwen2.5-0.5B 1.339 / 4.804 / 11.919 / 1.672. Tied on
  en and cm, but Qwen costs **1.89x** more tokens per Telugu word and 1.35x per
  Hindi word, which directly weakens curvature in the Indic buckets. Even with
  mGPT, Telugu runs 4.64x English — that ratio is the F2 result, not just setup.
  gemma-2-2b and Llama-3.1-8B are gated and recorded as `status=gated`.
- **`src/utils/resumable.py`** — the shared spine: the output file is the
  progress record, ids already present are skipped, rows are appended and
  flushed after every batch of 8, `--max-minutes` stops between batches, and a
  failing batch is logged and skipped rather than ending a multi-hour run.
- **`src/utils/modelload.py`** — one loading rule: >3B params loads 4-bit
  bitsandbytes (nf4, double quant), everything else fp16 on CUDA. Size comes
  from a known-sizes table, then the model id, then Hub metadata; unknown is
  treated as large, since guessing fp16 for a 9B model means an OOM hours in.
- **`src/data/generate.py`** — prompt-matched machine text: the prompt is the
  human passage's first sentence (capped at 40 words so the model gets a topic,
  not a copy) and the requested length is that passage's length bin. cm gets its
  own instruction, since asking for "Hinglish" alone yields Devanagari or formal
  prose. Output carries `prompt_id` back to the human row.
- **`src/eval/score.py`** — one column per run into `results/scores.parquet`,
  via a registry; all eight columns registered as stubs that name the part that
  fills them in. Progress journals to `results/.score_<column>.jsonl` and folds
  into the parquet on exit. Convention fixed: **higher = more machine-like** for
  every column.
- **`src/data/attack.py`** — stub, but the CLI surface, id suffixes, attack_type
  values and output contract are fixed now.
- **Resume proven on the real path:** 8 en prompts through
  Qwen2.5-0.5B-Instruct at batch-size 4, process killed after the first batch
  landed (4 rows on disk), identical command re-run → "8 passages in scope, 4
  already done, 4 to do" → `complete — 8 of 8`, 8 unique ids, no duplicates.
  Found and fixed a real bug doing it (`RunReport.processed` had no default, so
  every run crashed on its first report).
- Also recorded the two cm decisions: a supplementary **40–120 word band table**
  across all buckets, and **cm calibration restricted to cmu_hinglish_dog +
  HinGE** (1,025 rows vs the 1,000 floor) with COMI-LINGUA train/test only.
  `configs/data.yaml` carries `calibration_eligible` per source and
  `freeze_splits.py` has the enforcement contract written into it.

**Verify:** run `tests/utils/test_resumable.py` and `tests/data/test_generate.py`
with the venv python → 12 PASS offline; `python -m src.eval.score --list` prints
the eight registered columns.

**Next:** Stage 3 — run `src.data.generate` for the five generators. Note that
lengths overshoot (a 150-word request returned 144–391 words), which is normal
and is exactly what Part 13's length-matching to the human bin distribution
corrects.

---

## 2026-09-13 — Part 3: cm human corpus + corpus card (human side complete)

- **hi/te domain blend first:** IndicCorpV2 capped at `max_share: 0.75`, both
  buckets rebuilt from scratch. Each is now 1,650 news_web + 550 wiki_general;
  mean length rose to 175.6 (hi) and 170.3 (te). Commit `882c1cc`.
- **cm** (`data/raw/human/cm.jsonl`): 2,200 rows, mean 55.2 words, median 25 —
  COMI-LINGUA `TN` 1,175 (social comments), cmu_hinglish_dog 917 (chat), HinGE
  108. Well past the >= 800 accept threshold. Script purity 1.000, zero
  duplicates, zero residual artefacts.
- **COMI-LINGUA is not Roman script, except in one config.** LID/MLI/NER/POS are
  Devanagari code-mixing; only `TN` has Roman-script sentences, and its raw
  `Sentences` column (not the annotator-normalised ones) is the authentic text.
- **New machinery:** per-bucket `passage_words` override (cm is 15-300, not
  120-300); a curated Romanised-Hindi function-word list with
  `min_hindi_word_ratio: 0.3` as the cm gate; `code_mix_ratio` now carries that
  measurement (median 0.39) instead of a hardcoded 0.0; `clean_informal`
  normalises chat spacing rather than rejecting it, which recovered ~20 % of the
  scarce pre-ChatGPT chat data.
- **`docs/data/corpus_card.md` written** — per bucket: sources, licences,
  counts, band split, domain split, mean/median length, cleaning stats, row
  schema, ethics, and eight numbered limitations.
- **Human corpus totals 8,800 passages**, 2,200 per bucket. Every bucket clears
  the 1,000-passage calibration floor with >= 1,200 left for train/test.

**Two things that need a decision before the tables are built.** cm averages
55.2 words against 170-211 elsewhere and is internally bimodal, so a
cross-bucket comparison confounds language with length; raising the floor to 50
words would leave only the chat source (~900 rows, single domain). And
COMI-LINGUA states no collection date, so machine text cannot be ruled out in
53 % of cm — every other source in the corpus predates ChatGPT. Both are in
`docs/decisions.md` and the corpus card's limitations.

**Verify:** `.env\Scripts\python.exe tests\data	est_build_human_corpus.py`
→ 14 × PASS (offline). Then
`.env\Scripts\python.exe -m src.data.build_human_corpus --bucket cm --target 2200`
→ prints the source table and `complete`; re-running is a no-op.

**Next:** Part 4 — tokenizer fertility across mGPT / Qwen2.5-0.5B / gemma-2-2b,
lock the scorer in `configs/models.yaml`, and build the three resumable CLIs
(`generate`, `score`, `attack`).

---

## 2026-09-13 — Part 2: hi and te human corpora (2,200 passages each)

- Added `indiccorp_v2` to the loader registry and hi / te entries to
  `configs/data.yaml`. Both buckets use a single band, `writer_L1_band=native`
  (added to `WRITER_L1_BANDS`), with IndicCorpV2 primary and Wikipedia as a
  pure fallback.
- **hi** (`data/raw/human/hi.jsonl`): 2,200 rows, mean 164.0 words, median 150.
  All from IndicCorpV2 `hin_Deva`. 32,646 documents read, 93.3 % rejected —
  length 30,057 (92.1 %), artefact 364 (1.1 %), wrong script 25 (0.1 %).
- **te** (`data/raw/human/te.jsonl`): 2,200 rows, mean 161.3 words, median 147.
  All from IndicCorpV2 `tel_Telu`. 106,950 documents read, 97.9 % rejected —
  length 104,251 (97.5 %), artefact 474 (0.4 %), wrong script 24 (0.0 %),
  near-duplicate 1. Telugu needs ~3x the streaming of Hindi because only ~2 % of
  its documents reach 120 words.
- Wikipedia hi / te were configured but never read: IndicCorpV2 filled both
  targets on its own.
- **Same cleaning discipline as en, extended for Indic:** space before a danda
  (` ।`) now counts as an artefact; `wikipedia_prose` no longer mistakes a short
  danda-terminated line for a heading; a new per-source script gate
  (`script` + `min_script_ratio: 0.8`) keeps each bucket monolingual while still
  allowing the Latin loanwords normal in Indian news.
- **Fixed a real bug in the new script check:** counting the Devanagari block
  directly put combining vowel signs in the numerator but not the denominator,
  so ordinary Hindi scored above 1.0. The numerator is now the subset of letters
  in range.
- The run summary now prints a per-source "documents read vs rejected" table
  with a reason breakdown.
- **Quality:** both buckets are 100 % within 120-300 words, zero residual
  artefacts, zero exact-duplicate texts, script ratio mean 0.99 and min 0.80.
- **Resume verified:** re-running either bucket at the same target is a 2-3 s
  no-op that leaves the file hash unchanged.

**Verify:** `.env\Scripts\python.exe tests\data	est_build_human_corpus.py`
→ 11 × PASS (offline). Then
`.env\Scripts\python.exe -m src.data.build_human_corpus --bucket hi --target 2200`
(and `--bucket te`) → each prints its band table and `complete` within seconds.

**Next:** Part 3 — cm sources (COMI-LINGUA, cmu_hinglish_dog, a HinGE mirror),
then the corpus card. Note for Part 13: en averages 210.6 words vs ~163 for
hi/te, and hi/te are single-domain news while en is mixed — both need handling
before cross-bucket comparison.

---

## 2026-09-13 — Part 1: en human corpus (2,200 passages, two writer bands)

- `src/data/build_human_corpus.py` implemented as a resumable CLI
  (`python -m src.data.build_human_corpus --bucket <en|hi|te|cm> --target N
  [--config configs/data.yaml] [--max-minutes M]`). Per band, sources are pulled
  in order (`human_corpus` section of `configs/data.yaml`); one 120–300-word
  passage per source doc, cut at a sentence boundary with a deterministic per-doc
  target length; MinHash LSH dedup (datasketch, 128 perms, word 5-grams, J≥0.8);
  rows appended to `data/raw/human/<bucket>.jsonl` one at a time with
  deterministic ids, so re-running skips what is on disk and pulls the shortfall.
- **en result** (`data/raw/human/en.jsonl`, gitignored): 2,200 rows, mean 210.6
  words. `writer_L1_band=general` 1,100 (mean 202.8 words): HC3 reddit_eli5 304,
  wiki_csai 300, open_qa 1, Wikipedia `20231101.en` page id ≤ 1,853 → 495.
  `writer_L1_band=indian` 1,100 (mean 218.4 words): Samanantar en side (hi
  config) 1,100. Ids unique; passages 120–300 words; zero LaTeX / wiki-markup /
  template-hole artefacts after cleaning.
- **Cleaning that turned out to be necessary:** wikimedia dump text has
  template holes (`Albedo (; ) is`, `an area of , making`) in ~1/3 of pages and
  wiki_csai has LaTeX; HC3 ELI5/WikiQA answers are PTB-tokenised (`word ,
  it 's`). Benign holes are repaired, PTB spacing undone, and any passage that
  still carries an artefact is rejected (21 % of Wikipedia docs, ~3 % of HC3).
  Without this, human text would be separable from machine text by surface cues.
- **Resume verified on real sources:** `--target 1000` then `--target 2200`
  gave 1,000 → 2,200 rows with every earlier id reported `already_present`; a
  third identical run pulled nothing (2 s, file hash unchanged).
- **Caveats:** HC3 open_qa answers are almost all < 120 words (7 of 1,187), so
  that subset contributes ~nothing. Samanantar on HF is shuffled sentences, not
  documents — `indian` passages are runs of filtered sentences (see
  `docs/decisions.md`). Samanantar is CC-BY-NC-4.0.
- Also: `src/utils/{config,io,logging}.py` stubs implemented (needed here);
  `WRITER_L1_BANDS` → `general / indian / unknown`; `datasets==5.0.1`,
  `datasketch==2.0.0`, `pyarrow==25.0.1` pinned; `venv/` rebuilt (it had been
  moved from another folder and had no interpreter) — CPU torch for now.

**Verify:** `.env\Scripts\python.exe tests\data	est_build_human_corpus.py`
→ 9 × PASS (offline). Then
`.env\Scripts\python.exe -m src.data.build_human_corpus --bucket en --target 2200`
→ prints the band table above and `complete` within seconds (nothing to pull).

**Next:** Part 2 — hi and te sources (IndicCorpV2 / Wikipedia hi, te) in
`configs/data.yaml`, run for both buckets.

## 2026-09-02 — exp00 HC3 smoke test: Fast-DetectGPT AUROC = 0.9445

- Ran `experiments/exp00_smoke.py` on 100 human + 100 ChatGPT HC3 English answers
  (Hello-SimpleAI/HC3 `all.jsonl`, pulled via huggingface_hub, prompt-matched by
  question). Scorer = Qwen/Qwen2.5-0.5B, fp16 on CUDA (RTX 4060, venv python).
- **AUROC = 0.9445.** Mean curvature — human −0.6951, chatgpt +1.6754 (machine
  clearly higher, as expected on genuine generations). Wrote
  `results/exp00_smoke.csv` (gitignored). First Rubric-7 evidence for Review-1.
- Fixed a Windows `os error 1455` (paging-file) during the CUDA weight load by
  deferring pandas/sklearn imports until after scoring.

---

## 2026-09-02 — Head A (stylometric extractor) implemented

- `src/features/stylometric.py` implemented per its scaffolded interface
  (`fit`/`transform`/`fit_transform`/`feature_names`/`score`, all unchanged).
- **25 features, fixed schema:** lexical diversity (TTR, root-TTR, MTLD, hapax,
  bigram-repeat), length/shape + burstiness (sentence- and word-length),
  language-aware function-word ratio + function/content transition proxies,
  clause-depth proxy, punctuation profile, char-class ratios.
- **Dependency-free (stdlib + numpy):** no spaCy/stanza model load → no download,
  no crash on Devanagari/Telugu. Unicode-aware tokenisation; danda (`।`) sentence
  splitting. Per-bucket function-word lists (`en`/`hi`/`te`/`cm`); `fit` stores
  global + per-bucket norms (`bucket_norms_`) for the §2.2.2 importance work.
- **POS n-grams / syntactic depth are proxies for now** (function/content-word
  transitions; punctuation-delimited clause depth). Real per-bucket taggers/
  parsers deferred to Phase 2 §2.2.2 — see `docs/decisions.md`.
- `score()` is a **provisional, uncalibrated** unsupervised head (bursty/less-
  repetitive → human), a placeholder for P1's trained fusion head (non-neg #8).

**Verify:** `PYTHONIOENCODING=utf-8 PYTHONPATH=. python src/features/stylometric.py`
→ prints `feature matrix shape: (5, 25)`, the 25 ordered names, per-sample scores.
Runs on all 5 samples (en×2, hi, te, hinglish); edge cases (empty/whitespace/
single-word/punct-only/Indic-only) all finite.

**Next:** Phase 1 Day 2 — P2 runs Head A on 50 hi + 50 te + 50 hinglish samples
for the per-language feature-distribution plot (Rubric-7 evidence).

---

## 2026-09-02 — Scaffold complete

- Repository scaffolded to match `docs/project-context-master.md` §8.
- **src/** (pure Python, zero Django imports): `data/`, `features/`, `fusion/`,
  `calibration/`, `baselines/`, `eval/`, `utils/` — full signatures with type
  hints, every body `raise NotImplementedError`, `# TODO(phase-N step-X)` refs.
- Shared interfaces enforced: `score(texts) -> np.ndarray` on every head and
  baseline; `StylometricExtractor.fit/transform/feature_names`;
  `Fuser.fit/predict_proba`; `ConformalCalibrator.fit/threshold`;
  `AbstentionGate.decide`.
- **experiments/** exp00–exp08, each takes `--config` and writes
  `results/<name>.csv` (mapped to T1–T6 / F1 in the README).
- **tests/** mirror `src/` with stub cases.
- **scripts/check_hardware.py** is real (RAM / GPU / VRAM / torch+CUDA);
  `download_datasets.py` is a stub.
- **webapp/** Django 5 + DRF: `Submission` + `Decision` models, admin with
  list_display/filters, `services.py` bridge (placeholder, correctly shaped),
  forms with CSRF, six navigable pages + admin. Migration `0001_initial`
  hand-authored (Django not installed at scaffold time).
- **configs/** default/models/data YAML; **requirements.txt** pinned;
  **README.md** with setup + experiment→table map.
- No dependencies installed; server not run (scaffold-only session).

**Verify:** `python -c "import src"` (needs nothing installed);
after `pip install -r requirements.txt` then `python manage.py migrate` and
`python manage.py runserver`, click through all pages.

**Next:** Phase 1 Day 1 tasks — Phase 0 emails (P1), dataset downloads (P2),
run `scripts/check_hardware.py` and post output (P3).
