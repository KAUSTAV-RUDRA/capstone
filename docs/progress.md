# Progress log

Dated daily log — the evidence trail for Rubric #7 and every review.
Newest entries at the top.

---

## 2026-09-24 — Head A scored: first real result

Review-2 sprint Day 2, Head A only (no GPU; headB / headB_word next sitting).
`src/features/stylometric.py` now has 40 features: the 25 surface features
unchanged, plus 15 from real parsers (stanza hi/te, spaCy `en_core_web_sm` en/cm,
CPU): 10 UPOS ratios, POS bigram entropy / type ratio, POS trigram repeat rate,
mean tree depth, mean dependency distance. `HeadA` fits one standardise→logistic
regression per bucket on the train split (`class_weight=balanced`); column `headA`
is in `results/scores.parquet`, the model in `results/models/headA.joblib`, the
feature cache in `results/features/stylometric.parquet`. Parsing 8,896 rows took
49 min on CPU.

| bucket | test h/m | AUROC | 95 % CI | train h/m | train 5-fold CV AUROC |
|---|---|---|---|---|---|
| en | 272/361 | **0.991** | [0.985, 0.995] | 271/359 | 0.984 ± 0.004 |
| hi | 392/233 | **0.984** | [0.974, 0.994] | 392/233 | 0.991 ± 0.008 |
| te | 349/165 | **0.989** | [0.981, 0.995] | 349/163 | 0.993 ± 0.006 |
| cm | 596/83 | **0.910** | [0.883, 0.934] | 596/82 | 0.902 ± **0.054** |

- **cm is unstable, as expected with 82 machine train rows.** Its CV spread is 7–13×
  the other buckets' (± 0.054 vs ± 0.004–0.008), and its test CI is ~5× wider. Its
  leading coefficients (uppercase ratio, punctuation diversity) partly reflect
  register (lower-case social comments vs. punctuated machine chat), and cm goes
  through an English parser. Read cm's 0.91 as provisional until gemma/mistral
  add machine rows. Its bound is chat-register (decisions.md 2026-09-24).
- **Shortcut check.** No single feature separates the classes (best single-feature
  test AUROC 0.83–0.92: sentence-length burstiness, MTLD, comma rate). en does not
  hinge on samanantar's shuffled-sentence construction: headA vs machine is 0.983 for
  hc3 humans and 0.986 for Wikipedia, against 0.998 for samanantar.
- **Scope of the number.** One generator (qwen7b), seen in training. This is
  same-generator performance: high by construction, and not the paper's headline
  (non-negotiable #4; T3 needs llama/phi).

**Verify:** `python -m src.eval.score --column headA` (reuses the feature cache,
~10 s) → prints the table above. `pytest tests/features/test_stylometric.py`.

**Next:** Day 2 GPU half: headB and headB_word (mGPT, fp16) on mains power.

---

## 2026-09-24 — Part 13: corpus v1 frozen (qwen7b)

Review-2 sprint Day 1. `clean_artifacts.py` and `freeze_splits.py` implemented;
`splits.json` + `corpus.jsonl` written and frozen on human 8,800 + qwen7b 2,331.
Decisions and the cm finding: `docs/decisions.md` 2026-09-24 (later). Counts,
hashes and cleaning table: `docs/data/corpus_card.md` §0.

- **Cleaning** drops 259 rows (qwen7b en 1.4 %, hi 6.8 %, te 18.0 %, cm 7.4 %; human
  0–0.8 %, Indic Wikipedia 5.5–7.1 %). Caught and fixed on the way: qwen7b cm rows
  carried `code_mix_ratio = 0.0` against ≥ 0.30 for human, a perfect class separator
  in metadata; recomputed for both classes.
- **First dry run stopped the freeze**: trimming machine to human lengths left en 302,
  hi 269, te 98, cm 290 machine rows. Resolved by trimming whichever side is the
  surplus: human for en/hi/te, machine for cm.
- **Final (human cal/train/test · machine train/test)**: en 1000/271/272 · 359/361;
  hi 1000/392/392 · 233/233; te 1000/349/349 · 163/165; cm 1000/596/596 · 82/83.
  Test length medians match in every bucket; 0 prompts straddle train/test.
- **Finding**: cm calibration is chat-register (cmu_hinglish_dog) while cm test is
  98 % comments (comi_lingua), so cm's bound holds for chat-register Hinglish. It's a
  worked example of the paper's calibration–deployment-match argument.
- `pytest` added to requirements.txt; 10 data tests pass (freeze guard, cal
  eligibility, cal floor, length matching both ways, symmetric drop rules).

**Verify:** `python -m src.data.freeze_splits --config configs/data.yaml` → refuses
with `FileExistsError` (frozen). `sha256sum data/processed/splits.json` →
`1a71fa2f…7471`. `pytest tests/data`.

**Next:** Day 2 — score headA / headB / headB_word on the frozen corpus.

---

## 2026-09-23 — gemma: the hi gate abort, per-generator compliance, and all four buckets passing

A 110-minute gemma run aborted at the `hi` gate (median 1.48 against a 1.25
ceiling, 83 % of rows over). Cause, fix, scope cuts and re-run are in
`docs/decisions.md` 2026-09-23 and commit `afd8342`. This entry is the evidence
trail and the numbers.

### 1. The abort and its cause

`compliance_ratio {hi 0.55, te 0.62}` was measured on **qwen** and applied to
**gemma** unchanged, because the config keys the ratio on the bucket alone. A
ratio is the fraction of the ask a model writes, so it belongs to the generator
*and* the bucket. gemma's measured raw hi ratio is 0.73, so under a 0.55 ask
(×1.82 inflation) it wrote 0.73 × 1.82 = 1.33 bins ≈ **1.46 of human**, against
**1.48 observed** — the arithmetic predicts the failure to within 0.02.

`decisions.md` 2026-09-18 already required every regime to be measured on its own
probe and never carried across by analogy. The rule was prose; the config could
not express it, so it carried the value across silently.

### 2. Measured — gemma's own ratios

New script `src/data/probe_compliance.py`: 12 passages per bucket, current
template, ask = the length bin with **no** compensation, `num_predict` at 3×
headroom so over-production is measured rather than clipped. 0 of 24 rows
truncated. Words counted after `strip_boilerplate()`, as the gate counts them.

| bucket | machine/ask | mean | p10 | p90 | qwen's value |
|---|---|---|---|---|---|
| hi | **0.73** | 0.74 | 0.60 | 0.87 | 0.55 |
| te | **0.84** | 0.79 | 0.58 | 0.93 | 0.62 |

Both machine/**ask**, the convention every existing value uses.

### 3. Re-run — all four gates pass

`python -m src.data.generate --generator gemma --buckets cm,en,hi,te`,
22:16–23:50 (94 min, stopped on schedule rather than at the 110-minute exit).

| bucket | gate median | before | verdict |
|---|---|---|---|
| cm | 1.00 | 1.00 | PASS |
| en | 0.80 | 0.80 | PASS |
| hi | **1.07** (mean 1.10, p90 1.23) | 1.48 FAIL | **PASS** |
| te | **0.80** (mean 0.86, p90 1.00) | never reached | **PASS** |

0 % at the token cap in every bucket. Predicted medians before generating were
hi 1.05 and te 0.89, from the fitted ask→output regression.

Throughput roughly doubled as a side effect: **24.6 s/row on hi against 57.4 s/row**
under the old ratio, because the ×1.82 inflation was paying to decode ~450 tokens
a row where ~300 was called for. Run average 15.4 tok/s aggregate, 0 retries,
0 failed requests.

### 4. `te` passes on a median its short half carries — for Limitations

The `te` gate median of 0.80 conceals a split. Across the 24 probe rows, by human
passage length:

| human length | n | median | range | below 0.75 |
|---|---|---|---|---|
| ≤ 160 words | 12 | 0.95 | 0.77–1.32 | **0** |
| > 160 words | 12 | **0.75** | 0.58–0.96 | **6** |
| all | 24 | 0.80 | — | 6 |

Rows above 200 human words read 0.78, 0.66, 0.73, 0.58, 0.73, 0.68, 0.75 — five of
seven under the floor.

**This is a corpus property, not a defect to tune.** Regressing output words on ask
words gives hi a slope of 0.65 with intercept 14 (r = 0.86), but te a slope of
**0.36 with intercept 69** (r = 0.73): gemma writes roughly 69 Telugu words plus a
third of whatever is asked, so its output is far less responsive to the ask than
hi's. Because the ask barely moves it, **no compliance ratio can correct the long
tail** — the ratio only inflates the ask, and at slope 0.36 that inflation is
mostly absorbed. Raising the ratio further would over-produce the short half
before it fixed the long half.

Consequence to state in **Limitations**: in `te`, gemma's machine passages are
length-matched to human ones at the short end and systematically shorter at the
long end. Length is therefore a weaker matched covariate for `te` than for the
other buckets, and any `te` result should be read with that in mind rather than
treated as a like-for-like length match. Correcting it needs a different Telugu
template, measured on its own probe — not a ratio.

### 5. Tally at the stop

| bucket | rows | target | |
|---|---|---|---|
| cm | 676 | 790 | 85.6 % |
| en | 24 | 712 | 3.4 % |
| hi | 24 | 250 | 9.6 % |
| te | 24 | 200 | 12.0 % |
| **total** | **748** | **1952** | **38.3 %** |

700 rows generated this sitting in 94 min (7.4 rows/min). en/hi/te stand at their
24 gate-probe rows; the bulk phase worked through `cm` first.

**Integrity after a forced stop:** 0 duplicate `id`, 0 duplicate `prompt_id`, 0
truncated rows, 0 malformed lines. `run_concurrent` flushes each row under a lock,
so a hard stop loses only in-flight requests, which the next resume regenerates.

**cm across all 676 rows** (not just the probe): median 1.02, mean 1.05, p90 1.40,
0 % at cap, romanised-Hindi share 0.32 against a human 0.32 — and **3 rows carry an
in-body emoji**, counted after stripping. qwen7b's finished cm bucket reads 3 of
701. These are Part 13 drops, not gate failures: the gate judges the 24-row probe,
which was clean.

### 6. Corpus state, all generators

| generator | role | cm | en | hi | te | total |
|---|---|---|---|---|---|---|
| qwen7b | seen | 701/701 | 730/730 | 500/500 | 400/400 | 2331/2331 |
| gemma | seen | 676/790 | 24/712 | 24/250 | 24/200 | 748/1952 |
| mistral | seen | 0/709 | 0/758 | 0/250 | 0/200 | 0/1917 |
| llama | heldout | 0/647 | — | 0/500 | 0/400 | 0/1547 |
| phi | heldout | 0/629 | — | 0/500 | 0/400 | 0/1529 |

**3,079 / 9,276 machine rows (33.2 %).** Targets reflect the 2026-09-23 scope cuts:
gemma and mistral capped at hi 250 / te 200, and held-out generators skip `en`.

**Verify:** `PYTHONPATH=. python tests/data/test_generate.py` → 40 PASS, 0 FAIL.
Resume gemma with the same command; it picks up from the 748 rows on disk.

**Next:** gemma has ~1,200 rows left, ~8 h at the observed rate. mistral, llama and
phi are all still on qwen's compliance ratios — the exact condition that caused
tonight's abort — and each needs `probe_compliance.py` run on its own buckets
before any bulk generation.

---

## 2026-09-22 — cm regenerated under the long-ask template; the gate gains an emoji check

Step 4 of the cm plan. The emoji check is committed (`e17674e`). v0's 701 rows are
moved aside, not deleted. qwen7b cm is regenerated at 701/701. On the finished
bucket, length and code-mix now pass, and the **emoji check fails on 3 of 701 rows**.
Those 3 rows are an open decision (below).

### 1. Emoji check in the gate (`e17674e`)

Human cm has emoji in **0 of 2,200** rows, so one emoji in a machine cm row is
a class shortcut. The template's "No emoji" does not guarantee none: two of the three
2026-09-21 probe variants still put one in, and 28 of v0's 701 rows carry one. For
`gate.emoji_buckets` (`[cm]`), a single probe row with an emoji fails the gate.
`generate.has_emoji()` uses the probe scripts' ranges plus ⌚⌛⏩–⏺, and **drops
the lone ZWJ**. ZWJ joins Devanagari conjuncts, and counting it scored 118 of the
2,200 human hi passages as emoji. Under the new ranges human cm stays at 0/2,200,
and en/hi/te read 2/2,200 each.

### 2. v0 moved to `data/raw/discarded/`

All 701 v0 cm rows are in `data/raw/discarded/qwen7b_cm_v0.jsonl`. They were moved
as raw bytes, not re-serialised. `data/raw/machine/qwen7b.jsonl` then held
**1,630 rows** (en 730, hi 500, te 400), and those lines are byte-identical to
before (sha256 checked). The two files' sizes add up to the original's exactly.
`data/` is gitignored, so the discard file exists only on this machine.

### 3. Regeneration

`python -m src.data.generate --generator qwen7b --buckets cm --max-minutes 30`,
with the Ollama app started first (it was not running; `OLLAMA_NUM_PARALLEL:4` in
server.log). **701/701 in 23.2 min**, 81,643 tokens, 58.7 tok/s, concurrency
x2.8–3.8. The run hit no errors and needed no retries. cm covers the same 701
prompt_ids as v0. The file is back to 2,331 rows, with zero duplicate `id` or
`prompt_id`.

**Gate on the 24-row stratified probe: PASS.** Length median 1.19 (mean 1.18, p90
1.61), 0 % at the cap, Hindi share 0.33 against 0.29, 0 emoji. Inside the probe,
0–40 read **1.22** (n = 14) and 40+ read 1.06 (n = 10).

**Settled, all 701 rows.** Hindi share is the gate's median romanised-Hindi share,
with the paired human value in brackets.

| human length | n | v0 median | **v1 median** | v1 in band | v1 > 1.25 | Hindi share | emoji | at cap |
|---|---|---|---|---|---|---|---|---|
| 0–40 | 413 | 1.75 | **1.18** | 229 (55 %) | 159 | 0.33 (0.32) | 1 | 0 |
| 40–100 | 152 | 1.05 | **1.07** | 111 (73 %) | — | 0.35 (0.30) | 0 | 1 |
| 100+ | 136 | 1.06 | **1.11** | 85 (62 %) | — | 0.37 (0.30) | 2 | 12 |
| 40+ | 288 | 1.05 | **1.09** | 196 (68 %) | 79 | 0.36 (0.30) | 2 | 13 |
| **all** | 701 | 1.39 | **1.13** | 425 (61 %) | 238 | 0.35 (0.32) | 3 | 13 |

By n_words bin: 25 → 1.18 (408), 50 → 1.03 (101), 100 → 1.09 (108), 150 → 1.13 (56),
200 → 1.19 (14), **250 → 1.70 (11), 300 → 1.93 (3)**.

**The short bin: 1.22 on the probe, 1.18 settled, against a ceiling of 1.25.** It
held, with 0.07 of headroom on the settled median. The probe-to-settled gap (0.04)
is within the ±0.06 run-to-run noise measured 2026-09-21. The median hides
dispersion: 159 of 413 short rows (38 %) sit above 1.25, and only 25 below 0.75
(p25 1.00, p75 1.38, p90 1.63). The short bin still leans long; it no longer sits
out of band.

### 4. The gate re-run on the finished bucket (the check v0 failed)

Same `gate_stats` / `gate_reasons` code and thresholds as the run, applied to all 701
rows with each row paired to its own human passage:

| | length median | cap | Hindi share | emoji | verdict |
|---|---|---|---|---|---|
| v0 (discarded) | 1.39 ✗ (62 % of rows > 1.25) | 0 % | 0.03 vs 0.32 ✗ | 28 ✗ | FAIL ×3 |
| **v1** | **1.13** ✓ (34 % > 1.25) | 2 % ✓ | **0.35 vs 0.32** ✓ | **3** ✗ | **FAIL (emoji only)** |

The 3 emoji rows: `cmu_hinglish_dog_train:187-216` (😂), `…:6898-6903` (🌍), and
`comi_lingua_TN:train:743` (😂😉, which also echoes "roman script"). At 3/701 the
24-row probe had about a 10 % chance of catching one, so passing there was the
expected outcome, not a gate defect. With a zero limit, the check fails on any
nonzero rate once it is applied to a whole bucket.

### Row-level defects in v1, for Part 13 (`clean_artifacts.py` is still a stub)

- **Truncated: 13 rows**, all from passages of 100+ words, 8 of them in the 250/300
  bins. Those bins over-produce (1.70 / 1.93, n = 14): at 200+ words the long-ask
  inflation (human / 0.75) overshoots. It was probed on 10 long rows with human
  length at most 170. This is 4 % of the bucket and does not move the medians.
- **Instruction echo: ~23 rows (3.3 %)**, mostly "roman script mein likha hai". The
  regex `\broman\b|hinglish|\bemoji` hits 25 v1 rows, 7 v0 rows and 3 of 2,200
  human rows. All but 1–2 of the v1 hits are real echo ("roman numeral watch" is
  not). The probe saw 0 of 24, and the long-ask probe saw 1 of 24. This is a
  detector shortcut, as flagged 2026-09-21.
- **Stray script: 13 rows with Devanagari**, against 9 of 2,200 human rows. Most are
  single Devanagari characters inside Roman words ("banayा", "samjhा"). A few also
  carry Cyrillic or Hangul fragments ("sitацию", "Jayeग즈。"). Two are substantially
  Devanagari (228/458 and 92/133 characters).
- Zero-Hindi rows: 5 (v0: 134).

### Open: what to do with the 3 emoji rows

Choices: (a) drop them in Part 13, with the truncated and echo rows; (b) regenerate
those 3 ids (parallel decoding is not seed-deterministic, so they would probably
come back different); (c) strip the emoji and keep the text. **Not yet decided,
nothing has been changed.** decisions.md does not yet record the cm template
adoption, the long-ask rule, or the emoji check. The 2026-09-21 plan had that as
part of step 4.

**Verify:** `PYTHONPATH=. venv\Scripts\python.exe tests\data\test_generate.py` → 34
PASS. Row counts: en 730 / hi 500 / te 400 / cm 701 in `data/raw/machine/qwen7b.jsonl`,
701 cm in `data/raw/discarded/qwen7b_cm_v0.jsonl`.

---

## 2026-09-21 (late) — cm: code-mix gate built, long-ask template adopted; rows not yet regenerated

Steps 1–3 of the cm plan. Step 4 (record in decisions.md, delete qwen7b's 701 cm
rows, regenerate) is **next sitting**. Nothing was deleted or generated into
`data/` tonight.

1. **Probe evidence committed** (`0e006b3`) to `scripts/probes/`. The scripts,
   their rows and summaries now take `--config`, and write to `results/probes/` so
   a rerun cannot overwrite the committed copies.
2. **Code-mix gate check** (`a796da6`).
   `src/features/language_id.py: romanised_hindi_share()` over
   `ROMANISED_HINDI_FUNCTION_WORDS`, with yaar/yar/bhai excluded. In the gate, each
   probe row is paired with its own human passage. For `gate.codemix_buckets`
   (`[cm]`) the median share must be ≥ `codemix_min_ratio` (2/3) of the human
   median. On real data, the original first-24 cm probe (length 1.02, passed) now
   **fails at 0.02 against 0.31**, and all three probed variants pass at
   0.32–0.33 against 0.29.
3. **v1 with the long bin's ask inflated**: same 24 passages, same seeds, ask =
   human for < 40 words, human / 0.75 for ≥ 40. The sentence clause follows the
   ask. The adoption rule was set before the run.

| | length median | in band | 0–40 median | 40+ median | Hindi share (human 0.29) | emoji | gate |
|---|---|---|---|---|---|---|---|
| v1, ask = human | 0.97 | 15/24 | 1.13 | **0.73** | 0.32 | 0 | pass |
| **v1, long-ask** | **1.15** | 12/24 | **1.19** | **1.09** | 0.37 | 0 | **pass → ADOPTED** |

Shares here use the gate's word list (without yaar/bhai); v1's 0.32 is unchanged
by that. Long passages, human → v1 → long-ask: 69→46→99, 95→59→111, 170→140→296,
42→25→23.

**Adopted with three caveats, for the regeneration to watch:**
- **Dispersion widened.** Only 12/24 rows are in band against v1's 15, and the
  long rows scatter 0.55–1.74 (4 of 10 above 1.25). Both bin medians are in band;
  individual rows often are not.
- **Run-to-run noise is about ±0.06 on these medians.** The short-bin prompts were
  identical to v1's and only 3 of 14 rows reproduced (parallel decoding is not
  seed-deterministic), so 1.13 → 1.19 is noise. With the short bin at 1.19, the
  upper bound (1.25) is close.
- **1 row in 24 echoes the instruction** ("roman script"), and 1 contains
  Devanagari.

**In code:** `PROMPT_TEMPLATES["cm"]` is the probed v1 text. The new
`ASK_FROM_HUMAN` / `budget.ask_from_human: {cm: {long_from_words: 40, long_ratio: 0.75}}`
and `ask_words_for()` make cm ask from the human length rather than the bin. The
compliance ratio is not applied to cm; other buckets are unchanged. Checked by
rebuilding every qwen7b prompt: **en 730/730, hi 500/500, te 400/400 identical to
the rows on disk**, cm 0/701 (the new template), and the 24 long-ask probe prompts
24/24 identical to what the code now builds.

**Do not resume `cm` before step 4.** The 701 old rows still hold every cm id, so
`generate --buckets cm` reports "complete" without reaching the gate. They must be
deleted first.

**Verify:** `PYTHONPATH=. venv\Scripts\python.exe tests\data\test_generate.py` →
31 PASS; `PYTHONPATH=. venv\Scripts\python.exe tests\features\test_language_id.py`
→ 3 PASS + 1 TODO (the Phase-2 placeholder).

---

## 2026-09-21 — stratified gate probe committed; `cm` fails on code-mixing, not just length

- **Committed `7d955c6`**: `run_with_gates` draws its probe with
  `stratified_probe` (decisions.md 2026-09-20). **Verify:**
  `PYTHONPATH=. venv\Scripts\python.exe tests\data\test_generate.py` → 27 PASS
  (without `PYTHONPATH` the `src` import fails).

### Probe 1: short-passage variant, 12 passages from `cm`'s 0–40 bin

Variant: "one or two short sentences … at most N words", N = the bin (25). The
control is the rows already on disk for the same 12 ids, with their prompts
checked identical to the live template, so only 12 rows were generated.

| | median m/h | in band | machine words, median |
|---|---|---|---|
| live template (on disk) | 2.07 | 2/12 | 36.5 |
| short variant | **0.747** | 6/12 | 14 |

Machine length is near-flat inside the bin under both templates (~36 and ~14
words), so the ratio tracks 1/human length. The variant also lost the Hindi
(7/12 rows with none) and added emoji to 5/12; the human `cm` bucket has emoji in
0 of 2,200 rows.

### The real `cm` defect is code-mixing

Hindi function-word share: the share of a text's words found in a fixed
romanised-Hindi lexicon (hai, ka, nahi, yaar, …). Crude, but held constant across
every figure below.

| human-length bin | n | human | qwen7b `cm` on disk | machine rows with zero Hindi |
|---|---|---|---|---|
| 0–40 | 413 | 0.33 | 0.04 | 75 |
| 40–100 | 152 | 0.31 | 0.02 | 34 |
| 100+ | 136 | 0.30 | 0.02 | 3 |

The same lexicon scores **0.028 on English human text**, so the live `cm` rows sit
at the English floor in every bin. A detector could tell `cm` human from machine
on the amount of Hindi alone. **Decision:** all 701 qwen7b `cm` rows will be
regenerated once a template passes. Regenerating only the 0–40 bin would leave the
code-mixing defect in place. Nothing has been deleted.

### Probe 2: 3 templates × 24 passages, stratified across `cm`'s length bins

The 24 are `stratified_probe(cm, 24)`, n_words bins 25:14, 50:3, 100:4, 150:2,
200:1, human median 26 words, human Hindi share 0.29. Same model, decoding,
`num_predict` and per-passage seed as the real run. All three variants use "about
N words" with N = the **human** length (not the bin), "mix Hindi and English words
within every sentence, with the Hindi written in Roman script, casual, like
students texting each other", and "No emoji." Each variant changes one thing
against v1.

| variant | length median | in band | 0–40 / 40+ median | Hindi share (human 0.29) | zero-Hindi | emoji | verdict |
|---|---|---|---|---|---|---|---|
| **v1** mix + sentence clause `max(2, n/15)` | **0.97** | 15/24 | 1.13 / **0.73** | 0.32 | 0 | 0 | **PASS** |
| v2 = v1 without the sentence clause | 0.80 | 8/24 | 1.02 / 0.62 | 0.32 | 0 | 1 | FAIL (emoji) |
| v3 = v1 + "roughly half the words Hindi" | 0.88 | 11/24 | 1.18 / 0.68 | 0.33 | 0 | 1 | FAIL (emoji) |

Pass rule: length median in 0.75–1.25 AND Hindi share ≥ 0.20 AND zero emoji rows.
No row hit the token cap, and every row stopped naturally.

**v1 carries two caveats that the pass rule does not test:**
- **Its 0.97 median hides a split in opposite directions.** Short passages read
  1.13, and passages of 40+ words read 0.73: 8 of those 10 rows are below 0.85
  (42→25, 69→46, 95→59). This is the bin-local pattern again, smaller and
  flipped. The 40+ group is only n = 10.
- **Instruction echo.** 1 of 24 v1 rows contains "roman script mein likha gaya";
  v2 and v3 each have 2, and the human texts have 0. A leaked prompt phrase is a
  detector shortcut, so either Part 13 cleaning or the template has to handle it.

v2 shows the sentence clause still matters for length. v3's quantified mix
bought nothing (0.33 against 0.32) and put IAST diacritics in 2 rows.

### Proposed, not implemented: a code-mix check in the gate for `cm`

The probe rows' median Hindi share must be at least 2/3 of the median share of
the same passages' human texts. The live regime reads ~0.1 of human and would
have failed at 24 rows. All three probed variants read ≥ 1.1. Length alone let
the defect through. Open: where the lexicon lives, and whether discourse markers
(yaar, bhai) count.

State: nothing generated into `data/` and nothing deleted. Probe scripts, rows and
summaries are committed under `scripts/probes/` (`cm_short_probe.*`,
`cm_codemix_probe.*`). Next: the user decides on v1 (and its 40+ split) before
the 701 rows are regenerated.

---

## 2026-09-20 — qwen7b complete at 2,331 rows, and `cm` settled out of band

All four buckets reached target in one sitting: `te` 382 → 400, `cm` 67 → 701,
`hi` 256 → 500. 1,484,227 generated tokens on disk, zero duplicate `id`, zero
duplicate `prompt_id`, zero malformed lines.

| bucket | rows | median | mean | p90 | at cap | gen-tokens/row |
|---|---|---|---|---|---|---|
| en | 730/730 | 0.87 | 0.87 | 1.16 | 0.0 % | 235 |
| hi | 500/500 | 1.10 | 1.10 | 1.41 | 3.0 % | 975 |
| te | 400/400 | 0.94 | 0.96 | 1.25 | 0.8 % | 1,899 |
| cm | **701/701** | **1.39** | **1.52** | **2.37** | 0.0 % | 94 |

### `cm` is out of band and the gate never saw it

**`cm`'s settled median is 1.39 against a gate reading of 1.02** — outside the
0.75–1.25 the gate itself enforces, with **67 % of rows outside the band**. Run
the gate against the finished bucket and it fails. This is not a small drift; it
is the 2026-09-18 over-production fault surviving its fix at reduced amplitude.

**The cause is that the probe was not a sample of the bucket.** The gate judges
the *first* 24 rows, and for `cm` those are nothing like the other 677:

| | human words, median | share in the 0–40 bin |
|---|---|---|
| first 24 (the gate probe) | **82** | 12.5 % (3 of 24) |
| all 701 | **26** | **59 % (413 of 701)** |

And `cm`'s failure lives entirely in the short bin:

| human-length bin | n | median | p90 | human → machine (median words) |
|---|---|---|---|---|
| **0–40** | **413** | **1.75** | 2.59 | **20 → 36** |
| 40–70 | 74 | 0.94 | 1.26 | 54 → 51 |
| 70–100 | 78 | 1.21 | 1.55 | 80 → 107 |
| 100–150 | 79 | 1.07 | 1.31 | 120 → 127 |
| 150–250 | 49 | 1.00 | 1.49 | 171 → 172 |
| 250+ | 8 | 1.36 | 1.48 | 266 → 368 |

Every bin from 40 words up is inside the band. The bucket fails because the bin
that fails is the bin that *is* the bucket. The `max(2, n/15)` sentence floor
did reduce the original defect — 4.0x on a 21-word passage became 1.75 at 20 —
but did not remove it: the model will not write 20 words when asked.

**The finding for the corpus section is sharper than "24 rows is too few".** The
probe is the *head* of a list sorted by human id, not a random or stratified
draw, so it inherits whatever length distribution that ordering happens to
front-load. `te` (0.99 → 0.94) and `hi` (1.00 → 1.10) drifted modestly because
their probes were roughly representative; `cm`'s probe drew a median passage 3x
longer than its bucket and was therefore blind by construction. A probe
stratified across the length bins would have read ~1.75 on the short bin and
failed `cm` at 24 rows, which is exactly what the gate exists to do.

`en` is the control: gate 0.87 → settled 0.87 at 730 rows, drift 0.00.

| bucket | gate (24) | settled | n | drift | outside 0.75–1.25 |
|---|---|---|---|---|---|
| en | 0.87 | 0.87 | 730 | +0.00 | 34.7 % |
| te | 0.99 | 0.94 | 400 | −0.04 | 24.8 % |
| hi | 1.00 | 1.10 | 500 | +0.10 | 34.0 % |
| cm | 1.02 | **1.39** | 701 | **+0.38** | **67.0 %** |

**`cm`'s 701 rows are not usable as they stand** and the decision on them —
regenerate the 0–40 bin under a corrected prompt, drop the bin, or accept and
document — is open. Nothing has been discarded.

### The sitting itself

- **`te` closed at 400/400**, median 0.94, and needed only its last 18 rows.
- **Two processes were briefly running at once** — one started by me before the
  user's message arrived, one by the user — and the corpus was checked for
  damage: 0 duplicate ids, 0 duplicate `prompt_id`, 0 malformed lines. Only one
  had written. No rewrite was needed, and a rewrite would itself have been
  unsafe with a live appender holding the file open.
- **The first user run was killed at 00:56 by Claude**, ~47 minutes of its
  55-minute budget unused. A second `generate` process (PID 11292) looked inert
  — 0 CPU, 1 thread, 4.3 MB, no socket to Ollama — and was killed as a duplicate
  hazard; it was the **parent** of the working process, which died with it. The
  giveaway was in the same `Win32_Process` query that was used to confirm the
  command line: `ParentProcessId` was not checked. No rows were lost (rows flush
  as they land); the cost was the unused budget. The run was restarted by the
  user and completed.
- **This sitting's rows were generated under the anaconda interpreter**,
  `C:\Users\harik\anaconda3\python.exe`, not the project venv: the user's
  PowerShell had the venv activated but `python` resolved to anaconda first.
  Immaterial to the rows themselves — Ollama does the work server-side and the
  client only posts HTTP — but recorded here so the corpus card is not
  reconstructed wrongly later. **Invoke the venv explicitly in future:
  `.\venv\Scripts\python.exe`, not `python`.**

**Verify:** `.\venv\Scripts\python.exe -m src.data.generate --generator qwen7b
--buckets en,hi,te,cm` → prints `complete — 2331 of 2331 (nothing to generate)`
and exits without contacting Ollama.

**Next:** decide `cm`. Then the remaining four generators (gemma, mistral as
seen; llama, phi held out) — and the `cm` prompt regime must be settled first,
because every generator will otherwise reproduce the same short-passage defect.

---

## 2026-09-20 — qwen7b: te reordered first, 382/400, and the 11.7 tok/s question closed

Ran `te` first so the expensive bucket took the fresh budget. Stopped by
`--max-minutes` at 00:44:54, exit 0. **One warning in the whole run and it was
the budget stop itself. Zero failed requests** — against four lost to the Ollama
restart last sitting.

| bucket | rows | median | mean | p90 | at cap | gen-tokens/row |
|---|---|---|---|---|---|---|
| en | 730/730 | 0.87 | 0.87 | 1.16 | 0.0 % | 235 |
| hi | 256/500 | 1.07 | 1.08 | 1.40 | 2.3 % | 869 |
| te | **382/400** | 0.94 | 0.96 | 1.25 | 0.8 % | 1,893 |
| cm | 24/701 | 1.02 | 1.09 | 1.47 | 0.0 % | 161 |

1,392 rows total, **+326 this sitting, all `te`**. 618,807 tokens in 111.3 min =
92.7 tok/s run average.

- **`te` decodes as fast as `hi` — the 11.7 tok/s that halted Run 1 was not the
  bucket.** Warm-up climbed 23.1 → 31.0 tok/s per stream, then settled at
  **20.4–20.5** for the last half hour, against a 15 floor and `hi`'s 20.8. The
  fall off the peak is decode slowing under sustained load, not lost concurrency
  (x3.3–4.2 throughout) and not the long passages arriving — the final rows
  averaged 1,874 gen-tokens against the bucket's 1,893, on ordinary-length
  inputs. See `decisions.md`.
- **All three gates re-passed off disk in 0.1 s** — `te` 0.99, `cm` 1.02, `hi`
  1.00, judged from rows already written, no probe generated, zero cost. The
  resume path working exactly as designed.
- **The 24-row gate under-reads in both directions.** `te` 0.99 → 0.94 at 382
  rows, `hi` 1.00 → 1.07 at 256, with 24.6 % and 31.6 % of rows outside
  0.75–1.25. Last sitting called this on `hi` alone; `te` confirms it and drifts
  the *other* way. The median is a tripwire for a broken regime, never an
  estimate of a bucket's final ratio — `decisions.md` carries the
  dispersion-check decision.
- **The `te` cap worry did not materialise.** 17 % of `te` rows had
  `num_predict` clamped to the 3600 cap and **none of them truncated**; the only
  2 truncations in 382 rows were *short* passages overrunning their own smaller
  budget (129w → 211w, 184w → 287w). `te` truncation fell as n grew:
  1.7 → 0.7 → 0.6 → 0.8 %.
- **Throughput climbed, then plateaued.** Run average 69.4 → 104.3 tok/s over
  the first 80 minutes as the worker pool filled (concurrency x3.0 → x4.2), then
  eased to 92.7 by the end. Finish estimates drawn from warm-up windows were
  ~40 % optimistic and had to be revised three times; the runner's own ETA
  stayed wrong in the other direction for the known reason — it spreads `te`'s
  1,893 gen-tokens/row uniformly across `cm`'s 161-token rows.

**Remaining for qwen7b: `te` 18, `cm` 677, `hi` 244** — 939 of the 1,601 in
scope. At this sitting's rates that is ~85–90 min, so one more sitting finishes
the generator.

**Verify / resume:** `python -m src.data.generate --generator qwen7b --buckets
te,cm,hi --max-minutes 110` — re-judges each bucket's first 24 rows off disk in
well under a second, then continues where it stopped.

**Next:** finish `te`'s 18 rows, then `cm` and `hi` on the same sitting.

---

## 2026-09-19 — qwen7b sitting: the gate's first real run, hi to 51 %

First generation run with the gate in the code. **All three gates fired within
six minutes of start and all three passed** — `hi` 0.99 (20 rows), `cm` 1.02
(24 rows), `te` 0.99 (24 rows, judged from the rows already on disk without
generating a probe of its own). Paused by hand at 47 min of the 110-minute
budget, not by the budget.

| bucket | rows | median | mean | p90 | at cap | gen-tokens/row |
|---|---|---|---|---|---|---|
| en | 730/730 | 0.87 | 0.87 | 1.16 | 0.0 % | 235 |
| hi | **256/500** | 1.07 | 1.08 | 1.40 | 2.3 % | 869 |
| cm | 24/701 | 1.02 | 1.09 | 1.47 | 0.0 % | 161 |
| te | 56/400 | 0.97 | 0.98 | 1.26 | 1.8 % | 1,864 |

- **Both fixes hold.** `cm` sits at 1.02 median where the `max(8, n/12)` sentence
  floor produced 1.88 across 470 rows, and `hi`'s plain+0.55 regime came in at
  0.99 on its probe. Each was predicted by its own 12-passage probe.
- **A 24-row gate reads low, and that is the finding of this sitting.** `hi`
  passed at 0.99 on 20 rows and sits at **1.07 median / 1.40 p90** at 256 rows.
  Still inside 0.75–1.25, so the regime is sound and the gate was right to pass
  it — but the gate is a tripwire for a *broken* regime, not an estimate of a
  bucket's final ratio, and it should not be quoted as one. Re-check `hi` at 500.
- **The Ollama server died once, at 17:53, and restarted itself** (clean restart
  in its log, 7.3 GiB free, no OOM). The four `hi` requests in flight failed and
  are left for resume; those are the only 4 failures of the sitting. Cost: four
  retryable rows and zero written rows. Rows are flushed as they land, so the
  kill at pause also left no partial line.
- **Throughput:** 80.8 tok/s run average, per-stream 20.8, concurrency x3.8–4.4.
  `OLLAMA_NUM_PARALLEL:4` confirmed from the server's own startup banner — the
  user environment does not carry it, so check the banner, not `HKCU`.
- **Remaining for qwen7b: hi 244, cm 677, te 344.** The bulk queue runs
  hi → cm → te, so this sitting only ever worked `hi`. `te` is the cost: 14 % of
  the remaining rows but ~60 % of the remaining tokens, at 1,864 gen-tokens/row
  against `hi` 869 and `cm` 161. Estimate ~135 min for `te` alone, ~225 min for
  all three — the runner's own ETA understates it, since it extrapolates rows/sec
  uniformly across buckets whose per-row cost differs 11x.

**Verify / resume:** `python -m src.data.generate --generator qwen7b --buckets
hi,cm,te --max-minutes 110` — reads the 1,066 rows on disk, re-judges each
bucket's first 24 rows against the gate, and continues where it stopped.

**Next:** finish `hi` and `cm`, then `te` on its own sitting. Watch per-stream
decode when `te` starts — Run 1 was halted on 2026-09-18 at 11.7 tok/s against a
15 tok/s floor, and this sitting never reached `te`.

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

## 2026-09-24 (night) — Day 2 GPU half: headB + headB_word scored, corpus-wide

Fast-DetectGPT curvature (`src/features/curvature.py`), scorer mGPT-1.3B, fp16 CUDA.
One forward pass yields both `headB` (token-level) and `headB_word` (per-word
standardised, Patent 2) into `results/scores.parquet`, all 8896 rows.

Test AUROC: en 0.859/0.832, cm 0.661/0.635, hi 0.187/0.192, te 0.120/0.122
(headB/headB_word). hi and te are inverted, not just weak — diagnosed as a
likely pretraining-contamination effect specific to low-resource-language
human corpora (both available hi/te human sources, IndicCorp and Wikipedia,
score the same way relative to qwen7b; en's three sources don't). Sign is
**not** flipped — fusion (Day 4) handles it. Full writeup and the evidence
table: `docs/decisions.md` 2026-09-24 (night).

headB_word is a negative result: indistinguishable from headB everywhere,
including on the worst-fragmentation buckets (te, hi). Patent 2 needs
attention — flagged for P2.

Also fixed mid-run: a real CUDA OOM at row 216/6784 from allocator
fragmentation (not true VRAM usage — card was idle right after). Token-budget
batching, `empty_cache()` per group, and an OOM-safe bisecting retry; re-run
completed clean.

**Verify:** `python -m src.eval.score --report headA,headB,headB_word`;
`pytest tests/features/test_curvature.py` → 9 passed.

**Next:** Day 3 — headC (MuRIL), ppl and binoculars baselines (Qwen2.5-0.5B),
and `fastdetectgpt_en` — the decisive test for whether the hi/te inversion is
scorer-specific (different pretraining) or a property of curvature on Indic
text generally. Prioritise `fastdetectgpt_en` first.
