# Decisions log

Every locked decision with date and reason. Append-only; supersede rather than
delete. See `docs/project-context-master.md` §4 (non-negotiables) and §5
(locked decisions) for the authoritative source.

---

## 2026-09-02 — Locked decisions carried into the scaffold

| Decision | Value | Why |
|---|---|---|
| Scorer model (Head B) | `ai-forever/mGPT`; fallback `Qwen/Qwen2.5-0.5B` | Only multilingual causal LM with real Telugu coverage at a runnable size |
| Language buckets | `en`, `hi`, `te`, `cm` (Hinglish primarily) | Four separate calibrations, four fairness rows |
| Min calibration data | ≥ 1000 human texts per bucket, separate from train/test | Conformal quantile unstable below ~500 |
| Web layer | Django 5 + DRF, SQLite in dev | Admin = free audit interface; ORM = decision-evidence trail |
| Fusion | Logistic regression first; GBM only if it wins on cal AUROC | Interpretability is a feature |
| Seen generators | 3 open models via local Ollama (Llama-3-8B, Gemma-2-9B, Mistral-7B) | Free, reproducible |
| Held-out generators | 2 models, test only | Non-negotiable #4 |
| Adversarial attacks | Back-translation (IndicTrans2), LLM paraphrase, 20%-human-edited hybrids | Matches RAID/DetectRL practice |
| Real student data | Undecided; proxy corpora primary; go/no-go at week 6 | Approvals uncertain |

### Head A (stylometric) engineering decisions — 2026-09-02

- **POS n-grams and syntactic depth are script-agnostic proxies in Phase 1**, not
  true taggers/parsers. POS n-grams → function/content-word transition ratios;
  syntactic depth → punctuation-delimited clauses per sentence. Reason: real
  per-bucket POS/dependency models (spaCy/stanza) need model downloads that are
  not in `requirements.txt` and would crash on Devanagari/Telugu when absent
  (violates non-negotiable #1). The proxies keep Head A running today with
  stdlib + numpy only and a fixed feature schema.
  - **TODO (Phase 2 §2.2.2):** replace the two proxy feature groups with real
    per-bucket POS-tag n-grams and dependency-tree depth, behind the *same*
    `StylometricExtractor` interface (schema may grow; `feature_names()` stays
    the source of truth). Pick taggers per bucket: en → spaCy `en_core_web_sm`;
    hi/te → stanza Indic models; cm → romanised handling TBD. Add whichever
    models are chosen to `requirements.txt` (ask before installing).
- **`score()` is a provisional unsupervised head**, not a trained classifier —
  placeholder until P1's fusion head exists; never surfaced as a verdict
  (non-negotiable #8).

### Scaffold-time engineering decisions

- **`src/` third-party imports are guarded under `TYPE_CHECKING`** so the package
  imports with nothing installed (upholds non-negotiable #7: runs with Django
  uninstalled). Type hints (`np.ndarray`, etc.) still read correctly via
  `from __future__ import annotations`.
- **Migration `0001_initial` hand-authored** because Django was not installed at
  scaffold time; it matches `models.py` so a later `makemigrations` is a no-op.
- **App registered as `webapp.detector` with label `detector`** (app nested in the
  project package per §8).

## 2026-09-13 — Human corpus (Part 1): en sources, bands, dedup

| Decision | Value | Why |
|---|---|---|
| `writer_L1_band` vocabulary | `general` / `indian` / `unknown` (was `native` / `non_native` / `unknown`) | T5 reports the general–indian FPR gap on en; "indian" = Indian-English writers as the L2 proxy, since real L1 metadata does not exist in any public source. hi/te/cm band names are decided in Parts 2–3. |
| en / general sources | HC3 human answers (reddit_eli5, open_qa, wiki_csai; ≤ 55 % of the band) + wikimedia/wikipedia `20231101.en` with page id ≤ 2,000,000 (filler) | Both pre-ChatGPT in provenance: ELI5 2011–19, WikiQA 2015, Wikipedia intros; page ids ≤ 2M were created before mid-2005 (text as of the 2023-11-01 dump, so "pre-2020" is a proxy, not a guarantee). |
| en / indian source | English side of ai4bharat/samanantar (`hi` config), regrouped into passages | Only large, openly licensed, unambiguously Indian-authored English (news / PIB / PMIndia ≤ 2020). Licence CC-BY-NC-4.0 → research use only; note in the corpus card. |
| Passage rule | ONE passage per source doc, 120–300 whitespace words, cut at a sentence boundary, per-doc target length drawn deterministically from the id | Length-matching later happens per bucket, so sources should not all sit on the 300-word cap. Deterministic target ⇒ identical passages on re-run. |
| Dedup | MinHash LSH (datasketch), 128 permutations, word 5-gram shingles, Jaccard ≥ 0.8 | Catches near-copies (shared Wikipedia intros between wiki_csai and wikipedia, reposted answers) without dropping merely similar text. Index is rebuilt from disk on every run. |
| HC3 cleaning | ELI5 / open_qa answers are PTB-detokenised; open_qa sentence spacing repaired; passages with U+FFFD dropped | Space-before-punctuation is a surface artefact that would make human text trivially separable from machine text. |

**Known limitation — Samanantar is shuffled sentences, not documents.** Checked
at offsets 0, 200k, 600k, 1.2M, 4M, 9M of the `hi` config and 0 / 2M of `te`:
every region is an unordered mix of unrelated sentences. The `indian` passages
are therefore runs of consecutive *filtered* sentences (complete, English-script,
6–60 words, not headlines): authentically Indian-authored at sentence level but
topically incoherent at passage level. Consequence for T5: Head A's discourse
features (burstiness, TTR) will see unusually diverse text in this band, so the
general–indian comparison must be read with that in mind and stated in the
paper's Limitations. If a document-level Indian-English source becomes
available (e.g. a licensed ICE-India slice, or PIB releases crawled with
document boundaries), it replaces this loader via `configs/data.yaml` only.

**Environment.** `venv/` had been moved from another folder and had no
interpreter; rebuilt with Anaconda Python 3.12.4 (`pyvenv.cfg` home) and
`requirements.txt` + `datasets`, `datasketch`, `pyarrow` (now pinned). torch is
the CPU build from `requirements.txt`; GPU parts need the CUDA build installed
over it (see the note at the top of `requirements.txt`).

## 2026-09-13 — Head C enabled; CUDA torch locked

**Head C enabled — RTX 4060 8 GB (7.9956 GiB); gate set to 7.5 GiB.**

The `>= 8.0` gate in `scripts/check_hardware.py` compared torch's
`total_memory`, which is binary GiB, against a threshold written in decimal GB.
The card reports 8,585,216,000 bytes = 8.59 decimal GB = 7.9956 GiB, so it
failed by 0.0044 GiB on units alone, not on capability. Threshold lowered to
7.5 GiB in `scripts/check_hardware.py` and in `configs/models.yaml`
(`head_c_vram_threshold_gb`), which carried the same gate. 7.5 GiB admits
genuine 8 GB cards and still excludes 6 GB ones (5.6 GiB). The
non-negotiable #5 intent ("Head C only if VRAM >= 8GB") is unchanged; only the
unit is corrected. Head C (MuRIL, encoder-only, non-negotiable #9) is therefore
IN scope — parts-plan Part 16.

**Host profile** (`python scripts/check_hardware.py`): Windows 11, Intel 20
logical CPUs, 15.7 GB RAM, 1 x NVIDIA GeForce RTX 4060 Laptop GPU, 7.9956 GiB
VRAM, compute capability 8.9, ~6.85 GiB free at idle.

**torch pinned to the CUDA build.** `torch==2.6.0` from PyPI is CPU-only; every
GPU stage (generation, scoring, attacks) would have run on CPU without warning.
The venv now has `2.6.0+cu126` from `https://download.pytorch.org/whl/cu126`
(`torch.cuda.is_available()` True, fp16 matmul verified on device,
`bitsandbytes` 0.45.3 imports). `requirements.txt` keeps the `torch==2.6.0` pin
and its header now names the CUDA index command, since a plain
`pip install -r requirements.txt` silently reinstates the CPU wheel.

## 2026-09-13 — Human corpus (Part 2): hi and te sources

| Decision | Value | Why |
|---|---|---|
| `writer_L1_band` for hi / te / cm | `native` (added to `WRITER_L1_BANDS`) | These buckets are first-language writers; the general/indian L1-proxy split is an English-only construct for T5. |
| hi / te primary source | `ai4bharat/IndicCorpV2`, splits `hin_Deva` and `tel_Telu` | CC-0 (the most permissive licence in the corpus), crawled Indian news/web, large enough to fill both buckets alone. |
| hi / te fallback | `wikimedia/wikipedia` 20231101.hi / .te, page id <= 100,000 | Kept configured but UNUSED — IndicCorpV2 filled both targets, so neither bucket read a Wikipedia row. |
| Script purity gate | `script` + `min_script_ratio: 0.8` per source, checked on the finished passage | Per-language calibration is meaningless if the hi bucket contains English. 0.8 admits the Latin loanwords and acronyms that are normal in Indian news, and rejected 25 (hi) / 24 (te) passages. |
| Artefact rule extended to Devanagari | Space before danda (` ।` / ` ॥`) now counts as an artefact | Same class of scrape artefact as space-before-comma in English; only 3.4 % of candidate hi passages, 0 % of te. |

**IndicCorpV2 is addressed by split, not by config.** The brief assumed
`config hi`. The repo exposes ONE config, `indiccorp_v2`, whose splits are the
languages (`hin_Deva`, `tel_Telu`); `data/hi-*.txt` and `data/te.txt` are 26 GB
and 16 GB, so the loader always streams and never downloads. Format is one
document per line, blank-line separated.

**Yield is low, so `max_docs` is large.** Only ~7 % of Hindi documents and
~2 % of Telugu documents are long enough to give a 120-300 word passage, so the
"first ~20k docs" in the brief would have produced roughly 1,400 (hi) and 470
(te). `max_docs` is set to 120,000 (hi) and 400,000 (te); the actual runs read
32,646 and 106,950 documents before hitting 2,200 rows.

**Bug found and fixed in `script_ratio`.** Counting characters in the Devanagari
block directly put combining vowel signs in the numerator but not the
denominator, which made the ratio exceed 1.0 for ordinary Hindi. The numerator
is now the subset of the *letters* that are in the script's range.

### Two asymmetries to carry into the results

1. **Domain.** `en` is a mix (Reddit ELI5 QA, Wikipedia CS/AI, Wikipedia
   general, Indian news), while `hi` and `te` are 100 % `news_web`, because the
   brief made Wikipedia a pure fallback and IndicCorpV2 never fell short. Any
   per-bucket difference in T1/T5 is therefore partly a domain difference.
   Changing `max_share` on the IndicCorpV2 entries in `configs/data.yaml` (e.g.
   to 0.75) would mix Wikipedia in without touching code.
2. **Length.** Mean passage length is 210.6 words for `en` but 164.0 (`hi`) and
   161.3 (`te`), because IndicCorpV2 documents are short. Part 13's
   length-matching step must equalise this before any head is compared across
   buckets.

## 2026-09-13 — hi / te domain blend (IndicCorpV2 capped at 0.75)

`max_share` on the IndicCorpV2 entries for `hi` and `te` is now **0.75**, so
Wikipedia supplies the remaining ~25 % of each bucket. Both files were rebuilt
from scratch rather than resumed, so the mix is right throughout rather than
only in the rows added last.

| bucket | rows | IndicCorpV2 (news_web) | Wikipedia (wiki_general) | mean words | median |
|---|---|---|---|---|---|
| en | 2200 | — | 495 (22 %) | 210.6 | 207 |
| hi | 2200 | 1650 (75 %) | 550 (25 %) | 175.6 | 159 |
| te | 2200 | 1650 (75 %) | 550 (25 %) | 170.3 | 155 |

**Why.** With IndicCorpV2 uncapped, `hi` and `te` were 100 % crawled news while
`en` was a four-domain mix, so any per-bucket difference in T1/T5 would have
been partly a domain difference rather than a language one. The blend also
pulls mean length up (164.0 → 175.6 for hi, 161.3 → 170.3 for te), since
Wikipedia passages are longer, narrowing the gap to `en`. Part 13's
length-matching still has work to do.

**Wikipedia's rejection rate is much higher for Indic than for English.**
Hindi rejected 41 % of the articles it read (17 % artefacts) and Telugu 57 %
(39 % artefacts) — template holes of the `(; )` kind are far more common in the
Indic dumps than in `20231101.en`. Both still filled their 550-row quota from
936 (hi) and 1,263 (te) articles read, well inside the `max_page_id: 100000`
window, so the pre-2020 proxy is not strained.

## 2026-09-13 — Human corpus (Part 3): cm sources

| Decision | Value | Why |
|---|---|---|
| cm sources | COMI-LINGUA `TN` config (1175), cmu_hinglish_dog (917), HinGE (108) | The only three openly-licensed Romanised Hinglish corpora of usable size. Ordered pre-ChatGPT first, so 47 % of the bucket has provenance that rules out machine text. |
| COMI-LINGUA config | `TN` only, raw `Sentences` column | **The brief assumed COMI-LINGUA is Roman script; it is not.** LID/MLI/NER/POS carry *Devanagari* code-mixing, which does not belong in a Romanised bucket. Only `TN` has Roman-script sentences. The raw column is used rather than the annotator-normalised ones: normalisation is a third party's edit, and the unedited comment is the authentic human writing. |
| cm passage length | 15-300 words (bucket override of the global 120-300) | Romanised Hinglish occurs as comments and chat turns. At a 120-word floor the whole bucket yields well under 800 rows. |
| Code-mixing gate | `min_hindi_word_ratio: 0.3` against a curated Romanised Hindi **function-word** list | These sources all contain English-only rows that would otherwise land in the cm bucket. Function words carry the grammar, are a closed class, and romanise fairly stably. Forms colliding with English ("to", "is", "us", "me", "he", "main", "hi") are excluded, so English text scores ~0.00. |
| `code_mix_ratio` | Populated from the same measurement | The schema field finally has a real value: min 0.30, median 0.39, p90 0.50, max 0.73. Zero in every other bucket. |
| Informal spacing | Normalised (`clean_informal`), not rejected | `kaisa hai ?` is genuine informal typing, not corrupt text. Rejecting it cost ~20 % of the scarce pre-ChatGPT chat data; leaving it in would have been a shortcut feature no LLM reproduces. **`clean_artifacts.py` must apply the same normalisation to machine cm text.** |

**Curated list over transliteration.** The brief allowed either. A list
transliterated from the `hi` bucket would produce ITRANS-style forms
(`kyonki`, `kiyaa`) that real Hinglish spells differently (`kyunki`, `kiya`),
so recall would be poor where it matters. The curated list is ~200 function
words with their common spelling variants side by side.

**Result:** 2,200 rows, mean 55.2 words, median 25, script purity 1.000, zero
duplicates, well above the >= 800 accept threshold.

### The cm length problem — needs a decision before T1/T5

`cm` averages 55.2 words against 170-211 for the other buckets, and is
internally bimodal (chat 103 words, comments ~21). Curvature is strongly
length-dependent, so comparing `cm` to `en` confounds language with length, and
Part 13's length matching only equalises machine against human *within* a
bucket. Raising the cm floor is not an escape: at 50 words only the chat source
survives, giving ~900 rows and a single-domain bucket. The realistic options are
to run a length-controlled sub-analysis for cm, or to caveat the cm row
explicitly wherever it appears. Recorded rather than decided.

**Also of note:** the Hindi function-word list detects Hindi-matrix code-mixing
but not English-matrix. HinGE inserts Hindi *content* words into English
sentences ("a part of vikas of the adhosanrachna"), scoring a median 0.18, so
42 % of it was rejected and the bucket under-represents that mixing pattern.

## 2026-09-13 — cm: length-band table and calibration-source restriction

Both decisions answer problems raised when the cm bucket was built (see the
Part 3 entry below): cm is much shorter than the other buckets, and 53 % of it
comes from a source with no stated collection date.

### (a) Supplementary length-band table, 40-120 words, across all buckets

Every per-bucket table that includes `cm` is accompanied by a supplementary
table restricted to passages of **40-120 words**, computed for all four
buckets. The main tables keep the full corpus.

**Why.** cm averages 55.2 words against 170-211 elsewhere, and curvature scores
depend strongly on length, so a full-corpus per-bucket comparison confounds
language with length. A common length band is the cheapest control that needs
no new data: it holds length roughly constant and lets the language difference
be read on its own. 40-120 is the widest window with real mass in every
bucket - it covers the cm chat transcripts (median 93) and the shorter tail of
en/hi/te, while excluding the ~21-word comment mode that has no counterpart
elsewhere.

**Consequences.** The band is a *reporting* decision, not a corpus change: no
rows are dropped. Sub-band counts per bucket must be printed alongside the
table, because a band with too few rows in some bucket is not interpretable.
Part 22 produces this as a companion to T1 and T5. The headline claim still
comes from the full corpus, with the band table as the robustness check.

### (b) cm calibration uses only cmu_hinglish_dog + HinGE

The cm conformal calibration split draws **only** from `cmu_hinglish_dog` and
`hinge`. `comi_lingua` rows are routed to train/test only.

**Why.** The conformal threshold is fitted on human-only text and its guarantee
is exactly as good as the certainty that the text is human. COMI-LINGUA states
no collection date, so machine-written comments cannot be ruled out in it; both
other cm sources predate ChatGPT (2018-2021). A machine-written row inside the
calibration set inflates the human score distribution and silently loosens the
threshold - the one failure that would quietly void the project's central
claim. The other three buckets are unaffected, since every en/hi/te source
predates ChatGPT.

**Feasibility.** cmu_hinglish_dog (917) + hinge (108) = **1,025 rows**, against
the 1,000-row calibration floor (locked §5). That clears it by 25 rows, so the
cm calibration set is the whole of both sources and the floor is met exactly.
If either source shrinks on a rebuild, cm calibration drops below the floor and
the restriction has to be revisited rather than quietly broken.

**Mechanism.** `configs/data.yaml` carries `calibration_eligible` on every
source spec (default `true`, `false` on `comi_lingua`).
`src/data/freeze_splits.py` MUST enforce it: any row whose `source` is not
calibration-eligible is barred from the `cal` split and assigned to
train/test. The flag is per *source*, and rows already carry `source`, so no
rebuild is needed. `freeze_splits` must also assert the per-bucket calibration
floor after routing, so a violation fails loudly instead of producing a corpus
that looks fine.

## 2026-09-13 — Head B scorer LOCKED: ai-forever/mGPT (tokenizer fertility)

`scripts/tokenizer_fertility.py`, 200 human passages per bucket from
`data/raw/human/`, tokenizers only. Full table in
`results/tokenizer_fertility.csv`.

| tokens per word | en | hi | te | cm |
|---|---|---|---|---|
| **ai-forever/mGPT** | 1.363 | 3.560 | 6.319 | 1.714 |
| Qwen/Qwen2.5-0.5B | 1.339 | 4.804 | 11.919 | 1.672 |
| mGPT advantage | — | **1.35x** | **1.89x** | — |

| tokens per char | en | hi | te | cm |
|---|---|---|---|---|
| ai-forever/mGPT | 0.226 | 0.684 | 0.776 | 0.324 |
| Qwen/Qwen2.5-0.5B | 0.222 | 0.923 | 1.464 | 0.316 |

**Decision: mGPT is the Head B scorer.** The two tokenizers are
indistinguishable on `en` (1.363 vs 1.339) and on `cm` (1.714 vs 1.672, since
Romanised Hinglish is Latin script and both handle it as English-like). They
diverge sharply on the Indic scripts: Qwen needs 1.89x as many tokens per
Telugu word and 1.35x per Hindi word. Fast-DetectGPT curvature is computed from
per-token conditional log-probabilities, so heavier fragmentation spreads the
same information across more, individually less predictable tokens and weakens
the signal in precisely the buckets this project exists to serve. `Qwen2.5-0.5B`
stays as the documented fallback for hosts where mGPT-1.3B will not fit, at a
known cost in Indic sensitivity.

**Fertility is itself a result, not just a setup detail.** Even with mGPT,
Telugu costs 4.64x as many tokens per word as English and Hindi 2.61x. This is
the quantitative explanation for any per-bucket AUROC gap Head B shows, and it
is what F2 (fertility vs AUROC) plots. Report it before explaining a weak
Telugu number.

**gemma-2-2b and Llama-3.1-8B were skipped: both are gated** and this host has
no HF token. The script records them as `status=gated` rather than failing.
Running `huggingface-cli login` and accepting both licences would fill those
rows; it is not required, since the scorer choice is settled by the two models
that did run.

---

## 2026-09-13 — Machine text: Ollama backend, one seen generator per passage, fertility-aware budget

**Why.** The first Stage 3 run (`Qwen/Qwen2.5-7B-Instruct`, transformers +
bitsandbytes nf4, batch 8, RTX 4060 laptop at 88 of 90 W) managed 0.12–0.13
passages/s, about 26 kept tokens/s. The GPU was 75–97 % busy and Python used
one core, so the limit is per-token nf4 dequantisation, not idle hardware.
Projected: ~32 h for one generator over 8,800 prompts, ~18 sittings against the
2 planned. The same run exposed a budget bug: `words x 4 + 64` new tokens cut
every hi and te passage short, because Qwen needs 4.8 tokens per Hindi word and
11.9 per Telugu word. Its 72 rows (en only) were deleted so that no generator
file mixes quantisations. Supersedes the 2026-09-02 seen/held-out generator rows.

### 1. Ollama is the generation backend; HF stays as fallback

- `src/data/generate.py --backend ollama` (default) posts to `/api/generate`
  with `stream=false` and 4 concurrent requests from a thread pool. The server
  only runs them in parallel with `OLLAMA_NUM_PARALLEL=4`: `setx
  OLLAMA_NUM_PARALLEL 4`, then quit and reopen the desktop app; `server.log`
  prints `OLLAMA_NUM_PARALLEL:4` when it took effect. Every batch logs tokens/s
  and the effective concurrency (summed decode time / wall time), so a server
  that is queueing shows up as `x1.0` within one batch.
- `--backend hf` (transformers + bitsandbytes 4-bit) is for a generator Ollama
  cannot serve. It replaces a generator wholesale: generate.py refuses to
  append to a file written by another backend or model.
- stdlib `urllib` only, so no new dependency. Per-request seed =
  `sha256(seed:id)`.
- **Persistent worker pool (same day, after the first minutes of qwen7b).**
  The first client sent 8-row batches, 4 requests at a time, and waited for
  the whole batch before starting the next, so slots idled behind each
  batch's longest request: batch 2 ran at 79.7 tok/s (concurrency x2.6), batch
  3 at 46.7 tok/s (x2.1). Now `run_concurrent` (src/utils/resumable.py) keeps 4
  worker threads that each pull the next prompt from a shared queue and post
  on their own. Each row is written under a lock the moment it completes, and
  resume semantics are unchanged. The model is loaded once before the workers
  start (an empty-prompt request), so the first requests no longer queue
  behind a ~75 s load. Progress and tokens/s are logged every 8 completed
  requests.

### 2. Quantisation: Q4_K_M for all five generators

Tags resolved against `registry.ollama.ai` on 2026-09-13:

| alias | requested tag | tag used | quant | size | model blob |
|---|---|---|---|---|---|
| qwen7b | `qwen2.5:7b-instruct` | same | Q4_K_M | 4.68 GB | `2bada8a74506` |
| gemma | `gemma2:9b` | `gemma2:9b-instruct-q4_K_M` | Q4_K_M | 5.76 GB | `cb654129f57b` |
| mistral | `mistral:7b-instruct` | same (= v0.3-q4_K_M) | Q4_K_M | 4.37 GB | `f5074b1221da` |
| llama | `llama3.1:8b-instruct` | `llama3.1:8b-instruct-q4_K_M` | Q4_K_M | 4.92 GB | `667b0c1932bc` |
| phi | `phi3.5:3.8b` | `phi3.5:3.8b-mini-instruct-q4_K_M` | Q4_K_M | 2.39 GB | `3ef532675b66` |

`gemma2:9b` and `phi3.5:3.8b` resolve to **Q4_0**, and `llama3.1:8b-instruct`
does not exist (404), so those three use explicit Q4_K_M tags. The whole point
of the switch is one quantisation throughout. Every row records
`decoding.quantization` (from `/api/show`) and the model digest.

**For the paper (limitations):** the corpus is 4-bit K-quant output, not fp16
or API output. It is the same for seen and held-out generators, so it is not a
seen-vs-held-out confound, but transfer to unquantised or commercial generators
is untested.

**Gemma VRAM, expected:** 5.76 GB of weights plus KV cache for 4 slots x 3,072
tokens (~2.1 GB at 42 layers x 8 KV heads x 256 dims) exceeds 8 GB, so Ollama
will offload layers to CPU. Check its first batches; if concurrency collapses,
lower `parallel` or `num_ctx` for that run.

### 3. Assignment: one seen generator per human passage; held-out 30 % slices

- **Seen:** `sha256(prompt_id) mod 3` → qwen7b / gemma / mistral. Python's
  `hash()` is salted per process, so it would reshuffle the assignment each run.
  On the 8,800-passage corpus: qwen7b 2,874, gemma 2,997, mistral 2,929 (sums to
  8,800; each bucket 701–790 per generator).
- **Held-out:** an independent `sha256("heldout:" + prompt_id)` mapped to [0, 1):
  below 0.30 → llama, 0.30–0.60 → phi, else none. Disjoint by construction and
  spread evenly across the three seen generators: llama 2,727, phi 2,605. Test
  only; `configs/data.yaml heldout_generators: [llama, phi]`.
- **Why:** every human passage gets exactly one seen machine counterpart, so
  human : seen-machine is 1:1 per bucket with prompt-matched pairs intact, and
  seen generation costs 8,800 calls instead of 26,400.
- **Carry into Part 13:** a held-out passage shares its prompt with one seen
  machine passage and one human passage. `freeze_splits` should keep each
  `prompt_id` group inside one split, so a topic never sits in train on one side
  and test on the other. Decide there.

### 4. Token budget: `requested words x fertility(bucket) x 1.3 + 64`, cap 2048

Fertility is each generator's own tokenizer (tokenizer-only downloads; all five
official repos are accessible to this HF account). Mistral loads from its
`tokenizer.json`, because AutoTokenizer asks for sentencepiece, which is not in
`requirements.txt`. The Qwen2.5 row is the fallback for a generator without
rows. `results/tokenizer_fertility.csv` is regenerated with them; the mGPT and
Qwen2.5-0.5B rows reproduce exactly, so the scorer lock is untouched.

| tokens per word | en | hi | te | cm |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct (qwen7b) | 1.339 | 4.804 | 11.919 | 1.672 |
| gemma-2-9b-it (gemma) | 1.297 | 2.023 | 4.942 | 1.506 |
| Mistral-7B-Instruct-v0.3 (mistral) | 1.460 | 5.359 | 13.309 | 1.889 |
| Llama-3.1-8B-Instruct (llama) | 1.306 | 2.706 | 13.819 | 1.651 |
| Phi-3.5-mini-instruct (phi) | 1.495 | 5.562 | 20.286 | 1.908 |

The previously gated gemma-2-2b row now measures too (same tokenizer as
gemma-2-9b-it). It fragments Indic text less than mGPT, but at 2.6B params it
is above the 2B cap (non-negotiable #5), so it is not a scorer candidate.

**The cap still binds on Telugu.** Budgets clipped at 2,048: qwen7b te 647 of
712, mistral te 650 + hi 42, llama te 624, phi te 687 + hi 33, gemma none. A
clipped budget is not automatically a cut passage, since the model stops when
it is done. Real cuts are recorded per row as `done_reason == "length"` /
`truncated`. At exactly the requested length, 2,048 tokens hold about 170 Telugu
words for Qwen and about 100 for Phi, so Qwen's te requests of 200–300 words (754 of
2,200 human te passages) and almost all of Phi's te will be cut. Part 13 drops
or length-matches truncated rows; if te machine counts fall too low, the lever
is a larger te cap with `num_ctx` raised to match (3,072 now covers the longest
prompt, ~860 tokens for Phi te, plus 2,048).

---

## 2026-09-18 — Indic prompt regime: compliance, not truncation (Part 6)

Telugu generation was stopped at 11.7 tok/s per stream (below the 15 tok/s
floor) after `en` 730/730 and `hi` 731/731. Chasing the slowdown surfaced a
larger problem: machine passages are far shorter than the human passages they
are matched to, and the gap scales with how Indic the bucket is.

| bucket | n | requested | machine | human | machine/human |
|---|---|---|---|---|---|
| en | 730 | 211 | 190 | 210 | **0.90** |
| hi | 731 | 174 | 99 | 174 | **0.57** |
| te | 12 | 158 | 56 | 153 | **0.37** |

### It is under-production, not truncation

`done_reason` is `stop` on 100 % of those rows and none hit `num_predict`. The
fertility-aware budget is working as designed; the model simply declines to
write long Indic text. This is a *different* failure from the cap-clipping
recorded in section 4 of 2026-09-13, and the lever recorded there — a larger
`te` cap with `num_ctx` raised — does not address it on its own, because we
never reach the cap to begin with.

### Prompt probe (12 te passages x 4 variants, qwen2.5:7b-instruct Q4_K_M, temp 0.8)

| variant | words | natural stop | telugu script | 5-gram repeat | m/human |
|---|---|---|---|---|---|
| v0 plain (the live template) | 59 | 12/12 | 0.96 | 0.00 | 0.31 |
| v1 instruction written in Telugu | 76 | 12/12 | 0.99 | 0.01 | 0.40 |
| v2 compensated target (3x) | 166-170 | 1/12 | 1.00 | 0.01 | 0.88 |
| v3 explicit "do not stop early" | 116 | 11/12 | 1.00 | 0.00 | 0.62 |

Two readings matter. **v2's 0.88 is not compliance** — 11 of 12 hit the 2,048
cap at a mean of 2,041 tokens and the tail is cut mid-word, so its true ratio is
unmeasured. **v3 is a real gain**: it nearly doubles output and still terminates
naturally 11 times in 12. Quality does not pay for the longer framing — Telugu
script purity *rises* (0.96 -> 1.00) and 5-gram repetition stays near 0.01, so
neither script drift nor looping is a cost.

### The cap is a second, independent constraint

At 11.919 tokens/word the human `te` mean of 170 words needs ~2,025 tokens and
the 300-word tail ~3,575, against a 2,048 cap. Even perfect compliance could not
represent the upper half of the bucket. Compliance and cap have to move
together; fixing either alone changes nothing.

### Decision

- **New template for `hi`, `te`, `cm`** (`PROMPT_TEMPLATES["indic"]` plus a
  rewritten `cm` entry): explicit "at least N words, N/12 or more full
  sentences, do not stop early and do not summarise".
- **`en` stays on the plain template.** At 0.90 it needs no fix, and changing it
  would invalidate 730 finished rows for no gain. **The corpus therefore uses
  two prompt templates, which the paper's corpus section must state:** `en` is
  not comparable to `hi`/`te`/`cm` at the prompt level.
- **Compensated target**: the prompt asks for `human_words / compliance_ratio`,
  clamped to `cap / fertility` so the ask can never exceed what the cap holds.
  Ratios are measured *under the template the bucket actually uses* — te 0.62
  from v3, not 0.31 from v0. Compensating a control-template ratio on top of the
  new framing would double-count and drive every passage into the cap. hi (0.85)
  and cm (0.85) are estimates, gated by the check below.
- **Budget raised**: `num_predict = human_words x fertility x 1.5 + 128`, cap
  3,600, `num_ctx` 4,608. The base is now the human length, not the requested bin.
- **Out-of-memory at load drops parallelism** (`load_with_fallback`) instead of
  ending a multi-hour run; `gemma2:9b` at Q4_K_M is the expected trigger on 8 GB
  (2026-09-13 section 2). Caveat recorded in the function: the server's
  `OLLAMA_NUM_PARALLEL` governs slot allocation, so this reduces pressure but
  cannot shrink an allocation the server already made.
- **`hi` and `te` rows for qwen7b are discarded and regenerated**, `en` kept.
  Two prompt regimes inside one bucket would be a confound in exactly the
  per-language comparison this project reports.

### Telugu scope reduced to 400 passages per generator

Telugu costs 11.919 tokens per word against English's 1.339 — about 8.9x — so a
te passage of the same word length costs roughly nine times the decode. At the
measured 46 tok/s aggregate the full 712-passage te assignment is on the order of
10 h for one generator, and would dominate generation across all five. te is
therefore capped at **400 passages per generator**, selected stratified across
the human length bins (largest-remainder allocation, then a deterministic
`sha256(id)` ordering inside each bin) so the bin mix is preserved:

| bin | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|
| source share (2,200) | 10.0 % | 55.7 % | 19.3 % | 10.8 % | 4.2 % |
| selected of qwen7b's 712 | 37 | 228 | 76 | 43 | 16 |

The chosen ids are frozen to `data/processed/selection/<generator>__te.json` and
reused by every later run, so the corpus cannot change silently if the selection
rule is ever edited.

**Section 3 is preserved.** A shared 400-passage panel generated by all five
generators was considered and declined: it would have made te human:seen-machine
1:3 rather than 1:1 and broken the one-seen-generator-per-passage rule locked in
section 3 of 2026-09-13. Generators therefore still cover **disjoint** te
passages, and cross-generator comparison on te stays unpaired.

**For the paper:** this belongs in the corpus section (te n per generator) and in
Limitations — te is the one bucket whose machine side is a subsample, and the
reason is decode cost, not data availability.

**Gate before committing to the full run:** after the first 24 rows of each
bucket, stop if machine/human is below 0.75 or more than 20 % hit the cap.

---

## 2026-09-18 — Three scope reductions (Part 6)

All three shrink cost without touching the contribution. The four buckets, Head
C, the fairness table, conformal calibration and the held-out generators are
explicitly unchanged. **All three belong in the paper's Limitations section.**

### 1. Hindi capped at 500 passages per generator (from 731)

Same treatment as te's 400: each generator caps its **own disjoint** assignment,
selected stratified across the human hi length bins, ids frozen under
`data/processed/selection/`. Hindi runs 4.804 tokens/word on Qwen's tokenizer
(3.560 on mGPT, the scorer) against English's 1.339, so it is the second-largest
decode cost after Telugu. 500 per generator still yields ~1,500 hi machine rows
across the three seen generators — ample for a stable per-bucket AUROC and well
above the 1,000-text floor that per-language conformal calibration needs.

| bucket | was | now | why |
|---|---|---|---|
| te | 712 | **400** | 11.919 tokens/word, ~8.9x en |
| hi | 731 | **500** | 4.804 tokens/word, ~3.6x en |
| en, cm | full | full | cheap to decode |

### 2. Vanilla DetectGPT dropped as a baseline

The `detectgpt` column is removed from `src/eval/score.py`'s registry and Parts
19–20 are deleted from the plan. Fast-DetectGPT is its efficient successor and
computes an identical statistic analytically rather than by 100 T5-large
mask-fill perturbations per text; running both cost two GPU sittings and
produced no evidence the other did not. **The paper still cites DetectGPT
(Mitchell et al. 2023) as the origin of the curvature hypothesis** and
benchmarks against Fast-DetectGPT. Baselines are now **ppl, fastdetectgpt_en,
binoculars, xlmr**, and `results/scores.parquet` carries 7 method columns, not 8.

`src/baselines/detectgpt.py` and its stub test are left in place but
unregistered — dead code to remove deliberately later, not silently now.

### 3. Attacks sample the test split rather than covering it

`src/data/attack.py` attacks **TEST-split rows only**, and samples per bucket per
attack: at most **400 machine** rows and **150 human** rows, stratified by length
bin *and by generator* so no generator drops out of T4. T4 needs a precise
estimate of degradation, not the whole test set — 400 rows per cell holds the
standard error on AUROC under 0.02. Attacking `cal` rows would also contaminate
the conformal thresholds, which the TEST-only rule now rules out explicitly.
Stage 6 collapses from Parts 23–28 to Parts 23–26 (~2 attack sittings instead of
four).

---

## 2026-09-18 — cm over-production: the sentence floor, not the framing

The Indic regime was applied to `cm` on the assumption that it behaved like `hi`
and `te`. It does not, and 470 `cm` rows were generated at **1.88 of human
length** (median 1.98, p90 4.94, 91 % above 1.25x) before the gate caught it —
the gate as first written only tested the *lower* bound and waved a 1.72 through
as a pass.

### Attribution

| bucket | human | bin | ask | machine | ask/bin | mach/ask | mach/hum |
|---|---|---|---|---|---|---|---|
| en | 210 | 211 | 211 | 190 | 1.00 | 0.90 | 0.90 |
| cm | 74 | 75 | 88 | 139 | 1.18 | **1.59** | **1.88** |
| te | 163 | 166 | 250 | 154 | 1.50 | 0.62 | 0.94 |

Length binning was *not* the cause (`bin/hum` = 1.00). Splitting `cm` by whether
the `max(8, n/12)` sentence floor was active:

| group | n | human | ask | sentences | machine | mach/hum |
|---|---|---|---|---|---|---|
| floor binds (ask < 96) | 278 | 34 | 39 | 8.0 | 91 | **2.69** |
| floor free (ask >= 96) | 192 | 133 | 157 | 12.4 | 209 | 1.58 |

Eight sentences implies ~88 words regardless of the word target, so the sentence
clause silently overrode the word clause. Worst case: 21 human words asked for
"29 words / 8 or more sentences" produced 84 words, **4.0x**.

### Probe: 12 cm passages x 3 variants, ask = bin, no compensation

| variant | m/hum | m/ask | short subset (<40 w) |
|---|---|---|---|
| plain original template | 0.76 | 0.78 | 0.73x |
| sentence floor fixed, `max(2, n/15)` | **1.08** | 1.10 | 1.20x |
| sentence clause removed entirely | 0.78 | 0.79 | 0.71x |

**Both predictions were wrong, and that is the corpus-section finding.** The
expectation on both sides was that `cm` should revert to the plain template — that
the insistent framing was simply wrong for a bucket which had never
under-produced. The data says the opposite: without the sentence clause `cm`
under-produces at 0.76, and "do not stop early" on its own does nothing (0.78 is
within noise of 0.76). `cm` **needs** the sentence clause; only the floor was
wrong — `max(2, n/15)`, not `max(8, ...)`.

The general lesson, which the paper should state: prompt-length behaviour does
not transfer between buckets by analogy. `en` at 0.90 predicted nothing about
`cm`, and `te`'s under-production predicted nothing about either. Every bucket's
regime is now measured on a 12-passage probe **before** its rows are generated,
not diagnosed from 24 rows afterwards.

### Decision

- `sentence_target()` splits the rule: `cm` uses `max(2, n/15)`, hi/te keep
  `max(8, n/12)`. Their bins are all >= 100, so the floor never bound for them
  and their prompts are byte-identical — the 56 good `te` rows stay valid.
- `cm` compliance ratio 0.85 -> **1.0**. It writes 1.10x what it is asked for, so
  compensation would overshoot. Note `compensated_words` cannot express a ratio
  above 1.0 (it never asks for fewer words than the human passage); tuning `cm`
  below 1.0 would need that constraint relaxed.
- **The gate gains an upper bound.** Failing only below 0.75 is what let this run
  to 470 rows; it now fails outside ~0.75-1.25 in both directions.

**Carry into Limitations:** `cm` machine length is prompt-steered by a sentence
count rather than a word count, because the model ignores word targets on short
Romanised text.

---

## 2026-09-18 — hi regime, and the per-bucket prompt table

`hi` was probed **before** generating its 500 rows rather than after 24, which is
the practice cm's failure forced. It paid for itself immediately.

### Probe: 12 hi passages x 3 variants, ask = bin, no compensation

| variant | machine | m/hum | m/ask | done_reason |
|---|---|---|---|---|
| plain original | 101 | **0.57** | 0.55 | 12 stop |
| insistent + sentence clause | 248 | **1.40** | 1.35 | 10 stop, 2 length |
| insistent, no sentence clause | 215 | **1.21** | 1.17 | 11 stop, 1 length |

The regime `hi` was about to run — insistent framing with compensation 0.85 —
would have produced **~1.59** of human length: a 1.35 over-producer with its ask
inflated a further 1.18x. That is cm's failure repeated, caught before the rows
existed. Note also that the plain template reproduces **0.57**, exactly the ratio
the real 731-row `hi` corpus showed, which is independent confirmation that a
12-passage probe predicts bucket behaviour well enough to configure from.

### Why `hi` takes the plain template, not the best raw ratio

On raw numbers v2 (1.21) is closest to 1.0, and the obvious move is to set
`compliance_ratio: 1.17` and divide it back down. That does not work:
`compensated_words` only ever *inflates* the ask — it clamps `ratio <= 1.0` and
never asks for fewer words than the human passage. A 1.17 in the config would be
silently clamped to 1.0, the ask would be unchanged, and `hi` would generate at
1.21 while the config claimed otherwise.

Relaxing that clamp was considered and **rejected**: it widens a core function's
contract to serve one bucket, and the alternative needs no code change at all.
Since compensation can only correct *under*-production, the right template for
`hi` is the one it under-produces under — the plain one, at 0.55. The ask is
inflated x1.8, exactly the mechanism `te` already uses (x1.6 -> 0.94 measured on
56 real rows).

`hi` is therefore the plain template at ratio 0.55, and the rule generalises:
**pick the template whose bias the existing mechanism can correct.** Two
independent measurements support the value — the probe (0.55) and the real
731-row `hi` corpus generated under that template (0.57).

### Per-bucket prompt regime — for the paper's corpus section

**Every value measured on that bucket's own probe. None of it transfers by
analogy:** `en` at 0.90 predicted nothing about `cm`, and `te`'s under-production
predicted nothing about `hi`.

| bucket | template | compliance ratio | effect on the ask | measured machine/human |
|---|---|---|---|---|
| en | plain "write about N words" | 1.00 | unchanged | **0.90** (730 rows) |
| hi | plain "write about N words" | 0.55 | inflates to 1.82n | 0.55-0.57 of ask → ~1.00 expected |
| te | insistent + sentence clause | 0.62 | inflates to 1.61n | **0.94** (56 rows) |
| cm | insistent + sentence clause, floor `max(2, n/15)` | 1.00 | unchanged | **1.08** (probe) |

Four buckets, three different templates and three different ratios. The corpus
section must state this: prompt regime is a per-bucket empirical choice, not a
single design applied uniformly, and `en` in particular is not comparable to the
other three at the prompt level.

---

## 2026-09-19 — the gate is code, and its statistic is the median

The 0.75–1.25 rule decided on 2026-09-18 lived in this file and in a watcher run
by hand in a second terminal. It is now `run_with_gates` in
`src/data/generate.py`, with thresholds in `configs/data.yaml` under
`machine_corpus.gate`.

**Why it moved.** A check that is a habit rather than code runs only when
someone remembers to run it. That is the whole of the cm failure: the rule
existed, and 470 rows were generated anyway.

**Where it sits.** A preflight phase inside `generate.py`, not a new callback in
`src/utils/resumable.py`. Adding a row hook to the spine would widen the
contract of a file `score.py` and `attack.py` also depend on, to serve one
caller — the same reasoning that rejected relaxing `compensated_words`' clamp
for `hi` on 2026-09-18. The preflight also aborts *before* any bulk work exists,
which a hook cannot: it can only stop a run already in progress.

**The statistic is the median of per-row ratios.** cm's over-production was
concentrated in the short passages (21 human words → 84 machine), so a
length-weighted ratio of totals dilutes exactly the rows that are wrong, and a
mean is dragged by one runaway row. On the 470 discarded cm rows the median read
1.98 against a mean of 1.88. `cap %` is the share of rows with
`done_reason == "length"`.

**Deliberately awkward, in two places.** Failed probe rows stay on disk and make
every resume fail the same gate until they are removed, and there is no
`--skip-gate` flag. Both are there so the gate cannot be walked past the way the
watcher could.

---

## 2026-09-20 — the 24-row gate under-reads in both directions

The gate locked on 2026-09-19 passes a bucket on the median ratio of its first
24 rows. Two buckets have now been carried far past that probe, and in both the
median moved — in opposite directions.

| bucket | gate (24 rows) | settled | n | rows outside 0.75–1.25 |
|---|---|---|---|---|
| `te` | 0.99 | **0.94** | 382 | 24.6 % |
| `hi` | 1.00 | **1.07** | 256 | 31.6 % |

**The median is the right tripwire and the wrong estimator.** It caught `cm` at
1.98 on the 470 discarded rows, which is the job it exists for: a structurally
broken regime moves the median far outside the band, fast, and 24 rows are
enough to see it. What 24 rows cannot do is predict where a *working* bucket
settles — `te` drifted −0.05 and `hi` +0.07 off the same reading. **A gate
figure must therefore never be quoted as a bucket's ratio** in the corpus card
or the paper; report the settled median at the bucket's final n.

**It also says nothing about spread, and the spread is large.** Both buckets
pass on a median comfortably inside 0.75–1.25 while a quarter to a third of
their rows sit outside it. A bucket could in principle hold a 1.00 median with
every row at 0.5 or 1.5 and pass untouched. That was a one-bucket suspicion on
2026-09-19; two buckets now support it.

**Decision: add a dispersion check to the gate — but not this sitting.**
> **Superseded 2026-09-20** by "the gate fix is a stratified probe, not a spread
> threshold" below: the drift is a selection effect, not a sample-size problem,
> and no dispersion threshold was added.
Deferred until `te` and `hi` are complete, for two reasons. The threshold should
be set from settled p90 values (`te` 1.25, `hi` 1.40) rather than guessed. And
changing the gate mid-corpus would leave one generator's rows judged under two
gate regimes — the same objection that makes the corpus's two prompt templates
something the paper must state outright. No change to `configs/data.yaml` yet.

---

## 2026-09-20 — Run 1's 11.7 tok/s was configuration, not Telugu

Run 1 was halted on 2026-09-18 when Telugu per-stream decode fell to 11.7 tok/s
against a 15 tok/s floor. The open question since was whether `te` is simply too
slow to generate on this box.

It is not. Across 326 `te` rows in one 111-minute sitting:

| | per-stream |
|---|---|
| Run 1 (halted) | 11.7 tok/s |
| floor | 15 tok/s |
| `te`, warm-up peak | 31.0 tok/s |
| **`te`, settled plateau** | **20.4–20.5 tok/s** |
| `hi`, for comparison | 20.8 tok/s |

**`te` sustains the same per-stream rate as `hi`.** The plateau held at 20.5 for
the last 30 minutes at concurrency x3.3–4.2, so the fall from the warm-up peak
is decode slowing under sustained load (most likely thermal), not workers going
idle. It was also checked against row length rather than assumed: the final rows
averaged 1,874 gen-tokens against the bucket's 1,893, on ordinary-length inputs,
so the queue had not reached the long passages.

**Consequence:** `te` needs no throughput special-casing, no floor change and no
`--max-minutes` change. `te` is expensive in *tokens* — 1,893 gen-tokens/row
against `hi`'s 869 and `cm`'s 161 — which is a budget fact, not a speed fault.
Run 1's halt belongs to that run's configuration and the corpus section should
describe it that way.

---

## 2026-09-20 — the gate fix is a stratified probe, not a spread threshold

**Supersedes the deferred dispersion check** recorded earlier on 2026-09-20.
That entry read the drift between a bucket's gate figure and its settled median
as a sample-size problem and proposed adding a spread threshold once `te` and
`hi` finished. Finishing all four buckets shows the diagnosis was wrong, and the
correct fix is cheaper and catches strictly more.

### What the completed corpus shows

| bucket | gate (24) | settled | n | drift | outside 0.75–1.25 |
|---|---|---|---|---|---|
| en | 0.87 | 0.87 | 730 | **+0.00** | 34.7 % |
| te | 0.99 | 0.94 | 400 | −0.04 | 24.8 % |
| hi | 1.00 | 1.10 | 500 | +0.10 | 34.0 % |
| cm | 1.02 | **1.39** | 701 | **+0.38** | **67.0 %** |

`en` is the control and it settles exactly where its probe read it. A gate whose
problem was sample size would drift on `en` too. It does not, so the drift is
not noise — it is a selection effect that varies by bucket.

### The cause: the probe is the head of the list, not a sample of the bucket

Probe rows were `pending[:24]` — the head of a list sorted by human id — so the
probe inherits whatever length mix that ordering front-loads. For `cm` the two
distributions are not close:

| | human words, median | share in the 0–40 bin |
|---|---|---|
| first 24 (the probe) | **82** | 12.5 % (3 of 24) |
| all 701 | **26** | **59 % (413 of 701)** |

### The defect is bin-local, which is why this mattered

| human-length bin | n | median | in band |
|---|---|---|---|
| **0–40** | **413** | **1.75** | **no** |
| 40–70 | 74 | 0.94 | yes |
| 70–100 | 78 | 1.21 | yes |
| 100–150 | 79 | 1.07 | yes |
| 150–250 | 49 | 1.00 | yes |
| 250+ | 8 | 1.36 | no (n = 8) |

Every bin from 40 words up is inside the band. `cm` fails as a bucket because
the bin that fails is 59 % of it. A probe that under-samples that bin by 5x
cannot see the defect at any sample size — and a spread threshold would not have
saved it either, since the probe's own 24 rows are mostly well-behaved passages
whose spread is unremarkable. **The fix has to change which rows are probed, not
what statistic is computed over them.**

### Decision

`run_with_gates` draws its probe with `stratified_probe`, allocating the 24 rows
proportionally across the bucket's `n_words` bins. It reuses `stratified_cap`'s
largest-remainder allocation, so the probe and the per-generator bucket caps
stratify by the same rule and a resume draws the same passages. Prompts without
`n_words` keep the previous head order.

On `cm`'s real shape this puts ~14 of 24 probe rows in the 0–40 bin, which reads
1.75 and **fails the bucket at 24 rows** — what the gate exists to do. Covered by
`test_a_bin_local_defect_fails_the_gate`, with
`test_head_order_probe_would_have_missed_the_bin_local_defect` asserting the old
selection passes the same bucket, so the regression cannot come back silently.

**No dispersion threshold is added.** The spread figures above (24–35 % of rows
outside the band even in healthy buckets) are a property of prompt-matched
generation at these lengths, not a defect signal, and thresholding them would
fail `en` — a bucket with zero drift and a correct regime. Revisit only if a
bucket is ever found whose median and bins are all in band while its tails are
not.

**This does not retroactively fix `cm`'s 701 rows.** They were generated under
the old probe and sit at 1.39; the decision on them — regenerate the 0–40 bin
under a corrected prompt, drop the bin, or accept and document — is open, and
nothing has been discarded.

---

## 2026-09-22 — the cm regime, its two new gate checks, and v0 → v1

What the 2026-09-21 and 2026-09-22 sittings settled. The code landed in `61fe922`
(template + ask rule), `a796da6` (code-mix check) and `e17674e` (emoji check); this
is the decision record those commits were missing.

### The `cm` prompt regime

| Decision | Value | Why |
|---|---|---|
| `cm` template | `PROMPT_TEMPLATES["cm"]`: "about N words … in Hinglish: mix Hindi and English words within every sentence, with the Hindi written in Roman script, casual, like students texting each other, as S or more full sentences. No emoji. Continue naturally and do not stop early." | The template it replaced wrote English in all but name: romanised-Hindi share 0.02–0.04 against a human 0.30–0.33. Probed 3 variants × 24 stratified passages; this one (v1) was the only one with zero emoji rows and a length median in band. |
| `cm` ask comes from the human length, not the bin | `ASK_FROM_HUMAN` / `budget.ask_from_human: {cm: {long_from_words: 40, long_ratio: 0.75}}` | 59 % of `cm` sits in the 0–40 bin, where the bin value 25 is already 1.25× a 20-word passage before the model writes a word. One ask ratio does not fit both ends: at ask = human, 0–40 read 1.13 and 40+ read 0.73, so only passages of 40+ words are inflated (ask = human / 0.75). |
| `compliance_ratio` is not applied to `cm` | `cm: 1.0` | The ask rule already sets the target. Applying both would double-count, the same error the te 0.62 note warns about. |
| Sentence clause for `cm` | `max(2, n_words // 15)`, following the ask | A floor of 8 sentences implies ~88 words whatever the word clause says, and silently overrides it. |

**Every figure above is measured on `cm`'s own probe.** No bucket's behaviour
transfers to another by analogy (2026-09-18, still holds).

### Two new gate checks, both `cm`-only by config

| Check | Rule | Why |
|---|---|---|
| Code-mix (`a796da6`) | For `gate.codemix_buckets` (`[cm]`), the probe rows' median romanised-Hindi function-word share must be ≥ `codemix_min_ratio` (2/3) of the median for **the same passages'** human text | Length alone passed 701 rows that were English in all but name. Paired, because a 24-row probe against a whole bucket's human share mixes a sample with a population. Function words, not content words: Hindi grammar has to be running through the text. yaar/yar/bhai are excluded — they are the cheapest way to look Hinglish without being it. |
| Emoji (`e17674e`) | For `gate.emoji_buckets` (`[cm]`), **any** probe row containing an emoji fails | Human `cm` has emoji in 0 of 2,200 rows, so one in a machine row is a class shortcut, not style. "No emoji" in the prompt is not enough: 2 of the 3 probed variants still produced one. |

**The emoji regex excludes the lone ZWJ (U+200D)**, which the probe scripts
counted. ZWJ joins Devanagari conjuncts: counting it scored 118 of the 2,200 human
`hi` passages as emoji. Inside a real emoji sequence the pictographs match anyway.

### v0 → v1

The 701 v0 `cm` rows were **moved to `data/raw/discarded/qwen7b_cm_v0.jsonl`, not
deleted** (`data/` is gitignored, so that file is local to this machine). This
supersedes the 2026-09-20 line "nothing has been discarded". v1 regenerated the
same 701 prompt_ids in 23.2 min.

| | overall | 0–40 | 40–100 | 100+ | Hindi share (human) | emoji |
|---|---|---|---|---|---|---|
| v0 | 1.39 | 1.75 | 1.05 | 1.06 | 0.03 (0.32) | 28 |
| **v1** | **1.13** | **1.18** | 1.07 | 1.11 | **0.35** (0.32) | 3 |

**The gate re-run on the whole finished bucket** — the check v0 never got, since a
24-row probe judged it and the bucket then settled elsewhere — gives v0 FAIL on all
three checks and **v1 FAIL on emoji alone, 3 of 701**. At 3/701 a 24-row probe has
about a 10 % chance of catching one, so the probe passing was arithmetic, not a
gate defect. A zero limit fails any nonzero rate once a whole bucket is judged.

**Decided: the residue is a cleaning problem, not a template problem.** v1's 3
emoji rows, ~23 instruction-echo rows and 13 truncated rows are row-level defects
in a regime whose length and code-mix both pass. They are dropped in Part 13
rather than fixed by another template round or another regeneration. The rules are
specified in `src/data/clean_artifacts.py`'s docstring and apply to every
generator, since gemma and mistral will produce their own.

**Not decided:** whether the long-ask rule needs a third band above 200 words. The
250 and 300 bins over-produce (1.70 / 1.93) with 8 of their 14 rows truncated;
`long_ratio: 0.75` was probed on 10 long rows whose human length reached 170. It is
4 % of `cm` and does not move the bucket median, so it is recorded and left alone.

---

## 2026-09-22 (later) — the gate judges the text cleaning will keep, not raw output

**What happened.** gemma's first `cm` gate probe failed: emoji in 2 of 24 rows.
Everything else was healthy — length median 0.97 (0–40 1.05, 40+ 0.92), Hindi share
0.29 against a human 0.33, 0 % at the cap, zero instruction echo, zero Devanagari.
Both emoji were inside a **trailing line of assistant chatter**: "Let me know if
you'd like me to continue the conversation! 😊". Neither was in the passage.

**The inconsistency this exposed.** The Part 13 spec written the same day says drop
rules run *after* stage-1 stripping, so a row whose only emoji sits in boilerplate
survives cleaning. The gate was applying a drop rule to raw output. The two orders
disagreed, and the gate's order is the one that aborts a 110-minute run.

| Decision | Value | Why |
|---|---|---|
| `strip_boilerplate()` lives in `src/data/clean_artifacts.py` | Imported by `generate.gate_stats` | One rule, one place. Cleaning owns it; the gate borrows it, so the gate cannot drift from what Part 13 will actually keep. |
| The gate strips before **every** check | Length (words recounted, not read from `length_words`), code-mix, emoji | A postamble inflates all three. Counting `length_words` would let chatter push a passage over the ratio ceiling. |
| The stored row keeps the **raw** text | Only the checks see the stripped version | Cleaning is Part 13's job and is measured there. The corpus keeps what the model produced, so the strip rule can change later without regenerating anything. |
| The rule matches a **whole final line**, up to 3 of them | `_BOILERPLATE_LINE` in `clean_artifacts.py` | Both anchors are load-bearing. In the same 24 rows, two passages use the same words *in character* — "You know kya film dekhne ka mood hai? Let me know!" — and a substring rule eats text the prompt asked for. The 3-line cap stops a pathological row from being stripped to nothing. |

**Evidence.** The stripper changes 3 of gemma's 24 probe rows, all of them the
offer-to-continue line; the probe then passes at length 0.96, Hindi 0.29 vs 0.33,
0 emoji. Across qwen7b's whole 2,331-row corpus it changes **1 row** — an `en`
passage ending "Would you like to delve deeper into a specific aspect…". qwen7b's
3 `cm` emoji rows are in-body (3 %, 11 % and 98 % through the text), so they still
fail, and the bucket's full gate figures are unchanged: 1.13 length, 0.35 vs 0.32
Hindi, 3 emoji.

**Recorded for the remaining generators.** Closing packaging is a per-model habit,
not a bucket property: gemma emits it in ~12 % of `cm` passages, qwen7b in 1 row
in 2,331. mistral, llama and phi each need their own measurement rather than an
assumption, and the same rule is what Part 13 will apply to all of them.

**Unchanged:** the emoji rule is still zero-tolerance *on the passage*. Stripping
decides what counts as the passage; it does not raise the limit.

---

## 2026-09-23 — a compliance ratio belongs to a generator, not just a bucket

**What happened.** gemma's `hi` gate aborted a 110-minute run after 24 rows:
median machine/human length **1.48**, against a ceiling of 1.25, with 83 % of rows
above it. `cm` (1.00) and `en` (0.80) had passed minutes earlier on rows already on
disk.

**The cause.** `compliance_ratio {en 1.0, hi 0.55, te 0.62, cm 1.0}` was measured on
**qwen** and applied to **gemma** unchanged, because the key is the bucket alone. A
compliance ratio is the fraction of the ask a model actually writes, so it is a
property of the generator *and* the bucket. gemma writes longer on Indic and shorter
on English than qwen (`en` 0.80 against qwen's 0.87), so hi's ×1.82 inflation — sized
for a model that under-produces at 0.55 — was applied to one that does not.

The arithmetic is exact. gemma's measured raw ratio on hi is 0.73, so under a 0.55
ask it writes 0.73 × 1.82 = 1.33 bins ≈ **1.46 of human**. Observed: 1.48.

decisions.md 2026-09-18 already required every regime to be measured on its own probe
and never carried across by analogy. The rule was in the prose; the config could not
express it, so it carried the value across silently. **The same trap is still armed
for mistral, llama and phi.**

### Measured — gemma's own raw ratios

`src/data/probe_compliance.py`, new: 12 passages per bucket, current template,
ask = the length bin with **no** compensation, `num_predict` at 3× headroom so
over-production is measured rather than truncated (0 of 24 rows truncated). Words
counted after `strip_boilerplate()`, the way the gate counts them.

| bucket | machine/ask (median) | mean | p10 | p90 | qwen's value | was |
|---|---|---|---|---|---|---|
| hi | **0.73** | 0.74 | 0.60 | 0.87 | 0.55 | ×1.82 over-inflated |
| te | **0.84** | 0.79 | 0.58 | 0.93 | 0.62 | ×1.35 over-inflated |

Both are machine/**ask**, the convention every existing value uses. Mixing in a
machine/human figure for one generator would be the same silent-config bug in a new
form.

**How the ask actually lands.** Output is regressed on ask across the 12 rows:

| bucket | slope (words out per word of ask) | intercept | r | predicted gate median |
|---|---|---|---|---|
| hi | 0.65 | 14 | 0.86 | **1.05** (0.84–1.17) |
| te | 0.36 | 69 | 0.73 | **0.89** (0.67–1.06) |

hi responds near-proportionally, so compensation does what it assumes. **te is
damped**: a large intercept and a shallow slope mean gemma writes ~69 words plus a
third of whatever is asked, so long te passages under-produce however the ask is
sized — 3 of 12 predicted rows fall below 0.75 even though the median passes. The
gate tests the median, so te is expected to pass, but te's long tail is a corpus
property to report, not something a ratio fixes.

| Decision | Value | Why |
|---|---|---|
| `compliance_ratio_by_generator` | New key: generator → bucket → ratio | The flat `compliance_ratio` stays every generator's default, so a generator absent from it resolves exactly as before and qwen7b's frozen rows are untouched. Falls back per bucket, so overriding hi alone leaves en and cm shared. |
| gemma's ratios | `{hi: 0.73, te: 0.84}` | Measured on gemma's own probe. en and cm are deliberately absent: gemma passed both under the shared defaults. |
| The run logs its regime | `compliance <bucket>: <value> measured on <generator>` / `shared default` | gemma inherited qwen's 0.55 for a whole run without one line of the log naming it. |
| A changed selection cap aborts | `load_or_record_selection` compares the recorded limit | A frozen record outranks the config and was reused verbatim, so a recut cap would have been read, logged as "reusing", and ignored — generating the old 500 passages while the config said 250. Same bug class, found while fixing this one. |

### Scope cuts — a compute constraint, not a statistical one

Measured on this card: gemma decodes hi at ~10 tok/s aggregate across 4 workers
against qwen's ~40, roughly 4× slower. At that rate gemma's remaining assignment
alone projected to **22.3 hours** (cm 1.8 h, en 4.1 h, hi 6.9 h, te 9.4 h) — te is
42 % of it, because Telugu runs 4.942 tokens/word against English's 1.297.

| Decision | Value | Why |
|---|---|---|
| gemma and mistral caps | hi 500 → **250**, te 400 → **200** | Wall-clock. qwen keeps 500/400 — already generated and frozen. Across three seen generators this still leaves ~1,150 hi and ~1,000 te machine rows. |
| Held-out generators drop `en` | `heldout_drop_buckets: [en]` | llama and phi are test-only and exist to prove unseen-generator generalisation (non-negotiable #4). hi, te and cm carry that test; en is the best-covered bucket in the corpus, so generating it twice more buys nothing. |
| Keyed on the **role**, not the alias | `spec["role"] == "heldout"` | The reason is the role, so a generator whose role changes carries the behaviour with it instead of stranding a per-alias list. |

The caps are a compute decision and are recorded as one: they do not touch the ≥1,000
**human** texts per bucket that calibration requires (§5), and `splits.json` is
untouched.

**Housekeeping.** gemma's 24 failed hi rows moved to
`data/raw/discarded/gemma__hi__gate-failed-2026-09-23.jsonl` rather than deleted;
`gemma.jsonl` back to 48 rows (24 cm + 24 en). The 500/400 selection records moved to
`data/processed/selection/superseded/` with a note — a recut changes *which* passages
are chosen, not only how many, and it is only safe here because no gemma te rows ever
existed and the hi rows were discarded.

**Still owed for the other generators.** mistral, llama and phi each need their own
probe before bulk generation. Their ratios are currently qwen's, which is exactly the
condition that produced this abort.

---

## 2026-09-24 — mistral does not generate `te`: the cap, not the template

**What the probe found.** mistral measured with `probe_compliance.py`, 12 passages
per bucket, current templates, ask = bin, no compensation:

| bucket | machine/ask | mean | p10 | p90 | machine/human |
|---|---|---|---|---|---|
| hi | **0.88** | 0.94 | 0.77 | 1.19 | 0.95 |
| te | **1.57** | 1.62 | 1.08 | 2.15 | **1.65** |

`hi` is healthy and close to the ask. `te` over-produces by ~60 %, and 1 of its 12
rows hit `num_predict`, so 1.57 is a **lower bound**.

**Why this is not a template problem.** mistral's tokenizer reads Telugu at
**13.309 tokens/word**, against gemma's 4.942 and qwen's 11.919. At that fertility
the 3600-token cap holds ~270 Telugu words, and `num_predict` for mistral te
computed to **3600..3600** — every single ask pinned at the cap regardless of
passage length.

That is the structural part. Length-matching requires the *passage* to set the
length; here the *budget* does. A prompt template changes what the model is asked
for, and a compliance ratio changes how much it is asked for, but neither can move
a ceiling that binds every row identically. Over-production at 1.65× is then a
second, independent failure on top of it — measured under both templates.

Raising the cap is not a way out either: 3600 tokens already sits at the `num_ctx`
budget the 4-worker configuration allows on this card, and a cap large enough to
hold long Telugu passages at 13.3 tokens/word would cut concurrency for every other
bucket.

| Decision | Value | Why |
|---|---|---|
| mistral does not generate `te` | `drop_buckets_by_generator: {mistral: [te]}` | The cap, not the template. Every te ask pins at 3600 tokens whatever the passage length, so no template or ratio reaches it. |
| A second, per-generator exclusion | Separate from `heldout_drop_buckets` | The reasons differ in kind. The held-out drop is about a generator's ROLE in the design; this one is about a specific model's tokenizer. Keying both on the same list would merge two unrelated arguments. |
| Both report their source | `generator X: bucket Y excluded (<source>)` | A bucket that silently fails to appear is the failure mode this whole line of work has been about. |
| mistral's te compliance ratio | **Removed**, not set to 1.00 | A generator that does not generate a bucket must not carry a ratio for it. The 1.00 placeholder recorded 2026-09-24 was a guard against inheriting the shared 0.62; with te excluded there is nothing to guard. |

### Coverage — reported, not uniform

mistral covers **en, hi, cm**. `te` is covered by **qwen7b (400) and gemma (200) =
600 machine rows across two seen generators**, against three for every other
bucket.

This is a deliberate, recorded asymmetry rather than a gap to be closed. Generator
coverage per bucket is a property of the corpus that gets **reported**; nothing in
the design requires it to be uniform, and non-negotiable #4 constrains the held-out
generators, not how many seen generators appear in each bucket. Forcing mistral into
`te` would buy uniformity with rows whose length is set by a token cap — worse
corpus, better-looking table.

**Goes in Limitations**, alongside the 2026-09-23 note that gemma's `te` passages
are length-matched at the short end and short at the long end. Both are statements
about how far `te` length-matching holds, and they belong together:

> Telugu coverage rests on two of the three seen generators. mistral was excluded
> from the bucket because its tokenizer's Telugu fertility (13.3 tokens/word) pins
> every generation budget at the context cap, making length-matched generation
> impossible rather than merely poorly tuned.

**Unchanged.** mistral's `hi` is ready to generate at 0.88. qwen7b's and gemma's te
selections, caps and rows are untouched.

---

## 2026-09-24 (later) — Part 13: corpus v1 frozen on qwen7b; cm as a worked example of the thesis

Review-2 sprint Day 1 (`docs/review2-sprint.md`): Part 13 ran on what exists —
human 8,800 + qwen7b 2,331 — without waiting for gemma/mistral. `splits.json` and
`corpus.jsonl` are written (hashes in `docs/data/corpus_card.md`) and frozen
(non-negotiable #6).

### Cleaning — measured, symmetric

`clean_artifacts.py` implements the spec in its docstring. Two changes to that spec,
both from measuring against the human baseline:

| Change | Why |
|---|---|
| hi/te echo patterns do **not** include `in hindi` / `in telugu` | They fire on 6 hi and 16 te human news bylines ("News18 Telugu - …") against 1–2 machine rows. hi/te echo is template leakage only: `<\|im_start\|>`-style tokens and "N words". |
| Echo also catches **meta responses** and **chat-template leaks** | "Certainly, please provide the topic…", "As an AI…", and one hi row ending `<\|im_start\|>user 请继续用 Hindi…`. Stripping cannot rescue them. Human rate: 0. |
| `code_mix_ratio` recomputed for every row | qwen7b cm rows arrived with 0.0 against ≥ 0.30 for every human cm row — a perfect class separator in the metadata. Recomputed with `hindi_word_ratio` on the cleaned text, identically for both classes. |

Drop rates: qwen7b en 1.4 %, hi 6.8 %, te **18.0 %** (mostly Devanagari inside Telugu,
43 rows), cm 7.4 %. Human sources 0–0.8 %, except Indic Wikipedia's stray scripts
(hi 5.5 %, te 7.1 %, still under the 2 % per-bucket guard). 259 rows dropped, copied to
`data/raw/discarded/part13_clean_v1.jsonl`.

### Length matching — trim the surplus side

| Decision | Value | Why |
|---|---|---|
| Match where the classes meet | Bins are quantiles of the **train/test** human rows; `cal` is drawn first and never trimmed | cal is human-only; test is where a length difference becomes a class signal. Matching against the whole bucket would have left cm's test lengths mismatched (see below). |
| en, hi, te: **trim human** to the machine distribution | `length_match_trim: {en: human, hi: human, te: human}` | What must hold is that the distributions match, not which side is cut. Trimming machine (the Part 13 draft) cut en 720 → 302 and te 328 → 98; machine is not the surplus. Human train/test is. |
| cm: **trim machine** | `length_match_trim: {cm: machine}` | cm's human train/test cannot be cut without disturbing the calibration-eligibility routing. |
| Buckets below 500 machine rows are accepted | `--allow-small-buckets`; final counts below | hi and te start under 500 (qwen7b caps 500/400, before cleaning). Recorded rather than fixed; gemma/mistral add rows after Review-2. |

Final counts: human cal / train / test and machine train / test —
en 1000/271/272, 359/361 · hi 1000/392/392, 233/233 · te 1000/349/349, 163/165 ·
cm 1000/596/596, 82/83. Test length medians match in every bucket (human vs machine:
en 194.5/193, hi 175/172, te 155/156, cm 20/20). 0 prompts straddle train/test.

### Finding — cm's bound is calibrated on chat-register Hinglish

**What happened.** The cm calibration set may draw only from cmu_hinglish_dog + HinGE
(2026-09-13 (b)): 1,021 eligible rows after cleaning, for 1,000 places. So cal takes
894 of 913 chat rows and all but 2 HinGE rows, and cm's human **train/test is 98 %
comi_lingua** — social comments, median 20 words, against a calibration set of chat
turns joined per conversation (median ~93).

**Why it is a finding, not a caveat.** A split-conformal FPR bound holds for test text
exchangeable with the calibration text. cm's calibration and test humans come from
different registers, so the cm bound is a statement about **chat-register Hinglish**,
and it is not guaranteed on comment-register Hinglish. That is precisely the argument
this project makes against English-only calibration: a threshold is only as good as the
match between the population it was calibrated on and the population it is deployed on.
cm is a worked example of our own thesis, inside our own corpus — the same mechanism
that makes a single global threshold misfire on Hindi also bounds how far a per-bucket
threshold reaches within a bucket.

| Decision | Value | Why |
|---|---|---|
| Source restriction and 1,000 floor **unchanged** | Option (a) | They are the guarantee. Admitting comi_lingua would put possibly-machine text in cal; shrinking cal below 1,000 weakens the quantile. |
| cm's bound is stated as chat-register | Wherever cm appears: corpus card, T1/T5/T6 captions, paper | Honest scope of the guarantee. |
| cm test FPR is **reported as measured** on comments | T5 | The gap between the nominal α and cm's empirical test FPR is itself evidence for the calibration–deployment-match argument, and goes in the paper as such. |

**The same mechanism, mildly, in en/hi/te.** Trimming human train/test to the machine
length distribution makes test humans a length-reweighted sample relative to cal
(median words, cal vs test: en 208 vs 194.5, hi 159 vs 175, te 155 vs 155). Any
per-bucket test FPR is therefore read against a slightly shifted population. The
trimmed human rows are not lost: their ids are in `splits.json` under
`excluded.length_match` with prompt groups assigned, so an unmatched human-test FPR can
be computed from `data/raw/human/` as a robustness check in T5.

**Consequence for later generators.** New machine rows take their prompt's recorded
group (`prompt_groups` in `splits.json`, one per prompt including those whose human is
in cal), so adding gemma/mistral does not need `splits.json` changed. How rows are
added to a frozen corpus (a v2 file keyed on the same groups) is decided when that
happens.

---

## 2026-09-24 (evening) — Head A: real parsers, per-bucket logistic head

| Decision | Value | Why |
|---|---|---|
| Parsers | stanza (UD HDTB / MTG) for hi, te; spaCy `en_core_web_sm` for en, cm; CPU only | Real UPOS + dependency trees in every bucket. CPU keeps Head A off the GPU that headB needs and inside the 4 GB floor. |
| cm uses the English parser | Documented as weaker evidence | No parser for romanised Hinglish exists. The same parser runs on both classes, so its errors are noise, not a class shortcut. |
| Keep the 25 surface features, including the old proxies | 25 + 15 = 40 | The request was to keep them. The proxies still carry signal; the real parser features now answer "POS n-grams, syntactic depth". |
| One logistic head per bucket, `class_weight=balanced` | `HeadA` | Feature baselines differ by language; train is not 1:1 (cm 596 : 82). |
| Instability is reported as 5-fold CV AUROC spread on train, and a bootstrap CI on test | `format_auroc` in score.py | cm's ± 0.054 CV spread (vs ± 0.004–0.008) is the measurable form of "82 machine rows is thin". |
| `click==8.1.8` pinned | requirements.txt | spaCy 3.8.4's CLI imports it and typer ≥ 0.20 stopped pulling it in; without it `import spacy` fails. |

Result (test AUROC): en 0.991, hi 0.984, te 0.989, cm 0.910. Same-generator, qwen7b
only; see docs/progress.md 2026-09-24.

---

## 2026-09-24 (night) — headB scored: a finding, not a bug — curvature inverts on hi/te

**Result (test AUROC, qwen7b, mGPT fp16 CUDA):**

| bucket | headA | headB | headB_word |
|---|---|---|---|
| en | 0.991 | 0.859 [0.832, 0.889] | 0.832 [0.801, 0.862] |
| cm | 0.910 | 0.661 [0.594, 0.728] | 0.635 [0.566, 0.703] |
| hi | 0.984 | **0.187** [0.153, 0.222] | 0.192 [0.159, 0.229] |
| te | 0.989 | **0.120** [0.090, 0.152] | 0.122 [0.092, 0.153] |

headB is below 0.5 in hi and te — inverted, not merely weak. This was the sprint's
one stop condition ("a column's AUROC is below 0.5 — sign error"), and it was
checked: same code, same sign convention, works correctly on en (0.859) and cm
(0.661, weak but right-signed). Confirmed against a full-logits reference in
`tests/features/test_curvature.py::test_padded_batch_matches_single_rows_and_reference`.
Not a bug — see the diagnostic below. **Sign is NOT flipped for hi/te**; the
fused head (Day 4) learns weights per bucket and a consistent inverted signal
is still usable there. Do not hand-correct it.

**Diagnostic — human headB by source, within-bucket, train+test:**

| bucket | source | n | mean | vs. machine mean |
|---|---|---|---|---|
| en | hc3 | 169 | −0.724 | machine +0.701 (**above**, correct direction) |
| en | samanantar_en | 254 | −1.105 | — |
| en | wikipedia | 120 | −0.795 | — |
| hi | indiccorp_v2 | 565 | −1.565 | machine −3.889 (**below**, inverted) |
| hi | wikipedia | 219 | −2.398 | — |
| te | indiccorp_v2 | 546 | −5.124 | machine −9.473 (**below**, inverted) |
| te | wikipedia | 152 | −5.998 | — |

Within-bucket spread across human sources (0.38 en, 0.83 hi, 0.87 te) is small next
to the human-vs-machine gap in every bucket (1.4 en, 2.3 hi, 3.5–4.3 te) — which
human corpus you pick barely moves the number. What differs is *direction*: in en
every available human source sits below qwen7b's mean (expected — machine text
scores as more machine-like); in hi and te, **both** available human sources
(IndicCorp v2 and Wikipedia — there is no third option at this corpus size) sit
above qwen7b's mean. The inversion isn't one contaminated source contaminating the
bucket; it's that every real-world hi/te human corpus available to this project
behaves the same way against mGPT.

**Working hypothesis:** curvature-based detection assumes the human calibration
text is *outside* the scorer's own pretraining distribution, so the scorer finds
genuinely-human text more "surprising" than the model's own likely continuations.
For en, pretraining data is broad enough that no single available corpus (HC3,
Samanantar, Wikipedia) dominates what shaped mGPT's English distribution. For hi
and te, IndicCorp and Wikipedia-derived text *are* plausibly a substantial fraction
of what a multilingual LM this size was trained on for those languages — there
isn't much else at scale. If so, mGPT finds real hi/te text unsurprising not
because it's human, but because it has effectively seen it (or its close relatives)
before; qwen7b's Hindi/Telugu output, despite being machine-written, is a
different model's distribution and reads as comparatively more surprising to
mGPT. This is not verified against mGPT's actual training manifest (not public);
it is the explanation consistent with every number above and with nothing else
we've checked.

**Why this matters beyond this dataset:** the project's own provenance rule (human
calibration text predates the scorer / is out-of-distribution, so an LLM can't
have memorised it) is usually enforced by picking recent or held-out text. For
low-resource languages that rule runs into a wall: the small set of large hi/te
corpora that exist at all (IndicCorp, Wikipedia, CommonCrawl-derived) *are*, in
practice, close to what any sizable multilingual LM was pretrained on for that
language, because there is little else. There may be no way to build a hi/te
human calibration set that is simultaneously (a) large enough (≥1000/bucket,
locked §5) and (b) provably outside mGPT's pretraining. To our knowledge this
tension is unreported in the curvature-detection literature (Fast-DetectGPT,
Binoculars, DetectGPT are evaluated almost entirely on English/high-resource
text). Candidate for the paper's limitations section and worth checking whether
it independently motivates Patent 1's abstention gate: if a bucket's curvature
head is structurally unreliable, that's exactly the case abstention exists for.

**headB_word — negative result.** headB_word ≈ headB everywhere (en 0.832 vs
0.859, te 0.122 vs 0.120, hi 0.192 vs 0.187, cm 0.635 vs 0.661) — word-level
aggregation changes nothing measurable, including on te/hi where tokenizer
fertility is worst (6.3, 3.6 tokens/word) and a fragmentation-robustness
argument would predict the biggest gap. **Patent 2's central claim — that
per-word standardisation recovers signal fragmentation destroys — does not
hold on this evidence.** Recorded as a finding, not hidden: `headB_word` stays
in the registry and the parquet (it's cheap, one extra column from the same
forward pass), but Patent 2 needs a different central claim or needs to be
retargeted before the December filing. Flagged for P2, who owns Patent 2.

**Engineering, same session:**
- `max_length` 512 → 2048: at 512, 66% of hi and 100% of te texts were truncated
  (te median is 988 mGPT tokens). 2048 scores >99% of texts whole.
- Batching is by **token budget** (`max_batch_tokens`), not row count: `8 rows x
  2048 tokens` peaked at 7.88 GiB of the 7.9956 GiB card in isolation. A
  sustained run still hit a genuine CUDA OOM at row 216/6784 with the token
  budget in place — allocator fragmentation across ~800 differently-shaped
  batches, not true usage (the card was at 0 MiB immediately after the crash).
  Fixed three ways: `max_batch_tokens` 8192 → 6144, `empty_cache()` after every
  group, and an OOM-safe retry that bisects a group and retries rather than
  losing the run. Re-run completed all 8896 rows clean. This was the sprint's
  other stop condition ("mGPT won't load in 8GB") — it loaded and fit; the
  failure was allocator behaviour over a long run, an engineering fix, not a
  capacity or scope question, so it did not require stopping to ask.

**Verify:** `python -m src.eval.score --report headA,headB,headB_word` (reads
`results/scores.parquet`, no GPU needed); `pytest tests/features/test_curvature.py`
→ 9 passed.

---

## 2026-09-25 — Day 3: the hi/te inversion is mGPT-specific, not a property of curvature

**Result (test AUROC, qwen7b, full corpus):**

| bucket | headA | headB (mGPT) | headB_word | fastdetectgpt_en (Qwen2.5-0.5B) | ppl (Qwen2.5-0.5B) | binoculars | headC (MuRIL) |
|---|---|---|---|---|---|---|---|
| en | 0.991 | 0.859 | 0.832 | 0.794 [0.759, 0.830] | 0.915 [0.889, 0.936] | 0.804 [0.770, 0.837] | 1.000 [0.999, 1.000] |
| cm | 0.910 | 0.661 | 0.635 | 0.696 [0.632, 0.762] | 0.711 [0.644, 0.773] | 0.661 [0.596, 0.728] | 0.927 [0.895, 0.953] |
| hi | 0.984 | **0.187** | 0.192 | **0.728** [0.689, 0.768] | 0.816 [0.780, 0.850] | 0.732 [0.691, 0.772] | 1.000 [0.999, 1.000] |
| te | 0.989 | **0.120** | 0.122 | **0.555** [0.502, 0.604] | 0.847 [0.812, 0.880] | 0.564 [0.513, 0.615] | 1.000 [1.000, 1.000] |

**fastdetectgpt_en (same statistic as headB, scorer = Qwen2.5-0.5B instead of
mGPT) is the answer to last night's open question: is the hi/te inversion
scorer-specific, or a property of curvature on this corpus's Indic text?**
It's scorer-specific. With a different scorer, hi and te are no longer
inverted — hi jumps from 0.187 to 0.728, te from 0.120 to 0.555. Neither
matches headA or even mGPT's own en/cm numbers, but both are on the correct
side of 0.5, with CIs that exclude it. **headB's inversion is mGPT-specific**,
consistent with last night's contamination hypothesis (mGPT's Indic pretraining
overlapping this project's only available Indic human corpora) rather than a
structural property of curvature-based detection on Indic text in general —
that door is not closed, but this rules out the strongest, corpus-wide version
of it.

**Source breakdown confirms it, same method as last night:**

| bucket | source | n | mean fastdetectgpt_en | vs. machine mean |
|---|---|---|---|---|
| hi | indiccorp_v2 | 565 | −0.364 | machine +0.484 (**above**, correct direction) |
| hi | wikipedia | 219 | −0.492 | — |
| te | indiccorp_v2 | 546 | −1.515 | machine −1.459 (**above**, correct, barely) |
| te | wikipedia | 152 | −2.615 | — |

For hi, both human sources now sit clearly below qwen7b's mean — the mirror
image of headB's numbers. For te, IndicCorp is now essentially tied with
qwen7b (−1.515 vs −1.459) while Wikipedia is well below it — direction is
correct on average (AUROC 0.555, CI excludes 0.5) but the margin is thin,
matching the weak-not-inverted AUROC. **A second, compounding effect is visible
here and worth separating from the contamination story**: Qwen2.5-0.5B's
tokenizer fragments Telugu far worse than mGPT's (11.919 vs 6.319 tokens/word,
locked-decision table above), and curvature is a statistic over per-token
log-probabilities — so te's *weak* signal with this scorer is plausibly a
fragmentation effect layered on top of a real but modest curvature signal,
independent of which scorer produced it. This is exactly the tokenizer-
fertility argument that locked mGPT as Head B's scorer in the first place
(2026-09-13); it now also explains why swapping to Qwen2.5-0.5B doesn't fully
recover te even once the inversion is gone.

**ppl (negated mean log-perplexity, Qwen2.5-0.5B) is the strongest single
column on hi and te** (0.816, 0.847) — stronger than its own curvature
sibling on the same scorer. Not surprising in hindsight: perplexity is a
first-order statistic (needs no reference distribution, no variance
normalisation), so it's less exposed to a fragmented tokenizer's noisy
per-token variance estimates than curvature is. This is exactly the raw-
perplexity failure mode non-negotiable #2 exists to avoid (§2, §4) — ppl is
in the registry as the intentionally-naive baseline T1 needs, not a candidate
for the fusion head.

**binoculars tracks fastdetectgpt_en closely in both direction and magnitude
across all four buckets** (cm 0.661/0.696, en 0.804/0.794, hi 0.732/0.728, te
0.564/0.555) — expected, since both are variants of the same cross-entropy-
style statistic (§ module docstring, `src/baselines/binoculars.py`) and share
an observer/performer pair from the same base model.

**headC (MuRIL) is suspiciously strong — flagged, not fully trusted yet.**
AUROC 1.000 in en/hi/te, 0.927 in cm. This is same-generator (qwen7b only,
no held-out generator — non-negotiable #4 explicitly requires held-out-
generator evaluation before any result counts) and it is far stronger than
every other head including headA. A semantic encoder achieving *perfect*
separation is the kind of result that's usually a shortcut — length, register,
or a residual generation artifact clean_artifacts.py didn't strip — rather
than genuine machine-vs-human semantics holding up under paraphrase or a
different generator. Not chased further today (out of scope for a scoring
session), but headC's numbers should not be reported anywhere as a headline
result until they've been checked against a held-out generator (T3) and,
ideally, an adversarial paraphrase (T4). Noted for P1/P2 before fusion weights
are fit on it.

**Engineering, same session:** the OOM-safe token-budget batching and retry
built for headB last night (`token_budget_groups`, `run_group_with_oom_retry`
in `src/features/curvature.py`) is now a free function, reused as-is by
`CurvatureScorer` (mGPT, Qwen2.5-0.5B — same class, different `scorer_model`),
`BinocularsDetector` (two simultaneous models), and `MurilEmbedder`. All four
full-corpus runs today (fastdetectgpt_en+ppl, binoculars, headC) completed
with zero OOMs. `CurvatureScorer.full_stats()` now returns a third column,
`ll_mean` (mean observed log-likelihood), free from the same forward pass that
computes curvature — that's what `ppl` reads; `stats()`'s existing 2-column
contract (headB/headB_word) is unchanged, verified by a new test asserting
`stats() == full_stats()[:, :2]`.

One trailing bug, caught by binoculars, fixed: `score.py`'s final summary
print assumed every column's `ctx.cache` entry was a `(head, feats)` tuple
(true for headA/headC only); binoculars caches the detector object itself,
which crashed the print *after* all 8896 rows were already scored and merged
— cosmetic (exit 1 on a run that fully succeeded), fixed with an `isinstance`
guard.

**Verify:** `python -m src.eval.score --report headA,headB,headB_word,fastdetectgpt_en,ppl,binoculars,headC`
(reads `results/scores.parquet`, no GPU); `pytest tests/baselines/test_binoculars.py
tests/features/test_semantic.py` → 3 + 4 passed.

---

## 2026-09-28 — phi does not generate `cm`: instruction compliance, not the template

**What the 24-row gate run found.** phi's cm gate failed hard: length ratio 3.57
against human, 100 % of rows at the token cap, 14 of 24 rows carrying an emoji
despite the template's "No emoji" clause.

**Probe 1 — is this a compliance-ratio problem?** `probe_compliance.py`, 12
passages, current template, ask = bin, **no compensation** (`results/probes/
phi__cm__askbin.jsonl`):

| | machine/ask | mean | p10 | p90 | machine/human | at cap |
|---|---|---|---|---|---|---|
| cm | **4.64** | 4.20 | 3.01 | 5.18 | 4.50 | 8/12 (67 %) |

No. A compliance ratio only ever inflates the ask (`compensated_words` never
shrinks it), so the smallest value on the table is 1.0 — and 1.0 already
overshoots by 4.5×. Whatever produced the 3.57×/100 %-at-cap numbers in the
24-row run, it was the *compensated* ask making an already-runaway model
worse, not a mistuned ratio. The length side has to move at the template
level (or phi drops cm), not through `compliance_ratio_by_generator`.

**Probe 2 — is the emoji rate a template-wording problem?** Human cm is 0 of
2,200 rows with an emoji, so any non-zero machine rate is a class shortcut, not
noise. A paired two-variant probe (`results/probes/phi_cm_emoji_probe.jsonl`,
same 12 passages, ask = bin, no compensation, so the length lever is held
constant and only the constraint wording changes):

| variant | emoji rows |
|---|---|
| `v0_current` — existing template, "No emoji" mid-prompt | 7/12 |
| `v1_end` — same content, moved to the end and strengthened to "Plain text only: no emoji, no emoticons, no symbols standing in for words" | 6/12 |

58 % → 50 %. Repositioning the constraint to the most recency-weighted part of
the prompt *and* strengthening its wording moved the rate by one row. That
rules out placement and phrasing as the cause: phi is not failing to notice or
parse the instruction, it is not complying with it. This is model behaviour,
not a prompt-engineering gap — the same qualitative distinction the mistral/te
finding drew between a structural cap and a template fix (2026-09-24, above).

| Decision | Value | Why |
|---|---|---|
| phi does not generate `cm` | `drop_buckets_by_generator: {mistral: [te], phi: [cm, te]}` | Two independent, unfixable-at-the-template-level failures: length (4.64× ask median under zero compensation, 67 % at cap) and emoji (a controlled two-variant probe shows compliance, not wording, is the cause). |
| phi's cm rows from the failed gate run | Moved to `data/raw/discarded/phi__cm__gate-failed-2026-09-28.jsonl` | Kept for the writeup, not deleted; same pattern as `gemma_cm_gateprobe_v0.jsonl` and `gemma__hi__gate-failed-2026-09-23.jsonl`. |
| phi's cm compliance ratio | Never set | Same reasoning as mistral/te (2026-09-24): a generator that does not generate a bucket carries no ratio for it. |

### Finding: generator-bucket fit is a property of the pair, not the language

Two of the five generators now fail a bucket each, for structurally unrelated
reasons. **mistral/te** fails because its tokenizer's Telugu fertility (13.309
tokens/word) pins every generation budget at the context cap — the *budget*,
not the *passage*, sets the length, so no template or compliance ratio can
reach it. **phi/cm** fails because the model does not comply with an
instruction it demonstrably can read (repositioning and strengthening it moved
the rate by less than one row in twelve) — an *instruction-compliance* limit,
not a capacity or budget one. Both symptoms look similar from the gate's
output (length and/or emoji failures on the probe), but the mechanism, and
therefore the fix that doesn't exist, is different in each case. The lesson
this sets for the rest of generation: a bucket failing on one generator is not
evidence about the bucket, and does not predict which other generator/bucket
pairs will fail or why — each has to be probed and diagnosed on its own terms,
the same way mistral's te and phi's cm were.

### phi does not generate `te` either: the same cap-pinning as mistral/te

Probed the same session (`probe_compliance.py`, 12 passages per bucket,
current templates, ask = bin, no compensation, `results/probes/
phi__hi-te__askbin.jsonl`):

| bucket | machine/ask | mean | p10 | p90 | machine/human | at cap |
|---|---|---|---|---|---|---|
| hi | **1.60** | 1.65 | 1.21 | 2.09 | 1.47 | 1/12 (8 %) |
| te | **3.33** | 4.16 | 2.99 | 5.33 | 3.49 | 8/12 (67 %) |

te's signature is the mistral/te finding again, not a new one: phi's
tokenizer reads Telugu at **20.286 tokens/word** (worse than mistral's
13.309), and the probe log shows `num_predict` computed to **3600..3600 for
every one of the 12 passages** — the budget, not the passage, sets the
length, so no template or compliance ratio reaches it. `phi: [cm, te]` added
to `drop_buckets_by_generator`, same mechanism and same reasoning as
`mistral: [te]` (2026-09-24, above). **phi now generates `hi` only.**

hi is a genuine, milder overshoot (1.47× human even with the ratio clamped
to 1.0) rather than a cap artifact — only 1/12 rows hit the cap. Whether that
clears the gate depends on the template, not a ratio; hi generation was not
launched this session pending that check and pending llama's three-bucket
numbers below.

### Coverage — cm, and what zero held-out coverage costs T3

cm is covered by **qwen7b (701, seen) + gemma (seen) + mistral (seen)** on the
seen side. On the held-out side: see the next entry — phi is dropped
entirely, and llama's raw probe clears all three of hi, te and cm, so llama
alone carries held-out coverage for every bucket that needs it rather than cm
being left at zero.

**Verify:** `results/probes/phi__cm__askbin.jsonl`, `results/probes/
phi_cm_emoji_probe.jsonl`, `results/probes/phi__hi-te__askbin.jsonl`, and
`_summary.txt` alongside the first two; `configs/data.yaml`
`machine_corpus.drop_buckets_by_generator` no longer lists `phi` (dropped as
a generator entirely — next entry).

---

## 2026-09-28 (later) — phi dropped entirely; llama is the sole held-out generator

Three of phi's four buckets failed, each for a different structural reason,
all measured this session:

| bucket | failure | evidence |
|---|---|---|
| cm | instruction compliance (emoji), not fixable by prompt wording | paired probe: constraint moved to end + strengthened, 7/12 → 6/12 emoji rows (above) |
| te | cap-pinning, not fixable by template or ratio | 20.286 tok/word, `num_predict` 3600..3600 on all 12 probe rows — same signature as mistral/te (2026-09-24) |
| hi | 1.47× over-production even with the compliance ratio clamped to 1.0 (the ratio can only inflate) | `results/probes/phi__hi-te__askbin.jsonl`: machine/ask 1.60, machine/human 1.47, 1/12 at cap |

hi is the one bucket that did not fail on a structural, unfixable-by-config
axis — but fixing it needs template work (the same kind of change cm's
2026-09-21 template redesign took a full sitting to land), for a generator
that would then cover exactly one bucket out of four. Rather than spend that
sitting, **phi is dropped as a generator entirely**: removed from
`machine_corpus.generators` and from `heldout_generators` in
`configs/data.yaml`. Its `drop_buckets_by_generator` entries for cm and te are
removed as moot (a dropped generator needs no per-bucket exclusion list).

**llama covers all three held-out buckets on its own raw-compliance probe**
(`results/probes/llama__hi-te-cm__askbin.jsonl`, 12 passages/bucket, current
templates, ask = bin, no compensation):

| bucket | machine/ask | machine/human | at cap | verdict |
|---|---|---|---|---|
| hi | 0.61 (mean 0.62, p10 0.44, p90 0.81) | 0.63 | 0/12 | healthy under-production, ordinary ratio correction |
| te | 0.52 (mean 0.66, p10 0.26, p90 1.09) | 0.51 | 1/12 (8 %) | healthy under-production; 1 row's ratio is a lower bound |
| cm | 1.17 (mean 1.51, p10 0.51, p90 2.89) | 1.09 | — | already close to human with no compensation |

No cap-pinning, no compliance-ratio ceiling, nothing that looks like phi's
failures on any of the three. `compliance_ratio_by_generator.llama` set to
`{hi: 0.61, te: 0.52, cm: 1.17}`. This is a raw 12-row probe, not the 24-row
gate — hi/te/cm still each have to clear their own gate (length band +
codemix + emoji, for cm) once the bulk run reaches the probe threshold.

| Decision | Value | Why |
|---|---|---|
| phi | **Dropped as a generator** | Not a per-bucket exclusion — 3 of 4 buckets failed structurally, and fixing the 4th (hi) buys one bucket's worth of held-out coverage for a full sitting of template work. |
| Held-out generator set | `[llama]`, down from `[llama, phi]` | The only generator left standing on held-out evaluation. |
| llama compliance ratios | `{hi: 0.61, te: 0.52, cm: 1.17}` | Measured 2026-09-28, raw probe, current templates — see table above. |

### What a single held-out generator costs T3

Non-negotiable #4 requires held-out-generator evaluation; the original design
locked **2** held-out generators specifically so T3 could claim generalisation
to a *family* of unseen models, not one specific model's quirks (decisions.md
2026-09-13). With phi dropped, **T3 now measures generalisation to exactly one
unseen generator (llama3.1:8b)**. A detector that generalises to llama but
happens to exploit some llama-specific artifact would pass T3 with this
design in a way it would not if a second, architecturally different held-out
model were also available to disagree with it. This is a real reduction in
what T3's result can claim, not a bookkeeping change — **goes in Limitations**
alongside the te/mistral note:

> Held-out generalisation (T3) is evaluated against a single unseen generator,
> llama3.1:8b. A second held-out generator (phi3.5:3.8b-mini-instruct) was
> attempted and dropped after failing three of its four buckets for
> structurally distinct reasons (§ below); the result is a narrower
> generalisation claim than the original two-generator design intended.

### phi's failures as evidence for the generator-bucket-fit finding

The 2026-09-28 (earlier) entry above named the finding: generator-bucket fit
is a property of the *pair*, not the language. phi's three failures are
themselves a second, independent demonstration of exactly that claim, this
time all inside one generator rather than across two: cm failed on
instruction compliance, te failed on tokenizer-fertility cap-pinning, and hi
failed on plain over-production — three different mechanisms, same model,
different buckets. No single "phi is bad at Indic" or "phi is bad at
constrained generation" story covers all three; each had to be probed and
diagnosed on its own terms, same as mistral/te and phi/cm were.

**Verify:** `results/probes/llama__hi-te-cm__askbin.jsonl` and its printed
report; `configs/data.yaml` — `machine_corpus.generators` has no `phi` key,
`heldout_generators: [llama]`, `compliance_ratio_by_generator.llama` set.

---

## 2026-09-28 (later still) — llama/cm: deferred, not failed-and-abandoned

The bulk run (`generate --generator llama --buckets cm,hi,te`) hit cm's 24-row
gate first and failed it: median machine/human ratio **1.97** (mean 2.44, p90
5.12, 75 % of rows above 1.25), against the 12-passage raw probe's 1.09
median. Split by length bin:

| bin | n | ratio median | range |
|---|---|---|---|
| human < 40 words | 15 | **2.22** | 0.21 – 6.12 |
| human ≥ 40 words | 9 | 1.32 | 0.66 – 2.27 |

The failure is concentrated entirely in the short bin, and it is *variance*,
not *bias*: individual short-passage rows range from 15 words (against an ask
of 25) to 110 words on the same ask, both directions represented, no
consistent over- or under-shoot to correct with a ratio or a template tweak.
This is the short-passage defect `probe_compliance.py`'s docstring already
names from 2026-09-20 ("a head-of-list sample hid cm's short-passage defect")
— a 12-passage probe can miss it by luck of the draw; this 24-row stratified
gate sample didn't.

**Not investigated further, on instruction.** The 24 rows are moved to
`data/raw/discarded/llama__cm__gate-failed-2026-09-28.jsonl` (kept for the
writeup, not deleted); `data/raw/machine/llama.jsonl` is now empty of cm rows
so a resume starts clean rather than re-reading and re-failing the same gate.

**Status: deferred for llama, not failed-and-abandoned.** The distinction
matters because it differs in kind from every other exclusion recorded today
(mistral/te's cap-pinning, phi/cm's instruction compliance, phi/te's
cap-pinning, phi/hi's over-production) — those are properties of a specific
generator's fit to a bucket. cm's 20-word-passage variance instead looks like
it could be a property of *short-passage instruction following under a
word-count ask*, independent of which model is asked: qwen needed a dedicated
sentence-count floor fix for cm short passages (`sentence_target`,
2026-09-18) precisely because the naive template drove short cm passages to
4.0x; llama's failure, high-variance rather than one-directional, may be the
same underlying instability showing up differently. Whether it recurs on
llama specifically, or is a general short-cm-passage phenomenon, is unresolved
and not chased this session — **goes in Limitations** as an open note on
prompt-controlled corpus construction at short lengths, not as a settled
per-generator exclusion the way phi's buckets were.

cm's held-out coverage is therefore open pending this bucket being revisited,
not zero by design and not resolved by llama the way hi/te are expected to be.

**Verify:** `data/raw/discarded/llama__cm__gate-failed-2026-09-28.jsonl` (24
rows); `data/raw/machine/llama.jsonl` absent/empty until te/hi regenerate it.

---

## 2026-09-28 (later still) — te has no held-out generator: compensation assumes a response that Telugu doesn't give

llama's te gate failed next: median machine/human **0.37** (mean 0.46, p90
0.78), still under-producing after `compliance_ratio_by_generator.llama.te`
(0.52, measured the same session on a 12-passage raw probe) was applied. The
ratio should have corrected the ask to land near 1.0; it landed at little
over a third of human length instead.

**Why the correction didn't take.** Compensation (`compensated_words`)
assumes output scales with the ask: inflate the ask by `1/ratio` and the
model writes proportionally more. That assumption is exactly what gemma's te
regression already showed failing, measured 2026-09-23 (above): output
regressed on ask gave a **slope of 0.36** words-out per word-of-ask, against
hi's near-proportional 0.65 — gemma writes ~69 words plus roughly a third of
whatever is asked, so inflating the ask moves the output much less than the
inflation factor assumes. llama's 24-row gate result is the same non-response
in a different generator: a ratio measured at 0.52 (already accounting for
some of this) still left the *gated, ask-inflated* run at 0.37, well short of
where 1/0.52 inflation should have landed it if output scaled with the ask.

**Three of the four generators now show the same shape of te failure, for
three different reasons, none of them a wrong ratio:**

| generator | te outcome | mechanism |
|---|---|---|
| mistral | never tested for over/under-production | tokenizer reads Telugu at 13.309 tok/word; every `num_predict` pinned at the 3600 cap before compliance could even be measured (2026-09-24) |
| gemma | passed on the gate's median, but flagged | slope 0.36, intercept 69 — a shallow, non-proportional response to the ask; the long tail under-produces even though the median clears (2026-09-23) |
| llama | failed the gate outright | ratio measured at 0.52, applied, still landed at 0.37 — the same shallow-response shape as gemma's, just severe enough to fail the median too |

qwen (the fourth, seen generator) is the only one whose te ratio (0.62) was
ever a clean, working correction — and it was measured and frozen before this
line of investigation existed. Telugu is not merely harder to length-match;
for most of these models the *mechanism* compensation relies on — write more
when asked for more — is weak or absent, and a ratio computed from a single
probe (a point estimate of a relationship that isn't linear) cannot be
expected to generalise the way it does for hi or en.

| Decision | Value | Why |
|---|---|---|
| llama does not generate `te` | (pending: `drop_buckets_by_generator` update deferred until the concurrently-running `llama --buckets hi` run finishes — the 24 failed te rows are not yet moved out of `data/raw/machine/llama.jsonl`, to avoid rewriting a file a live process is still appending to) | Gate failure at 0.37 despite a measured, applied compliance ratio; compensation's proportional-response assumption does not hold for this generator/bucket either. |

### Consequence: te has no held-out generator

qwen (seen, frozen) and gemma (seen, median passes with a flagged long tail)
are te's only two working sources. Both held-out generators have now failed
te for independent reasons — mistral was never even a held-out candidate
(seen role) but is the third generator to fail it regardless. **te's held-out
coverage is zero.** Non-negotiable #4's generalisation claim (T3) therefore
covers **en, hi, and cm only** — te is a seen-generator-only bucket, the same
status structurally as the te coverage gap already recorded for mistral
(2026-09-24), except now no held-out generator covers it at all rather than
two-of-three seen generators covering it.

This is a second, compounding instance of the session's central finding
(generator-bucket fit is a property of the pair): te specifically is now
0-for-3 on every generator that wasn't qwen, each failing by a different
route (cap-pinned, shallow-response-median-passes, shallow-response-median-
fails). **Goes in Limitations** alongside the mistral/te and phi/cm/te notes:

> Held-out generalisation (T3) is evaluated against llama on en, hi and cm
> only. llama's `te` gate failed (machine/human 0.37) despite an applied,
> measured compliance ratio; Telugu length compensation assumes output scales
> with the ask, and te's low, often sub-proportional response to that ask
> (gemma's regression slope 0.36, 2026-09-23) means no held-out generator in
> this study produced usable Telugu machine text. te rests entirely on the
> two seen generators (qwen, gemma) that do.

**Verify:** `results/probes/llama__hi-te-cm__askbin.jsonl` (raw ratio 0.52);
gate-failure output from `python -m src.data.generate --generator llama
--buckets te,hi --max-minutes 240` (median 0.37, 24 rows). File cleanup and
`drop_buckets_by_generator` update are pending, not yet applied.

---

## 2026-09-29 — mistral/cm dropped: fourth generator, same short-passage signature

mistral's cm gate failed at median machine/human **3.43x**, 62 % of its 24
probe rows cap-pinned — the same signature as phi/cm (2026-09-28) and llama's
high-variance cm failure (2026-09-28). Its 24 rows are moved to
`data/raw/discarded/mistral__cm__gate-failed-2026-09-29.jsonl` and
`drop_buckets_by_generator.mistral` now reads `[te, cm]`.

**Four generators tested on cm short passages, three failed on length
variance.** Human cm median is ~20 words, and prompt-controlled length
matching is unreliable at that scale: qwen and gemma are the only two that
hold. cm coverage is now **qwen + gemma only, with no held-out generator** —
consistent with te's held-out gap (2026-09-28, above), but for cm even the
seen-generator side has shrunk to two of four.

---

## 2026-09-29 (later) — Part 13 re-run: corpus v2 frozen on all generators; v1 archived

Generation finished (qwen all four buckets, gemma all four, mistral en+hi, llama hi).
Part 13 was re-run end to end over every generator's rows, same rules as v1 (1,000 human
cal per bucket, cm cal from calibration-eligible sources only, 5 length bins, trim side
per bucket, pairs never split, held-out test-only, seed 20260924).

| Decision | Value | Why |
|---|---|---|
| **v1 is archived, not overwritten** | `data/processed/archive/v1/{splits.json,corpus.jsonl}`, sha256 re-checked after the move against the 2026-09-24 card | Review-2's numbers (`results/scores.parquet`, fitted heads) were computed on v1 and reproduce only against it. |
| v2 supersedes v1 at the same paths | `data/processed/splits.json`, `corpus.jsonl`; `corpus.version: v2` | Non-negotiable #6 is honoured per version: the overwrite guard is untouched and v2 was written once. This is the "v2 file keyed on the same groups" the 2026-09-24 entry left open, except the groups were **re-drawn, not reused**, because Part 13 was re-run end to end as instructed (calibration draw, length-match trim and stratified halves all depend on which rows are in the pool). I did not test a variant that pins v1's qwen assignments. v1 ids move between splits. |
| Held-out prompts are forced to `test` | `assign_splits` / `_halve(forced_test)` | The old code would have put a llama prompt's human and seen-generator rows in `train` while llama's row went to `test`, splitting the pair. 479 llama prompts now sit wholly in test; 0 prompts straddle train/test. |
| Halves are stratified by seen generator | stratum = {seen generator or none} x {cal} x {trimmed} | Was {has a seen row} x ..., which halved seen machine only in aggregate (hi gemma was 135/114). Each seen generator is now within one row of 50/50. |
| `--allow-small-buckets` | te machine 449, cm machine 464 | Below `min_rows_per_side` 500; human sides are all above. Recorded, not fixed. |

**Not yet done, needed before any v2 number exists.** `results/scores.parquet` and the
fitted heads are v1. They must be re-scored on v2 (headA, headB, headB_word,
fastdetectgpt_en, ppl, binoculars, headC) and T1/T2/T5/T6 re-run. Anything that reads
`data/processed/splits.json` now sees v2 while the cached scores are still v1.
`scripts/t3_preliminary_llama_hi.py` was written against the v1 corpus and its
llama rows were then outside the frozen file; llama hi is now inside it.

**Findings carried to the paper.** (1) Length matching is against the pooled machine
distribution, so per generator it is off: en test median words human 200, mistral 247,
gemma 163.5. (2) Stray-script drops are large in two Indic cells, mistral hi 31.6 % and
gemma te 39.5 %; the surviving rows are the ones the generator wrote cleanly.
(3) cm machine is 464 rows after trimming 968 long machine rows to human's ~20-word
distribution; cm has no held-out generator, te neither. T3 is hi only.

**Unrelated, pre-existing.** 8 tests in `tests/data` fail identically before and after
this change (stale registry assertions such as "two held-out generators", plus
`NotImplementedError` stubs in test_loaders / test_schema).

**Verify:** `sha256sum data/processed/splits.json data/processed/corpus.jsonl
data/processed/archive/v1/*` against `docs/data/corpus_card.md` §0; `pytest
tests/data/test_freeze_splits.py` (5 passed).

## 2026-09-30 — Newline artefact: whitespace normalisation at scoring time; Head C stays excluded

**Finding.** In corpus v2 every human row has 0 newlines; every generator's rows
average 3.7-9.8 per text (en gemma 5.55, mistral 7.58, qwen7b 3.71; hi llama 7.14; te
gemma 9.83). Newline count alone separates human from machine at test AUROC 0.922 (en),
0.947 (hi), 0.938 (te); cm has no newlines on either side. The human sources are single
paragraph / flattened passages; a real student essay has paragraph breaks, so this is a
property of how the proxy corpus was built, not of human writing. Found while diagnosing
Head C (`scripts/diagnose_headc_v2.py`); Head C and TF-IDF are blind to it (MuRIL ids
identical with `\n` -> space on 300/300 texts), but Head A's sentence splitter counts
`\n` as a terminator (`src/features/stylometric.py:94`) and the mGPT/Qwen tokenisers see it.

**Decision.** Collapse newlines and whitespace runs to one space, in memory, before any
feature extraction or tokenisation, identically for human and machine text
(`src/utils/text.py`, switched on by `text_normalisation: whitespace` in
`configs/models_norm.yaml`, applied in `src/eval/score.py`). `corpus.jsonl` and
`splits.json` are untouched. Normalised caches, models and scores live under
`results/norm/`; the raw-text results are kept for comparison. Only headA, headB,
headB_word and headC were re-scored normalised; fastdetectgpt_en, ppl and binoculars
were NOT, and still saw the newlines.

**Effect (test AUROC, raw -> normalised).** Head A, seen: en 0.989->0.986, hi 0.973->0.973,
te 0.977->0.973, cm 0.928->0.928; llama hi 0.940->0.947. Head A does not depend on the
newline gap. Head B, seen: hi 0.350->0.336, te 0.251->0.235 (still inverted); llama hi
0.819->0.785. Head C: unchanged everywhere (0.961 on llama hi).
The nine-number surface-format probe falls but does not reach chance: en 0.989->0.918,
hi 0.970->0.836, te 0.980->0.871, cm 0.913->0.913 (unchanged); llama hi 0.982->0.713.
What remains is digit share, punctuation share and word length, i.e. register/source
differences between scraped news/wiki text and prompted generations, not formatting.

**cm terminal punctuation: kept, not normalised.** cm human rows end in terminal
punctuation 38% of the time (cmu_hinglish_dog 39%, comi_lingua 33%, hinge 80%); machine
97-100% in every domain including chat, where humans are at 39%. Within-domain AUROC of
this one feature is 0.83-0.89. It is a genuine register difference (informal Hinglish
omits final punctuation, LLMs complete sentences), not a corpus-processing artefact like
newlines, and stripping it from real text would alter what the scorers are meant to see.
It is not what drives the lexical signal: stripping trailing punctuation from all cm
texts leaves TF-IDF AUROC at 0.941 (word) and 0.947 (char, from 0.959). It is a
fairness risk: a careful writer who punctuates every sentence looks machine-like on this
cue. Reliance of Head A on it is untested. Revisit with an independent cm human sample.

**Head C stays excluded; reported as a diagnostic only.** TF-IDF also transfers to
held-out llama hi (word 0.911, char 0.921 vs Head C 0.961), across held-out generators
(leave-one-generator-out 0.91-1.00 en/hi) and across domains (0.90-1.00 en/hi/te), so
the held-out test cannot separate a real signal from a provenance difference. Every
human row comes from one scraped source per domain, every machine row from a prompted
generation. Pooled length matching explains none of it (length alone 0.507 on en seen
test; Head C per generator 0.999-1.000 raw and length-stratified).

**Not done.** Tables, fusion, calibration untouched. Web inference does not yet
normalise whitespace, so it would not match a normalised Head A if that were adopted.

**Verify:** `python scripts/report_norm_compare.py --config configs/data.yaml
--norm-scores results/norm/scores_a.parquet results/norm/scores_gpu.parquet`.

---

## 2026-09-30 — Normalisation adopted; Head A is not surface artefact (except cm); provenance confound bounds all AUROCs

**Adopted.** Whitespace normalisation (previous entry) is now the scoring and inference
default: `configs/models_norm.yaml` is the canonical models config, `webapp/detector/
services.py` reads it (`MODELS_CONFIG_PATH`), collapses whitespace on submitted text
before language ID, Head A and Head B, and loads Head A from `results/norm/models/`.
`configs/models.yaml` and `results/` (raw-text) are kept only for raw-vs-normalised
comparison. `fastdetectgpt_en` and `ppl` were re-scored normalised (test AUROC raw -> normalised, seen:
fastdetectgpt_en en 0.807->0.822, hi 0.718->0.752, te 0.576->0.589, cm 0.714->0.714; ppl en
0.900->0.901, hi 0.776->0.786, te 0.816->0.832, cm 0.748->0.748; llama hi 0.908->0.927 and
0.882->0.887). **`binoculars` was NOT re-scored** (stopped for time, ~35 min needed); its
`results/scores.parquet` column is still raw-text and not comparable with the rest. To finish:
`python -m src.eval.score --config configs/models_norm.yaml --scores results/norm/scores_gpu.parquet --column binoculars`.

**Is Head A reducible to the surface-format probe?** (`scripts/headA_surface_ablation.py`,
normalised text, test AUROC, seen / held-out llama hi.) (a) 9-number surface probe;
(b) Head A, 40 features; (c1) Head A minus 14 format-adjacent features (digit, uppercase,
punctuation shares, comma/period/question/exclaim/quote, punct diversity, word-length
mean/std/burstiness, clause counts); (c2) c1 minus 3 sentence-length features;
syntax-only = 15 parser features.

| bucket | (a) probe | (b) Head A | (c1) | (c2) | syntax-only |
|---|---|---|---|---|---|
| en | 0.918 | 0.986 | 0.982 | 0.970 | 0.945 |
| hi | 0.836 | 0.973 | 0.964 | 0.942 | 0.905 |
| hi, llama | 0.713 | 0.947 | 0.932 | 0.916 | 0.831 |
| te | 0.871 | 0.973 | 0.962 | 0.942 | 0.855 |
| cm | 0.913 | 0.928 | **0.806** | **0.748** | 0.708 |

Reading: in en, hi and te, Head A with every format-adjacent feature removed keeps
0.96-0.98 seen and 0.93 on held-out llama, well above the probe; the stylometric signal
is real beyond surface format. In cm it is not: removing them drops Head A to 0.806
(0.748 with sentence length), below the probe's 0.913, so cm Head A is largely surface
form (chiefly terminal punctuation) plus what the English parser makes of romanised text.
Do not present cm Head A as a stylometric result without this caveat.

**Residual surface separability is a provenance confound.** After normalisation the
9-number probe still reaches 0.84-0.92 (en 0.918, hi 0.836, te 0.871, cm 0.913). This is
not removable by normalisation: every human row is scraped, single-source-per-domain text
(news, wiki, forum, chat) and every machine row is a prompted generation, so any
difference between those two pipelines (register, digits, punctuation, word length) is
learnable. It bounds what any AUROC on this corpus can claim: a number here is
"scraped human vs prompted generation", not "human vs machine writing". An independent
human sample per bucket (e.g. student essays) is the only clean test. Head C's held-out
llama transfer (0.961) and TF-IDF's (0.911) are consistent with this, so Head C stays
excluded and is reported as a diagnostic only.

**cm terminal punctuation: kept (real register difference).** Informal Hinglish omits
final punctuation, LLMs complete sentences; stripping it would alter real text.
**Limitations (paper / report):** a careful writer who punctuates every sentence will
look machine-like on this cue, a fairness risk for exactly the students the project
protects. Head A's reliance on it is untested beyond the cm ablation above.

**Known consequence.** `results/models/fuser_ab.joblib` was archived to
`data/processed/archive/v1/results/models/` in the v2 re-score, and no v2 fuser exists,
so the webapp's `_get_fuser()` raises until fusion is refit on the normalised scores.
Tables, fusion and calibration are deliberately not rebuilt yet.

---

## 2026-09-30 — Downstream pipeline rebuilt on v2 normalised; fusion does not beat Head A; tests could clobber artefacts

**Done.** binoculars re-scored on normalised text (all 7 columns now match); merged to
`results/norm/scores.parquet` (`scripts/merge_norm_scores.py`). Fusion `[headA, headB, bucket
one-hot]` refit on the v2 train split and saved (`results/models/fuser_ab.joblib`;
`results/calibration.json`, model_version v2.0-norm; webapp `analyse_text` runs again). Per-bucket
temperature, per-bucket conformal tau at alpha 0.01/0.05 and a pooled global tau refit. All tables
rewritten (T1, T2, T3 new, T5 incl. the unmatched-human check now computed, T6, F1, F2, S1 Head A
ablation) in `results/` and `docs/results/`; `docs/results/README.md` rewritten. Reproduce commands
are at the top of that README.

**Findings to carry into the paper.**
- Fusion AUROC en 0.989, hi 0.961, te 0.963, cm 0.928 beats every zero-shot baseline but **not Head A
  alone** (0.986 / 0.960 / 0.973 / 0.928); te is 0.010 worse. Head B is inverted on seen hi (0.336) and
  te (0.235), and one global headB weight cannot use it in every bucket. Per-bucket headB sign or
  interaction terms is a structural option, not done.
- T3 (llama, hi): fused 0.969 seen -> 0.953 held-out; Head A 0.973 -> 0.947; zero-shot baselines and
  Head B *rise* on llama (e.g. fastdetectgpt_en 0.752 -> 0.927). Only hi has a held-out generator.
- Gate at alpha=0.01: human FPR ~1% in every bucket, but coverage 0.37-0.73 and it flags only 31% (seen)
  / 46% (llama) of hi machine text MACHINE; the rest abstains. Calibration gap is generator shift (hi ECE
  0.025 seen, 0.076 with llama), not temperature (temperatures 0.81-1.20, ECE barely moves).
- T5: en L1 disparity is reversed (general 0.085 vs indian 0.016 at alpha 0.05); unmatched humans are not
  flagged materially more than matched.
- All AUROCs are "scraped human vs prompted generation" (provenance confound above).

**Bug found and fixed.** `run_ablation()` and `fit_full_pipeline()` defaulted to `configs/default.yaml`,
which has no scores path, so the test suite (`tests/eval/test_ablation.py`) refit on the raw-text scores and
overwrote `results/models/fuser_ab.joblib`, `temperature.csv` and `conformal_thresholds.csv` with raw-text
fits. Defaults now point at `configs/models_norm.yaml`; the suite leaves the artefacts byte-identical. All
tables were regenerated after the fix. 16 other tests fail on master too (NotImplemented stubs for
detectgpt/xlmr/explain/loaders etc. and stale generator-count assertions), unchanged by this work.

**Not done.** T4 (adversarial), XLM-R supervised baseline, any held-out generator outside hi.

---

## Decisions still open (fill as resolved)

- [ ] Phase 0.1 — what "patent" means (disclosure / IPR-cell / IPO provisional).
- [ ] Phase 0.2 — what "consultancy" means; internal client acceptable?
- [ ] Scorer lock — after the mGPT vs Qwen tokenizer-fertility test (Day 3).
- [x] Head C — **included**; 7.9956 GiB card vs a 7.5 GiB gate (2026-09-13, above).
- [ ] Fusion — logistic vs GBM, decided on calibration AUROC.
- [ ] Paper venue — ICON / IEEE-Springer / journal fallback.
- [x] Held-out generators — **llama** (`llama3.1:8b-instruct-q4_K_M`) and **phi**
  (`phi3.5:3.8b-mini-instruct-q4_K_M`), disjoint 30 % slices (2026-09-13, above).
