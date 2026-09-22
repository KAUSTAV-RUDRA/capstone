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

## Decisions still open (fill as resolved)

- [ ] Phase 0.1 — what "patent" means (disclosure / IPR-cell / IPO provisional).
- [ ] Phase 0.2 — what "consultancy" means; internal client acceptable?
- [ ] Scorer lock — after the mGPT vs Qwen tokenizer-fertility test (Day 3).
- [x] Head C — **included**; 7.9956 GiB card vs a 7.5 GiB gate (2026-09-13, above).
- [ ] Fusion — logistic vs GBM, decided on calibration AUROC.
- [ ] Paper venue — ICON / IEEE-Springer / journal fallback.
- [x] Held-out generators — **llama** (`llama3.1:8b-instruct-q4_K_M`) and **phi**
  (`phi3.5:3.8b-mini-instruct-q4_K_M`), disjoint 30 % slices (2026-09-13, above).
