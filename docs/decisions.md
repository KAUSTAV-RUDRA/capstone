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

## Decisions still open (fill as resolved)

- [ ] Phase 0.1 — what "patent" means (disclosure / IPR-cell / IPO provisional).
- [ ] Phase 0.2 — what "consultancy" means; internal client acceptable?
- [ ] Scorer lock — after the mGPT vs Qwen tokenizer-fertility test (Day 3).
- [x] Head C — **included**; 7.9956 GiB card vs a 7.5 GiB gate (2026-09-13, above).
- [ ] Fusion — logistic vs GBM, decided on calibration AUROC.
- [ ] Paper venue — ICON / IEEE-Springer / journal fallback.
- [ ] Held-out generators — the two names to put in `configs/*.yaml`.
