# Test Plan

**Project:** A Stylometric and Perplexity-Curvature Framework for Detecting Machine-Generated Text in Multilingual Student Submissions with Calibrated Abstention
**Version:** v1.0-qwen7b · **Date:** 25 September 2026
**Scope:** the research pipeline (`src/`), the generation toolchain, and the Django reviewer application (`webapp/`)

---

## 1. Testing objectives

This project makes a statistical guarantee — that the false-positive rate on human-written text stays below a chosen α within each language. A guarantee is only as trustworthy as the data and code behind it, so testing here has to answer three questions that ordinary unit testing does not:

1. **Is the code correct?** Conventional unit and integration testing.
2. **Is the corpus free of shortcuts?** A detector that separates classes on length, punctuation spacing, emoji, or a generator's vocabulary would report excellent numbers and mean nothing. Validating the *data* is as important as validating the code.
3. **Does a reported result survive scrutiny?** Any result that looks too good is treated as a defect until proven otherwise.

Each of the three is covered by a different mechanism: the unit suite, the generation gate, and the diagnostic probes.

---

## 2. Test levels and coverage

| Level | Mechanism | Count | Runs |
|---|---|---|---|
| Unit | `pytest` across `tests/` | **146 tests**, all passing | Every commit |
| Integration (research) | Shared-fit pipeline tests over the frozen corpus | Included above | Every commit |
| Integration (web) | `python manage.py test webapp.detector` | 10 tests, offline, no model loads | Every commit |
| Data validation | Five-check abort gate, run on production data before bulk generation | 5 checks × 4 buckets × N generators | Every generation run |
| Diagnostic | Targeted probes against a suspicious result | Ad hoc, documented per finding | On demand |
| Non-functional | VRAM, latency, reproducibility, provenance | Measured, recorded in `docs/decisions.md` | Per milestone |

### 2.1 Unit test distribution

| Module under test | What is verified |
|---|---|
| `tests/data/` | Corpus loaders; artefact stripping; length-binning; split freezing including the overwrite guard; generator assignment determinism; token-budget arithmetic; resume semantics |
| `tests/features/` | Stylometric feature extraction on all four scripts and on degenerate inputs; romanised-Hindi share; language ID and bucket routing; curvature against the published reference formula |
| `tests/baselines/` | Perplexity, Binoculars and the English-scorer Fast-DetectGPT variant; the shared higher-is-machine sign convention |
| `tests/fusion/`, `tests/calibration/` | Fusion fit and per-sample head contributions; temperature scaling monotonicity; conformal quantile arithmetic; abstention thresholds and the coverage sweep |
| `tests/eval/` | Metric computation; ablation configuration count; risk–coverage construction; fairness table assembly |
| `webapp/detector/tests.py` | End-to-end `analyse_text()` contract; persistence of `Submission` and `Decision`; batch CSV parsing; page rendering — all without loading a model |

### 2.2 Notable test cases

| Test | Why it exists |
|---|---|
| Batched curvature == single-row curvature == published reference formula | The core statistic must be right under padding and batching, not merely self-consistent |
| `splits.json` overwrite guard raises `FileExistsError` | Non-negotiable #6 says the splits are frozen; an assertion in prose is not a control |
| Generator assignment is `sha256(prompt_id) mod 3` and deterministic | Guarantees each human passage maps to exactly one seen generator, reproducibly |
| Resume skips rows already on disk and produces no duplicates | Verified by killing a real run mid-batch and re-running the identical command |
| A bucket with a bin-local defect fails the gate; head-ordered probing would miss it | Encodes a real defect (see §4) as a regression test |
| `src/` contains zero Django imports | Non-negotiable #7, checked by grep in CI and by inspection at each commit |
| Stylometric extraction is finite on empty, whitespace-only, single-word, punctuation-only and Indic-only input | Edge cases that would otherwise surface as NaNs deep in the fusion |

---

## 3. Data validation: the generation gate

Conventional testing cannot catch a corpus that is subtly separable. The gate is a validation harness that runs against **production data** before any bulk work: each bucket is taken to 24 rows drawn *stratified across its length bins*, and judged before the remaining rows are generated. Any failure aborts the run and reports what failed, what it cost, and how to undo it. There is no bypass flag, by design.

| Check | Criterion | Rationale |
|---|---|---|
| Length | Median machine/human word ratio within **0.75–1.25**, both bounds | A length gap alone would let a classifier separate the classes |
| Stratification | Probe rows drawn proportionally across length bins | A defect confined to one bin is invisible to a head-ordered probe |
| Code-mix | Romanised-Hindi function-word share ≥ **2/3** of the paired human text | "Hinglish" that is really English is a class signal |
| Emoji | **Zero** emoji rows in the probe | Human code-mixed text has emoji in 0 of 2,200 rows |
| Token cap | ≤ **20%** of rows hitting `num_predict` | Truncated endings are a learnable artefact |

Thresholds live in `configs/data.yaml` under `machine_corpus.gate`, not in code.

**Cleaning is symmetric.** Every drop rule is applied identically to human and machine rows, with a 2% cap on human-side drops. Cleaning that touched only one class would itself become the signal it was meant to remove.

---

## 4. Defect log

Every row below is a real defect caught by the mechanism named, with the measurement that exposed it and the fix that closed it.

| # | Found by | Defect | Measurement | Resolution |
|---|---|---|---|---|
| 1 | Length gate | Code-mixed machine text ran at 1.88× human length; 470 rows already written | Median ratio 1.98 | Sentence-count floor corrected (`max(8, …)` → `max(2, n/15)`); rows moved to `data/raw/discarded/`, not deleted |
| 2 | Gate design review | The probe read the head of an id-sorted list, so a defect confined to short passages passed | cm 0–40-word bin at 1.75 against a bucket median of 1.39 | Probe rows now drawn proportionally across length bins; regression test added |
| 3 | Code-mix check (added after #1) | Machine "Hinglish" was English in all but name | Romanised-Hindi share 0.03 vs 0.30 in the paired human text | Template rewritten after a three-variant probe; share now 0.35. Check added to the gate |
| 4 | Emoji check | Machine code-mixed rows carried emoji; the human side has none | 28 of 701 rows | Gate check added. ZWJ deliberately excluded — including it flagged 118 of 2,200 human Hindi passages, since ZWJ joins Devanagari conjuncts |
| 5 | Strip-order review | The gate judged raw text while the cleaning spec strips first, so rows cleaning would have saved were failing | 5 chatter matches, 2 of them legitimate in-dialogue text | `strip_boilerplate()` shared by the gate and the cleaning stage; anchored to the final line only |
| 6 | Cross-generator audit | A compliance ratio keyed on bucket alone applied Qwen's value to every generator | Gemma Hindi at 1.51× target under Qwen's 0.55 | Ratio made per generator; each model measured before bulk generation (Qwen 0.55, Gemma 0.73, Mistral 0.82) |
| 7 | Metadata audit | Every machine code-mixed row carried `code_mix_ratio = 0.0` against ≥ 0.30 for human rows | Perfect separability on a metadata field | Recomputed identically for both classes during corpus assembly |
| 8 | Selection-limit review | `load_or_record_selection` reused a frozen id list and ignored a changed cap | Would have silently generated the old count | Limit now part of the selection key |
| 9 | Result plausibility | Head C (MuRIL) scored 1.000 AUROC on three buckets | TF-IDF bag-of-words alone reaches 0.99 on the same data | Head C excluded from the adopted fusion; diagnosis documented in `docs/results/headc_diagnosis.md` |
| 10 | Result plausibility | Curvature below chance on Hindi and Telugu — suspected tokenizer fragmentation | Head B AUROC 0.187 / 0.120 | Hypothesis disproved by scorer swap (see §5); cause is scorer pretraining overlap, recorded as a finding |
| 11 | Structural review | Mistral's Telugu fertility pins every request at the token cap | 13.3 tokens/word vs Gemma's 4.9 | Telugu dropped for that generator, with the reason recorded |
| 12 | End-to-end web test | First request stalled over five minutes | `huggingface_hub` performed an HTTPS etag check per model load despite a warm cache | `HF_HUB_OFFLINE=1` defaulted in `services.py`; cold start now 23–29 s, warm under 10 s |

---

## 5. Diagnostic testing

Two results were treated as defects until investigated. Both investigations changed what the project reports.

### 5.1 Head C — a result rejected

Head C reached 1.000 AUROC on English, Hindi and Telugu. Rather than report it, we tested what it was separating on:

| Probe | en | hi | te | cm |
|---|---|---|---|---|
| Head C (MuRIL + logistic) | 1.000 | 1.000 | 1.000 | 0.927 |
| Length alone | 0.565 | 0.502 | 0.521 | 0.500 |
| Domain alone | 0.532 | 0.518 | 0.499 | 0.600 |
| **TF-IDF bag-of-words, no embeddings** | **0.991** | **0.986** | **0.997** | **0.897** |

Length and domain were ruled out — corpus matching held. But a plain lexical classifier reproduces Head C almost exactly, and the embedding's first principal component correlates with the label at |r| 0.75–0.90. With one generator scored, Head C is memorising that generator's vocabulary. **Head C was excluded from the adopted fusion** and is reported standalone only so the exclusion is quantitative.

*The same argument has not yet been closed for Head A, which carries the adopted fusion. Stylometric features are not lexical memorisation, but with a single generator the distinction is untested. Resolving it is the purpose of the held-out-generator evaluation.*

### 5.2 Head B — a hypothesis falsified

Curvature scored below chance on Hindi and Telugu. The obvious explanation was tokenizer fragmentation. A controlled scorer swap falsified it:

| Scorer | Fertility hi / te (tokens/word) | AUROC hi | AUROC te |
|---|---|---|---|
| mGPT-1.3B | 3.56 / 6.32 | 0.187 | 0.120 |
| Qwen2.5-0.5B | 4.80 / 11.92 | 0.728 | 0.555 |

The scorer with *worse* fertility performs *better*. Fragmentation does not explain it. A per-source breakdown showed every Indic human source sitting above the machine mean under mGPT and below it under Qwen — consistent with mGPT having been pretrained on those corpora. The effect is scorer-specific and is reported as a finding rather than a limitation.

---

## 6. Non-functional testing

| Property | Target | Measured |
|---|---|---|
| VRAM ceiling | Fits 8 GB | mGPT fp16 2.73 GiB; longest Telugu batch peaked 7.88 GiB. Token-budget batching plus an OOM-safe bisecting retry added after a fragmentation OOM at row 216; the re-run completed all 8,896 rows clean |
| Inference latency | ≤ 10 s per submission (NFR-02) | 23–29 s cold start (model load), **under 10 s warm** |
| Reproducibility | One command per experiment | Six experiment entry points share one cached fit; re-running any one alone stays consistent with the others |
| Provenance | Frozen corpus verifiable | SHA-256 of both frozen files committed in the corpus card, since `data/` is gitignored |
| Data locality | No submission text leaves the host (NFR-04) | All models run locally; `HF_HUB_OFFLINE=1` by default |
| Framework isolation | `src/` runs with Django uninstalled | Verified by grep at each commit |
| Interface language | Decision-support framing only | UI strings audited — "assists a reviewer", never "cheated" |

---

## 7. Test execution

```bash
pytest                                   # 146 tests
python manage.py test webapp.detector    # 10 tests, offline
python -m src.eval.score --report headA,headB,headB_word,fastdetectgpt_en,ppl,binoculars,headC
python -m src.data.freeze_splits --config configs/data.yaml   # must raise FileExistsError
sha256sum data/processed/splits.json     # must match docs/data/corpus_card.md
```

Four placeholder tests in `tests/data/` fail with `NotImplementedError`; they cover Phase-2 stubs not yet implemented and are tracked, not ignored.

---

## 8. Known gaps

| Gap | Status |
|---|---|
| Held-out-generator evaluation (T3) | Pending — needs two more generators. This is the test that decides whether Head A generalises or fingerprints |
| Adversarial evaluation (T4) | Pending — paraphrase, back-translation, hybrid edit |
| Unmatched-human FPR robustness check | Pending a GPU sitting; recorded as `NaN` with a note rather than a fabricated number |
| PDF / DOCX upload extraction | `.txt` only; tracked TODO |
| Batch scale | Capped at 200 rows, synchronous — adequate for demonstration, not for deployment |
| Temperature fitted on train | The frozen calibration split is human-only by design, so temperature is optimistic by construction; documented in `src/calibration/temperature.py` |

---

## 9. Testing philosophy, in one line

Every check in this plan exists because something it now catches once got through. The gate's five checks, the symmetric cleaning rule, and the two diagnostic investigations are all consequences of defects found in this project — and each is now a regression test rather than a lesson.
