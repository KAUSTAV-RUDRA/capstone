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

---

## Decisions still open (fill as resolved)

- [ ] Phase 0.1 — what "patent" means (disclosure / IPR-cell / IPO provisional).
- [ ] Phase 0.2 — what "consultancy" means; internal client acceptable?
- [ ] Scorer lock — after the mGPT vs Qwen tokenizer-fertility test (Day 3).
- [ ] Head C — include only if host VRAM ≥ 8GB (`scripts/check_hardware.py`).
- [ ] Fusion — logistic vs GBM, decided on calibration AUROC.
- [ ] Paper venue — ICON / IEEE-Springer / journal fallback.
- [ ] Held-out generators — the two names to put in `configs/*.yaml`.
