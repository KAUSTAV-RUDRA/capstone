# Progress log

Dated daily log — the evidence trail for Rubric #7 and every review.
Newest entries at the top.

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
