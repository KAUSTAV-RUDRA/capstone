# Progress log

Dated daily log — the evidence trail for Rubric #7 and every review.
Newest entries at the top.

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
