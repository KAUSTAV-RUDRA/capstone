# SCAFFOLD PROMPT — paste into Claude Code after context is verified

Prerequisites: repo initialised, `docs/project-context-master.md` present, `CLAUDE.md` present. Run from the repo root.

---

```
Read docs/project-context-master.md fully before doing anything. Then scaffold the
complete project structure exactly as in its §8. This session is SCAFFOLD ONLY.

TWO PARTS, DIFFERENT RULES:

PART A — src/, experiments/, tests/, scripts/  (research code, stubs only)
Every Python file gets:
  - module docstring: what it implements and which section of
    docs/master-execution-plan.md it serves
  - full function/class signatures with type hints matching the interfaces in
    docs/project-context-master.md §3, §6, §7
  - every body is `raise NotImplementedError`
  - a `# TODO(phase-N step-X):` comment referencing the master plan step
  - __init__.py in every package
Common interfaces to enforce:
  - every baseline and every head exposes `score(texts: list[str]) -> np.ndarray`
  - StylometricExtractor has .fit(), .transform(), .feature_names()
  - Fuser has .fit(head_scores, labels), .predict_proba(head_scores)
  - ConformalCalibrator has .fit(scores_human_only, bucket), .threshold(bucket)
  - AbstentionGate.decide(prob, bucket) -> Literal["HUMAN","ABSTAIN","MACHINE"], float
  - every experiments/expNN_*.py takes --config and writes results/<name>.csv
Hard rule: src/ must contain ZERO Django imports.

PART B — webapp/  (Django, CLICKABLE, actually runs)
Create a working Django 5 project named `webapp` with one app `detector`.
This is NOT a stub — I want to run `python manage.py runserver` and click
through every page with placeholder content.

Pages (all navigable via a top nav bar, all must render without error):
  /                      landing: project title, one-paragraph description,
                         buttons to Analyse / Batch / Dashboard / About
  /analyse/              form: textarea OR file upload (txt/pdf/docx),
                         language dropdown (auto / en / hi / te / code-mixed).
                         On submit → redirect to /result/<id>/
  /result/<id>/          verdict card: HUMAN | ABSTAIN | MACHINE, confidence,
                         which head drove it, top-5 stylometric features,
                         curvature score, a "this is decision support, not a
                         verdict" banner. All values placeholder for now.
  /batch/                upload CSV → table of results (placeholder rows)
  /dashboard/            counts by verdict and by language, a table of recent
                         submissions, placeholder numbers
  /about/                method summary, team, disclaimer
  /admin/                default Django admin with Submission and Decision
                         registered

Models (webapp/detector/models.py):
  Submission: id (uuid), text, language, source_filename, uploaded_at,
              uploaded_by (nullable FK to User)
  Decision:   submission (OneToOne), verdict (choices HUMAN/ABSTAIN/MACHINE),
              confidence (float), driving_head, stylometric_score,
              curvature_score, semantic_score (nullable), explanation (JSON),
              model_version, created_at
  Register both in admin with list_display and filters on verdict + language.

Bridge (webapp/detector/services.py):
  The ONLY file that imports from src/. Expose
  `analyse_text(text: str, language: str | None) -> dict` that, for now,
  returns a hardcoded placeholder dict shaped exactly like the real output
  will be. Add a clear `# TODO(phase-3 step-3.8): wire to src.pipeline` comment.
  views.py calls services, never src directly.

Forms: forms.py with AnalyseForm and BatchUploadForm (Django forms, CSRF on).
Templates: templates/detector/base.html with nav + Bootstrap 5 via CDN,
  then landing.html, analyse.html, result.html, batch.html, dashboard.html,
  about.html extending base.
Static: static/detector/style.css with minimal overrides.
Settings: SQLite, DEBUG=True, SECRET_KEY read from env with a dev fallback,
  STATIC and MEDIA configured, `detector` in INSTALLED_APPS, sys.path insert
  so `src` is importable from webapp.
URLs: webapp/urls.py includes detector/urls.py with named routes.
Migrations: generate them (makemigrations) but do not run the server.

PART C — root and config files (real content, not stubs)
  - requirements.txt: pinned — torch, transformers, accelerate, bitsandbytes,
    scikit-learn, numpy, pandas, scipy, spacy, stanza, indic-nlp-library,
    pyyaml, tqdm, django, djangorestframework, python-docx, pypdf, matplotlib
  - configs/default.yaml, models.yaml, data.yaml with sensible defaults; every
    path parameterised; scorer_model: "ai-forever/mGPT", fallback
    "Qwen/Qwen2.5-0.5B"; buckets: [en, hi, te, cm]; alpha: [0.01, 0.05];
    heldout_generators: []  (to be filled)
  - README.md: description, setup (venv, install, migrate, runserver), and a
    table mapping experiments/expNN → results table (T1–T6, F1)
  - docs/progress.md: first dated entry "scaffold complete"
  - docs/decisions.md: header + the locked decisions from context §5
  - .gitignore: data/, results/, venv/, __pycache__/, *.pyc, .env, *.pt,
    *.bin, db.sqlite3, media/, wandb/
  - scripts/check_hardware.py: REAL — prints RAM, GPU name, VRAM, torch/CUDA

RULES
  - Do not install anything. Do not run the server. Do not write any
    implementation in src/ beyond signatures.
  - Do not invent modules not listed in context §8.
  - When finished: print the full tree, then print every function signature
    in src/ grouped by module as one flat list so I can review the
    architecture before anything is implemented.
  - Then stop.
```

---

## After it finishes — verification, in this order

```bash
pip install -r requirements.txt
python scripts/check_hardware.py
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Click every nav link. Submit the analyse form with dummy text and confirm you land on a result page. Open `/admin/` and confirm Submission and Decision are there.

Then:

```
Confirm src/ has zero Django imports: grep it and show me the output.
```

Then commit:

```bash
git add . && git commit -m "scaffold: full structure, clickable Django shell, research stubs"
git tag scaffold
```

## Follow-up sessions (one per session, in this order for Review-1)

```
Implement src/features/stylometric.py per its docstring and signatures.
Include a __main__ that runs on 5 samples (en×2, hi, te, hinglish). Nothing else.
```
```
Implement src/baselines/fast_detectgpt.py using the scorer from configs/models.yaml,
8-bit if VRAM < 8GB. __main__ scores 10 HC3 samples. Nothing else.
```
```
Wire webapp/detector/services.py to call the real StylometricExtractor and
fast_detectgpt.score(). Keep the placeholder shape. Result page must now show
real numbers for those two fields. Do not touch src/.
```
