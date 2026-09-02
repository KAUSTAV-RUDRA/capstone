# mgt-detect

**A Stylometric and Perplexity-Curvature Framework for Detecting Machine-Generated
Text in Multilingual Student Submissions with Calibrated Abstention.**

Distinguishes human-written from LLM-written student text across English, Hindi,
Telugu, and Romanised code-mixed input. It fuses stylometric features with
perplexity-**curvature** (Fast-DetectGPT), calibrates a distribution-free
false-positive bound **per language** via split-conformal thresholds, and
**abstains** when unsure — returning `HUMAN / ABSTAIN / MACHINE` as decision
support for a human reviewer (never an automatic accusation).

See `docs/project-context-master.md` for the complete project truth and the nine
non-negotiables, and `docs/master-execution-plan.md` for the phase-by-phase plan.

---

## Repository layout

```
src/           Pure-Python research code (ZERO Django imports).
experiments/   expNN_*.py runners; each takes --config, writes results/<name>.csv
tests/         Mirrors src/.
scripts/       check_hardware.py (real), download_datasets.py
webapp/        Django 5 + DRF demo/consultancy web layer (bridges to src via
               webapp/detector/services.py ONLY).
configs/       default.yaml, models.yaml, data.yaml (every path parameterised).
docs/          context, plans, progress, decisions, lit/, data/, ip/, hte/
data/          raw + processed corpora (gitignored).
results/       experiment CSVs and figures (gitignored).
```

> **Status:** scaffold. All of `src/` is signatures + `NotImplementedError`. The
> Django app is fully clickable with placeholder output.

---

## Setup

```bash
# 1. Create and activate a virtualenv
python -m venv venv
venv\Scripts\activate        # Windows (PowerShell: venv\Scripts\Activate.ps1)
# source venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Check hardware (RAM, GPU, VRAM, torch/CUDA)
python scripts/check_hardware.py

# 4. Apply database migrations (creates db.sqlite3)
python manage.py migrate

# 5. (optional) create an admin user for /admin/
python manage.py createsuperuser

# 6. Run the demo web app
python manage.py runserver
#    then open http://127.0.0.1:8000/
```

Pages: `/` landing · `/analyse/` · `/result/<id>/` · `/batch/` · `/dashboard/`
· `/about/` · `/admin/`.

Configuration is via environment variables (all optional in dev):
`DJANGO_SECRET_KEY`, `DJANGO_DEBUG`, `DJANGO_ALLOWED_HOSTS`,
`DETECTOR_MODEL_VERSION`.

---

## Experiments → result tables

Each runner takes `--config` and writes `results/<name>.csv`.

| Runner | Writes | Result (docs/project-context-master.md §7) |
|---|---|---|
| `experiments/exp00_smoke.py`       | `results/exp00_smoke.csv`       | Review-1 smoke AUROC on HC3 (Rubric-7 evidence) |
| `experiments/exp01_baselines.py`   | `results/exp01_baselines.csv`   | **T1** main results vs 5 baselines; **T3** on held-out generators |
| `experiments/exp02_stylometric.py` | `results/exp02_stylometric.csv` | Head A per-bucket AUROC (feeds **T2**) |
| `experiments/exp03_curvature.py`   | `results/exp03_curvature.csv`   | Head B per-bucket AUROC + te/cm diagnostic (feeds **T2**) |
| `experiments/exp04_fusion.py`      | `results/exp04_fusion.csv`      | Fusion beats every baseline (feeds **T1/T2**) |
| `experiments/exp05_abstention.py`  | `results/exp05_abstention.csv`  | **T6** calibration (ECE + coverage) and **F1** risk–coverage curve |
| `experiments/exp06_adversarial.py` | `results/exp06_adversarial.csv` | **T4** clean vs paraphrase vs back-translation vs hybrid |
| `experiments/exp07_fairness.py`    | `results/exp07_fairness.csv`    | **T5** FPR by bucket and L1/L2 band, ours vs baselines |
| `experiments/exp08_ablation.py`    | `results/exp08_ablation.csv`    | **T2** A / B / A+B / +cal / +abstain |

Example:

```bash
python experiments/exp01_baselines.py --config configs/default.yaml
```

Headline claim shape: *"at X% coverage, FPR ≤ Y% in every language bucket."*
Never lead with raw accuracy.

---

## Tests

```bash
pip install pytest        # not pinned in requirements.txt
pytest
```

---

## Non-negotiables (short form; full list in context §4)

1. Multilingual + code-mixed stays in scope. 2. Curvature, not raw perplexity.
3. Abstention + per-language conformal thresholds are the contribution.
4. Held-out generators for evaluation. 5. No fine-tuning; ≤2B params; 4GB VRAM
floor. 6. `data/processed/splits.json` frozen once written. 7. `src/` has zero
Django imports; `webapp/detector/services.py` is the only bridge. 8. Decision
support, never an accusation. 9. MuRIL is encoder-only (Head C only).
