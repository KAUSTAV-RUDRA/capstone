# mgt-detect (v1.0)

**A Stylometric and Perplexity-Curvature Framework for Detecting Machine-Generated
Text in Multilingual Student Submissions with Calibrated Abstention.**

Distinguishes human-written from LLM-written student text across English, Hindi,
Telugu, and Romanised code-mixed input. It fuses stylometric features (Head A) with
perplexity-**curvature** (Head B, Fast-DetectGPT on mGPT-1.3B), calibrates a
distribution-free false-positive bound **per language** via split-conformal
thresholds, and **abstains** when unsure, returning `HUMAN / ABSTAIN / MACHINE` as
decision support for a human reviewer (never an automatic accusation).

See `docs/project-context-master.md` for the project truth and the nine
non-negotiables, `docs/results/README.md` for every result with its caveats, and
`docs/progress.md` / `docs/decisions.md` for the dated log.

---

## Repository layout

```
src/           Pure-Python research code (ZERO Django imports).
experiments/   expNN_*.py runners; each takes --config and writes tables.
tests/         Mirrors src/.
scripts/       calibration export, results README builder, diagnostics.
webapp/        Django 5 + DRF demo/consultancy layer; bridges to src via
               webapp/detector/services.py ONLY.
configs/       default.yaml, data.yaml, models.yaml, models_norm.yaml (canonical).
docs/          context, plans, progress, decisions, results/ (T1-T7, F1-F2).
data/          raw + processed corpora (gitignored).
results/       scores, fitted models, intermediate CSVs (gitignored).
```

`data/` and `results/` are not in the repository, so a fresh clone must rebuild them
(commands below) before the tables can be reproduced. The finished tables are
committed under `docs/results/`.

---

## Setup

Windows 11 / Python 3.12 / NVIDIA GPU with at least 4 GB VRAM (developed on an 8 GB RTX 4060 laptop).

```bash
# 1. virtualenv
python -m venv venv
venv\Scripts\activate                  # PowerShell: venv\Scripts\Activate.ps1
                                       # macOS / Linux: source venv/bin/activate

# 2. dependencies (pinned)
pip install -r requirements.txt
# torch must come from the CUDA index, the PyPI wheel is CPU-only:
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# 3. parser and lexicon data used by Head A and the hybrid attack
python -c "import stanza; stanza.download('hi'); stanza.download('te')"
python -m nltk.downloader wordnet omw-1.4

# 4. hardware check, then the web database
python scripts/check_hardware.py
python manage.py migrate
```

### Models

Hugging Face models are downloaded on first use into the local HF cache. The webapp
runs offline (`HF_HUB_OFFLINE=1`) once they are cached.

| Role | Model | Used by |
|---|---|---|
| Head B scorer | `ai-forever/mGPT` (1.3B) | `src.eval.score --column headB` |
| Baselines | `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-0.5B-Instruct` | fastdetectgpt_en, ppl, binoculars |
| Head C (diagnostic only) | `google/muril-base-cased` | `--column headC`, excluded from fusion |
| Back-translation | `facebook/nllb-200-distilled-600M` | `src.data.attack --attack backtranslation` |

### Ollama (machine-text generation and the paraphrase attack)

Install [Ollama](https://ollama.com), then pull the generator tags used in
`configs/data.yaml` (`machine_corpus.generators`):

```bash
ollama pull qwen2.5:7b-instruct          # seen generator, also the paraphrase attack
ollama pull gemma2:9b-instruct-q4_K_M    # seen
ollama pull mistral:7b-instruct          # seen
ollama pull llama3.1:8b                  # held-out (test only)
set OLLAMA_NUM_PARALLEL=4                # before starting the server; matches machine_corpus.ollama.parallel
```

Ollama is only needed to rebuild the corpus or re-run the paraphrase attack, not to run
the detector or the webapp. Close Ollama models (`ollama stop <tag>`) before GPU scoring or
back-translation: the attack CLI refuses to start while one is holding VRAM.

---

## Running the web app

```bash
python manage.py runserver               # http://127.0.0.1:8000/
```

Needs the fitted artefacts from the pipeline below (`results/norm/models/headA.joblib`,
`results/models/fuser_ab.joblib`, `results/calibration.json`). Pages: `/` · `/analyse/` ·
`/result/<id>/` · `/batch/` · `/dashboard/` · `/about/` · `/admin/`.

Deployed behaviour (`webapp/detector/services.py`): Head A + Head B, fusion, per-language
temperature and conformal gate at **alpha = 0.05** (set `DETECTOR_ALPHA=0.01` for the stricter
level). One **deployment rule**, not a research result: for `cm` text under 50 words, a
HUMAN verdict is changed to ABSTAIN with the reason "too short to confirm as human" (T7 shows
all cm false clears are under 50 words). MACHINE verdicts are never changed. No results table
depends on this rule. Other environment variables: `DJANGO_SECRET_KEY`, `DJANGO_DEBUG`,
`DJANGO_ALLOWED_HOSTS`, `DETECTOR_MODEL_VERSION`.

---

## Reproducing the corpus and scores

```bash
# human corpus per bucket, machine text, then the frozen splits (splits.json is never regenerated)
python -m src.data.build_human_corpus --bucket en --target 2200      # also hi, te, cm
python -m src.data.generate --generator qwen7b --buckets en,hi,te,cm  # also gemma, mistral, llama
python -m src.data.freeze_splits --config configs/data.yaml

# score every row (canonical, whitespace-normalised config); each column is resumable
C=configs/models_norm.yaml
python -m src.eval.score --config $C --scores results/norm/scores_a.parquet   --column headA
python -m src.eval.score --config $C --scores results/norm/scores_gpu.parquet --column headB
python -m src.eval.score --config $C --scores results/norm/scores_gpu.parquet --column fastdetectgpt_en
python -m src.eval.score --config $C --scores results/norm/scores_gpu.parquet --column ppl
python -m src.eval.score --config $C --scores results/norm/scores_gpu.parquet --column binoculars
python -m src.eval.score --config $C --scores results/norm/scores_gpu.parquet --column headC
python -m scripts.merge_norm_scores --config $C      # -> results/norm/scores.parquet
python -m experiments.exp04_fusion --config $C       # fits fusion + temperature + conformal (fuser_ab.joblib)
python -m scripts.export_calibration --config $C     # results/calibration.json for the webapp
```

Add `--max-minutes N` to any `src.eval.score` or `src.data.*` command to stop cleanly and resume later.

---

## Tables: one command each

All use `C=configs/models_norm.yaml`. Fitted thresholds and weights are loaded from disk by
every script after `exp04_fusion`; T4 and T7 never refit.

| Table | Command | Writes |
|---|---|---|
| **T1** main results vs baselines | `python -m experiments.exp01_baselines --config $C` | `results/T1_main_results.csv` |
| **T2** ablation A / B / A+B / +cal / +abstain | `python -m experiments.exp08_ablation --config $C` | `results/T2_ablation.csv` |
| **T3** held-out generator (llama) | `python -m experiments.exp10_heldout --config $C` | `results/T3_heldout*.csv` |
| **T4** adversarial, frozen detector | `python -m experiments.exp06_adversarial --config $C` | `docs/results/T4_adversarial.{csv,md}` |
| **T5** fairness by bucket and L1/L2 band | `python -m experiments.exp07_fairness --config $C` | `results/T5_fairness.csv` |
| **T6** calibration, plus F1 risk-coverage | `python -m experiments.exp05_abstention --config $C` | `results/T6_calibration.csv`, `F1_*` |
| **T7** error analysis (clean test split) | `python -m experiments.exp14_error_analysis --config $C` | `docs/results/T7_*` |

Extras: F2 `exp09_fertility_curvature`, both alpha operating points `exp13_alpha_operating_points`,
per-bucket fusion weights `exp11_perbucket_fusion`, coverage targets `exp12_coverage_targets`.
`python -m scripts.build_results_readme` copies the tables into `docs/results/` and rebuilds its README.
T1 and T2 use the clean corpus only and do not include attacked rows.

### Adversarial attacks (inputs to T4)

Test-split rows only, appended to `data/processed/attacked.jsonl`. Each CLI is resumable.

```bash
python -m src.data.attack --attack hybrid                                   # 20% of sentences edited; CPU
python -m src.data.attack --attack paraphrase       --max-per-bucket 250 100  # Qwen2.5-7B-Instruct via Ollama
python -m src.data.attack --attack backtranslation  --max-per-bucket 250 100  # en/hi/te only; cm not applicable

# score the attacked rows with the same heads, then build T4
python -m src.eval.score --config $C --only-attacked --scores results/norm/attacked_scores_a.parquet   --column headA
python -m src.eval.score --config $C --only-attacked --scores results/norm/attacked_scores_gpu.parquet --column headB
python -m experiments.exp06_adversarial --config $C
```

`--max-per-bucket MACHINE HUMAN` caps the rows per bucket (the first N in seeded order). The T4
run used 250 machine / 100 human for paraphrase and back-translation and the defaults for hybrid.

**IndicTrans2 was replaced by NLLB.** The project plan named IndicTrans2 for back-translation.
Its two `ai4bharat` model repositories are gated on Hugging Face (our token has no access) and its
`IndicTransToolkit` does not build on Windows without the MSVC C++ build tools. Back-translation
therefore uses `facebook/nllb-200-distilled-600M` (codes `eng_Latn` / `hin_Deva` / `tel_Telu`),
which covers all three languages, runs on the 4 GB VRAM floor and needs no gated access. The
`indictrans2` backend is still wired in `src/data/attack.py` and falls back to NLLB. Generation
is bounded (about 1.5x input tokens), with a script check and a greedy retry, because unbounded
NLLB output looped and hallucinated extra text; rows that still fail are kept as source text and
flagged `bt_failed`. Details: `docs/progress.md`, 2026-10-21.

---

## Tests

```bash
pytest                           # src/ and experiments: 162 passed, 7 skipped
python manage.py test webapp     # web layer: 13 passed
```

The 7 skips are unimplemented scaffold stubs whose function does not exist (vanilla DetectGPT
was dropped for Fast-DetectGPT; the supervised XLM-R baseline was never built; `loaders`,
`schema` constructors, `explain` and `build_indic_corpus` are stubs). Each skip states its reason.

---

## Non-negotiables (short form; full list in context section 4)

1. Multilingual + code-mixed stays in scope. 2. Curvature, not raw perplexity.
3. Abstention + per-language conformal thresholds are the contribution.
4. Held-out generators for evaluation. 5. No fine-tuning; at most 2B params; 4 GB VRAM
floor. 6. `data/processed/splits.json` frozen once written. 7. `src/` has zero
Django imports; `webapp/detector/services.py` is the only bridge. 8. Decision
support, never an accusation. 9. MuRIL is encoder-only (Head C only).
