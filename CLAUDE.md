# Project
Detecting machine-generated text in multilingual student submissions (en/hi/te/code-mixed).
Stylometric + perplexity-curvature fusion, per-language conformal calibration,
three-way output: human / abstain / machine. Django web layer for demo and consultancy.

# Read first, every session
docs/project-context-master.md   <- complete current truth. Read fully before any task.
docs/master-execution-plan.md    <- phase-by-phase steps and owners
docs/execution-plan.md           <- literature, datasets, architecture detail

# Non-negotiables (full list with reasons in context §4)
- Multilingual + code-mixed stays in scope. Never English-only.
- Curvature (Fast-DetectGPT), not raw perplexity.
- Abstention + per-language conformal thresholds are the contribution. Never simplify away.
- Held-out generators only for evaluation.
- No fine-tuning of large models. Nothing above 2B params. 4GB VRAM floor.
- data/processed/splits.json is frozen once written.
- src/ has ZERO Django imports. webapp/detector/services.py is the only bridge.
- Output is decision support, never an automatic accusation.
- MuRIL is encoder-only: Head C only, never for curvature.

# Locked decisions
- Scorer: ai-forever/mGPT (fallback Qwen/Qwen2.5-0.5B)
- Buckets: en, hi, te, cm. Four separate calibrations.
- Min 1000 human calibration texts per bucket.
- Web: Django 5 + DRF, SQLite in dev.

# Conventions
- Scripts only, no notebooks in main
- Every script takes --config pointing at configs/*.yaml; no hardcoded paths
- One component per session; stop when done and say how to verify
- Ask before installing anything not in requirements.txt
- Before anything structural: give 2-3 options with trade-offs, then wait
