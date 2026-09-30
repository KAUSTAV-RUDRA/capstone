"""The ONLY bridge between the Django web layer and the research package.

NON-NEGOTIABLE #7: no other file under ``webapp/`` may import from ``src/``.
``views.py`` calls into this module, never ``src`` directly.

Pipeline: language_id -> Head A (stylometric) + Head B (curvature) -> fusion
(A+B; Head C excluded, docs/results/headc_diagnosis.md) -> temperature ->
conformal -> abstention -> explanation. Everything heavy (Head A's model,
Head B's mGPT scorer, the fusion model, the calibration snapshot) is loaded
ONCE per process into the module-level ``_cache`` on first use, never per
request.

Requires, ahead of time (Day 2-4 pipeline outputs, all gitignored under
``results/``):
  results/norm/models/headA.joblib     python -m src.eval.score --config configs/models_norm.yaml --column headA
  results/models/fuser_ab.joblib       python -m experiments.exp04_fusion --config configs/models_norm.yaml
  results/calibration.json             python -m scripts.export_calibration --config configs/models_norm.yaml
Missing any of these raises a clear RuntimeError naming the command to run.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Models (mGPT, MuRIL) are assumed already present in the local HF cache from
# the Day 2-4 scoring runs. Without this, huggingface_hub still makes an etag
# HTTPS round-trip per load to check for updates even with a warm cache, which
# stalled for minutes rather than seconds on this dev box's connection.
# setdefault, not assignment: an operator who genuinely wants online
# revalidation can still set HF_HUB_OFFLINE=0 before starting Django.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.calibration.abstention import AbstentionGate, Verdict  # noqa: F401  (Verdict re-exported)
from src.calibration.conformal import ConformalCalibrator
from src.data.schema import LANGUAGE_BUCKETS
from src.features.language_id import LanguageIdentifier
from src.fusion.fuser import Fuser
from src.utils.config import load_config
from src.utils.text import collapse_whitespace

if TYPE_CHECKING:
    from src.features.curvature import CurvatureScorer
    from src.features.stylometric import HeadA, StylometricExtractor

log = logging.getLogger("detector.services")

# Whitespace-normalised variant: Head A / Head B are fitted and cached on text with
# newlines and whitespace runs collapsed (decisions.md 2026-09-30, newline artefact),
# so inference must collapse them too or it feeds the models a distribution they
# were not fitted on. The flag is read from this config so the two cannot drift.
MODELS_CONFIG_PATH = "configs/models_norm.yaml"
DEFAULT_CONFIG_PATH = "configs/default.yaml"
CALIBRATION_PATH = Path("results/calibration.json")
HEAD_A_DEFAULT_PATH = "results/norm/models/headA.joblib"
FUSER_AB_DEFAULT_PATH = "results/models/fuser_ab.joblib"

#: Loaded once per process, on first use. Never cleared during the process's life.
_cache: dict[str, Any] = {}


def _missing(path: str | Path, how_to_fix: str) -> RuntimeError:
    return RuntimeError(f"{path} not found. {how_to_fix}")


def _resolve_device(configured: str) -> str:
    """Fall back to CPU if CUDA was configured but isn't actually available.

    The research config (configs/models.yaml) assumes a GPU dev box; the
    webapp is a demo/consultancy layer that should still start (slowly) on a
    CPU-only host rather than crash.
    """
    if configured != "cuda":
        return configured
    import torch

    if torch.cuda.is_available():
        return "cuda"
    log.warning("configs/models.yaml requests device=cuda but no CUDA device is visible; falling back to cpu")
    return "cpu"


def _get_calibration() -> dict[str, Any]:
    if "calibration" not in _cache:
        if not CALIBRATION_PATH.exists():
            raise _missing(CALIBRATION_PATH,
                          "Run `python -m scripts.export_calibration --config configs/models_norm.yaml`.")
        _cache["calibration"] = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    return _cache["calibration"]


def _get_language_identifier() -> LanguageIdentifier:
    if "language_identifier" not in _cache:
        _cache["language_identifier"] = LanguageIdentifier(load_config(DEFAULT_CONFIG_PATH))
    return _cache["language_identifier"]


def _get_stylometric_extractor() -> "StylometricExtractor":
    if "stylometric_extractor" not in _cache:
        from src.features.stylometric import StylometricExtractor

        cfg = load_config(MODELS_CONFIG_PATH).get("head_a", {})
        _cache["stylometric_extractor"] = StylometricExtractor(config=cfg, syntax=True)
    return _cache["stylometric_extractor"]


def _get_head_a() -> "HeadA":
    if "head_a" not in _cache:
        from src.features.stylometric import HeadA

        path = load_config(MODELS_CONFIG_PATH).get("head_a", {}).get("model_path", HEAD_A_DEFAULT_PATH)
        if not Path(path).exists():
            raise _missing(path, "Run `python -m src.eval.score --column headA`.")
        _cache["head_a"] = HeadA.load(path)
    return _cache["head_a"]


def _get_curvature_scorer() -> "CurvatureScorer":
    if "curvature_scorer" not in _cache:
        from src.features.curvature import CurvatureScorer

        cfg = load_config(MODELS_CONFIG_PATH).get("head_b", {})
        device = _resolve_device(cfg.get("device", "cpu"))
        _cache["curvature_scorer"] = CurvatureScorer(
            cfg.get("scorer", "ai-forever/mGPT"), device=device,
            load_in_8bit=bool(cfg.get("load_in_8bit", False)), config=cfg)
    return _cache["curvature_scorer"]


def _get_fuser() -> Fuser:
    if "fuser" not in _cache:
        if not Path(FUSER_AB_DEFAULT_PATH).exists():
            raise _missing(FUSER_AB_DEFAULT_PATH,
                          "Run `python -m experiments.exp04_fusion --config configs/models_norm.yaml`.")
        _cache["fuser"] = Fuser.load(FUSER_AB_DEFAULT_PATH)
    return _cache["fuser"]


def _get_abstention_gate(bucket: str) -> AbstentionGate:
    """A one-bucket AbstentionGate built from the calibration.json snapshot.

    Reuses :class:`src.calibration.abstention.AbstentionGate`'s tested
    three-way rule rather than re-implementing it here.
    """
    key = f"abstention_gate:{bucket}"
    if key not in _cache:
        bucket_cal = _get_calibration()["buckets"][bucket]
        calibrator = ConformalCalibrator(alpha=0.01)
        calibrator.thresholds[bucket] = bucket_cal["tau_machine"]
        _cache[key] = AbstentionGate(calibrator, lower_threshold={bucket: bucket_cal["human_median"]})
    return _cache[key]


def _resolve_language(text: str, language: str | None) -> str:
    """Resolve the submitted language choice to a calibration bucket.

    An explicit, valid bucket choice is honoured as-is; ``"auto"``/``None``/
    anything else falls back to :class:`LanguageIdentifier`'s script + code-mix
    detection.
    """
    if language and language in LANGUAGE_BUCKETS:
        return language
    return _get_language_identifier().identify(text)["bucket"]


def analyse_text(text: str, language: str | None) -> dict[str, Any]:
    """Analyse a single text and return a decision-support payload.

    The returned dict is the stable contract consumed by ``views.py`` and
    persisted into :class:`webapp.detector.models.Decision`.

    Args:
        text: The submitted text.
        language: One of auto/en/hi/te/cm (or None).

    Returns:
        A dict with keys: verdict, confidence, language, driving_head,
        stylometric_score, curvature_score, semantic_score, explanation,
        model_version.
    """
    text = text or ""
    if "normalise" not in _cache:
        _cache["normalise"] = load_config(MODELS_CONFIG_PATH).get("text_normalisation") == "whitespace"
    if _cache["normalise"]:
        text = collapse_whitespace(text)
    detected = _get_language_identifier().identify(text)
    bucket = _resolve_language(text, language)

    extractor = _get_stylometric_extractor()
    raw_features = extractor.raw_features([text], [bucket])
    head_a = _get_head_a()
    stylometric_score = float(head_a.predict(raw_features, [bucket])[0])

    scorer = _get_curvature_scorer()
    curvature_score = float(scorer.score([text])[0])

    fuser = _get_fuser()
    head_scores = np.array([[stylometric_score, curvature_score]])
    logit = float(fuser.decision_function(head_scores, [bucket])[0])

    calibration = _get_calibration()
    temperature = calibration["buckets"][bucket]["temperature"]
    calibrated_prob = float(1.0 / (1.0 + np.exp(-logit / temperature)))

    gate = _get_abstention_gate(bucket)
    verdict, confidence = gate.decide(calibrated_prob, bucket)

    contributions = fuser.head_contributions(head_scores, [bucket])[0]
    driving_head = "stylometric" if abs(contributions["headA"]) >= abs(contributions["headB"]) else "curvature"

    top_features = [
        {"name": f["name"], "value": round(f["value"], 4), "contribution": round(f["contribution"], 4)}
        for f in head_a.top_deviating_features(raw_features[0], bucket, k=5)
    ]

    return {
        "verdict": verdict,
        "confidence": confidence,
        "language": bucket,
        "driving_head": driving_head,
        "stylometric_score": stylometric_score,
        "curvature_score": curvature_score,
        "semantic_score": None,  # Head C excluded from the adopted pipeline
        "explanation": {
            "top_features": top_features,
            "note": "Decision support, not an automatic accusation.",
            "is_decision_support": True,
            "language": detected["language"],
            "bucket": bucket,
            "code_mix_ratio": detected["code_mix_ratio"],
            "calibrated_probability": calibrated_prob,
            "tau_machine": calibration["buckets"][bucket]["tau_machine"],
            "human_median": calibration["buckets"][bucket]["human_median"],
            "head_contributions": contributions,
        },
        "model_version": calibration["model_version"],
    }


def analyse_batch(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Analyse several ``{row, filename, text, language}`` dicts.

    Returns the batch-table rows (``row``, ``filename``, ``language``,
    ``verdict``, ``confidence``, ``driving_head``) plus the full payload under
    ``result`` for callers that need it (e.g. to persist Decisions).
    """
    out = []
    for r in rows:
        try:
            result = analyse_text(r["text"], r.get("language"))
        except Exception as exc:  # noqa: BLE001 - one bad row must not sink the batch
            log.exception("batch row %s (%s) failed", r.get("row"), r.get("filename"))
            out.append({"row": r.get("row"), "filename": r.get("filename", ""), "language": "-",
                       "verdict": "ERROR", "confidence": None, "driving_head": str(exc), "result": None})
            continue
        out.append({
            "row": r.get("row"), "filename": r.get("filename", ""), "language": result["language"],
            "verdict": result["verdict"], "confidence": result["confidence"],
            "driving_head": result["driving_head"], "result": result,
        })
    return out
