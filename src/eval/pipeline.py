"""Shared fit-once pipeline: fusion -> temperature -> conformal -> abstention.

Every review2-sprint Day 4 table (T1/T2/T5/T6/F1/F2) needs the same fitted
fusion model, temperature scaler and conformal thresholds. This module fits
them once, caches the results as new columns on ``results/scores.parquet``
plus small CSV/joblib artefacts, and returns everything the experiment
scripts need -- so each ``experiments/exp0*.py`` stays independently runnable
(project convention) without re-fitting from scratch or drifting out of sync
with each other.

Head C is excluded from the adopted fusion (``fused_ab``) per
``docs/results/headc_diagnosis.md``: its 1.000 AUROC on en/hi/te is a lexical
fingerprint of the single seen generator, not a generalisable signal
(non-negotiable #4). ``fused_abc`` is still fit and stored for T1/T2 reference
only, never used downstream of fusion.

Serves docs/master-execution-plan.md Phase 2 §2.2.6, Phase 3 §3.1-§3.3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.calibration.abstention import AbstentionGate
from src.calibration.conformal import ConformalCalibrator
from src.calibration.temperature import TemperatureScaler
from src.data.schema import LANGUAGE_BUCKETS
from src.fusion.fuser import Fuser
from src.utils.io import read_jsonl

CORPUS_PATH = "data/processed/corpus.jsonl"
SCORES_PATH = "results/scores.parquet"
FUSER_AB_PATH = "results/models/fuser_ab.joblib"
FUSER_ABC_PATH = "results/models/fuser_abc.joblib"
CONFORMAL_CSV = "results/conformal_thresholds.csv"
TEMPERATURE_CSV = "results/temperature.csv"

HEADS_AB = ("headA", "headB")
HEADS_ABC = ("headA", "headB", "headC")
ALPHAS = (0.01, 0.05)


@dataclass
class PipelineArtifacts:
    corpus: list[dict[str, Any]]
    scores: pd.DataFrame  # indexed by id, includes fused_ab/_logit/_calibrated, fused_abc/_logit
    fuser_ab: Fuser
    fuser_abc: Fuser
    temperature: TemperatureScaler
    conformal: dict[float, ConformalCalibrator]  # alpha -> calibrator
    cal_median: dict[str, float]  # bucket -> median fused_ab_calibrated on cal split (human-only)
    abstention: AbstentionGate
    rows_by_split: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def bucket_of(self, row_id: str) -> str:
        return self._by_id[row_id]["language"]

    def __post_init__(self) -> None:
        self._by_id = {r["id"]: r for r in self.corpus}
        for split in ("train", "cal", "test"):
            self.rows_by_split[split] = [r for r in self.corpus if r["split"] == split]


def _needs_fit(scores: pd.DataFrame) -> bool:
    required = {"fused_ab", "fused_ab_logit", "fused_ab_calibrated", "fused_abc", "fused_abc_logit"}
    return not required.issubset(scores.columns) or scores[list(required)].isna().any().any()


def fit_full_pipeline(config_path: str = "configs/default.yaml", force: bool = False) -> PipelineArtifacts:
    """Fit (or load cached) fusion + temperature + conformal + abstention.

    Idempotent: if ``results/scores.parquet`` already has the fused columns
    and ``results/conformal_thresholds.csv`` exists, everything is loaded from
    disk instead of refit, unless ``force=True``.
    """
    corpus = read_jsonl(CORPUS_PATH)
    scores = pd.read_parquet(SCORES_PATH).set_index("id")

    if force or _needs_fit(scores) or not Path(CONFORMAL_CSV).exists():
        artifacts = _fit(corpus, scores)
        _persist(artifacts)
        return artifacts
    return _load(corpus, scores)


def _fit(corpus: list[dict[str, Any]], scores: pd.DataFrame) -> PipelineArtifacts:
    train = [r for r in corpus if r["split"] == "train"]
    cal = [r for r in corpus if r["split"] == "cal"]

    def X(rows: list[dict[str, Any]], heads: tuple[str, ...]) -> np.ndarray:
        return scores.loc[[r["id"] for r in rows], list(heads)].to_numpy(dtype=float)

    y_train = np.array([r["label"] for r in train])
    b_train = [r["language"] for r in train]

    fuser_ab = Fuser(method="logistic", buckets=LANGUAGE_BUCKETS).fit(
        X(train, HEADS_AB), y_train, b_train, head_names=list(HEADS_AB))
    fuser_abc = Fuser(method="logistic", buckets=LANGUAGE_BUCKETS).fit(
        X(train, HEADS_ABC), y_train, b_train, head_names=list(HEADS_ABC))

    ids_all = [r["id"] for r in corpus]
    buckets_all = [r["language"] for r in corpus]
    fused_ab_logit = fuser_ab.decision_function(X(corpus, HEADS_AB), buckets_all)
    fused_ab = 1.0 / (1.0 + np.exp(-fused_ab_logit))
    fused_abc_logit = fuser_abc.decision_function(X(corpus, HEADS_ABC), buckets_all)
    fused_abc = 1.0 / (1.0 + np.exp(-fused_abc_logit))

    scores = scores.copy()
    scores.loc[ids_all, "fused_ab"] = fused_ab
    scores.loc[ids_all, "fused_ab_logit"] = fused_ab_logit
    scores.loc[ids_all, "fused_abc"] = fused_abc
    scores.loc[ids_all, "fused_abc_logit"] = fused_abc_logit

    temperature = TemperatureScaler()
    for bucket in LANGUAGE_BUCKETS:
        b_rows = [r for r in train if r["language"] == bucket]
        logits = scores.loc[[r["id"] for r in b_rows], "fused_ab_logit"].to_numpy(dtype=float)
        labels = np.array([r["label"] for r in b_rows])
        temperature.fit(logits, labels, bucket=bucket)

    fused_ab_calibrated = np.array([
        temperature.transform(np.array([logit]), bucket=bucket)[0]
        for logit, bucket in zip(fused_ab_logit, buckets_all)
    ])
    scores.loc[ids_all, "fused_ab_calibrated"] = fused_ab_calibrated

    conformal: dict[float, ConformalCalibrator] = {}
    cal_median: dict[str, float] = {}
    for alpha in ALPHAS:
        calibrator = ConformalCalibrator(alpha=alpha)
        pooled: list[float] = []
        for bucket in LANGUAGE_BUCKETS:
            b_cal = [r for r in cal if r["language"] == bucket]
            assert all(r["label"] == 0 for r in b_cal), f"cal split for {bucket} is not human-only"
            human_scores = scores.loc[[r["id"] for r in b_cal], "fused_ab_calibrated"].to_numpy(dtype=float)
            calibrator.fit(human_scores, bucket)
            pooled.extend(human_scores.tolist())
            if alpha == ALPHAS[0]:
                cal_median[bucket] = float(np.median(human_scores))
        calibrator.fit_global(np.array(pooled))
        conformal[alpha] = calibrator

    abstention = AbstentionGate(conformal[0.01], lower_threshold=cal_median)

    return PipelineArtifacts(
        corpus=corpus, scores=scores, fuser_ab=fuser_ab, fuser_abc=fuser_abc,
        temperature=temperature, conformal=conformal, cal_median=cal_median, abstention=abstention,
    )


def _persist(a: PipelineArtifacts) -> None:
    Path(SCORES_PATH).parent.mkdir(parents=True, exist_ok=True)
    a.scores.reset_index().to_parquet(SCORES_PATH, index=False)
    a.fuser_ab.save(FUSER_AB_PATH)
    a.fuser_abc.save(FUSER_ABC_PATH)

    temp_rows = [{"bucket": b, "temperature": a.temperature.temperature(b)} for b in LANGUAGE_BUCKETS]
    pd.DataFrame(temp_rows).to_csv(TEMPERATURE_CSV, index=False)

    conf_rows = []
    for alpha, calibrator in a.conformal.items():
        for bucket in list(LANGUAGE_BUCKETS) + ["_global"]:
            conf_rows.append({
                "alpha": alpha, "bucket": bucket, "tau": calibrator.threshold(bucket),
                "n_calibration": calibrator.n_calibration.get(bucket),
                "cal_median": a.cal_median.get(bucket) if bucket != "_global" else None,
            })
    pd.DataFrame(conf_rows).to_csv(CONFORMAL_CSV, index=False)


def _load(corpus: list[dict[str, Any]], scores: pd.DataFrame) -> PipelineArtifacts:
    fuser_ab = Fuser.load(FUSER_AB_PATH)
    fuser_abc = Fuser.load(FUSER_ABC_PATH)

    temp_df = pd.read_csv(TEMPERATURE_CSV)
    temperature = TemperatureScaler()
    temperature.temperatures = dict(zip(temp_df["bucket"], temp_df["temperature"]))

    conf_df = pd.read_csv(CONFORMAL_CSV)
    conformal: dict[float, ConformalCalibrator] = {}
    cal_median: dict[str, float] = {}
    for alpha in ALPHAS:
        calibrator = ConformalCalibrator(alpha=alpha)
        sub = conf_df[np.isclose(conf_df["alpha"], alpha)]
        for _, row in sub.iterrows():
            calibrator.thresholds[row["bucket"]] = float(row["tau"])
            calibrator.n_calibration[row["bucket"]] = int(row["n_calibration"])
            if alpha == ALPHAS[0] and row["bucket"] != "_global" and pd.notna(row["cal_median"]):
                cal_median[row["bucket"]] = float(row["cal_median"])
        conformal[alpha] = calibrator

    abstention = AbstentionGate(conformal[0.01], lower_threshold=cal_median)

    return PipelineArtifacts(
        corpus=corpus, scores=scores, fuser_ab=fuser_ab, fuser_abc=fuser_abc,
        temperature=temperature, conformal=conformal, cal_median=cal_median, abstention=abstention,
    )
