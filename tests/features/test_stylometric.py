"""Tests for src.features.stylometric - Head A. Serves Phase 2 §2.2.2 / parts-plan Part 14.

No test here loads a parser: the syntax features are checked on hand-built parses,
and the extractor on ``syntax=False``, so the suite runs with no models downloaded.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.features.stylometric import (
    _FEATURE_NAMES, SYNTAX_FEATURE_NAMES, HeadA, StylometricExtractor, syntax_features,
)

TEXTS = ["The cat sat on the mat. It was warm.", "भारत एक विशाल देश है।",
         "భారతదేశం ఒక పెద్ద దేశం.", "Yaar kal main college gaya tha."]


def test_fit_transform_and_feature_names_align() -> None:
    ext = StylometricExtractor(syntax=False)
    X = ext.fit_transform(TEXTS, languages=["en", "hi", "te", "cm"])
    assert X.shape == (4, len(ext.feature_names())) == (4, 25)
    assert isinstance(ext.score(TEXTS), np.ndarray)


def test_syntax_adds_fifteen_named_features() -> None:
    assert len(SYNTAX_FEATURE_NAMES) == 15
    names = StylometricExtractor(syntax=True).feature_names()
    assert names == _FEATURE_NAMES + SYNTAX_FEATURE_NAMES and len(names) == 40


def test_syntax_features_on_a_hand_built_tree() -> None:
    # "The cat sat on the mat ." : sat is root; cat->sat, The->cat, mat->sat, on->mat, the->mat
    sent = [("DET", 2), ("NOUN", 3), ("VERB", 0), ("ADP", 6), ("DET", 6), ("NOUN", 3), ("PUNCT", 3)]
    f = dict(zip(SYNTAX_FEATURE_NAMES, syntax_features([sent])))
    assert f["upos_noun_ratio"] == pytest.approx(2 / 6)          # PUNCT excluded from the denominator
    assert f["upos_det_ratio"] == pytest.approx(2 / 6)
    assert f["mean_tree_depth"] == 3.0                            # sat -> mat -> on
    assert f["mean_dep_distance"] == pytest.approx((1 + 1 + 2 + 1 + 3 + 4) / 6)
    assert f["pos_bigram_type_ratio"] == pytest.approx(5 / 6)     # DET-NOUN occurs twice


def test_syntax_features_empty_and_cyclic_are_finite() -> None:
    assert syntax_features([]) == [0.0] * 15
    cyclic = [("NOUN", 2), ("NOUN", 1)]                           # malformed parse must not hang
    assert all(np.isfinite(syntax_features([cyclic])))


def test_head_a_per_bucket_fit_predict_and_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = np.array([0, 1] * 40)
    X[y == 1, 0] += 2.0
    buckets = ["en"] * 40 + ["cm"] * 40
    head = HeadA(["a", "b", "c"]).fit(X, y, buckets)
    p = head.predict(X, buckets)
    assert p.shape == (80,) and ((p >= 0) & (p <= 1)).all()
    assert p[y == 1].mean() > p[y == 0].mean()
    assert head.top_features("en", 1)[0][0] == "a"
    head.save(tmp_path / "h.joblib")
    assert np.allclose(HeadA.load(tmp_path / "h.joblib").predict(X, buckets), p)
    with pytest.raises(KeyError):
        head.predict(X[:1], ["te"])

    dev = head.top_deviating_features(X[y == 1][0], "en", k=2)
    assert len(dev) == 2
    assert {"name", "value", "z", "contribution"} <= dev[0].keys()
    assert abs(dev[0]["contribution"]) >= abs(dev[1]["contribution"])
