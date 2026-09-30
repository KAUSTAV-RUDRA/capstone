"""exp11 - RQ2 experiment: per-bucket head weights vs the adopted global-weight fusion.

Adopted fusion (``fused_ab``) has one weight per head and a bucket intercept, so Head B, which
points opposite ways across buckets, cannot be used in every one. Three variants, all fit on the
train split only, all over standardised ``[headA, headB]``:

  global        the adopted fuser (one weight per head + bucket one-hot)
  interaction   headA, headB, bucket one-hot, headA x bucket, headB x bucket (one model)
  separate      one logistic regression per bucket on [headA, headB]

Reports test AUROC per bucket (all test rows, seen only, held-out llama on hi) for each variant,
Head A alone and Head B alone, with 95% bootstrap CIs and the paired bootstrap difference vs
Head A. Nothing adopted is changed: the saved fuser/calibration are not touched.

Writes: results/exp11_perbucket_fusion.csv, results/exp11_perbucket_weights.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.schema import LANGUAGE_BUCKETS
from src.eval.pipeline import fit_full_pipeline, generator_roles

N_BOOT = 1000


def _lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))


def _interaction(X, b):
    oh = np.array([[1.0 if x == k else 0.0 for k in LANGUAGE_BUCKETS] for x in b])
    return np.hstack([X, oh] + [X[:, [i]] * oh for i in range(X.shape[1])])


def run(config_path: str) -> pd.DataFrame:
    a = fit_full_pipeline(config_path)
    _, heldout = generator_roles()
    S = a.scores
    tr = [r for r in a.corpus if r["split"] == "train"]
    te = [r for r in a.corpus if r["split"] == "test"]
    X = lambda rows: S.loc[[r["id"] for r in rows], ["headA", "headB"]].to_numpy(float)  # noqa: E731
    btr, bte = [r["language"] for r in tr], np.array([r["language"] for r in te])
    ytr, yte = np.array([r["label"] for r in tr]), np.array([r["label"] for r in te])
    Xtr, Xte = X(tr), X(te)

    scores = {"global (adopted)": a.fuser_ab.decision_function(Xte, list(bte))}
    m = _lr().fit(_interaction(Xtr, btr), ytr)
    scores["interaction"] = m.decision_function(_interaction(Xte, list(bte)))
    sep = np.zeros(len(te))
    wrows = []
    for b in LANGUAGE_BUCKETS:
        mt, me = np.array(btr) == b, bte == b
        mb = _lr().fit(Xtr[mt], ytr[mt])
        sep[me] = mb.decision_function(Xte[me])
        coef = mb[-1].coef_[0]
        wrows.append({"bucket": b, "headA_weight": coef[0], "headB_weight": coef[1]})
    scores["separate per bucket"] = sep
    # Same variants with OUT-OF-FOLD Head A on train (5-fold per bucket, same features/model as Head A):
    # the adopted fuser sees in-sample Head A scores on train and may over-trust it.
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from src.features.stylometric import HeadA
    feats = pd.read_parquet("results/norm/features/stylometric.parquet").set_index("id")["features"]
    F = np.stack([np.asarray(feats.loc[r["id"]], float) for r in tr])
    oof = np.zeros(len(tr))
    ha = HeadA([f"f{i}" for i in range(F.shape[1])])
    for b in LANGUAGE_BUCKETS:
        mt = np.array(btr) == b
        oof[mt] = cross_val_predict(ha._new_model(), F[mt], ytr[mt], method="predict_proba",
                                    cv=StratifiedKFold(5, shuffle=True, random_state=0))[:, 1]
    Xoof = np.column_stack([oof, Xtr[:, 1]])
    g = _lr().fit(np.hstack([Xoof, np.array([[1.0 if x == k else 0.0 for k in LANGUAGE_BUCKETS] for x in btr])]), ytr)
    scores["global, OOF Head A"] = g.decision_function(
        np.hstack([Xte, np.array([[1.0 if x == k else 0.0 for k in LANGUAGE_BUCKETS] for x in bte])]))
    scores["interaction, OOF Head A"] = _lr().fit(_interaction(Xoof, btr), ytr).decision_function(_interaction(Xte, list(bte)))
    sep2 = np.zeros(len(te))
    for b in LANGUAGE_BUCKETS:
        mt, me = np.array(btr) == b, bte == b
        mb = _lr().fit(Xoof[mt], ytr[mt])
        sep2[me] = mb.decision_function(Xte[me])
        c = mb[-1].coef_[0]
        wrows.append({"bucket": b + " (OOF Head A)", "headA_weight": c[0], "headB_weight": c[1]})
    scores["separate, OOF Head A"] = sep2
    scores["Head A alone"] = Xte[:, 0]
    scores["Head B alone"] = Xte[:, 1]

    gen = np.array([r["generator"] for r in te])
    rng = np.random.default_rng(0)
    rows = []
    for b in LANGUAGE_BUCKETS:
        for scope in ("all", "seen", "heldout"):
            m_ = bte == b
            if scope == "seen":
                m_ &= ~np.isin(gen, list(heldout))
            if scope == "heldout":
                m_ &= (yte == 0) | np.isin(gen, list(heldout))
                if not (np.isin(gen, list(heldout)) & (bte == b)).any():
                    continue
            y = yte[m_]
            h, mc = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
            idx = [np.concatenate([rng.choice(h, len(h)), rng.choice(mc, len(mc))]) for _ in range(N_BOOT)]
            base = {k: v[m_] for k, v in scores.items()}
            boots_a = np.array([roc_auc_score(y[i], base["Head A alone"][i]) for i in idx])
            for name, s in base.items():
                boots = np.array([roc_auc_score(y[i], s[i]) for i in idx])
                lo, hi = np.percentile(boots, [2.5, 97.5])
                dl, dh = np.percentile(boots - boots_a, [2.5, 97.5])
                rows.append({"bucket": b, "scope": scope, "method": name, "auroc": roc_auc_score(y, s),
                             "lo": lo, "hi": hi, "diff_vs_headA": roc_auc_score(y, s) - roc_auc_score(y, base["Head A alone"]),
                             "diff_lo": dl, "diff_hi": dh})
    df = pd.DataFrame(rows)
    df.to_csv("results/exp11_perbucket_fusion.csv", index=False)
    pd.DataFrame(wrows).to_csv("results/exp11_perbucket_weights.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    df = run(ap.parse_args().config)
    piv = df.pivot_table(index=["bucket", "scope"], columns="method", values="auroc")
    print(piv.round(3).to_string())
    print(pd.read_csv("results/exp11_perbucket_weights.csv").round(2).to_string(index=False))
    d = df[~df.method.isin(["Head A alone", "Head B alone"])]
    print(d[["bucket", "scope", "method", "diff_vs_headA", "diff_lo", "diff_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
