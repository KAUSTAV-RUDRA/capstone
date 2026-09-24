"""Diagnose Head C's 1.000 AUROC (en/hi/te) before it enters fusion.

Review-2 sprint Day 4, pre-fusion step. Head C (MuRIL embeddings + per-bucket
logistic head, src/features/semantic.py) scores 1.000 AUROC on en/hi/te and
0.927 on cm (python -m src.eval.score --report headC). That is suspiciously
higher than Head A (0.910-0.991) and Head B (0.12-0.86) on the same corpus, so
before it is allowed into fusion we check whether it is separating on the real
signal or on a shortcut: length, source/domain, or a trivial direction in the
embedding space.

Checks, per bucket, on the test split:
  1. A single-feature logistic head on length_words alone (mirrors Head A's
     shortcut check in progress.md 2026-09-24).
  2. A single-feature logistic head on length_tokens alone.
  3. domain/source as a classifier: does knowing the domain alone predict the
     label near-perfectly (a leaked construction artefact, not length)?
  4. Embedding-space check: fit logistic regression on the raw MuRIL embedding
     restricted to its top-k PCA components, and separately regress each
     embedding's L2 norm against length -- if headC's separation collapses
     onto a length- or norm-correlated direction, PC1 should correlate with
     length_words far more than it correlates with the label.
  5. Lexical-fingerprint check: a plain TF-IDF unigram+bigram bag-of-words
     logistic classifier (no embeddings, no MuRIL) on the same train/test
     split. Only one generator (qwen7b) is scored so far; if a crude
     word-presence classifier matches headC's AUROC, headC is separating on
     that generator's word choices, not on a generalisable AI-vs-human
     signal (non-negotiable #4: same-generator evaluation is worthless).

Usage: python -m scripts.diagnose_headc --config configs/default.yaml
Writes: docs/results/headc_diagnosis.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.schema import LANGUAGE_BUCKETS
from src.utils.io import read_jsonl

CORPUS = "data/processed/corpus.jsonl"
EMBEDDINGS = "results/features/muril_embeddings.parquet"
SCORES = "results/scores.parquet"
OUT_MD = "docs/results/headc_diagnosis.md"


def _fit_eval(X_train, y_train, X_test, y_test) -> float:
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))
    clf.fit(X_train, y_train)
    return float(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))


def run(config_path: str) -> str:
    corpus = read_jsonl(CORPUS)
    by_id = {r["id"]: r for r in corpus}
    emb_raw = pd.read_parquet(EMBEDDINGS).set_index("id")["embedding"]
    embeddings = {k: np.asarray(v, dtype=float) for k, v in emb_raw.items()}
    scores = pd.read_parquet(SCORES).set_index("id")

    lines = ["# Head C shortcut diagnosis\n",
             "Pre-fusion check (Review-2 Day 4): is Head C's 1.000 AUROC (en/hi/te) a "
             "real semantic signal or a shortcut on length / domain / a trivial "
             "embedding direction? All numbers are test-split, per bucket.\n"]

    verdicts: dict[str, str] = {}

    for bucket in LANGUAGE_BUCKETS:
        train = [r for r in corpus if r["language"] == bucket and r["split"] == "train"]
        test = [r for r in corpus if r["language"] == bucket and r["split"] == "test"]
        y_tr = np.array([r["label"] for r in train])
        y_te = np.array([r["label"] for r in test])

        len_words_tr = np.array([r["length_words"] for r in train], dtype=float)
        len_words_te = np.array([r["length_words"] for r in test], dtype=float)
        len_tok_tr = np.array([r["length_tokens"] for r in train], dtype=float)
        len_tok_te = np.array([r["length_tokens"] for r in test], dtype=float)

        auroc_len_words = _fit_eval(len_words_tr, y_tr, len_words_te, y_te)
        auroc_len_tokens = _fit_eval(len_tok_tr, y_tr, len_tok_te, y_te)

        # Domain as a one-hot classifier: can domain alone predict label?
        domains = sorted({r["domain"] for r in train} | {r["domain"] for r in test})
        dom_tr = np.array([[1.0 if r["domain"] == d else 0.0 for d in domains] for r in train])
        dom_te = np.array([[1.0 if r["domain"] == d else 0.0 for d in domains] for r in test])
        auroc_domain = _fit_eval(dom_tr, y_tr, dom_te, y_te)

        # Embedding-space check.
        X_tr = np.stack([embeddings[r["id"]] for r in train])
        X_te = np.stack([embeddings[r["id"]] for r in test])
        from sklearn.decomposition import PCA

        pca = PCA(n_components=10, random_state=42).fit(X_tr)
        pc_tr = pca.transform(X_tr)
        pc_te = pca.transform(X_te)
        # correlation of each of the first 3 PCs with length_words vs with label, on train
        pc_len_corr = [float(np.corrcoef(pc_tr[:, k], len_words_tr)[0, 1]) for k in range(3)]
        pc_label_corr = [float(np.corrcoef(pc_tr[:, k], y_tr)[0, 1]) for k in range(3)]
        auroc_pc10 = _fit_eval(pc_tr, y_tr, pc_te, y_te)
        # embedding L2 norm alone
        norm_tr = np.linalg.norm(X_tr, axis=1)
        norm_te = np.linalg.norm(X_te, axis=1)
        auroc_norm = _fit_eval(norm_tr, y_tr, norm_te, y_te)
        norm_len_corr = float(np.corrcoef(norm_tr, len_words_tr)[0, 1])
        norm_label_corr = float(np.corrcoef(norm_tr, y_tr)[0, 1])

        headc_auroc = float(roc_auc_score(y_te, scores.loc[[r["id"] for r in test], "headC"]))

        # Lexical-fingerprint check: crude TF-IDF bag-of-words, no embeddings.
        tfidf_clf = make_pipeline(
            TfidfVectorizer(max_features=20000, ngram_range=(1, 2)),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        tfidf_clf.fit([r["text"] for r in train], y_tr)
        auroc_tfidf = float(roc_auc_score(y_te, tfidf_clf.predict_proba([r["text"] for r in test])[:, 1]))

        lines.append(f"\n## {bucket}\n")
        lines.append(f"- Head C (full MuRIL embedding, per-bucket logistic): **{headc_auroc:.3f}**")
        lines.append(f"- length_words alone: {auroc_len_words:.3f}")
        lines.append(f"- length_tokens alone: {auroc_len_tokens:.3f}")
        lines.append(f"- domain one-hot alone: {auroc_domain:.3f} ({len(domains)} domains: {', '.join(domains)})")
        lines.append(f"- embedding, top-10 PCA components: {auroc_pc10:.3f}")
        lines.append(f"- embedding L2 norm alone: {auroc_norm:.3f}")
        lines.append(f"- TF-IDF unigram+bigram bag-of-words (no embeddings): **{auroc_tfidf:.3f}**")
        lines.append(f"- corr(PC1..3, length_words) = {[round(c, 3) for c in pc_len_corr]}; "
                     f"corr(PC1..3, label) = {[round(c, 3) for c in pc_label_corr]}")
        lines.append(f"- corr(embedding norm, length_words) = {norm_len_corr:.3f}; "
                     f"corr(embedding norm, label) = {norm_label_corr:.3f}")

        length_or_domain_shortcut = (auroc_len_words > 0.85 or auroc_len_tokens > 0.85
                                     or auroc_domain > 0.85 or auroc_norm > 0.90)
        lexical_shortcut = auroc_tfidf > 0.95 or (headc_auroc - auroc_tfidf) < 0.05
        if length_or_domain_shortcut:
            verdicts[bucket] = "SHORTCUT (length/domain)"
        elif lexical_shortcut:
            verdicts[bucket] = "SHORTCUT (lexical fingerprint of the single seen generator)"
        else:
            verdicts[bucket] = "no shortcut found in these checks"
        lines.append(f"- **verdict: {verdicts[bucket]}**")

    Path(OUT_MD).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    report = run(args.config)
    print(report)


if __name__ == "__main__":
    main()
