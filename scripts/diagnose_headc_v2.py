"""Head C shortcut diagnosis on corpus v2, with a held-out-generator test.

v1's diagnosis (scripts/diagnose_headc.py) excluded Head C because a TF-IDF
bag-of-words matched it on the single seen generator. v2 has held-out llama on
hi, and Head C scores ~0.96 there, which a lexical memoriser of the seen
generators should not do. This script re-asks the question properly.

Every probe is fit on the TRAIN split (seen generators only) per bucket and
scored on two test sets that share the same human rows:
  SEEN     human + seen-generator machine rows
  HELD-OUT human + llama rows (hi only)

Probes: Head C (as scored), an embedding LR refit here (sanity), length_words,
length_tokens, domain one-hot, TF-IDF word 1-2gram, TF-IDF char_wb 2-4gram,
surface-format features, PCA-10 of the embedding, embedding norm.

Extra checks:
  * leave-one-generator-out (LOGO): train without generator g, test human vs g.
  * length control: per-generator AUROC of headC/tfidf, and AUROC stratified
    within length_words quintile bins (pairs only compared at similar length).
  * top TF-IDF tokens, to say what is separating.

Usage: python scripts/diagnose_headc_v2.py --config configs/data.yaml
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config  # noqa: E402
from src.utils.io import read_jsonl  # noqa: E402

BUCKETS = ["en", "hi", "te", "cm"]
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def auc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))


def fmt_features(texts) -> np.ndarray:
    out = []
    for t in texts:
        n = max(len(t), 1)
        out.append([t.count("\n"), sum(c.isdigit() for c in t) / n, len(_PUNCT.findall(t)) / n,
                    sum(ord(c) < 128 for c in t) / n, float(t.rstrip()[-1:] in ".!?।"),
                    float(t[:1].isupper()), sum(c in "\"'“”‘’" for c in t) / n,
                    sum(ord(c) > 0x2000 and ord(c) not in range(0x0900, 0x0D80) for c in t) / n,
                    np.mean([len(w) for w in t.split()] or [0])])
    return np.asarray(out, dtype=float)


def strat_auc(y, s, length, bins: int = 5) -> float:
    """AUROC within length_words quantile bins, averaged by number of h/m pairs."""
    edges = np.quantile(length, np.linspace(0, 1, bins + 1)[1:-1])
    b = np.digitize(length, edges)
    num = den = 0.0
    for k in range(bins):
        m = b == k
        h, mc = int(((y == 0) & m).sum()), int(((y == 1) & m).sum())
        if h > 5 and mc > 5:
            num += auc(y[m], s[m]) * h * mc
            den += h * mc
    return num / den if den else float("nan")


class Probes:
    """Fit on a train frame, score any frame. One instance per (bucket, train set)."""

    def __init__(self, train: pd.DataFrame, emb: dict, domains: list[str]):
        self.emb, self.domains = emb, domains
        y = train["label"].to_numpy()
        self.m = {}
        self.m["length_words"] = lr().fit(train[["length_words"]], y)
        self.m["length_tokens"] = lr().fit(train[["length_tokens"]], y)
        self.m["domain"] = lr().fit(self._dom(train), y)
        self.m["format"] = lr().fit(fmt_features(train["text"]), y)
        self.m["emb_lr"] = lr().fit(self._emb(train), y)
        self.pca = PCA(n_components=10, random_state=42).fit(self._emb(train))
        self.m["pca10"] = lr().fit(self.pca.transform(self._emb(train)), y)
        self.m["emb_norm"] = lr().fit(np.linalg.norm(self._emb(train), axis=1, keepdims=True), y)
        self.m["tfidf_word"] = make_pipeline(
            TfidfVectorizer(max_features=20000, ngram_range=(1, 2)),
            LogisticRegression(max_iter=2000, class_weight="balanced")).fit(train["text"], y)
        self.m["tfidf_char"] = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=50000, sublinear_tf=True),
            LogisticRegression(max_iter=2000, class_weight="balanced")).fit(train["text"], y)

    def _dom(self, d):
        return np.array([[float(x == k) for k in self.domains] for x in d["domain"]])

    def _emb(self, d):
        return np.stack([self.emb[i] for i in d["id"]])

    def score(self, d: pd.DataFrame) -> dict[str, np.ndarray]:
        p = lambda k, X: self.m[k].predict_proba(X)[:, 1]  # noqa: E731
        E = self._emb(d)
        return {"length_words": p("length_words", d[["length_words"]]),
                "length_tokens": p("length_tokens", d[["length_tokens"]]),
                "domain": p("domain", self._dom(d)), "format": p("format", fmt_features(d["text"])),
                "emb_lr": p("emb_lr", E), "pca10": p("pca10", self.pca.transform(E)),
                "emb_norm": p("emb_norm", np.linalg.norm(E, axis=1, keepdims=True)),
                "tfidf_word": p("tfidf_word", d["text"]), "tfidf_char": p("tfidf_char", d["text"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--scores", default="results/scores.parquet")
    ap.add_argument("--embeddings", default="results/features/muril_embeddings.parquet")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    gens = load_config(args.config)["machine_corpus"]["generators"]
    seen = {g for g, v in gens.items() if v["role"] == "seen"}
    held = {g for g, v in gens.items() if v["role"] == "heldout"}
    corpus = pd.DataFrame(read_jsonl(args.corpus))
    corpus["gen"] = corpus["generator"].fillna("human")
    headc = pd.read_parquet(args.scores).set_index("id")["headC"]
    corpus["headC"] = corpus["id"].map(headc)
    emb = {k: np.asarray(v, dtype=float) for k, v in
           pd.read_parquet(args.embeddings).set_index("id")["embedding"].items()}
    domains = sorted(corpus["domain"].unique())
    order = ["headC", "emb_lr", "length_words", "length_tokens", "domain", "format", "tfidf_word",
             "tfidf_char", "pca10", "emb_norm"]

    def row(y, sc, label):
        return f"{label:22}" + "".join(f"{auc(y, sc[c]):>13.3f}" for c in order)

    header = f"{'':22}" + "".join(f"{c:>13}" for c in order)
    fitted: dict[str, Probes] = {}
    for b in BUCKETS:
        tr = corpus[(corpus.language == b) & (corpus.split == "train")]
        te = corpus[(corpus.language == b) & (corpus.split == "test")]
        P = fitted[b] = Probes(tr, emb, domains)
        print(f"\n===== {b}: train {len(tr)} (h {int((tr.label == 0).sum())}/m {int((tr.label == 1).sum())}) =====")
        print(header)
        for name, sub in (("SEEN test", te[(te.label == 0) | te.gen.isin(seen)]),
                          ("HELD-OUT test", te[(te.label == 0) | te.gen.isin(held)])):
            if sub.label.nunique() < 2:
                continue
            sc = P.score(sub)
            sc["headC"] = sub["headC"].to_numpy()
            print(row(sub.label.to_numpy(), sc, f"{name} ({int((sub.label == 1).sum())}m)"))
        Xp = P.pca.transform(P._emb(tr))
        print("  train corr(PC1..3,length) =", [round(float(np.corrcoef(Xp[:, k], tr.length_words)[0, 1]), 3) for k in range(3)],
              " corr(PC1..3,label) =", [round(float(np.corrcoef(Xp[:, k], tr.label)[0, 1]), 3) for k in range(3)])

    print("\n\n===== PER-GENERATOR (human vs one generator, test) and LENGTH-STRATIFIED AUROC =====")
    print("columns: headC / tfidf_word / tfidf_char / length_words, raw then stratified within length quintiles")
    print(f"{'bucket gen':16}{'n_m':>5}{'med_len h/m':>13} | " + "".join(f"{c:>10}" for c in
          ["C raw", "tfw raw", "tfc raw", "len raw", "C strat", "tfw strat", "tfc strat"]))
    for b in BUCKETS:
        te = corpus[(corpus.language == b) & (corpus.split == "test")]
        hum = te[te.label == 0]
        for g in sorted(te[te.label == 1].gen.unique()):
            sub = pd.concat([hum, te[te.gen == g]])
            y, ln = sub.label.to_numpy(), sub.length_words.to_numpy(dtype=float)
            sc = fitted[b].score(sub)
            c = sub["headC"].to_numpy()
            print(f"{b + ' ' + g + ('*' if g in held else ''):16}{int((y == 1).sum()):>5}"
                  f"{np.median(hum.length_words):>7.0f}/{np.median(te[te.gen == g].length_words):<5.0f} | "
                  f"{auc(y, c):>10.3f}{auc(y, sc['tfidf_word']):>10.3f}{auc(y, sc['tfidf_char']):>10.3f}"
                  f"{auc(y, sc['length_words']):>10.3f}{strat_auc(y, c, ln):>10.3f}"
                  f"{strat_auc(y, sc['tfidf_word'], ln):>10.3f}{strat_auc(y, sc['tfidf_char'], ln):>10.3f}")

    print("\n\n===== LEAVE-ONE-GENERATOR-OUT: train without g (seen gens only), test human vs g =====")
    print("(train/test humans are disjoint; g's own train rows are dropped; '*' = held-out, already never trained on)")
    print(f"{'bucket gen':16}" + "".join(f"{c:>13}" for c in ["emb_lr", "tfidf_word", "tfidf_char", "format", "domain"]))
    for b in BUCKETS:
        tr_all = corpus[(corpus.language == b) & (corpus.split == "train")]
        te = corpus[(corpus.language == b) & (corpus.split == "test")]
        hum_te = te[te.label == 0]
        seen_here = sorted(set(tr_all.gen) & seen)
        if len(seen_here) < 2:
            continue
        for g in seen_here:
            P = Probes(tr_all[tr_all.gen != g], emb, domains)
            sub = pd.concat([hum_te, te[te.gen == g]])
            sc = P.score(sub)
            y = sub.label.to_numpy()
            print(f"{b + ' -' + g:16}" + "".join(f"{auc(y, sc[c]):>13.3f}" for c in
                                                 ["emb_lr", "tfidf_word", "tfidf_char", "format", "domain"]))

    print("\n\n===== TOP TF-IDF WORD TOKENS (train), + = machine, - = human =====")
    for b in ["en", "hi", "te", "cm"]:
        m = fitted[b].m["tfidf_word"]
        vec, clf = m.steps[0][1], m.steps[1][1]
        names = np.array(vec.get_feature_names_out())
        w = clf.coef_[0]
        idx = np.argsort(w)
        print(f"{b}  +machine: {', '.join(names[idx[-15:]][::-1])}")
        print(f"{b}  -human:   {', '.join(names[idx[:15]])}")


if __name__ == "__main__":
    main()
