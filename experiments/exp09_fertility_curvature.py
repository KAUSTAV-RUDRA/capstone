"""exp09 - F2: tokenizer fertility vs curvature AUROC per bucket.

mGPT is Head B's scorer (locked §5, review2-sprint tokenizer-fertility
decision), Qwen2.5-0.5B is fastdetectgpt_en's. Both are Fast-DetectGPT-style
curvature statistics computed from per-token log-probabilities, so heavier
tokenizer fragmentation on Indic text is the leading hypothesis for Head B's
hi/te collapse (0.187 / 0.120 AUROC, progress.md 2026-09-24: "the inversion is
mGPT-specific"). This plots both scorers' tokens/word fertility
(results/tokenizer_fertility.csv) against their own curvature AUROC per
bucket, to see whether fragmentation tracks the collapse.

Writes: results/F2_fertility_vs_auroc.csv, results/F2_fertility_vs_auroc.png
Serves: docs/project-context-master.md §7 (Figure F2).
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

DATA_CSV = "results/F2_fertility_vs_auroc.csv"
PNG = "results/F2_fertility_vs_auroc.png"

SCORER_BY_METHOD = {"headB": "ai-forever/mGPT", "fastdetectgpt_en": "Qwen/Qwen2.5-0.5B"}
BUCKET_MARKERS = {"en": "o", "hi": "s", "te": "^", "cm": "D"}
METHOD_COLORS = {"headB": "#4C72B0", "fastdetectgpt_en": "#C44E52"}


def run(config_path: str) -> "pd.DataFrame":
    import pandas as pd

    from src.data.schema import LANGUAGE_BUCKETS
    from src.eval.metrics import auroc
    from src.eval.pipeline import fit_full_pipeline
    from src.utils.io import write_csv

    artifacts = fit_full_pipeline(config_path)
    fertility = pd.read_csv("results/tokenizer_fertility.csv")

    rows = []
    for method, scorer in SCORER_BY_METHOD.items():
        fert_rows = fertility[fertility["model"] == scorer].set_index("bucket")["tokens_per_word"]
        for bucket in LANGUAGE_BUCKETS:
            test = [r for r in artifacts.corpus if r["split"] == "test" and r["language"] == bucket]
            y = [r["label"] for r in test]
            s = artifacts.scores.loc[[r["id"] for r in test], method].to_numpy(dtype=float)
            rows.append({
                "method": method, "scorer": scorer, "bucket": bucket,
                "tokens_per_word": float(fert_rows[bucket]), "auroc": auroc(y, s),
            })
    df = pd.DataFrame(rows)
    write_csv(df, DATA_CSV)
    _plot(df, PNG)
    return df


def _plot(df: "pd.DataFrame", out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    for method, color in METHOD_COLORS.items():
        sub = df[df["method"] == method]
        ax.plot(sub["tokens_per_word"], sub["auroc"], color=color, linewidth=1, linestyle="--", alpha=0.6,
               label=f"{method} ({SCORER_BY_METHOD[method]})")
        for _, row in sub.iterrows():
            ax.scatter(row["tokens_per_word"], row["auroc"], color=color,
                      marker=BUCKET_MARKERS[row["bucket"]], s=90, zorder=3)
            ax.annotate(row["bucket"], (row["tokens_per_word"], row["auroc"]),
                       textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance (AUROC = 0.5)")
    ax.set_xlabel("Tokenizer fertility (tokens / word)")
    ax.set_ylabel("Test AUROC")
    ax.set_title("Curvature AUROC vs tokenizer fertility, per bucket")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    args = parser.parse_args()
    df = run(args.config)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
