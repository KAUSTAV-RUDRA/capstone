"""exp00 - Fast-DetectGPT smoke test on HC3 English.

Runs the locked scorer on ~200 HC3 samples (100 human / 100 machine) and
reports AUROC. This number is the Rubric-7 evidence for Review-1.

Writes: results/exp00_smoke.csv
Serves: docs/master-execution-plan.md Phase 1, Day 4 (P3).

Data: Hello-SimpleAI/HC3 English `all.jsonl`, pulled with huggingface_hub (no
`datasets` dependency). Human and ChatGPT answers are paired by question, so the
two classes are prompt-matched. Scorer = Qwen/Qwen2.5-0.5B (the locked fallback,
§5), fp16 on CUDA.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import yaml
from huggingface_hub import hf_hub_download

from src.baselines.fast_detectgpt import FastDetectGPTDetector

# NOTE: pandas / sklearn are imported lazily inside run(), AFTER the model has
# loaded and scored. Importing scipy+sklearn+pandas up front inflates the Windows
# commit charge and can trip "paging file too small" (os error 1455) during the
# CUDA weight load on a small pagefile.

RESULTS_CSV = "results/exp00_smoke.csv"

HC3_REPO = "Hello-SimpleAI/HC3"
HC3_FILE = "all.jsonl"
SCORER = "Qwen/Qwen2.5-0.5B"   # locked fallback (§5); user-requested for this smoke test
N_PER_CLASS = 100
MIN_WORDS = 5                  # skip trivially short answers


def _load_hc3_pairs(n_per_class: int, seed: int) -> list[dict]:
    """Return up to ``n_per_class`` prompt-matched {question, human, chatgpt} rows."""
    path = hf_hub_download(HC3_REPO, HC3_FILE, repo_type="dataset")
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    random.Random(seed).shuffle(rows)  # domain diversity, reproducible

    pairs: list[dict] = []
    for row in rows:
        humans = row.get("human_answers") or []
        bots = row.get("chatgpt_answers") or []
        if not humans or not bots:
            continue
        human = (humans[0] or "").strip()
        bot = (bots[0] or "").strip()
        if len(human.split()) < MIN_WORDS or len(bot.split()) < MIN_WORDS:
            continue
        pairs.append({"question": (row.get("question") or "").strip(),
                      "human": human, "chatgpt": bot})
        if len(pairs) >= n_per_class:
            break
    if len(pairs) < n_per_class:
        raise RuntimeError(f"HC3 yielded only {len(pairs)} usable pairs (< {n_per_class}).")
    return pairs


def run(config_path: str) -> "pd.DataFrame":
    """Run the smoke test and return the results table (also written to CSV)."""
    import torch

    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    seed = int(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "fp16" if device == "cuda" else "fp32"
    print(f"scorer={SCORER}  device={device}  dtype={dtype}  seed={seed}")

    pairs = _load_hc3_pairs(N_PER_CLASS, seed)
    # Interleave so the CSV keeps each prompt-matched pair together.
    texts = [p["human"] for p in pairs] + [p["chatgpt"] for p in pairs]
    labels = np.array([0] * len(pairs) + [1] * len(pairs))  # 0=human, 1=chatgpt
    questions = [p["question"] for p in pairs] * 2
    pair_ids = list(range(len(pairs))) * 2

    detector = FastDetectGPTDetector(scorer_model=SCORER, device=device, config=cfg)
    scores = detector.score(texts)

    import pandas as pd  # lazy: keep commit charge low during the CUDA load above
    from sklearn.metrics import roc_auc_score

    df = pd.DataFrame({
        "pair_id": pair_ids,
        "label": ["human"] * len(pairs) + ["chatgpt"] * len(pairs),
        "curvature": scores,
        "question": questions,
    })
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")

    finite = np.isfinite(scores)
    auroc = roc_auc_score(labels[finite], scores[finite])
    human_mean = float(np.nanmean(scores[labels == 0]))
    chatgpt_mean = float(np.nanmean(scores[labels == 1]))

    print(f"n = {len(pairs)} human / {len(pairs)} chatgpt "
          f"({int(finite.sum())} finite scores)")
    print(f"mean curvature — human: {human_mean:.4f} | chatgpt: {chatgpt_mean:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"wrote {RESULTS_CSV}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
