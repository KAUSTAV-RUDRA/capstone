"""cm v1 with the long bin's ask inflated: "about N words", N = human / 0.75 for passages of 40+ words.

The 2026-09-21 code-mix probe passed v1 overall (length median 0.97) but split it
by bin: 0-40 at 1.13, 40+ at 0.73. This probe keeps the short bin's ask at the
human length and inflates only the long bin's, on the same 24 stratified passages
with the same per-passage seeds. The sentence clause follows the ask
(``max(2, ask/15)``), so the prompt probed is exactly what ``build_prompt`` would
produce if this ask were adopted. The adoption rule, set before the run: both
bins' median machine/human ratio in 0.75-1.25, overall length median in band,
Hindi share >= 0.20, zero emoji rows, and the gate (length + code-mix) passes.

Read-only against the corpus; rows go to --out-dir only.

Evidence (run 2026-09-21 23:07, 24/24 rows, ADOPTED as the cm template):
cm_longask_probe.jsonl and cm_longask_probe_summary.txt in this directory;
docs/progress.md 2026-09-21.

Run::

    python -m scripts.probes.cm_longask_probe [--config configs/data.yaml]
        [--out-dir results/probes] [--max-minutes 0] [--long-ratio 0.75]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median

from src.data import generate as g
from src.features.language_id import romanised_hindi_share
from src.utils.config import load_config

REPO = Path(__file__).resolve().parents[2]
N_PASSAGES = 24
GENERATOR = "qwen7b"
SEED = 42
LONG_FROM_WORDS = 40
V1_ROWS = Path(__file__).with_name("cm_codemix_probe.jsonl")

V1 = ("{first_sentence}\n\nWrite about {n_words} words on this topic in Hinglish: mix Hindi and "
      "English words within every sentence, with the Hindi written in Roman script, casual, like "
      "students texting each other, as {n_sentences} or more full sentences. No emoji. "
      "Continue naturally and do not stop early.")

EMOJI = re.compile(r"[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
ECHO = re.compile(r"roman|script|hinglish|emoji", re.I)


def ask_words(human_words: int, long_ratio: float) -> int:
    return int(round(human_words / long_ratio)) if human_words >= LONG_FROM_WORDS else human_words


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out-dir", default="results/probes")
    parser.add_argument("--max-minutes", type=float, default=0)
    parser.add_argument("--long-ratio", type=float, default=0.75)
    args = parser.parse_args()
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cm_longask_probe.jsonl"
    summary_path = out_dir / "cm_longask_probe_summary.txt"
    deadline = (dt.datetime.now() + dt.timedelta(minutes=args.max_minutes) if args.max_minutes > 0
                else dt.datetime.max)

    config = load_config(str(REPO / args.config))
    mc = config["machine_corpus"]
    spec = mc["generators"][GENERATOR]
    keep = g.make_assignment(GENERATOR, mc["generators"], float(mc["assignment"]["heldout_share"]))
    prompts = g.load_prompts(REPO / config["human_corpus"]["output_dir"], ["cm"], 1.0, SEED,
                             GENERATOR, None, keep)
    budget = mc["budget"]
    fertility, _ = g.load_fertility(REPO / budget["fertility_csv"], spec.get("tokenizer"),
                                    budget["fallback_tokenizer"], ["cm"])
    sample = g.stratified_probe(prompts, N_PASSAGES)
    human_text = {p["id"]: p["text"] for p in sample}
    v1_rows = {r["id"]: r for r in map(json.loads, open(V1_ROWS, encoding="utf-8"))
               if r["variant"] == "v1_mix"}
    assert set(v1_rows) == set(human_text), "not the same 24 passages as the v1 probe"

    items = []
    for p in sample:
        h = p["human_length_words"]
        ask = ask_words(h, args.long_ratio)
        item = dict(p)
        item["ask_words"] = ask
        item["prompt"] = V1.format(first_sentence=g.first_sentence(p["text"]), n_words=ask,
                                   n_sentences=g.sentence_target("cm", ask))
        # Budget the passage we want (the human length), exactly as the real run does.
        item["num_predict"] = g.token_budget(h, fertility["cm"], float(budget["overshoot"]),
                                             int(budget["pad_tokens"]), int(budget["cap"]))
        if h < LONG_FROM_WORDS:
            assert item["prompt"] == v1_rows[p["id"]]["prompt"], p["id"]   # short bin unchanged
        items.append(item)

    ollama = mc["ollama"]
    client = g.OllamaClient(ollama["host"], float(ollama["request_timeout_s"]))
    model = spec["ollama"]
    info = g.describe_ollama_model(client, model)
    worker = g.make_ollama_worker(client, GENERATOR, spec["role"], model, info,
                                  temperature=float(mc["decoding"]["temperature"]),
                                  top_p=float(mc["decoding"]["top_p"]),
                                  num_ctx=int(ollama["num_ctx"]), keep_alive="5m", seed=SEED)
    lock = threading.Lock()
    rows: list[dict] = []
    with open(out_path, "w", encoding="utf-8") as fh:
        def run(item):
            if dt.datetime.now() >= deadline:
                return
            row = worker(item)
            row["variant"] = f"v1_longask_{args.long_ratio}"
            row["ask_words"] = item["ask_words"]
            with lock:
                rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()

        with ThreadPoolExecutor(4) as pool:
            list(pool.map(run, items))

    lines: list[str] = []
    out = lines.append

    def ratio(r):
        return r["length_words"] / r["human_length_words"]

    def describe(label, rs):
        short = [ratio(r) for r in rs if r["human_length_words"] < LONG_FROM_WORDS]
        long_ = [ratio(r) for r in rs if r["human_length_words"] >= LONG_FROM_WORDS]
        stats = g.gate_stats("cm", rs, human_texts={r["id"]: human_text[r["id"]] for r in rs})
        reasons = g.gate_reasons(stats)
        shares = [romanised_hindi_share(r["text"]) for r in rs]
        emoji = sum(bool(EMOJI.search(r["text"])) for r in rs)
        in_band = lambda xs: sum(0.75 <= x <= 1.25 for x in xs)  # noqa: E731
        bins_ok = bool(short and long_) and all(0.75 <= median(x) <= 1.25 for x in (short, long_))
        verdict = (not reasons and bins_ok and median(shares) >= 0.20 and emoji == 0)
        out(f"\n{label}: n={len(rs)}  {'ADOPT' if verdict else 'DO NOT ADOPT'}")
        out(f"  length median {stats.ratio:.2f} (mean {stats.mean:.2f}, p90 {stats.p90:.2f}), "
            f"in band {in_band([ratio(r) for r in rs])}/{len(rs)}, cap {stats.cap_pct:.0f} %")
        out(f"  0-40 median {median(short):.2f} (in band {in_band(short)}/{len(short)}), "
            f"40+ median {median(long_):.2f} (in band {in_band(long_)}/{len(long_)})")
        out(f"  Hindi share median {stats.codemix:.2f} vs human {stats.human_codemix:.2f}, "
            f"zero-Hindi {sum(s == 0 for s in shares)}, emoji {emoji}, "
            f"Devanagari {sum(bool(DEVANAGARI.search(r['text'])) for r in rs)}, "
            f"instruction echo {sum(bool(ECHO.search(r['text'])) for r in rs)}, "
            f"done {sorted({str(r['done_reason']) for r in rs})}")
        out(f"  gate (length + code-mix): {'PASS' if not reasons else 'FAIL - ' + '; '.join(reasons)}")

    out(f"generated {len(rows)}/{len(items)} rows; long bin = human >= {LONG_FROM_WORDS} words, "
        f"ask = human / {args.long_ratio}; short bin ask = human")
    describe("v1 (2026-09-21 probe, ask = human)", list(v1_rows.values()))
    describe(f"v1 long-ask (ask = human / {args.long_ratio} for 40+)", rows)
    out("\nlong passages, paired by passage (human -> v1 -> long-ask, ask):")
    by_id = {r["id"]: r for r in rows}
    for pid in sorted(v1_rows, key=lambda i: v1_rows[i]["human_length_words"]):
        v1 = v1_rows[pid]
        if v1["human_length_words"] >= LONG_FROM_WORDS and pid in by_id:
            r = by_id[pid]
            out(f"  {v1['human_length_words']:>4} -> {v1['length_words']:>4} ({ratio(v1):.2f}) -> "
                f"{r['length_words']:>4} ({ratio(r):.2f})  ask {r['ask_words']}")
    same = sum(1 for pid, r in by_id.items() if r["human_length_words"] < LONG_FROM_WORDS
               and r["text"] == v1_rows[pid]["text"])
    out(f"\nshort-bin rows identical to the v1 run (same prompt, same seed): "
        f"{same}/{sum(1 for r in rows if r['human_length_words'] < LONG_FROM_WORDS)}")
    for r in sorted(rows, key=lambda r: -r["human_length_words"])[:2]:
        out(f"  [{r['human_length_words']}->{r['length_words']}] {r['text'][:200]}")
    text = "\n".join(lines)
    summary_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
