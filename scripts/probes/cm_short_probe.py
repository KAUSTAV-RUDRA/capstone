"""cm 0-40 bin probe: short-passage variant vs the live template (read from disk).

Read-only against the corpus: the control is the rows already in qwen7b.jsonl for
the same 12 ids; only the variant is generated, and it is written to --out-dir.
Same model, decoding, num_predict and per-item seed as the real run.

Evidence (run 2026-09-21 22:46): cm_short_probe.jsonl and cm_short_probe_summary.txt
in this directory; docs/progress.md 2026-09-21. The control is the qwen7b cm rows
on disk at the time — once those are regenerated a rerun fails its prompt assertion.

Run::

    python -m scripts.probes.cm_short_probe [--config configs/data.yaml] [--out-dir results/probes]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import median

from src.data import generate as g
from src.utils.config import load_config

REPO = Path(__file__).resolve().parents[2]
N_PASSAGES = 12
BIN_MAX = 40          # human words, exclusive — the "0-40" bin of the 2026-09-20 analysis
GENERATOR = "qwen7b"
SEED = 42

VARIANT = ('{first_sentence}\n\nWrite one or two short sentences on this topic in casual '
           'Hindi-English Hinglish, Roman script, the way students text, at most {n_words} words.')

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LATIN = re.compile(r"[A-Za-z]")


def devanagari_share(text: str) -> float:
    dev, lat = len(DEVANAGARI.findall(text)), len(LATIN.findall(text))
    return dev / (dev + lat) if dev + lat else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out-dir", default="results/probes",
                        help="never this directory: a rerun must not overwrite the committed evidence")
    args = parser.parse_args()
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    OUT = out_dir / "cm_short_probe.jsonl"
    config = load_config(str(REPO / args.config))
    mc = config["machine_corpus"]
    spec = mc["generators"][GENERATOR]
    keep = g.make_assignment(GENERATOR, mc["generators"], float(mc["assignment"]["heldout_share"]))
    prompts = g.load_prompts(REPO / config["human_corpus"]["output_dir"], ["cm"], 1.0, SEED,
                             GENERATOR, None, keep)
    budget = mc["budget"]
    fertility, source = g.load_fertility(REPO / budget["fertility_csv"], spec.get("tokenizer"),
                                         budget["fallback_tokenizer"], ["cm"])
    for p in prompts:
        p["num_predict"] = g.token_budget(p["human_length_words"], fertility["cm"],
                                          float(budget["overshoot"]), int(budget["pad_tokens"]),
                                          int(budget["cap"]))

    on_disk = {}
    with open(REPO / mc["output_dir"] / f"{g.generator_slug(GENERATOR)}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if row["language"] == "cm":
                    on_disk[row["id"]] = row

    in_bin = [p for p in prompts if p["human_length_words"] < BIN_MAX and p["id"] in on_disk]
    print(f"cm prompts {len(prompts)}, on disk {len(on_disk)}, 0-{BIN_MAX} bin {len(in_bin)}; "
          f"fertility {fertility['cm']:.3f} ({source})")
    all_bin_ratios = [on_disk[p["id"]]["length_words"] / p["human_length_words"] for p in in_bin]
    print(f"whole bin on disk: median {median(all_bin_ratios):.2f}, "
          f"human median {median(p['human_length_words'] for p in in_bin)}")

    sample = sorted(in_bin, key=lambda p: g._hash64(p["id"]))[:N_PASSAGES]
    for p in sample:
        # the live prompt must be what is on disk, or the control is not a control
        assert on_disk[p["id"]]["prompt"] == p["prompt"], p["id"]
        p["prompt"] = VARIANT.format(first_sentence=g.first_sentence(p["text"]), n_words=p["n_words"])

    ollama = mc["ollama"]
    client = g.OllamaClient(ollama["host"], float(ollama["request_timeout_s"]))
    model = spec["ollama"]
    info = g.describe_ollama_model(client, model)
    print(f"model {model} {info}")
    worker = g.make_ollama_worker(client, GENERATOR, spec["role"], model, info,
                                  temperature=float(mc["decoding"]["temperature"]),
                                  top_p=float(mc["decoding"]["top_p"]),
                                  num_ctx=int(ollama["num_ctx"]), keep_alive="5m", seed=SEED)
    variant_rows = []
    with open(OUT, "w", encoding="utf-8") as fh:
        for p in sample:
            row = worker(p)
            variant_rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    control_rows = [on_disk[p["id"]] for p in sample]

    print(f"\n{'id':<28} {'hum':>4} {'ask':>4} {'live':>5} {'var':>5} {'live/h':>7} {'var/h':>6}  done")
    for p, c, v in zip(sample, control_rows, variant_rows):
        h = p["human_length_words"]
        print(f"{p['id']:<28} {h:>4} {p['n_words']:>4} {c['length_words']:>5} {v['length_words']:>5} "
              f"{c['length_words'] / h:>7.2f} {v['length_words'] / h:>6.2f}  {v['done_reason']}")

    for label, rows in (("live template (on disk)", control_rows), ("short variant", variant_rows)):
        stats = g.gate_stats("cm", rows)
        ratios = [r["length_words"] / r["human_length_words"] for r in rows]
        in_band = sum(0.75 <= r <= 1.25 for r in ratios)
        reasons = g.gate_reasons(stats)
        print(f"\n{label}: {stats.describe()}")
        print(f"  in band {in_band}/{len(rows)}, above 1.25 {sum(r > 1.25 for r in ratios)}, "
              f"below 0.75 {sum(r < 0.75 for r in ratios)}")
        print(f"  median words {median(r['length_words'] for r in rows)}, "
              f"devanagari share max {max(devanagari_share(r['text']) for r in rows):.2f}, "
              f"done {sorted({r['done_reason'] for r in rows})}")
        print(f"  gate: {'PASS' if not reasons else 'FAIL - ' + '; '.join(reasons)}")

    print("\nvariant texts:")
    for v in variant_rows:
        print(f"  [{v['human_length_words']}->{v['length_words']}] {v['text']}")


if __name__ == "__main__":
    main()
