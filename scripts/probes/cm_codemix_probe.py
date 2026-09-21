"""cm code-mix probe: 3 template variants x 24 passages stratified across cm's length bins.

Read-only against the corpus. Rows go to --out-dir only. Same model, decoding,
num_predict and per-item seed as the real run, so the variants are paired per passage.
Stops issuing requests after --max-minutes; rows are written as they complete.

Evidence (run 2026-09-21 22:57, 72/72 rows): cm_codemix_probe.jsonl and
cm_codemix_probe_summary.txt in this directory; docs/progress.md 2026-09-21.
The Hindi-share lexicon below is kept exactly as it was for that run (it still
counts yaar/bhai); the gate's lexicon lives in src/features/language_id.py.

Run::

    python -m scripts.probes.cm_codemix_probe [--config configs/data.yaml]
        [--out-dir results/probes] [--max-minutes 0]
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
from src.utils.config import load_config

REPO = Path(__file__).resolve().parents[2]
N_PASSAGES = 24
GENERATOR = "qwen7b"
SEED = 42

MIX = ("in Hinglish: mix Hindi and English words within every sentence, with the Hindi "
       "written in Roman script, casual, like students texting each other")
MIX_HALF = ("in Hinglish: mix Hindi and English words within every sentence, roughly half "
            "the words Hindi, with the Hindi written in Roman script, casual, like students "
            "texting each other")
# Each variant changes one thing against v1, so each lever's effect reads off directly.
VARIANTS = {
    "v1_mix": ("{first_sentence}\n\nWrite about {n_words} words on this topic " + MIX +
               ", as {n_sentences} or more full sentences. No emoji. "
               "Continue naturally and do not stop early."),
    "v2_mix_nosent": ("{first_sentence}\n\nWrite about {n_words} words on this topic " + MIX +
                      ". No emoji. Continue naturally and do not stop early."),
    "v3_mix_half": ("{first_sentence}\n\nWrite about {n_words} words on this topic " + MIX_HALF +
                    ", as {n_sentences} or more full sentences. No emoji. "
                    "Continue naturally and do not stop early."),
}

# Same lexicon as the 2026-09-21 0-40 probe, unchanged so the 0.20 threshold means the same thing.
HI = set("hai hain ho ka ki ke ko se mein me nahi nhi na kya bhi to toh tha thi yaar yar aur ek "
         "bhai kar karo karna raha rahe rhe hota hoga ab bas kuch sab log wala wali abhi apna apne "
         "mera meri tera tum aap hum ye yeh wo woh kaise kyun kyu jo agar par pe".split())
EMOJI = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]")
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
OTHER_SCRIPT = re.compile(r"[\u3040-\u9FFF\uAC00-\uD7AF\u0600-\u06FF]")


def hindi_share(text: str) -> float:
    toks = re.findall(r"[a-z]+", text.lower())
    return sum(w in HI for w in toks) / max(1, len(toks))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out-dir", default="results/probes",
                        help="never this directory: a rerun must not overwrite the committed evidence")
    parser.add_argument("--max-minutes", type=float, default=0, help="stop issuing requests after this long")
    args = parser.parse_args()
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    OUT = out_dir / "cm_codemix_probe.jsonl"
    SUMMARY = out_dir / "cm_codemix_probe_summary.txt"
    DEADLINE = (dt.datetime.now() + dt.timedelta(minutes=args.max_minutes) if args.max_minutes > 0
                else dt.datetime.max)
    config = load_config(str(REPO / args.config))
    mc = config["machine_corpus"]
    spec = mc["generators"][GENERATOR]
    human_dir = REPO / config["human_corpus"]["output_dir"]
    keep = g.make_assignment(GENERATOR, mc["generators"], float(mc["assignment"]["heldout_share"]))
    prompts = g.load_prompts(human_dir, ["cm"], 1.0, SEED, GENERATOR, None, keep)
    budget = mc["budget"]
    fertility, _ = g.load_fertility(REPO / budget["fertility_csv"], spec.get("tokenizer"),
                                    budget["fallback_tokenizer"], ["cm"])
    sample = g.stratified_probe(prompts, N_PASSAGES)
    for p in sample:
        p["num_predict"] = g.token_budget(p["human_length_words"], fertility["cm"],
                                          float(budget["overshoot"]), int(budget["pad_tokens"]),
                                          int(budget["cap"]))
    human_text = {p["id"]: p["text"] for p in sample}
    bins = {}
    for p in sample:
        bins[p["n_words"]] = bins.get(p["n_words"], 0) + 1
    print(f"{len(sample)} passages from {len(prompts)}, n_words bins {dict(sorted(bins.items()))}, "
          f"human median {median(p['human_length_words'] for p in sample)}")

    ollama = mc["ollama"]
    client = g.OllamaClient(ollama["host"], float(ollama["request_timeout_s"]))
    model = spec["ollama"]
    info = g.describe_ollama_model(client, model)
    worker = g.make_ollama_worker(client, GENERATOR, spec["role"], model, info,
                                  temperature=float(mc["decoding"]["temperature"]),
                                  top_p=float(mc["decoding"]["top_p"]),
                                  num_ctx=int(ollama["num_ctx"]), keep_alive="5m", seed=SEED)
    jobs = []
    for p in sample:                        # passage-major, so a deadline cut stays balanced
        h = p["human_length_words"]
        for name, template in VARIANTS.items():
            item = dict(p)
            item["prompt"] = template.format(first_sentence=g.first_sentence(p["text"]), n_words=h,
                                             n_sentences=g.sentence_target("cm", h))
            jobs.append((name, item))

    lock = threading.Lock()
    rows: list[dict] = []
    fh = open(OUT, "w", encoding="utf-8")

    def run(job):
        name, item = job
        if dt.datetime.now() >= DEADLINE:
            return
        row = worker(item)
        row["variant"] = name
        with lock:
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(run, jobs))
    fh.close()

    lines = []
    out = lines.append
    human_shares = [hindi_share(t) for t in human_text.values()]
    en_rows = sorted(g.read_jsonl(human_dir / "en.jsonl"), key=lambda r: r["id"])[:300]
    out(f"generated {len(rows)}/{len(jobs)} rows (deadline {DEADLINE:%H:%M})")
    out(f"human (same {len(sample)} passages): Hindi share median {median(human_shares):.2f}, "
        f"zero-Hindi {sum(s == 0 for s in human_shares)}, emoji {sum(bool(EMOJI.search(t)) for t in human_text.values())}")
    out(f"English floor (300 en human passages): Hindi-lexicon share median "
        f"{median(hindi_share(r['text']) for r in en_rows):.3f}")
    for name in VARIANTS:
        vr = [r for r in rows if r["variant"] == name]
        if not vr:
            out(f"\n{name}: no rows")
            continue
        stats = g.gate_stats("cm", vr)
        ratios = [r["length_words"] / r["human_length_words"] for r in vr]
        short = [r["length_words"] / r["human_length_words"] for r in vr if r["human_length_words"] < 40]
        rest = [r["length_words"] / r["human_length_words"] for r in vr if r["human_length_words"] >= 40]
        shares = [hindi_share(r["text"]) for r in vr]
        hum = [hindi_share(human_text[r["id"]]) for r in vr]
        emoji = sum(bool(EMOJI.search(r["text"])) for r in vr)
        passed = (0.75 <= stats.ratio <= 1.25) and median(shares) >= 0.20 and emoji == 0
        out(f"\n{name}: n={len(vr)}  {'PASS' if passed else 'FAIL'}")
        out(f"  length median {stats.ratio:.2f} (mean {stats.mean:.2f}, p90 {stats.p90:.2f}), in band "
            f"{sum(0.75 <= x <= 1.25 for x in ratios)}/{len(vr)}, above {sum(x > 1.25 for x in ratios)}, "
            f"below {sum(x < 0.75 for x in ratios)}, cap {stats.cap_pct:.0f} %")
        out(f"  0-40 median {median(short) if short else float('nan'):.2f} (n={len(short)}), "
            f"40+ median {median(rest) if rest else float('nan'):.2f} (n={len(rest)})")
        out(f"  Hindi share median {median(shares):.2f} vs human {median(hum):.2f} "
            f"(ratio {median(shares) / median(hum):.2f}), zero-Hindi {sum(s == 0 for s in shares)}, "
            f"emoji {emoji}, Devanagari {sum(bool(DEVANAGARI.search(r['text'])) for r in vr)}, "
            f"other-script {sum(bool(OTHER_SCRIPT.search(r['text'])) for r in vr)}, "
            f"done {sorted({str(r['done_reason']) for r in vr})}")
        for r in sorted(vr, key=lambda r: r["human_length_words"])[:3]:
            out(f"    [{r['human_length_words']}->{r['length_words']}] {r['text'][:160]}")
    text = "\n".join(lines)
    SUMMARY.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
