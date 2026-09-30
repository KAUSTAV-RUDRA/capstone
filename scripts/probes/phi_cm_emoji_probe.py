"""phi cm emoji-compliance probe: current template vs. a stronger no-emoji constraint.

2026-09-28: phi's 24-row cm gate run failed hard (length ratio 3.57, 100 % at
cap, 14/24 rows with emoji despite the template's "No emoji"). The 12-passage
raw-compliance probe (``src.data.probe_compliance --generator phi --buckets cm``,
ask = bin, no compensation, current template) confirmed the length side is not
a compliance-ratio problem: median machine/ask is 4.64 (ratio 1.0 already
massively overshoots), 8/12 rows hit the token cap, and 8/12 still carry an
emoji despite the existing "No emoji" clause. That rules out re-tuning
compliance_ratio_by_generator; the length ask has to shrink at the template
level and the emoji clause has to be strengthened or restructured.

This probe isolates the emoji lever only, holding the same 12 passages and the
same ask (= bin, no compensation, matching the raw-compliance probe) so the
two runs are paired and the only variable is the constraint wording:

  * v0_current   -- the unmodified cm template (PROMPT_TEMPLATES["cm"]).
  * v1_end       -- the same template with the no-emoji instruction removed
                    from the middle and restated, stronger, as the LAST
                    sentence of the prompt: "Plain text only: no emoji, no
                    emoticons, no symbols standing in for words."

Read-only against the corpus. Rows go to --out-dir only.

Run::

    python -m scripts.probes.phi_cm_emoji_probe [--config configs/data.yaml]
        [--out-dir results/probes] [--n 12]
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median

from src.data import generate as g
from src.data.clean_artifacts import strip_boilerplate
from src.features.language_id import romanised_hindi_share
from src.utils.config import load_config

REPO = Path(__file__).resolve().parents[2]
GENERATOR = "phi"
SEED = 42

EMOJI = re.compile(r"[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]")

MIX = ("in Hinglish: mix Hindi and English words within every sentence, with the Hindi "
       "written in Roman script, casual, like students texting each other")

VARIANTS = {
    # Byte-identical to PROMPT_TEMPLATES["cm"] in src/data/generate.py.
    "v0_current": ("{first_sentence}\n\nWrite about {n_words} words on this topic " + MIX +
                   ", as {n_sentences} or more full sentences. No emoji. "
                   "Continue naturally and do not stop early."),
    # Same content, no-emoji clause moved to last and strengthened.
    "v1_end": ("{first_sentence}\n\nWrite about {n_words} words on this topic " + MIX +
               ", as {n_sentences} or more full sentences. Continue naturally and do not "
               "stop early. Plain text only: no emoji, no emoticons, no symbols standing "
               "in for words."),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out-dir", default="results/probes")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--overshoot", type=float, default=3.0)
    args = parser.parse_args()
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phi_cm_emoji_probe.jsonl"
    summary_path = out_dir / "phi_cm_emoji_probe_summary.txt"

    config = load_config(str(REPO / args.config))
    mc = config["machine_corpus"]
    spec = mc["generators"][GENERATOR]
    keep = g.make_assignment(GENERATOR, mc["generators"], float(mc["assignment"]["heldout_share"]))
    prompts = g.load_prompts(REPO / config["human_corpus"]["output_dir"], ["cm"], 1.0, SEED,
                             GENERATOR, None, keep)
    budget = mc["budget"]
    fertility, _ = g.load_fertility(REPO / budget["fertility_csv"], spec.get("tokenizer"),
                                    budget["fallback_tokenizer"], ["cm"])
    sample = g.stratified_probe(prompts, args.n)
    human_text = {p["id"]: p["text"] for p in sample}

    for p in sample:
        # ask = bin, no compensation -- matches the raw-compliance probe, so the
        # two runs are comparable and this run isolates the emoji lever only.
        p["ask_words"] = p["n_words"]
        p["num_predict"] = g.token_budget(p["ask_words"], fertility["cm"], args.overshoot,
                                          int(budget["pad_tokens"]), int(budget["cap"]))

    ollama = mc["ollama"]
    client = g.OllamaClient(ollama["host"], float(ollama["request_timeout_s"]))
    model = spec["ollama"]
    info = g.describe_ollama_model(client, model)
    worker = g.make_ollama_worker(client, GENERATOR, spec["role"], model, info,
                                  temperature=float(mc["decoding"]["temperature"]),
                                  top_p=float(mc["decoding"]["top_p"]),
                                  num_ctx=int(ollama["num_ctx"]), keep_alive="5m", seed=SEED)

    jobs = []
    for p in sample:
        for name, template in VARIANTS.items():
            item = dict(p)
            item["prompt"] = template.format(first_sentence=g.first_sentence(p["text"]),
                                             n_words=p["ask_words"],
                                             n_sentences=g.sentence_target("cm", p["ask_words"]))
            jobs.append((name, item))

    lock = threading.Lock()
    rows: list[dict] = []
    with open(out_path, "w", encoding="utf-8") as fh:
        def run(job):
            name, item = job
            row = worker(item)
            row["variant"] = name
            row["ask_words"] = item["ask_words"]
            with lock:
                rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()

        with ThreadPoolExecutor(4) as pool:
            list(pool.map(run, jobs))

    lines: list[str] = []
    out = lines.append
    out(f"generated {len(rows)}/{len(jobs)} rows, n={len(sample)} passages, "
        f"ask = bin (no compensation), overshoot {args.overshoot}")

    def describe(name, vr):
        ratios_ask = [len(strip_boilerplate(r["text"]).split()) / r["ask_words"] for r in vr]
        ratios_human = [len(strip_boilerplate(r["text"]).split()) / r["human_length_words"] for r in vr]
        at_cap = sum(1 for r in vr if r.get("done_reason") == "length")
        emoji_rows = [r for r in vr if EMOJI.search(r["text"])]
        shares = [romanised_hindi_share(r["text"]) for r in vr]
        out(f"\n{name}: n={len(vr)}")
        out(f"  length ratio (to ask) median {median(ratios_ask):.2f}, (to human) median "
            f"{median(ratios_human):.2f}")
        out(f"  at cap (done_reason=length): {at_cap}/{len(vr)} ({100*at_cap/len(vr):.0f} %)")
        out(f"  emoji rows: {len(emoji_rows)}/{len(vr)}")
        out(f"  Hindi share median: {median(shares):.2f}")
        if emoji_rows:
            out(f"  emoji ids: {[r['id'] for r in emoji_rows]}")

    for name in VARIANTS:
        describe(name, [r for r in rows if r["variant"] == name])

    text = "\n".join(lines)
    summary_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
