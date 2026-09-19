"""Offline tests for src.data.generate (no model load, no Ollama server).

Run with pytest, or directly:  python tests/data/test_generate.py
Serves docs/parts-plan.md Part 4(a) and Stage 3.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from src.data import generate
from src.utils import modelload, resumable
from src.utils.config import load_config

DATA_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "data.yaml"
IDS = [f"en-{i:05d}" for i in range(3000)]
SEEN = ["qwen7b", "gemma", "mistral"]
HELDOUT = ["llama", "phi"]


def test_generator_slug_is_filesystem_safe() -> None:
    assert generate.generator_slug("Qwen/Qwen2.5-7B-Instruct") == "qwen-qwen2-5-7b-instruct"
    assert generate.generator_slug("ai-forever/mGPT") == "ai-forever-mgpt"
    assert generate.generator_slug("qwen7b") == "qwen7b"
    assert "/" not in generate.generator_slug("meta-llama/Llama-3.1-8B-Instruct")


def test_first_sentence_handles_danda_and_truncates() -> None:
    assert generate.first_sentence("This is one. This is two.") == "This is one."
    assert generate.first_sentence("यह पहला है। यह दूसरा है।") == "यह पहला है।"
    long_sentence = " ".join(f"word{i}" for i in range(100))
    assert len(generate.first_sentence(long_sentence, max_words=40).split()) == 40


def test_length_bin_picks_the_nearest_bin() -> None:
    assert generate.length_bin(20) == 25
    assert generate.length_bin(140) == 150
    assert generate.length_bin(1000) == 300


def test_prompt_is_matched_to_the_human_passage() -> None:
    prompt = generate.build_prompt("Antibodies are proteins. They bind antigens.", "en", 150)
    assert "Antibodies are proteins." in prompt      # topic seed
    assert "about 150 words" in prompt               # length match
    assert "in English" in prompt
    hi = generate.build_prompt("यह एक वाक्य है। और दूसरा।", "hi", 200)
    assert "in Hindi" in hi and "about 200 words" in hi   # hi is on the PLAIN template:
    assert "do not stop early" not in hi.lower()          # it over-produces under framing
    te = generate.build_prompt("ఇది ఒక వాక్యం. రెండు.", "te", 200)
    assert "in Telugu" in te and "at least 200 words" in te
    assert "do not stop early" in te.lower()     # compliance framing (decisions.md 2026-09-18)


def test_cm_prompt_asks_for_romanised_informal_hinglish() -> None:
    prompt = generate.build_prompt("Yaar yeh movie bahut acchi thi.", "cm", 50)
    for expected in ("Hinglish", "Roman script", "the way students text"):
        assert expected in prompt, f"cm prompt must say '{expected}'"
    assert "in English" not in prompt


def test_4bit_threshold_reads_the_size_from_the_model_id() -> None:
    assert modelload.estimate_params_b("Qwen/Qwen2.5-7B-Instruct") == 7.0
    assert modelload.estimate_params_b("Qwen/Qwen2.5-0.5B") == 0.5
    assert modelload.estimate_params_b("google/gemma-2-9b-it") == 9.0
    assert modelload.estimate_params_b("ai-forever/mGPT") == 1.3   # size not in the id
    assert modelload.should_use_4bit("Qwen/Qwen2.5-7B-Instruct") is True
    assert modelload.should_use_4bit("google/gemma-2-9b-it") is True
    assert modelload.should_use_4bit("Qwen/Qwen2.5-0.5B") is False
    assert modelload.should_use_4bit("ai-forever/mGPT") is False   # 1.3B < 3B


def test_config_registry_has_three_seen_and_two_heldout_generators() -> None:
    config = load_config(DATA_CONFIG)
    generators = config["machine_corpus"]["generators"]
    assert generate.generator_roster(generators) == (SEEN, HELDOUT)
    assert config["heldout_generators"] == HELDOUT      # what freeze_splits routes to test
    for alias, spec in generators.items():
        assert spec["ollama"] and spec["hf"] and spec["tokenizer"], alias


def test_seen_assignment_is_sha256_mod_3_and_gives_each_passage_one_generator() -> None:
    expected = SEEN[int.from_bytes(hashlib.sha256(b"en-00000").digest()[:8], "big") % 3]
    assert generate.seen_generator_for("en-00000", SEEN) == expected   # not salted like hash()
    picks = [generate.seen_generator_for(i, SEEN) for i in IDS]
    for alias in SEEN:
        share = picks.count(alias) / len(IDS)
        assert 0.30 < share < 0.37, (alias, share)
    generators = load_config(DATA_CONFIG)["machine_corpus"]["generators"]
    keeps = [generate.make_assignment(alias, generators, 0.30) for alias in SEEN]
    assert all(sum(keep(i) for keep in keeps) == 1 for i in IDS)


def test_heldout_generators_take_disjoint_30_percent_slices_independent_of_seen() -> None:
    generators = load_config(DATA_CONFIG)["machine_corpus"]["generators"]
    llama = generate.make_assignment("llama", generators, 0.30)
    phi = generate.make_assignment("phi", generators, 0.30)
    assert not any(llama(i) and phi(i) for i in IDS)                  # disjoint
    for keep in (llama, phi):
        share = sum(map(keep, IDS)) / len(IDS)
        assert 0.27 < share < 0.33, share
    llama_ids = [i for i in IDS if llama(i)]
    for alias in SEEN:                                               # independent second hash
        share = sum(generate.seen_generator_for(i, SEEN) == alias for i in llama_ids) / len(llama_ids)
        assert 0.27 < share < 0.40, (alias, share)
    try:
        generate.heldout_generator_for("x", HELDOUT, 0.6)
    except ValueError:
        pass
    else:
        raise AssertionError("2 x 60 % must be rejected")


def test_load_prompts_keeps_only_assigned_passages() -> None:
    rows = [{"id": f"en-{i:05d}", "text": "One sentence here. Two.", "length_words": 140, "domain": "d"}
            for i in range(60)]
    keep = lambda prompt_id: generate.seen_generator_for(prompt_id, SEEN) == "gemma"  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "en.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        prompts = generate.load_prompts(Path(tmp), ["en"], 1.0, 42, "gemma", keep=keep)
    assert prompts and all(p["id"].startswith("gemma__") for p in prompts)
    assert {p["prompt_id"] for p in prompts} == {r["id"] for r in rows if keep(r["id"])}


def test_token_budget_follows_fertility_and_caps() -> None:
    assert generate.token_budget(150, 1.339) == int(150 * 1.339 * 1.3 + 64)
    assert generate.token_budget(150, 4.804) == 1000     # hi; the old words x 4 + 64 budget gave 664
    assert generate.token_budget(150, 11.919) == 2048    # te still hits the cap
    assert generate.token_budget(25, 1.672, cap=100) == 100


def test_cm_short_passage_is_not_asked_for_eight_sentences() -> None:
    # Regression: a floor of 8 sentences implies ~88 words whatever the word target
    # says, and drove 21-word cm passages to 84 machine words (decisions.md 2026-09-18).
    short = generate.build_prompt("Yaar yeh movie bahut acchi thi.", "cm", 25)
    assert "2 or more full sentences" in short, short
    assert "8 or more" not in short
    assert generate.sentence_target("cm", 25) == 2
    # te keeps the indic floor — its bins are all >= 100, so the floor never bound
    # there, which is why the 56 te rows generated before the fix stay valid.
    assert generate.sentence_target("te", 100) == 8
    assert "8 or more full sentences" in generate.build_prompt("ఇది ఒక వాక్యం.", "te", 100)


def test_stratified_cap_preserves_the_bin_mix_and_is_deterministic() -> None:
    prompts: list[dict] = []
    for bin_words, n in ((100, 200), (150, 1000), (200, 400)):   # 1600 total
        prompts.extend({"id": f"te-{bin_words}-{i:05d}", "n_words": bin_words} for i in range(n))
    ids = generate.stratified_cap(prompts, 400)                  # exactly a quarter of each bin
    assert len(ids) == len(set(ids)) == 400
    chosen = set(ids)
    got = {b: sum(1 for p in prompts if p["id"] in chosen and p["n_words"] == b)
           for b in (100, 150, 200)}
    assert got == {100: 50, 150: 250, 200: 100}, got
    assert generate.stratified_cap(prompts, 400) == ids          # deterministic, not salted
    assert generate.stratified_cap(prompts, 5000) == [p["id"] for p in prompts]


def test_compensated_words_inflates_the_ask_but_never_past_the_cap() -> None:
    # te: 150 words at ratio 0.62 -> ask ~242, which still fits 3600 tokens at 11.919/word.
    assert generate.compensated_words(150, 11.919, 0.62, cap=3600) == 242
    # 300 words would ask 484, but the cap only holds 302 -> clamped, not truncated.
    assert generate.compensated_words(300, 11.919, 0.62, cap=3600) == 302
    # hi: 150 words at 0.55 -> ask 273, well inside the 749 words 3600 tokens holds.
    assert generate.compensated_words(150, 4.804, 0.55, cap=3600) == 273
    # en at ratio 1.0 is unchanged, and no bucket is ever asked for fewer words:
    # the ratio only inflates, so a value above 1 is a no-op, not a shrink.
    assert generate.compensated_words(150, 1.339, 1.0, cap=3600) == 150
    assert generate.compensated_words(150, 1.339, 2.0, cap=3600) == 150


def test_load_fertility_prefers_the_generator_tokenizer_then_falls_back() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fertility.csv"
        path.write_text("model,bucket,tokens_per_word,status\n"
                        "Qwen/Qwen2.5-0.5B,en,1.3,ok\nQwen/Qwen2.5-0.5B,te,11.9,ok\n"
                        "google/gemma-2-9b-it,en,1.2,ok\ngoogle/gemma-2-9b-it,te,4.9,ok\n"
                        "mistralai/Mistral-7B-Instruct-v0.3,-,,error:ValueError\n", encoding="utf-8")
        table, used = generate.load_fertility(path, "google/gemma-2-9b-it", "Qwen/Qwen2.5-0.5B", ["en", "te"])
        assert used == "google/gemma-2-9b-it" and table["te"] == 4.9
        table, used = generate.load_fertility(path, "mistralai/Mistral-7B-Instruct-v0.3",
                                              "Qwen/Qwen2.5-0.5B", ["en", "te"])
        assert used == "Qwen/Qwen2.5-0.5B" and table["te"] == 11.9


class FakeOllama:
    """Stands in for OllamaClient: records options, fails requests whose prompt names a failing id."""

    def __init__(self, fail: tuple[str, ...] = ()) -> None:
        self.calls: list[dict] = []
        self.fail = fail

    def generate(self, model: str, prompt: str, options: dict, keep_alive: str) -> dict:
        self.calls.append(options)
        if any(prompt.endswith(f) for f in self.fail):
            raise TimeoutError("slow")
        return {"response": " machine text ", "eval_count": 10, "eval_duration": 5e8,
                "prompt_eval_count": 20, "done_reason": "length" if options["num_predict"] < 100 else "stop"}


def _items(n: int) -> list[dict]:
    return [{"id": f"qwen7b__p{i}", "prompt_id": f"p{i}", "bucket": "en", "n_words": 50,
             "prompt": f"prompt p{i}", "domain": "d", "human_length_words": 48,
             "num_predict": 64 if i == 0 else 200} for i in range(n)]


def _worker(client: FakeOllama):
    return generate.make_ollama_worker(
        client, "qwen7b", "seen", "qwen2.5:7b-instruct", {"quantization": "Q4_K_M", "digest": "abc"},
        temperature=0.8, top_p=0.95, num_ctx=3072, keep_alive="1m", seed=42, retry_wait_s=0)


def test_ollama_worker_sends_the_passage_budget_and_records_truncation() -> None:
    client = FakeOllama()
    process = _worker(client)
    rows = [process(item) for item in _items(2)]
    assert [c["num_predict"] for c in client.calls] == [64, 200]
    assert client.calls[0]["seed"] != client.calls[1]["seed"]              # seeded per passage
    assert rows[0]["truncated"] is True and rows[1]["truncated"] is False
    row = rows[1]
    assert row["text"] == "machine text" and row["generator"] == "qwen7b" and row["backend"] == "ollama"
    assert row["generator_model"] == "qwen2.5:7b-instruct" and row["decoding"]["quantization"] == "Q4_K_M"
    assert row["prompt_id"] == "p1" and row["gen_tokens"] == 10
    assert process.meter.tokens == 20


def test_ollama_run_writes_each_row_and_leaves_a_failed_request_for_resume() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "qwen7b.jsonl"
        client = FakeOllama(fail=("p3",))
        report = resumable.run_concurrent(_items(8), id_of=lambda i: i["id"], process_item=_worker(client),
                                          out_path=out, workers=4)
        assert report.processed == 7 and report.failed == 1 and not report.complete
        assert len(client.calls) == 9                    # the failing request was retried once
        written = [json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()]
        assert len(written) == 7 and "qwen7b__p3" not in written
        again = resumable.run_concurrent(_items(8), id_of=lambda i: i["id"], process_item=_worker(FakeOllama()),
                                         out_path=out, workers=4)
        assert again.already_done == 7 and again.processed == 1 and again.complete


def test_refuses_to_append_to_a_file_from_another_backend() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "qwen7b.jsonl"
        generate.check_output_compatible(out, "ollama", "qwen2.5:7b-instruct")   # no file yet: fine
        out.write_text(json.dumps({"id": "x", "backend": "hf",
                                   "generator_model": "Qwen/Qwen2.5-7B-Instruct"}) + "\n", encoding="utf-8")
        generate.check_output_compatible(out, "hf", "Qwen/Qwen2.5-7B-Instruct")  # same writer: fine
        try:
            generate.check_output_compatible(out, "ollama", "qwen2.5:7b-instruct")
        except SystemExit:
            pass
        else:
            raise AssertionError("mixing backends in one file must be refused")


def _gate_rows(bucket: str, pairs: list[tuple[int, int]], truncated: int = 0) -> list[dict]:
    """Output rows from (human_words, machine_words) pairs; the first `truncated` hit the cap."""
    return [{"id": f"qwen7b__{bucket}-{n:03d}", "language": bucket, "length_words": machine,
             "human_length_words": human, "truncated": n < truncated}
            for n, (human, machine) in enumerate(pairs)]


def _fake_phase(out: Path, done: set[str], seen: list[int], human: int, machine: int):
    """A run_phase that writes one row per item at a fixed machine/human length ratio."""
    def run_phase(items, label="passages"):
        seen.append(len(items))
        with open(out, "a", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps({"id": item["id"], "language": item["bucket"],
                                     "length_words": machine, "human_length_words": human,
                                     "truncated": False}) + "\n")
        done.update(item["id"] for item in items)
        return resumable.RunReport(total=len(items), processed=len(items))
    return run_phase


def test_gate_uses_the_median_and_catches_cm_over_production() -> None:
    # The 470 discarded cm rows: short human passages answered at ~4x while the
    # long ones behaved (decisions.md 2026-09-18). The median is the statistic
    # because a length-weighted ratio of totals dilutes exactly those short rows.
    stats = generate.gate_stats("cm", _gate_rows("cm", [(21, 84)] * 12 + [(150, 190)] * 12))
    assert stats.n == 24 and stats.ratio > 1.25
    reasons = generate.gate_reasons(stats)
    assert len(reasons) == 1 and "over-production" in reasons[0]
    te = generate.gate_stats("te", _gate_rows("te", [(150, 141)] * 24))   # the real te, 0.94
    assert generate.gate_reasons(te) == []


def test_gate_tests_both_bounds_and_the_cap_independently() -> None:
    # Under-production: hi on the plain template at 0.55, before compensation.
    under = generate.gate_stats("hi", _gate_rows("hi", [(180, 99)] * 24))
    assert len(generate.gate_reasons(under)) == 1 and "under-production" in generate.gate_reasons(under)[0]
    # A 1.72 passed the watcher, which only tested the lower bound. It must not pass here.
    over = generate.gate_stats("cm", _gate_rows("cm", [(50, 86)] * 24))
    assert generate.gate_reasons(over), "the upper bound is what the watcher was missing"
    # Ratio 1.0 but half the rows truncated: the cap is a separate failure.
    capped = generate.gate_stats("te", _gate_rows("te", [(150, 150)] * 24, truncated=12))
    reasons = generate.gate_reasons(capped)
    assert len(reasons) == 1 and "token cap" in reasons[0]


def test_a_failing_bucket_aborts_before_the_bulk_run() -> None:
    # The cm regression: a bad prompt regime must cost 24 rows, not 470.
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "qwen7b.jsonl"
        prompts = [{"id": f"qwen7b__cm-{i:03d}", "bucket": "cm"} for i in range(470)]
        done: set[str] = set()
        seen: list[int] = []
        try:
            generate.run_with_gates(["cm"], prompts, done_ids=done, out_path=out,
                                    run_phase=_fake_phase(out, done, seen, human=21, machine=84))
        except SystemExit as exc:
            assert "cm" in str(exc) and "4.00" in str(exc), exc
        else:
            raise AssertionError("a 4.0x bucket must abort the run")
        assert seen == [24], seen          # the probe ran; the other 446 never did


def test_a_passing_bucket_probes_once_then_runs_the_rest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "qwen7b.jsonl"
        prompts = [{"id": f"qwen7b__te-{i:03d}", "bucket": "te"} for i in range(100)]
        done: set[str] = set()
        seen: list[int] = []
        report = generate.run_with_gates(["te"], prompts, done_ids=done, out_path=out,
                                         run_phase=_fake_phase(out, done, seen, human=150, machine=141))
        # probe, then the full list — the real runners skip the probe rows via done_ids
        assert seen == [24, 100] and report.total == 100


def test_rows_already_on_disk_count_towards_the_probe() -> None:
    # te resumed with 56 rows from an earlier sitting: judge those, generate no probe.
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "qwen7b.jsonl"
        prompts = [{"id": f"qwen7b__te-{i:03d}", "bucket": "te"} for i in range(100)]
        done = {p["id"] for p in prompts[:56]}
        with open(out, "w", encoding="utf-8") as fh:
            for row in _gate_rows("te", [(150, 141)] * 56):     # ids match prompts[:56]
                fh.write(json.dumps(row) + "\n")
        seen: list[int] = []
        generate.run_with_gates(["te"], prompts, done_ids=done, out_path=out,
                                run_phase=_fake_phase(out, done, seen, human=150, machine=141))
        assert seen == [100], seen         # the bulk phase only


def test_a_bucket_whose_rows_are_bad_keeps_failing_on_resume() -> None:
    # Deliberate: the probe rows stay on disk, so the next run reads them back and
    # fails the same gate until they are removed. No --skip-gate flag exists.
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "qwen7b.jsonl"
        prompts = [{"id": f"qwen7b__cm-{i:03d}", "bucket": "cm"} for i in range(470)]
        done: set[str] = set()
        for expected in ([24], []):        # first run generates the probe, second generates nothing
            seen: list[int] = []
            try:
                generate.run_with_gates(["cm"], prompts, done_ids=done, out_path=out,
                                        run_phase=_fake_phase(out, done, seen, human=21, machine=84))
            except SystemExit:
                pass
            else:
                raise AssertionError("must abort every time")
            assert seen == expected, (seen, expected)


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)
