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
    for expected in ("Hinglish", "Roman script", "about 50 words", "No emoji",
                     "mix Hindi and English words within every sentence", "like students texting"):
        assert expected in prompt, f"cm prompt must say '{expected}'"
    assert "in English" not in prompt
    assert "at least" not in prompt        # "at least" drove cm long (decisions.md 2026-09-20)


def test_cm_asks_from_the_human_length_and_inflates_only_the_long_bin() -> None:
    rule = generate.ASK_FROM_HUMAN["cm"]
    def ask(human, n_words):
        return generate.ask_words_for({"n_words": n_words, "human_length_words": human},
                                      1.672, 1.0, 3600, rule)
    assert ask(20, 25) == 20              # the human length, not the bin: 25 is 1.25x of 20
    assert ask(39, 50) == 39
    assert ask(40, 50) == 53              # 40 / 0.75: long passages under-produce at ask = human
    assert ask(69, 50) == 92 and ask(194, 200) == 259      # the 2026-09-21 probe's asks
    # Without a rule the ask is the compensated bin, exactly as before.
    assert generate.ask_words_for({"n_words": 100, "human_length_words": 97}, 4.804, 0.55, 3600) == \
        generate.compensated_words(100, 4.804, 0.55, 3600) == 182
    cfg = load_config(DATA_CONFIG)["machine_corpus"]["budget"]["ask_from_human"]
    assert set(cfg) == {"cm"} and cfg["cm"] == rule


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


def _cm_shaped_prompts() -> list[dict]:
    """cm as it finished: 288 long passages, then 413 short ones (59 % of the bucket).

    Ordered long-first so the head of the list is exactly the blind spot — the
    real cm's first 24 rows had a median human length of 82 against the bucket's
    26 (decisions.md 2026-09-20).
    """
    return ([{"id": f"qwen7b__cm-L{i:03d}", "bucket": "cm", "n_words": 150,
              "human_length_words": 150} for i in range(288)]
            + [{"id": f"qwen7b__cm-S{i:03d}", "bucket": "cm", "n_words": 25,
                "human_length_words": 20} for i in range(413)])


def _binned_phase(out: Path, done: set[str], seen: list[int]):
    """A run_phase with cm's real defect shape: short passages at 1.75, the rest at 1.0."""
    def run_phase(items, label="passages"):
        seen.append(len(items))
        with open(out, "a", encoding="utf-8") as fh:
            for item in items:
                human = item["human_length_words"]
                machine = round(human * (1.75 if human <= 40 else 1.0))
                fh.write(json.dumps({"id": item["id"], "language": item["bucket"],
                                     "length_words": machine, "human_length_words": human,
                                     "truncated": False}) + "\n")
        done.update(item["id"] for item in items)
        return resumable.RunReport(total=len(items), processed=len(items))
    return run_phase


def test_probe_is_stratified_across_length_bins() -> None:
    prompts = _cm_shaped_prompts()
    # The head of the list holds none of the short passages: the old blind spot.
    assert sum(1 for p in prompts[:24] if p["n_words"] == 25) == 0
    probe = generate.stratified_probe(prompts, 24)
    assert len(probe) == 24
    short = sum(1 for p in probe if p["n_words"] == 25)
    assert 13 <= short <= 15, f"413/701 of 24 is 14.1, got {short}"
    assert generate.stratified_probe(prompts, 24) == probe          # deterministic
    # Without bin information the previous head order is kept, so bare prompts still work.
    bare = [{"id": f"x-{i}", "bucket": "cm"} for i in range(50)]
    assert generate.stratified_probe(bare, 24) == bare[:24]


def test_a_bin_local_defect_fails_the_gate() -> None:
    # cm at 701 rows: 0-40 was 413 rows at median 1.75 while every bin from 40
    # words up sat in band, so the bucket settled at 1.39 after passing at 1.02.
    # A stratified probe carries ~14 short rows and fails it at 24.
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "qwen7b.jsonl"
        prompts = _cm_shaped_prompts()
        done: set[str] = set()
        seen: list[int] = []
        try:
            generate.run_with_gates(["cm"], prompts, done_ids=done, out_path=out,
                                    run_phase=_binned_phase(out, done, seen))
        except SystemExit as exc:
            assert "cm" in str(exc) and "over-production" in str(exc), exc
        else:
            raise AssertionError("a bin-local defect must abort the run")
        assert seen == [24], seen          # the probe ran; the other 677 never did


def test_head_order_probe_would_have_missed_the_bin_local_defect() -> None:
    # The regression this fix exists for: judging the first 24 of the same bucket
    # passes it, which is what happened to cm on 2026-09-19.
    prompts = _cm_shaped_prompts()
    rows = [{"id": p["id"], "language": "cm", "human_length_words": p["human_length_words"],
             "length_words": round(p["human_length_words"] * (1.75 if p["human_length_words"] <= 40 else 1.0)),
             "truncated": False} for p in prompts[:24]]
    assert generate.gate_reasons(generate.gate_stats("cm", rows)) == [], \
        "head-order probing passes the bucket — the blind spot being fixed"
    stratified = generate.stratified_probe(prompts, 24)
    rows = [{"id": p["id"], "language": "cm", "human_length_words": p["human_length_words"],
             "length_words": round(p["human_length_words"] * (1.75 if p["human_length_words"] <= 40 else 1.0)),
             "truncated": False} for p in stratified]
    assert generate.gate_reasons(generate.gate_stats("cm", rows)), "stratified probing must catch it"


HINGLISH_HUMAN = "yeh exam mera sabse tough tha, par main nahi ruka aur kal bhi padhai karunga"
ENGLISH_MACHINE = "The exam was the toughest one this term, but I kept going and will study again tomorrow"
HINGLISH_MACHINE = "exam bahut tough tha, par maine nahi chhoda aur kal phir se padhai karni hai"


def _cm_codemix_prompts(n: int = 60) -> list[dict]:
    return [{"id": f"qwen7b__cm-{i:03d}", "bucket": "cm", "n_words": 25, "human_length_words": 15,
             "text": HINGLISH_HUMAN} for i in range(n)]


def _text_phase(out: Path, done: set[str], seen: list[int], text: str, emoji_rows: int = 0):
    """A run_phase writing ``text`` at exactly human length: in band on length, always.

    The first ``emoji_rows`` rows of each phase end in an emoji.
    """
    def run_phase(items, label="passages"):
        seen.append(len(items))
        with open(out, "a", encoding="utf-8") as fh:
            for n, item in enumerate(items):
                row_text = text + (" \U0001F602" if n < emoji_rows else "")
                fh.write(json.dumps({"id": item["id"], "language": item["bucket"], "text": row_text,
                                     "length_words": item["human_length_words"],
                                     "human_length_words": item["human_length_words"],
                                     "truncated": False}) + "\n")
        done.update(item["id"] for item in items)
        return resumable.RunReport(total=len(items), processed=len(items))
    return run_phase


def test_codemix_gate_fails_english_cm_rows_and_passes_hinglish() -> None:
    # The 2026-09-21 defect: cm rows in band on length, English in all but name.
    prompts = _cm_codemix_prompts(24)
    human = {p["id"]: p["text"] for p in prompts}
    def rows(text):
        return [{"id": p["id"], "language": "cm", "text": text, "length_words": 15,
                 "human_length_words": 15, "truncated": False} for p in prompts]
    english = generate.gate_stats("cm", rows(ENGLISH_MACHINE), human_texts=human)
    reasons = generate.gate_reasons(english)
    assert len(reasons) == 1 and reasons[0].startswith("code-mix"), reasons   # length alone passes
    assert english.codemix == 0.0 and english.human_codemix > 0.3, english
    assert generate.gate_reasons(generate.gate_stats("cm", rows(HINGLISH_MACHINE), human_texts=human)) == []
    # Without human text there is nothing to be a fraction of: length only, as before.
    assert generate.gate_reasons(generate.gate_stats("cm", rows(ENGLISH_MACHINE))) == []
    # A human passage with no Hindi in it gives no floor, so it cannot fail the check.
    no_hindi = {k: ENGLISH_MACHINE for k in human}
    assert generate.gate_reasons(generate.gate_stats("cm", rows(ENGLISH_MACHINE), human_texts=no_hindi)) == []


def test_run_with_gates_aborts_cm_on_codemix_with_length_in_band() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "qwen7b.jsonl"
        done: set[str] = set()
        seen: list[int] = []
        try:
            generate.run_with_gates(["cm"], _cm_codemix_prompts(), done_ids=done, out_path=out,
                                    run_phase=_text_phase(out, done, seen, ENGLISH_MACHINE))
        except SystemExit as exc:
            assert "code-mix" in str(exc) and "over-production" not in str(exc), exc
        else:
            raise AssertionError("English cm rows must abort the run at the probe")
        assert seen == [24], seen          # the probe ran; the other 36 never did


def test_codemix_check_applies_only_to_listed_buckets() -> None:
    # The same English rows pass when cm is not code-mix checked, and Hinglish
    # rows pass when it is: the check keys on the bucket list, not on the text.
    for buckets, text in (((), ENGLISH_MACHINE), (("cm",), HINGLISH_MACHINE)):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "qwen7b.jsonl"
            done: set[str] = set()
            seen: list[int] = []
            generate.run_with_gates(["cm"], _cm_codemix_prompts(), done_ids=done, out_path=out,
                                    run_phase=_text_phase(out, done, seen, text),
                                    codemix_buckets=buckets)
            assert seen == [24, 60], (buckets, seen)    # probe, then the bulk run
    cfg = load_config(DATA_CONFIG)["machine_corpus"]["gate"]
    assert cfg["codemix_buckets"] == ["cm"] and abs(cfg["codemix_min_ratio"] - 2 / 3) < 0.001


def test_has_emoji_catches_pictographs_but_not_devanagari_joiners() -> None:
    for text in ("mast tha \U0001F602", "exam done ✅", "love it ❤️", "⭐ topper",
                 "family \U0001F468‍\U0001F469‍\U0001F467"):
        assert generate.has_emoji(text), text
    # U+200D joins Devanagari conjuncts: 118 of 2,200 human hi passages carry one.
    for text in (HINGLISH_MACHINE, "क्‍ष है", "marks: 95/100, rank #3!", ""):
        assert not generate.has_emoji(text), text


def test_emoji_gate_fails_one_emoji_row_in_cm() -> None:
    # Human cm has emoji in 0 of 2,200 rows, so one row in the probe is enough to fail.
    prompts = _cm_codemix_prompts(24)
    human = {p["id"]: p["text"] for p in prompts}
    rows = [{"id": p["id"], "language": "cm", "length_words": 15, "human_length_words": 15,
             "truncated": False, "text": HINGLISH_MACHINE + (" \U0001F602" if i == 0 else "")}
            for i, p in enumerate(prompts)]
    stats = generate.gate_stats("cm", rows, human_texts=human, check_emoji=True)
    reasons = generate.gate_reasons(stats)
    assert stats.emoji_rows == 1 and len(reasons) == 1 and reasons[0].startswith("emoji"), reasons
    assert "1 with emoji" in stats.describe()
    # Not measured unless asked: the same rows pass on length and code-mix alone.
    unchecked = generate.gate_stats("cm", rows, human_texts=human)
    assert unchecked.emoji_rows is None and generate.gate_reasons(unchecked) == []


def test_run_with_gates_aborts_cm_on_emoji_and_the_check_keys_on_the_bucket_list() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "qwen7b.jsonl"
        done: set[str] = set()
        seen: list[int] = []
        try:
            generate.run_with_gates(["cm"], _cm_codemix_prompts(), done_ids=done, out_path=out,
                                    run_phase=_text_phase(out, done, seen, HINGLISH_MACHINE, emoji_rows=1))
        except SystemExit as exc:
            assert "emoji" in str(exc) and "code-mix" not in str(exc), exc
        else:
            raise AssertionError("one emoji row in the cm probe must abort the run")
        assert seen == [24], seen          # the probe ran; the other 36 never did
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "qwen7b.jsonl"
        done = set()
        seen = []
        generate.run_with_gates(["cm"], _cm_codemix_prompts(), done_ids=done, out_path=out,
                                run_phase=_text_phase(out, done, seen, HINGLISH_MACHINE, emoji_rows=1),
                                emoji_buckets=())
        assert seen == [24, 60], seen      # not emoji checked: probe, then the bulk run
    assert load_config(DATA_CONFIG)["machine_corpus"]["gate"]["emoji_buckets"] == ["cm"]


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
