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
    assert "in Hindi" in hi and "at least 200 words" in hi
    assert "do not stop early" in hi.lower()     # compliance framing (decisions.md 2026-09-18)


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


def test_compensated_words_inflates_the_ask_but_never_past_the_cap() -> None:
    # te: 150 words at ratio 0.62 -> ask ~242, which still fits 3600 tokens at 11.919/word.
    assert generate.compensated_words(150, 11.919, 0.62, cap=3600) == 242
    # 300 words would ask 484, but the cap only holds 302 -> clamped, not truncated.
    assert generate.compensated_words(300, 11.919, 0.62, cap=3600) == 302
    # en at ratio 1.0 is unchanged, and no bucket is ever asked for fewer words.
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
