"""Generate prompt-matched machine text for one generator (resumable CLI).

Usage::

    python -m src.data.generate --generator qwen7b --buckets en,hi,te,cm --max-minutes 110
        [--backend ollama|hf] [--model TAG_OR_HF_ID] [--fraction 1.0]
        [--config configs/data.yaml] [--limit N] [--seed 42]

``--generator`` is an alias from ``machine_corpus.generators`` in
configs/data.yaml: ``qwen7b``, ``gemma``, ``mistral`` (seen) and ``llama``,
``phi`` (held-out, test only).

Prompt matching
    Every machine passage is matched to one human passage: the prompt is that
    passage's **first sentence**, and the requested length is its **length
    bin**, so human and machine text share topic and length distribution by
    construction. Without that, a detector can separate the classes on topic or
    length alone and the reported AUROC means nothing
    (docs/project-context-master.md §6).

Assignment
    Each human passage gets exactly ONE seen generator, ``sha256(prompt_id) mod
    3``, so each seen generator covers ~1/3 of every bucket and the seen machine
    side is 1:1 with the human side. Each held-out generator takes a disjoint
    30 % slice from a second, independent hash. Python's ``hash()`` is salted
    per process and would reshuffle the assignment on every run, so it is not
    used.

Token budget
    ``num_predict = requested_words x fertility(bucket) x 1.3 + 64``, capped at
    2048, with fertility read from results/tokenizer_fertility.csv for the
    generator's own tokenizer (the Qwen2.5 row when that is missing). A flat
    words-x-4 budget cut every Hindi and Telugu passage short: Qwen needs 4.8
    tokens per Hindi word and 11.9 per Telugu word. Rows record ``done_reason``
    and ``truncated`` so cleaning can drop the ones the cap still cuts.

Backends
    ``ollama`` (default) runs a persistent pool of
    ``machine_corpus.ollama.parallel`` worker threads
    (:func:`src.utils.resumable.run_concurrent`). Each pulls the next prompt and
    posts it to ``/api/generate`` (``stream=false``) on its own, so a server
    slot is refilled the moment its request finishes rather than waiting for a
    batch's slowest member. Each row is written the moment it completes. The
    server only runs requests concurrently if it was started with as many
    slots. On Windows, set it once and restart the Ollama app::

        setx OLLAMA_NUM_PARALLEL 4      # then Quit Ollama from the tray and reopen it

    ``%LOCALAPPDATA%/Ollama/server.log`` prints ``OLLAMA_NUM_PARALLEL:4`` at
    startup when it took effect. Without it the server queues requests one at a
    time: same output, slower. The progress line (every 8 completed requests)
    shows the effective concurrency, so a queueing server is visible at once.

    ``hf`` is the fallback: transformers + bitsandbytes 4-bit via
    src/utils/modelload.py, in batches. One output file never mixes backends or
    models, since that would mix quantisations; the run refuses to append to a
    file written by a different one.

Output is ``data/raw/machine/<generator>.jsonl``, one row per assigned human
passage, carrying ``prompt_id`` back to it. Ids already present are skipped, so
re-running the same command resumes.

Serves docs/parts-plan.md Stage 3.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence

from src.data.schema import LABEL_MACHINE, LANGUAGE_BUCKETS
from src.utils.config import load_config
from src.utils.io import read_jsonl
from src.utils.resumable import DEFAULT_BATCH_SIZE, read_done_ids, run_concurrent, run_resumable

log = logging.getLogger("generate")

BACKENDS: tuple[str, ...] = ("ollama", "hf")

#: Length bins (word counts) the prompt asks for. The human passage's own length
#: picks the bin, so the machine length distribution tracks the human one.
LENGTH_BINS: tuple[int, ...] = (25, 50, 100, 150, 200, 250, 300)

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "cm": "Hinglish",
}

#: Per-bucket instruction. cm needs its own: asking for "Hinglish" alone tends to
#: produce Devanagari Hindi or formal prose, neither of which matches the human
#: cm bucket (Romanised, informal).
PROMPT_TEMPLATES: dict[str, str] = {
    "default": ('{first_sentence}\n\nWrite about {n_words} words in {language} '
                'on this topic, continuing naturally.'),
    "cm": ('{first_sentence}\n\nWrite about {n_words} words on this topic in casual '
           'Hindi-English Hinglish, Roman script, the way students text. '
           'Continue naturally.'),
}

_SENT_END_RE = re.compile(r"(?<=[.!?।॥])\s+")
_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def generator_slug(generator: str) -> str:
    """Filesystem-safe name: ``Qwen/Qwen2.5-7B-Instruct`` -> ``qwen-qwen2-5-7b-instruct``."""
    return _SLUG_RE.sub("-", generator).strip("-").lower()


def first_sentence(text: str, max_words: int = 40) -> str:
    """Leading sentence of a passage, truncated to ``max_words``.

    This is the topic seed. Truncation matters: a very long first sentence would
    hand the model most of the human passage to copy rather than a topic.
    """
    parts = _SENT_END_RE.split(text.strip(), maxsplit=1)
    sentence = (parts[0] if parts else text).strip()
    words = sentence.split()
    return " ".join(words[:max_words]) if len(words) > max_words else sentence


def length_bin(n_words: int, bins: Sequence[int] = LENGTH_BINS) -> int:
    """Nearest length bin to ``n_words``."""
    return min(bins, key=lambda b: abs(b - n_words))


def build_prompt(text: str, bucket: str, n_words: int) -> str:
    """Prompt-matched instruction for one human passage."""
    template = PROMPT_TEMPLATES.get(bucket, PROMPT_TEMPLATES["default"])
    return template.format(
        first_sentence=first_sentence(text),
        n_words=n_words,
        language=LANGUAGE_NAMES.get(bucket, bucket),
    )


# --- assignment -------------------------------------------------------------

def _hash64(key: str) -> int:
    """Stable 64-bit hash of ``key`` (sha256), identical across runs and machines."""
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def seen_generator_for(prompt_id: str, seen: Sequence[str]) -> str:
    """The one seen generator for a human passage: ``sha256(prompt_id) mod len(seen)``."""
    return seen[_hash64(prompt_id) % len(seen)]


def heldout_generator_for(prompt_id: str, heldout: Sequence[str], share: float) -> str | None:
    """The held-out generator for a passage, or None: disjoint ``share`` slices of a second hash.

    The key is salted, so this hash is independent of the seen assignment and
    each held-out slice cuts evenly across all three seen generators.
    """
    if share * len(heldout) > 1:
        raise ValueError(f"{len(heldout)} held-out generators x {share} share exceeds 100 %")
    slot = int((_hash64("heldout:" + prompt_id) / 2**64) // share)
    return heldout[slot] if slot < len(heldout) else None


def generator_roster(generators: dict[str, dict[str, Any]]) -> tuple[list[str], list[str]]:
    """``(seen, heldout)`` aliases, each ordered by its ``slot``."""
    def by_role(role: str) -> list[str]:
        members = sorted((spec["slot"], alias) for alias, spec in generators.items()
                         if spec["role"] == role)
        slots = [slot for slot, _ in members]
        if slots != list(range(len(slots))):
            raise ValueError(f"{role} generator slots must be 0..{len(slots) - 1}, got {slots}")
        return [alias for _, alias in members]

    return by_role("seen"), by_role("heldout")


def make_assignment(generator: str, generators: dict[str, dict[str, Any]],
                    heldout_share: float) -> Callable[[str], bool]:
    """Predicate: is this human passage (by id) assigned to ``generator``?"""
    seen, heldout = generator_roster(generators)
    if generator in seen:
        return lambda prompt_id: seen_generator_for(prompt_id, seen) == generator
    if generator in heldout:
        return lambda prompt_id: heldout_generator_for(prompt_id, heldout, heldout_share) == generator
    raise KeyError(f"generator '{generator}' has no seen/heldout role")


# --- token budget -----------------------------------------------------------

def load_fertility(csv_path: str | Path, tokenizer: str | None, fallback: str,
                   buckets: Sequence[str] = LANGUAGE_BUCKETS) -> tuple[dict[str, float], str]:
    """Tokens per word by bucket for ``tokenizer``, else ``fallback``.

    Returns ``(table, model_used)``. A model counts only if it has an ``ok`` row
    for every requested bucket.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run: python scripts/tokenizer_fertility.py")
    by_model: dict[str, dict[str, float]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "ok" and row.get("tokens_per_word"):
                by_model.setdefault(row["model"], {})[row["bucket"]] = float(row["tokens_per_word"])
    for model in (tokenizer, fallback):
        if model and all(b in by_model.get(model, {}) for b in buckets):
            return by_model[model], model
    raise ValueError(f"{path} has no complete fertility rows for {tokenizer!r} or fallback {fallback!r}")


def token_budget(n_words: int, fertility: float, overshoot: float = 1.3,
                 pad: int = 64, cap: int = 2048) -> int:
    """Generation budget in tokens: ``n_words x fertility x overshoot + pad``, capped."""
    return min(cap, int(n_words * fertility * overshoot + pad))


# --- prompts and output -----------------------------------------------------

def load_prompts(human_dir: Path, buckets: Sequence[str], fraction: float,
                 seed: int, generator: str, limit: int | None = None,
                 keep: Callable[[str], bool] | None = None) -> list[dict[str, Any]]:
    """One prompt per human passage assigned to ``generator``, across the requested buckets.

    ``fraction`` samples deterministically from each bucket before assignment:
    the same fraction and seed always select the same passages, so a
    partly-generated corpus can be extended without reshuffling what is done.
    """
    slug = generator_slug(generator)
    prompts: list[dict[str, Any]] = []
    for bucket in buckets:
        path = human_dir / f"{bucket}.jsonl"
        if not path.exists():
            log.warning("no human corpus for bucket '%s' at %s — skipping", bucket, path)
            continue
        rows = read_jsonl(path, skip_bad_lines=True)
        rows.sort(key=lambda r: r["id"])  # stable order regardless of file order
        if fraction < 1.0:
            keep_n = max(1, int(round(len(rows) * fraction)))
            rows = random.Random(seed).sample(rows, keep_n)
            rows.sort(key=lambda r: r["id"])
        n_human = len(rows)
        if keep is not None:
            rows = [row for row in rows if keep(row["id"])]
        log.info("bucket %s: %d human passages, %d assigned to %s", bucket, n_human, len(rows), generator)
        for row in rows:
            n_words = length_bin(int(row["length_words"]))
            prompts.append({
                "id": f"{slug}__{row['id']}",
                "prompt_id": row["id"],
                "bucket": bucket,
                "n_words": n_words,
                "prompt": build_prompt(row["text"], bucket, n_words),
                "domain": row.get("domain"),
                "human_length_words": int(row["length_words"]),
            })
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def check_output_compatible(out_path: Path, backend: str, model: str) -> None:
    """Refuse to append to a file that another backend or model wrote (it would mix quantisations)."""
    if not out_path.exists():
        return
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            found = (row.get("backend"), row.get("generator_model"))
            if found != (backend, model):
                raise SystemExit(
                    f"{out_path} was written by backend={found[0]} model={found[1]}, but this run is "
                    f"backend={backend} model={model}. One file must not mix quantisations: move the "
                    f"file aside, or rerun with the original backend and model.")
            return


def machine_row(item: dict[str, Any], text: str, *, generator: str, role: str, model: str,
                backend: str, gen_tokens: int, done_reason: str | None,
                decoding: dict[str, Any]) -> dict[str, Any]:
    """One output row in the corpus schema, plus generation provenance."""
    return {
        "id": item["id"],
        "text": text,
        "label": LABEL_MACHINE,
        "language": item["bucket"],
        "code_mix_ratio": 0.0,          # measured in Part 13, after cleaning
        "generator": generator,
        "generator_role": role,
        "generator_model": model,
        "backend": backend,
        "domain": item["domain"],
        "length_words": len(text.split()),
        "attack_type": "clean",
        "writer_L1_band": None,
        "split": None,
        "prompt_id": item["prompt_id"],
        "prompt": item["prompt"],
        "requested_words": item["n_words"],
        "human_length_words": item["human_length_words"],
        "num_predict": item["num_predict"],
        "gen_tokens": gen_tokens,
        "done_reason": done_reason,
        "truncated": done_reason == "length",
        "decoding": decoding,
    }


class ThroughputMeter:
    """Thread-safe token counter that logs tokens/sec every ``log_every`` completed requests.

    With ``concurrent=True`` it also logs per-stream speed and the effective
    concurrency: summed per-request decode time over wall time in the window.
    About 4 means every server slot stayed busy; about 1 means the server is
    queueing requests.
    """

    def __init__(self, log_every: int = DEFAULT_BATCH_SIZE, concurrent: bool = True) -> None:
        self.log_every = max(1, log_every)
        self.concurrent = concurrent
        self.started = time.monotonic()
        self.tokens = 0
        self._lock = threading.Lock()
        self._reset_window(self.started)

    def _reset_window(self, now: float) -> None:
        self._window_start = now
        self._window_tokens = 0
        self._window_decode = 0.0
        self._window_requests = 0
        self._window_truncated = 0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def record(self, tokens: int, decode_seconds: float, truncated: int, requests: int = 1) -> None:
        with self._lock:
            self.tokens += tokens
            self._window_tokens += tokens
            self._window_decode += decode_seconds
            self._window_requests += requests
            self._window_truncated += truncated
            if self._window_requests < self.log_every:
                return
            now = time.monotonic()
            wall = now - self._window_start
            extra = ""
            if self.concurrent and self._window_decode and wall:
                extra = (f", per-stream {self._window_tokens / self._window_decode:.1f} tok/s, "
                         f"concurrency x{self._window_decode / wall:.1f}")
            log.info("%d requests: %d tokens in %.1fs = %.1f tok/s%s, %d hit num_predict (run %.1f tok/s)",
                     self._window_requests, self._window_tokens, wall,
                     self._window_tokens / wall if wall else 0.0, extra, self._window_truncated,
                     self.tokens / max(now - self.started, 1e-6))
            self._reset_window(now)


# --- ollama backend ---------------------------------------------------------

class OllamaClient:
    """Minimal Ollama HTTP client on the standard library (no new dependency)."""

    def __init__(self, host: str, timeout_s: float = 900) -> None:
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.host + path, data=data,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"ollama {path} HTTP {exc.code}: {detail}") from exc

    def version(self) -> str:
        return self._request("/api/version")["version"]

    def show(self, model: str) -> dict[str, Any]:
        return self._request("/api/show", {"model": model})

    def tags(self) -> list[dict[str, Any]]:
        return self._request("/api/tags").get("models", [])

    def load(self, model: str, num_ctx: int, keep_alive: str) -> None:
        """Load ``model`` with the run's context size; an empty prompt loads without generating."""
        self._request("/api/generate", {"model": model, "prompt": "", "stream": False,
                                        "options": {"num_ctx": num_ctx}, "keep_alive": keep_alive})

    def generate(self, model: str, prompt: str, options: dict[str, Any], keep_alive: str) -> dict[str, Any]:
        return self._request("/api/generate", {"model": model, "prompt": prompt, "stream": False,
                                               "options": options, "keep_alive": keep_alive})


def describe_ollama_model(client: OllamaClient, model: str) -> dict[str, Any]:
    """Quantisation and digest of a pulled model; exits with a pull hint if it is missing."""
    try:
        version = client.version()
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"no Ollama server at {client.host} ({exc}). Start the Ollama app.") from exc
    try:
        details = client.show(model).get("details") or {}
    except RuntimeError as exc:
        raise SystemExit(f"{exc}\nmodel not available - run: ollama pull {model}") from exc
    digest = next((m.get("digest", "") for m in client.tags()
                   if model in (m.get("name"), m.get("model"))), "")
    return {"backend": "ollama", "server_version": version, "quantization": details.get("quantization_level"),
            "parameter_size": details.get("parameter_size"), "digest": digest[:12]}


def make_ollama_worker(client: OllamaClient, generator: str, role: str, model: str,
                       info: dict[str, Any], *, temperature: float, top_p: float,
                       num_ctx: int, keep_alive: str, seed: int, retries: int = 1,
                       retry_wait_s: float = 2.0, log_every: int = DEFAULT_BATCH_SIZE,
                       ) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a thread-safe ``process_item`` that generates one passage with one Ollama request.

    Built for :func:`src.utils.resumable.run_concurrent`, whose worker threads
    each call it independently. A request that still fails after ``retries``
    raises; the runner logs it and leaves the passage for the next resume.
    """
    meter = ThroughputMeter(log_every, concurrent=True)
    decoding = {"temperature": temperature, "top_p": top_p, "num_ctx": num_ctx,
                "quantization": info["quantization"], "digest": info["digest"]}
    warned = threading.Event()

    def process_item(item: dict[str, Any]) -> dict[str, Any]:
        item_seed = _hash64(f"{seed}:{item['id']}") % 2**31
        options = {"num_predict": item["num_predict"], "num_ctx": num_ctx,
                   "temperature": temperature, "top_p": top_p, "seed": item_seed}
        for attempt in range(retries + 1):
            try:
                response = client.generate(model, item["prompt"], options, keep_alive)
                break
            except Exception as exc:  # noqa: BLE001 - timeouts, dropped connections, server restarts
                if attempt == retries:
                    raise
                log.warning("%s: %s — retrying", item["id"], exc)
                time.sleep(retry_wait_s)
        prompt_tokens = int(response.get("prompt_eval_count") or 0)
        if prompt_tokens + item["num_predict"] > num_ctx and not warned.is_set():
            warned.set()
            log.warning("%s: prompt %d + num_predict %d exceeds num_ctx %d — raise machine_corpus.ollama.num_ctx",
                        item["id"], prompt_tokens, item["num_predict"], num_ctx)
        row = machine_row(item, (response.get("response") or "").strip(), generator=generator, role=role,
                          model=model, backend="ollama", gen_tokens=int(response.get("eval_count") or 0),
                          done_reason=response.get("done_reason"), decoding={**decoding, "seed": item_seed})
        meter.record(row["gen_tokens"], int(response.get("eval_duration") or 0) / 1e9, int(row["truncated"]))
        return row

    process_item.meter = meter  # type: ignore[attr-defined]
    return process_item


# --- hf backend (fallback) --------------------------------------------------

def make_hf_processor(model, tokenizer, generator: str, role: str, model_id: str, info: dict[str, Any], *,
                      temperature: float, top_p: float, max_prompt_tokens: int = 1024):
    """Return a ``process_batch`` that generates a batch locally with transformers."""
    import torch

    meter = ThroughputMeter(log_every=1, concurrent=False)   # one log line per batch

    def process_batch(batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        started = time.monotonic()
        texts = [item["prompt"] for item in batch]
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            # Instruct models need their chat template or they continue the
            # instruction text instead of answering it.
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True
                )
                for t in texts
            ]
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                            max_length=max_prompt_tokens, add_special_tokens=not chat_template)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        # A batch runs until its longest request is done, so budget by the largest.
        max_new = max(item["num_predict"] for item in batch)
        with torch.no_grad():
            output = model.generate(
                **encoded, max_new_tokens=max_new, do_sample=True,
                temperature=temperature, top_p=top_p,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        generated = output[:, encoded["input_ids"].shape[1]:]
        completions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        decoding = {"temperature": temperature, "top_p": top_p, "max_new_tokens": max_new,
                    "quantization": info["dtype"]}

        rows: list[dict[str, Any]] = []
        for item, sequence, completion in zip(batch, generated, completions):
            n_tokens = int((sequence != tokenizer.pad_token_id).sum())
            rows.append(machine_row(item, completion.strip(), generator=generator, role=role, model=model_id,
                                    backend="hf", gen_tokens=n_tokens,
                                    done_reason="length" if n_tokens >= max_new else "stop",
                                    decoding=decoding))
        meter.record(sum(r["gen_tokens"] for r in rows), time.monotonic() - started,
                     sum(r["truncated"] for r in rows), requests=len(rows))
        return rows

    process_batch.meter = meter  # type: ignore[attr-defined]
    return process_batch


# --- CLI --------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", required=True,
                        help="alias from machine_corpus.generators: qwen7b, gemma, mistral, llama, phi")
    parser.add_argument("--backend", choices=BACKENDS, default=None, help="default: machine_corpus.backend")
    parser.add_argument("--model", default=None, help="override the alias's Ollama tag / HF id (fallbacks)")
    parser.add_argument("--buckets", default="en,hi,te,cm", help="comma-separated")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="fraction of human passages considered, before assignment")
    parser.add_argument("--max-minutes", type=float, default=0, help="stop cleanly after this long (0 = no limit)")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out-dir", default=None, help="default: machine_corpus.output_dir")
    parser.add_argument("--human-dir", default=None, help="default: human_corpus.output_dir")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="hf backend: prompts per batch")
    parser.add_argument("--parallel", type=int, default=None,
                        help="ollama backend: worker threads (default: machine_corpus.ollama.parallel)")
    parser.add_argument("--host", default=None, help="default: machine_corpus.ollama.host")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap prompts (smoke tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="hf backend: cuda | cpu (default: auto)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    from src.utils.logging import configure_logging

    configure_logging(args.log_level)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    config = load_config(args.config)
    mc = config.get("machine_corpus") or {}
    generators = mc.get("generators") or {}
    if args.generator not in generators:
        parser.error(f"unknown generator '{args.generator}'; valid: {list(generators)}")
    spec = generators[args.generator]
    backend = args.backend or mc.get("backend", "ollama")
    model = args.model or spec[backend]
    decoding_cfg = mc.get("decoding") or {}
    temperature = args.temperature if args.temperature is not None else decoding_cfg.get("temperature", 0.8)
    top_p = args.top_p if args.top_p is not None else decoding_cfg.get("top_p", 0.95)

    human_dir = Path(args.human_dir or (config.get("human_corpus") or {}).get("output_dir", "data/raw/human"))
    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]
    unknown = [b for b in buckets if b not in LANGUAGE_BUCKETS]
    if unknown:
        parser.error(f"unknown bucket(s) {unknown}; valid: {list(LANGUAGE_BUCKETS)}")

    keep = make_assignment(args.generator, generators, float((mc.get("assignment") or {}).get("heldout_share", 0.30)))
    prompts = load_prompts(human_dir, buckets, args.fraction, args.seed, args.generator, args.limit, keep)
    if not prompts:
        print("no prompts — is the human corpus built?")
        return

    budget = mc.get("budget") or {}
    fertility, fertility_source = load_fertility(budget.get("fertility_csv", "results/tokenizer_fertility.csv"),
                                                 spec.get("tokenizer"),
                                                 budget.get("fallback_tokenizer", "Qwen/Qwen2.5-0.5B"), buckets)
    cap = int(budget.get("cap", 2048))
    for prompt in prompts:
        prompt["num_predict"] = token_budget(prompt["n_words"], fertility[prompt["bucket"]],
                                             float(budget.get("overshoot", 1.3)),
                                             int(budget.get("pad_tokens", 64)), cap)
    for bucket in buckets:
        budgets = [p["num_predict"] for p in prompts if p["bucket"] == bucket]
        if budgets:
            log.info("budget %s: %.3f tokens/word (%s) -> num_predict %d..%d, %d of %d at the %d cap",
                     bucket, fertility[bucket], fertility_source, min(budgets), max(budgets),
                     sum(b >= cap for b in budgets), len(budgets), cap)

    out_path = Path(args.out_dir or mc.get("output_dir", "data/raw/machine")) / f"{generator_slug(args.generator)}.jsonl"
    check_output_compatible(out_path, backend, model)
    done_ids = read_done_ids(out_path)
    if len(done_ids) >= len(prompts) and all(p["id"] in done_ids for p in prompts):
        print(f"\ncomplete — {len(prompts)} of {len(prompts)} (nothing to generate)\n{out_path}")
        return

    if backend == "ollama":
        ollama_cfg = mc.get("ollama") or {}
        client = OllamaClient(args.host or ollama_cfg.get("host", "http://127.0.0.1:11434"),
                              float(ollama_cfg.get("request_timeout_s", 900)))
        info = describe_ollama_model(client, model)
        parallel = args.parallel or int(ollama_cfg.get("parallel", 4))
        num_ctx = int(ollama_cfg.get("num_ctx", 3072))
        keep_alive = str(ollama_cfg.get("keep_alive", "30m"))
        log.info("ollama %s: %s (%s, %s, digest %s), %d worker threads",
                 info["server_version"], model, info["parameter_size"], info["quantization"], info["digest"], parallel)
        load_started = time.monotonic()
        client.load(model, num_ctx, keep_alive)   # so the first requests do not queue behind the load
        log.info("model loaded in %.1fs", time.monotonic() - load_started)
        process_item = make_ollama_worker(client, args.generator, spec["role"], model, info,
                                          temperature=temperature, top_p=top_p, num_ctx=num_ctx,
                                          keep_alive=keep_alive, seed=args.seed)
        meter = process_item.meter  # type: ignore[attr-defined]
        precision = info["quantization"]
        report = run_concurrent(prompts, id_of=lambda p: p["id"], process_item=process_item,
                                out_path=out_path, done_ids=done_ids, workers=parallel,
                                max_minutes=args.max_minutes, label="passages")
    else:
        from src.utils.modelload import load_causal_lm

        hf_model, tokenizer, info = load_causal_lm(model, device=args.device)
        process_batch = make_hf_processor(hf_model, tokenizer, args.generator, spec["role"], model, info,
                                          temperature=temperature, top_p=top_p)
        meter = process_batch.meter  # type: ignore[attr-defined]
        precision = f"{info['dtype']} on {info['device']}"
        report = run_resumable(prompts, id_of=lambda p: p["id"], process_batch=process_batch,
                               out_path=out_path, done_ids=done_ids, batch_size=args.batch_size,
                               max_minutes=args.max_minutes, label="passages")

    resume = f"python -m src.data.generate --generator {args.generator} --buckets {args.buckets}"
    if args.fraction != 1.0:
        resume += f" --fraction {args.fraction}"
    if args.backend:
        resume += f" --backend {backend}"
    if args.model:
        resume += f" --model {model}"
    print(f"\ngenerator={args.generator} ({spec['role']}: {model}, {precision}, {backend})  out={out_path}")
    print(report.summary(f"resume with: {resume}"))
    if meter.tokens:
        print(f"throughput: {meter.tokens:,} tokens in {meter.elapsed / 60:.1f} min = "
              f"{meter.tokens / meter.elapsed:.1f} tokens/sec")
    on_disk = len(read_done_ids(out_path) & {p["id"] for p in prompts})
    if on_disk != report.done:
        print(f"on disk: {on_disk} of {len(prompts)}")
    if report.errors:
        print(f"first error: {report.errors[0]}")


if __name__ == "__main__":
    main()
