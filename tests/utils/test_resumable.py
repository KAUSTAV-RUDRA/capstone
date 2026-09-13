"""Offline tests for src.utils.resumable (no network, no models).

Run with pytest, or directly:  python tests/utils/test_resumable.py
Serves docs/parts-plan.md Part 4 and Stage 3.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from src.utils import resumable


def _items(n: int) -> list[dict]:
    return [{"id": f"item{i}"} for i in range(n)]


def _identity(item: dict) -> dict:
    return {"id": item["id"]}


def test_read_done_ids_tolerates_a_truncated_final_line() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, "out.jsonl")
        path.write_text('{"id": "a"}\n{"id": "b"}\n{"id": "c"', encoding="utf-8")
        assert resumable.read_done_ids(path) == {"a", "b"}
        assert resumable.read_done_ids(Path(tmp, "missing.jsonl")) == set()


def test_batched_covers_every_item() -> None:
    batches = list(resumable.batched(list(range(10)), 4))
    assert [len(b) for b in batches] == [4, 4, 2]
    assert [x for b in batches for x in b] == list(range(10))


def test_run_resumable_skips_done_and_appends_each_batch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, "out.jsonl")
        seen_batches: list[int] = []

        def process(batch):
            seen_batches.append(len(batch))
            return [{"id": it["id"], "value": 1} for it in batch]

        report = resumable.run_resumable(_items(10), id_of=lambda i: i["id"],
                                         process_batch=process, out_path=path, batch_size=4)
        assert report.processed == 10 and report.already_done == 0 and report.complete
        assert seen_batches == [4, 4, 2], "results must be appended after every batch"
        assert len(path.read_text(encoding="utf-8").splitlines()) == 10

        # Second run over the same output does nothing and writes nothing.
        seen_batches.clear()
        report2 = resumable.run_resumable(_items(10), id_of=lambda i: i["id"],
                                          process_batch=process, out_path=path, batch_size=4)
        assert report2.already_done == 10 and report2.processed == 0 and report2.complete
        assert seen_batches == []
        assert len(path.read_text(encoding="utf-8").splitlines()) == 10


def test_run_resumable_resumes_after_a_partial_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, "out.jsonl")
        # Simulate a killed run that completed one batch of 4.
        resumable.append_rows([{"id": f"item{i}"} for i in range(4)], path)
        processed_ids: list[str] = []

        def process(batch):
            processed_ids.extend(i["id"] for i in batch)
            return [{"id": i["id"]} for i in batch]

        report = resumable.run_resumable(_items(10), id_of=lambda i: i["id"],
                                         process_batch=process, out_path=path, batch_size=4)
        assert report.already_done == 4 and report.processed == 6 and report.complete
        assert processed_ids == [f"item{i}" for i in range(4, 10)], "must not redo finished items"
        ids = [json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()]
        assert len(ids) == len(set(ids)) == 10, "resume must not duplicate rows"


def test_a_failing_batch_does_not_end_the_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, "out.jsonl")

        def process(batch):
            if batch[0]["id"] == "item4":
                raise RuntimeError("simulated CUDA hiccup")
            return [{"id": i["id"]} for i in batch]

        report = resumable.run_resumable(_items(12), id_of=lambda i: i["id"],
                                         process_batch=process, out_path=path, batch_size=4)
        assert report.failed == 4 and report.processed == 8
        assert report.errors and "simulated CUDA hiccup" in report.errors[0]
        assert len(path.read_text(encoding="utf-8").splitlines()) == 8


def test_run_concurrent_writes_each_row_and_resumes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, "out.jsonl")
        resumable.append_rows([{"id": "item0"}, {"id": "item1"}], path)   # a killed earlier run
        calls: list[str] = []

        def process(item):
            calls.append(item["id"])
            return {"id": item["id"]}

        report = resumable.run_concurrent(_items(10), id_of=lambda i: i["id"], process_item=process,
                                          out_path=path, workers=3)
        assert report.already_done == 2 and report.processed == 8 and report.complete
        assert sorted(calls) == sorted(f"item{i}" for i in range(2, 10)), "must not redo finished items"
        ids = [json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()]
        assert len(ids) == len(set(ids)) == 10, "resume must not duplicate rows"

        again = resumable.run_concurrent(_items(10), id_of=lambda i: i["id"], process_item=process,
                                         out_path=path, workers=3)
        assert again.processed == 0 and again.complete and len(calls) == 8


def test_run_concurrent_never_holds_a_worker_behind_a_slow_item() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, "out.jsonl")

        def process(item):
            if item["id"] == "item0":
                # The slow item: it finishes only once the other worker has
                # written all nine remaining rows, which a batch runner never allows.
                deadline = time.monotonic() + 5
                while len(resumable.read_done_ids(path)) < 9:
                    if time.monotonic() > deadline:
                        raise TimeoutError("the other worker was held behind item0")
                    time.sleep(0.01)
            return {"id": item["id"]}

        report = resumable.run_concurrent(_items(10), id_of=lambda i: i["id"], process_item=process,
                                          out_path=path, workers=2)
        ids = [json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()]
        assert report.processed == 10 and report.failed == 0
        assert ids == [f"item{i}" for i in range(1, 10)] + ["item0"], "rows must land as each completes"


def test_run_concurrent_leaves_a_failed_item_for_resume() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, "out.jsonl")

        def flaky(item):
            if item["id"] == "item3":
                raise TimeoutError("server busy")
            return {"id": item["id"]}

        report = resumable.run_concurrent(_items(6), id_of=lambda i: i["id"], process_item=flaky,
                                          out_path=path, workers=2)
        assert report.failed == 1 and report.processed == 5 and not report.complete
        assert "server busy" in report.errors[0]
        again = resumable.run_concurrent(_items(6), id_of=lambda i: i["id"], process_item=_identity,
                                         out_path=path, workers=2)
        assert again.already_done == 5 and again.processed == 1 and again.complete


def test_run_concurrent_stops_taking_items_at_max_minutes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, "out.jsonl")

        def slow(item):
            time.sleep(0.05)
            return {"id": item["id"]}

        report = resumable.run_concurrent(_items(50), id_of=lambda i: i["id"], process_item=slow,
                                          out_path=path, workers=2, max_minutes=0.2 / 60)
        assert report.stopped_by_time and 0 < report.processed < 50
        assert len(path.read_text(encoding="utf-8").splitlines()) == report.processed
        assert report.summary().startswith(f"done {report.processed} of 50")


def test_summary_text_distinguishes_complete_from_resumable() -> None:
    done = resumable.RunReport(total=10, already_done=4, processed=6)
    assert done.summary() == "complete — 10 of 10"
    partial = resumable.RunReport(total=10, already_done=0, processed=4, stopped_by_time=True)
    text = partial.summary()
    assert text.startswith("done 4 of 10 — resume with the same command")
    assert "stopped at --max-minutes" in text


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
