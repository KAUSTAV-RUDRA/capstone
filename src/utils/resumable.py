"""Shared machinery for the three long-running CLIs.

`src/data/generate.py`, `src/eval/score.py` and `src/data/attack.py` all run for
hours against a GPU and all need the same three properties, so the logic lives
here once:

* **Resumable** - the output file is the progress record. On start the ids
  already present are read back and skipped, so re-running the same command
  continues rather than restarting or duplicating.
* **Crash-safe** - results are appended after every batch and flushed, so a
  kill -9 loses at most one batch.
* **Time-boxed** - ``--max-minutes`` stops cleanly between batches and prints
  how far it got, so a run fits in one sitting.

The unit of work is an item with an ``id``. What the id means differs per CLI
(a prompt for generate, a row for score), which is why the caller supplies both
the id function and the batch processor.

Serves docs/parts-plan.md Part 4.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from src.utils.io import ensure_dir

log = logging.getLogger(__name__)

T = TypeVar("T")

#: Batch size shared by every CLI (docs/parts-plan.md Part 4).
DEFAULT_BATCH_SIZE = 8


def read_done_ids(path: str | Path, id_field: str = "id") -> set[str]:
    """Ids already present in a JSONL output file (empty set if it does not exist).

    A partial final line left by a killed process is skipped with a warning
    rather than raising, so a hard kill never bricks a resume.
    """
    path = Path(path)
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)[id_field])
            except (json.JSONDecodeError, KeyError):
                log.warning("%s:%d: unreadable line skipped when resuming", path, lineno)
    return done


def append_rows(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Append rows to a JSONL file and flush, so a kill loses at most one batch."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()


def batched(items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    """Yield consecutive slices of ``size`` (the last may be shorter)."""
    for start in range(0, len(items), size):
        yield items[start:start + size]


@dataclass
class RunReport:
    """Outcome of one :func:`run_resumable` call."""

    total: int                      # units of work in scope
    already_done: int = 0           # skipped because the output already had them
    processed: int = 0              # completed during this run
    failed: int = 0
    stopped_by_time: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.already_done + self.processed >= self.total and not self.stopped_by_time

    @property
    def done(self) -> int:
        return self.already_done + self.processed

    def summary(self, resume_hint: str = "resume with the same command") -> str:
        """The single line every CLI prints when it exits."""
        if self.complete:
            head = f"complete — {self.done} of {self.total}"
        else:
            head = f"done {self.done} of {self.total} — {resume_hint}"
        extra = ""
        if self.failed:
            extra += f", {self.failed} failed"
        if self.stopped_by_time:
            extra += ", stopped at --max-minutes"
        return head + extra


def run_resumable(
    items: Sequence[T],
    *,
    id_of: Callable[[T], str],
    process_batch: Callable[[Sequence[T]], list[dict[str, Any]]],
    out_path: str | Path,
    done_ids: set[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_minutes: float = 0,
    label: str = "items",
    id_field: str = "id",
) -> RunReport:
    """Process ``items`` in batches, appending results and honouring a time budget.

    Args:
        items: Every unit of work in scope, including ones already done.
        id_of: Stable id for an item; must match ``id_field`` in the output rows.
        process_batch: Turns one batch into output rows. Raising is caught: the
            batch is recorded as failed and the run continues.
        out_path: JSONL file that results are appended to.
        done_ids: Pre-read done ids; read from ``out_path`` when omitted.
        batch_size: Units per batch (default 8).
        max_minutes: Stop cleanly after this long. 0 means no limit.
        label: Noun used in progress logs.
        id_field: Key holding the id in output rows.

    Returns:
        A :class:`RunReport`; print ``report.summary()`` before exiting.
    """
    if done_ids is None:
        done_ids = read_done_ids(out_path, id_field)
    pending = [item for item in items if id_of(item) not in done_ids]
    report = RunReport(total=len(items), already_done=len(items) - len(pending))
    log.info("%d %s in scope, %d already done, %d to do",
             report.total, label, report.already_done, len(pending))
    if not pending:
        return report

    deadline = time.monotonic() + max_minutes * 60 if max_minutes and max_minutes > 0 else None
    started = time.monotonic()
    for batch in batched(pending, batch_size):
        try:
            rows = process_batch(batch)
        except Exception as exc:  # noqa: BLE001 - one bad batch must not end a long run
            report.failed += len(batch)
            message = f"{type(exc).__name__}: {exc}"
            report.errors.append(message)
            log.error("batch of %d failed (%s) — continuing", len(batch), message)
        else:
            if rows:
                append_rows(rows, out_path)
            report.processed += len(batch)
            done_ids.update(id_of(item) for item in batch)

        elapsed = time.monotonic() - started
        rate = report.processed / elapsed if elapsed > 0 and report.processed else 0.0
        remaining = len(pending) - report.processed - report.failed
        eta = f", eta {remaining / rate / 60:.0f} min" if rate > 0 and remaining else ""
        log.info("%d/%d %s this run (%.2f/s%s)", report.processed, len(pending), label, rate, eta)

        if deadline is not None and time.monotonic() > deadline and remaining > 0:
            report.stopped_by_time = True
            log.warning("--max-minutes reached")
            break
    return report
