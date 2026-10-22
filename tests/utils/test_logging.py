"""Tests for src.utils.logging."""
from __future__ import annotations

import logging

from src.utils.logging import configure_logging, get_logger


def test_get_logger_returns_named_logger(tmp_path) -> None:
    log_file = tmp_path / "logs" / "run.log"
    configure_logging("DEBUG", str(log_file))
    log = get_logger("tests.logging")
    assert isinstance(log, logging.Logger) and log.name == "tests.logging"
    log.debug("hello %s", "world")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "hello world" in log_file.read_text(encoding="utf-8")
    configure_logging("INFO")                        # release the file handler
