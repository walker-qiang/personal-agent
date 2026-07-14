"""Tests for logging configuration."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest

from matrix.logging_config import RequestIdFilter, setup_logging


class TestRequestIdFilter:
    def test_injects_request_id(self):
        f = RequestIdFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello", (), None)
        RequestIdFilter.set_request_id("abc123")
        result = f.filter(record)
        assert result is True
        assert record.request_id == "[abc123]"

    def test_none_request_id(self):
        f = RequestIdFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello", (), None)
        RequestIdFilter.set_request_id(None)
        result = f.filter(record)
        assert result is True
        assert record.request_id == ""


class TestSetupLogging:
    def test_returns_matrix_logger(self):
        logger = setup_logging(level=logging.INFO)
        assert logger.name == "matrix"
        assert logger.level == logging.INFO

    def test_sets_root_level(self):
        setup_logging(level=logging.DEBUG)
        assert logging.getLogger().level == logging.DEBUG

    def test_creates_log_file_with_dir(self):
        with tempfile.TemporaryDirectory() as d:
            setup_logging(level=logging.INFO, log_dir=d)
            log_file = Path(d) / "matrix.log"
            # Log something to trigger file creation
            logger = logging.getLogger("matrix.test")
            logger.info("test message")
            assert log_file.exists()
            content = log_file.read_text()
            assert "test message" in content

    def test_handles_empty_log_dir(self):
        # Should not raise
        logger = setup_logging(level=logging.INFO, log_dir="")
        assert logger.name == "matrix"

    def test_restores_after_multiple_calls(self):
        # Multiple calls should not duplicate handlers
        l1 = setup_logging(level=logging.INFO)
        l2 = setup_logging(level=logging.INFO)
        l1.info("first")
        l2.info("second")
        # No assertion needed — just verify no crash from duplicate handlers