"""Tests for the logging configuration submodule."""

import logging

from litert_tunner import logging as litert_logging


def test__get_logger_creates_and_configures() -> None:
    """Verifies that get_logger creates a logger and configures a default handler."""
    logger_name = "test_litert_tunner_logger"
    logger = litert_logging.get_logger(logger_name, level=logging.DEBUG)

    assert logger.name == logger_name
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) > 0

    # Verify that calling it again with the same name returns the same logger
    # and doesn't add duplicate handlers
    num_handlers = len(logger.handlers)
    logger2 = litert_logging.get_logger(logger_name)
    assert logger2 is logger
    assert len(logger2.handlers) == num_handlers
