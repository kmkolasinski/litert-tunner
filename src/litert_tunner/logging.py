"""Logging configuration utilities for litert-tunner."""

import logging
import sys


def get_logger(name: str = "litert_tunner", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a logger with a default stdio stream handler.

    Args:
        name: Name of the logger. Defaults to "litert_tunner".
        level: Logging level. Defaults to logging.INFO.

    Returns:
        The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if the logger already has handlers to prevent duplicate logging
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
