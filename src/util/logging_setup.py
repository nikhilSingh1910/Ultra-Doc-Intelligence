"""Structured logging configuration for Ultra Doc-Intelligence.

Provides a consistent logger factory with JSON-friendly output.
Every module should use: `logger = get_logger(__name__)`
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Create a logger with consistent formatting.

    Args:
        name: Module name, typically ``__name__``.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def configure_root_logger(level: str = "INFO") -> None:
    """Set the root log level from configuration."""
    if isinstance(level, str):
        logging.getLogger().setLevel(getattr(logging, level.upper(), logging.INFO))
    else:
        logging.getLogger().setLevel(logging.INFO)
