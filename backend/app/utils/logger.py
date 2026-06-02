"""
Centralised logging configuration.
Import get_logger(__name__) in every module for consistent formatting.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Return a named logger with a consistent stdout handler.

    Args:
        name:  Logger name — pass __name__ from the calling module.
        level: Optional log-level override (default: INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger is re-requested
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)

    logger.setLevel(level or logging.INFO)
    logger.propagate = False
    return logger
