"""Centralised logging configuration.

This module configures the `loguru` logger with sensible defaults
for both console and file output.  It is imported by CLI entry
points so that logging is configured before any other modules are
loaded.
"""

from pathlib import Path
from typing import Optional

from loguru import logger


def configure_logging(log_file: Optional[str] = None, level: str = "INFO") -> None:
    """Configure the root logger.

    If ``log_file`` is provided, messages are written to that file in
    addition to stderr.  Logs are rotated once they reach 10 MB.

    Args:
        log_file: Optional path to a log file.
        level: The minimum severity level for console output.
    """
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=level)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(path, rotation="10 MB", encoding="utf-8", level=level)
