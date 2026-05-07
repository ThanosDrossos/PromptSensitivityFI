"""Loguru wiring. One log file per sprint/script under `logs/`."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from .config import REPO_ROOT


_DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level:<7}</level> | "
    "<cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>"
)


def configure_logging(name: str, *, level: str = "INFO") -> None:
    """Attach a sink writing to logs/<name>.log alongside stderr.

    Per the engineering standards (`Research_Design_v3` §0/§7.3), every script
    writes a logfile named after the sprint. Idempotent: re-calling with the
    same name reuses the file.
    """

    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.log"

    logger.remove()
    logger.add(sys.stderr, level=level, format=_DEFAULT_FORMAT, enqueue=False)
    logger.add(
        log_path,
        level="DEBUG",
        format=_DEFAULT_FORMAT,
        enqueue=True,
        rotation="50 MB",
        retention=10,
    )
    logger.info("logging configured -> {}", log_path)


def get_logger():
    """Return the shared loguru logger."""
    return logger


__all__ = ["configure_logging", "get_logger"]
