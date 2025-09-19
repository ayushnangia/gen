"""Shared logging configuration helpers."""

from __future__ import annotations

import logging
from typing import Optional


def configure_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.insert(0, logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers,
    )


__all__ = ["configure_logging"]
