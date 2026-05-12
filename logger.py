"""
omnirca/logger.py — Centralised logging configuration.

Call setup_logging() once at startup (done automatically in omnirca/__init__.py).
All modules obtain their logger with:

    from omnirca.logger import get_logger
    _log = get_logger(__name__)

Log levels:
    INFO  — major events: sub-agent start/end, tool calls, LLM requests/responses
    DEBUG — verbose details (full args, full results)

Format example:
    2026-03-18 14:32:05 [INFO ] omnirca.main_agent — [DataDetective] started
    2026-03-18 14:32:06 [INFO ] omnirca.llm        — LLM request — 3 messages, tools=[syscall_multi_service_compare, ...]
    2026-03-18 14:32:08 [INFO ] omnirca.llm        — LLM response — tool_calls: ['syscall_multi_service_compare']
    2026-03-18 14:32:08 [INFO ] omnirca.agent.DataDetective — [step 1] TOOL syscall_multi_service_compare | t_start=..., t_end=...
    2026-03-18 14:32:08 [INFO ] omnirca.agent.DataDetective — [step 1] RESULT (187 chars): backend_3 z=33.0 | database_2 z=31.8 | ...
"""
from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

_FMT     = "%(asctime)s [%(levelname)-5s] %(name)s — %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_ROOT    = "omnirca"


_LOG_FILE = Path(__file__).parent / "RCA.log"


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the omnirca logger to write to stdout AND to omnirca/RCA.log.
    Safe to call multiple times — adds handlers only if none exist yet.
    """
    logger = logging.getLogger(_ROOT)
    if logger.handlers:
        return  # already configured

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    # Console handler — use reconfigure if available (Python 3.7+) to force utf-8
    # so Unicode characters in log messages don't crash on Windows cp1252 terminals.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, io.UnsupportedOperation):
        pass
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # File handler — appends so previous runs are preserved
    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'omnirca' namespace."""
    # Ensure the name is rooted under omnirca so the handler is inherited
    if not name.startswith(_ROOT):
        name = f"{_ROOT}.{name}"
    return logging.getLogger(name)
