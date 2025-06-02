# pylint: disable=missing-module-docstring,missing-class-docstring,too-few-public-methods,global-statement, arguments-renamed
"""Logging configuration for Parallax."""
import logging
import os
import sys
import threading
from typing import Optional

__all__ = ["get_logger", "use_parallax_log_handler"]

_init_lock = threading.Lock()
_default_handler: logging.Handler | None = None


class _Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"


_LEVEL_COLOR = {
    "DEBUG": _Ansi.CYAN,
    "INFO": _Ansi.GREEN,
    "WARNING": _Ansi.YELLOW,
    "ERROR": _Ansi.RED,
    "CRITICAL": _Ansi.MAGENTA,
}


class CustomFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        levelname = record.levelname.upper()
        levelcolor = _LEVEL_COLOR.get(levelname, "")
        record.levelcolor = levelcolor
        record.bold = _Ansi.BOLD
        record.reset = _Ansi.RESET

        # caller_block: last path component + line no
        pathname = record.pathname.rsplit("/", 1)[-1]
        record.caller_block = f"{pathname}:{record.lineno}"

        return super().format(record)


def _enable_default_handler(target_module_prefix: str):
    root = logging.getLogger()

    # attach the handler only to loggers that start with target_module_prefix
    class _ModuleFilter(logging.Filter):
        def filter(self, rec: logging.LogRecord) -> bool:
            return rec.name.startswith(target_module_prefix)

    _default_handler.addFilter(_ModuleFilter())
    root.addHandler(_default_handler)


def _initialize_if_necessary():
    global _default_handler

    with _init_lock:
        if _default_handler is not None:
            return

        fmt = (
            "{asctime}.{msecs:03.0f} "
            "[{bold}{levelcolor}{levelname:<8}{reset}] "
            "{caller_block:<25} {message}"
        )
        formatter = CustomFormatter(fmt=fmt, style="{", datefmt="%b %d %H:%M:%S")
        _default_handler = logging.StreamHandler(stream=sys.stdout)
        _default_handler.setFormatter(formatter)

        # root level from env or INFO
        level_name = os.getenv("PARALLAX_LOGLEVEL", "INFO").upper()
        logging.getLogger().setLevel(level_name)

        _enable_default_handler("parallax")  # only parallax.* by default


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Grab a logger with parallax’s default handler attached.
    Call this in every module instead of logging.getLogger().
    """
    _initialize_if_necessary()
    return logging.getLogger(name)


def use_parallax_log_handler(for_root: bool = True):
    """
    Extend the custom handler to the root logger (so *all* libraries print
    with the same style) or to any logger you call this on.

    Example
    -------
        from parallax.logging import use_parallax_log_handler
        use_parallax_log_handler()            # now requests, hivemind, etc. share the style
    """
    del for_root
    _initialize_if_necessary()
    root = logging.getLogger()
    if _default_handler not in root.handlers:
        root.addHandler(_default_handler)
