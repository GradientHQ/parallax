"""Logging configuration for Parallax."""

import logging
import os
import sys
import threading
from typing import Optional

__all__ = ["get_logger", "use_parallax_log_handler", "set_log_level"]

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

_PACKAGE_COLOR = {
    "parallax": _Ansi.CYAN,
    "scheduling": _Ansi.GREEN,
    "backend": _Ansi.YELLOW,
    "sglang": _Ansi.MAGENTA,
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
        record.package = record.name.split(".")[0]
        record.packagecolor = _PACKAGE_COLOR.get(record.package, "")
        return super().format(record)


def _enable_default_handler(target_module_prefix):
    """Attach the default handler to the root logger with a name-prefix filter.

    Accepts either a single string prefix or an iterable of prefixes; a record
    passes the filter if its logger name starts with any provided prefix.
    """
    root = logging.getLogger()

    # attach the handler only to loggers that start with any of target prefixes
    class _ModuleFilter(logging.Filter):
        def __init__(self, prefixes):
            super().__init__()
            if isinstance(prefixes, str):
                self._prefixes = (prefixes,)
            else:
                try:
                    self._prefixes = tuple(prefixes)
                except TypeError:
                    self._prefixes = (str(prefixes),)

        def filter(self, rec: logging.LogRecord) -> bool:
            return any(rec.name.startswith(p) for p in self._prefixes)

    _default_handler.addFilter(_ModuleFilter(target_module_prefix))
    root.addHandler(_default_handler)


def _initialize_if_necessary():
    global _default_handler

    with _init_lock:
        if _default_handler is not None:
            return

        fmt = (
            "{asctime}.{msecs:03.0f} "
            "[{bold}{packagecolor}{package:<10}{reset}] "
            "[{bold}{levelcolor}{levelname:<8}{reset}] "
            "{caller_block:<25} {message}"
        )
        formatter = CustomFormatter(fmt=fmt, style="{", datefmt="%b %d %H:%M:%S")
        _default_handler = logging.StreamHandler(stream=sys.stdout)
        _default_handler.setFormatter(formatter)

        # root level from env or INFO
        logging.getLogger().setLevel("INFO")

        # Allow logs from our main packages by default
        _enable_default_handler(("parallax", "scheduling", "backend", "sglang"))


def set_log_level(level_name: str):
    """Set the root logger level."""
    _initialize_if_necessary()
    logging.getLogger().setLevel(level_name.upper())
    if level_name.upper() == "DEBUG":
        os.environ["RUST_LOG"] = "info"


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
