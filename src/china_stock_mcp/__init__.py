"""china-stock-mcp -- MCP server for Chinese stock market research.

This package exposes a process-wide :data:`logger` configured via
``CSM_LOG_LEVEL``. Modules should import the shared logger rather than
configure their own handlers::

    from china_stock_mcp import logger

    logger.info("ready")

The logger is built on `loguru <https://github.com/Delgan/loguru>`_ and
defaults to writing to ``stderr`` so that ``stdio`` MCP transport on
``stdout`` is not contaminated.

Logging policy (task 23.2 / Requirements 12.3, 12.4)
----------------------------------------------------

The package logger is intentionally configured to emit *operational*
information only. Concretely, log call sites across the codebase
SHALL log:

- Tool / prompt / resource name (e.g. ``"search_symbol"``).
- Timestamp (provided automatically by loguru's format string).
- Cache hit / miss status when relevant (descriptive verbs only --
  e.g. ``"hit"`` / ``"miss"``, not the cache key contents).
- Adapter source name on a fallback switch (``"akshare"`` /
  ``"tushare"`` / ``"efinance"``).
- Adapter / service initialization status booleans.
- Generic exception class names + sanitized messages from
  :class:`~china_stock_mcp.exceptions.ChinaStockMCPError` subclasses
  (these messages are crafted not to echo user query / token / PII).

Log call sites SHALL NOT emit:

- The raw user ``query`` string passed to ``search_symbol`` /
  ``screen_stocks`` / any other tool entry.
- The user's identity, holdings, watchlist, or any persistent
  account information (the server holds none of these).
- The tushare token (or any other secret) read from
  ``CSM_TUSHARE_TOKEN`` -- secrets are loaded from the environment
  exactly once into :class:`~china_stock_mcp.config.Settings` and
  forwarded directly to the third-party SDK.
- Symbol values, as a precaution: even though symbols are not PII,
  they are part of the user's research subject and the v1 logging
  policy stays on the conservative side. Operators can enable
  ``DEBUG`` logging in their own forks if they need symbol-level
  tracing for development.

Loguru's ``backtrace=False`` / ``diagnose=False`` configuration
(applied in :func:`_configure_logger`) ensures that exception logs
emitted via ``logger.exception(...)`` include only the canonical
Python stack trace -- never the values of local variables -- so a
caller cannot accidentally leak request data into the log stream by
catching an exception inside a tool body.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from loguru import logger as _loguru_logger

from china_stock_mcp.config import Settings, load_settings

if TYPE_CHECKING:
    from loguru import Logger

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Logger configuration
# ---------------------------------------------------------------------------

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def _configure_logger(settings: Settings) -> Logger:
    """Configure the package logger based on ``settings``.

    Removes loguru's default handler and installs a single ``stderr``
    sink at the configured level. Calling this function multiple times
    is safe; previous handlers are replaced on each call.
    """

    _loguru_logger.remove()
    _loguru_logger.add(
        sys.stderr,
        level=settings.log_level,
        format=_LOG_FORMAT,
        enqueue=False,
        backtrace=False,
        diagnose=False,
    )
    return _loguru_logger


# Configure the shared logger at import time using the current
# environment. Tests or callers that change ``CSM_LOG_LEVEL`` after
# import can call :func:`reconfigure_logger` to apply the change.
_settings: Settings = load_settings()
logger: Logger = _configure_logger(_settings)


def reconfigure_logger(settings: Settings | None = None) -> Logger:
    """Re-apply logger configuration.

    Useful after mutating ``CSM_*`` environment variables in tests or
    when the server boots and re-reads settings explicitly.
    """

    target = settings if settings is not None else load_settings()
    return _configure_logger(target)


__all__ = [
    "__version__",
    "logger",
    "reconfigure_logger",
]
