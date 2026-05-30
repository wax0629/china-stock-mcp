"""Primary / fallback orchestration for adapter calls.

This module implements :func:`fetch_with_fallback`, the helper used by
the Service layer to wrap every upstream data-source invocation. The
helper enforces a small but precise contract derived from
``design.md`` Algorithm 1 and Requirements 13.3 / 13.4 / 13.5:

* Only :class:`~china_stock_mcp.exceptions.NetworkError` and
  :class:`~china_stock_mcp.exceptions.RateLimitError` may trigger a
  fallback (Requirement 13.3).
* :class:`~china_stock_mcp.exceptions.DataNotFoundError` is **never**
  masked -- it propagates immediately so callers can prompt the user to
  adjust ``periods`` / ``report_type`` instead of silently switching
  sources (Requirement 13.4, Property 5).
* When the primary call succeeds, the fallback is **never** invoked
  (Requirement 13.5, Property 6).
* When ``fallback`` is ``None`` and the primary raises a fallback-eligible
  error, the original exception is re-raised so the failure type is
  preserved for upstream handlers.

The helper is deliberately synchronous: the existing Service layer is
not async (see ``design.md`` §"Service Layer" interface declarations),
and Algorithm 1 in the design document is presented synchronously. If
async adapters are introduced later, an ``afetch_with_fallback`` sibling
should be added rather than retrofitting this function.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from china_stock_mcp import logger
from china_stock_mcp.exceptions import NetworkError, RateLimitError

T = TypeVar("T")


def fetch_with_fallback(
    primary: Callable[[], T],
    fallback: Callable[[], T] | None,
    *,
    primary_name: str = "primary",
    fallback_name: str = "fallback",
) -> T:
    """Invoke ``primary`` and fall back to ``fallback`` on transient failure.

    Implements ``design.md`` Algorithm 1 (``fetch_with_fallback``).

    Parameters
    ----------
    primary:
        Zero-argument callable that performs the primary data-source
        call. Callers should use :func:`functools.partial` or a closure
        to bind arguments before invoking this helper.
    fallback:
        Zero-argument callable for the backup data source, or ``None``
        when no fallback is configured. When ``None`` and the primary
        raises a fallback-eligible error, the original exception is
        re-raised unchanged.
    primary_name:
        Short identifier of the primary adapter (e.g. ``"akshare"``).
        Used only in the warn-level log emitted when switching sources.
    fallback_name:
        Short identifier of the fallback adapter (e.g. ``"tushare"``).
        Used only in the warn-level log emitted when switching sources.

    Returns
    -------
    T
        The value produced by ``primary`` on success, or by ``fallback``
        when the primary raises :class:`NetworkError` or
        :class:`RateLimitError` and ``fallback`` is provided.

    Raises
    ------
    NetworkError, RateLimitError
        Re-raised verbatim when ``fallback`` is ``None`` (preserving the
        original exception type for upstream handlers).
    DataNotFoundError
        Propagated immediately from ``primary`` without invoking the
        fallback (Requirement 13.4 / Property 5).
    Exception
        Any other exception raised by ``primary`` -- including other
        :class:`~china_stock_mcp.exceptions.ChinaStockMCPError` subclasses
        such as :class:`~china_stock_mcp.exceptions.SymbolError` or
        :class:`~china_stock_mcp.exceptions.ValidationError` -- is
        propagated unchanged. Exceptions raised by ``fallback`` are also
        propagated unchanged.

    Notes
    -----
    Properties enforced by this implementation:

    * **P5 (fallback 不吞 NotFound)** -- ``DataNotFoundError`` is not
      caught here, so it cannot reach the fallback branch.
    * **P6 (fallback 透明性)** -- when ``primary`` returns a value, the
      function returns that value directly; ``fallback`` is never
      referenced on the success path.
    """

    try:
        return primary()
    except (NetworkError, RateLimitError) as exc:
        if fallback is None:
            # Preserve the original error type for upstream handlers
            # (Requirement 13.3 leaves the no-fallback case to the
            # caller; we do not wrap or rename the exception).
            raise
        logger.warning(
            "primary {primary} failed: {exc!s}; switching to fallback {fallback}",
            primary=primary_name,
            fallback=fallback_name,
            exc=exc,
        )
        return fallback()


__all__ = ["fetch_with_fallback"]
