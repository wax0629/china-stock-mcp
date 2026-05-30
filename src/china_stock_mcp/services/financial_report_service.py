"""FinancialReportService -- 多期财务报告 (年报 / 季报).

Implements the ``FinancialReportService`` half of *design.md*
Component 3 (Service Layer) for task 12.1. The service composes:

1. Strict input validation for ``report_type`` and ``periods``
   (Requirements 4.4) -- failures surface as :class:`ValidationError`
   listing the offending value plus the accepted set / range.
2. :func:`normalize_symbol` so callers may pass either a bare 6-digit
   A-share code or an already-standardized symbol.
3. A read-through :func:`cache_get_or_fetch` keyed by
   ``(symbol, report_type, periods)`` with :data:`TTL_FROZEN`
   (86 400s, the 财务数据 grade defined in Requirements 11.4 and
   Property 18) -- financial reports only refresh on report dates so
   a one-day TTL is safe.
4. A single rate-limit token per upstream call
   (Requirements 11.6 / 11.7 / Property 7) and
   :func:`fetch_with_fallback` for transient failure transparency
   (Requirements 13.3 / 13.5, Property 6); ``DataNotFoundError`` from
   the adapter (insufficient periods, Requirement 4.5) propagates
   verbatim because :func:`fetch_with_fallback` only switches sources
   on transient errors (Property 5).
5. A final stable sort of ``periods`` ascending by ``period_end``
   (Requirement 4.6) so the most recent reporting date is at the
   tail; the adapter already sorts but the service re-sorts
   defensively in case a future fallback adapter returns a different
   order.

The service deliberately re-raises every
:class:`ChinaStockMCPError` subclass verbatim -- the unified hierarchy
is the contract relied on by the Tool layer.
"""

from __future__ import annotations

from typing import Final, cast

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.adapters.fallback import fetch_with_fallback
from china_stock_mcp.cache import (
    TTL_FROZEN,
    Cache,
    cache_get_or_fetch,
    get_default_cache,
)
from china_stock_mcp.exceptions import ValidationError
from china_stock_mcp.models import FinancialReport, ReportType
from china_stock_mcp.normalizer import normalize_symbol
from china_stock_mcp.rate_limiter import RateLimiter, get_default_rate_limiter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Cache schema version for financial-report payloads. Bump whenever
#: the :class:`FinancialReport` shape changes so old entries are
#: invalidated (Requirement 11.3, Property 4).
_FIN_SCHEMA_VERSION: Final[int] = 1

#: Cache "tool" namespace for financial-report payloads.
_FIN_TOOL_NAMESPACE: Final[str] = "financial_report_service.report"

#: Accepted ``report_type`` values (Requirement 4.4).
_VALID_REPORT_TYPES: Final[frozenset[str]] = frozenset({"annual", "quarterly"})

#: ``periods`` lower / upper bounds (Requirement 4.4).
_MIN_PERIODS: Final[int] = 1
_MAX_PERIODS: Final[int] = 12


class FinancialReportService:
    """财务报告 service backed by an adapter pair.

    Parameters
    ----------
    primary:
        Primary :class:`BaseAdapter` used for upstream calls
        (typically :class:`AkshareAdapter`).
    fallback:
        Optional backup adapter; when ``None``, transient failures from
        ``primary`` propagate verbatim.
    cache:
        Optional :class:`Cache` injection. Defaults to the process-wide
        instance returned by :func:`get_default_cache`.
    rate_limiter:
        Optional :class:`RateLimiter` injection. Defaults to the
        process-wide instance from :func:`get_default_rate_limiter`.
    """

    __slots__ = ("_cache", "_fallback", "_primary", "_rate_limiter")

    def __init__(
        self,
        primary: BaseAdapter,
        fallback: BaseAdapter | None = None,
        *,
        cache: Cache | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._primary: BaseAdapter = primary
        self._fallback: BaseAdapter | None = fallback
        self._cache: Cache = cache if cache is not None else get_default_cache()
        self._rate_limiter: RateLimiter = (
            rate_limiter if rate_limiter is not None else get_default_rate_limiter()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def report(
        self,
        symbol: str,
        report_type: str,
        periods: int,
    ) -> FinancialReport:
        """Return ``periods`` of :class:`FinancialReport` for ``symbol``.

        Pipeline:

        1. Validate ``report_type`` and ``periods``. Each failure
           raises :class:`ValidationError` whose message lists the
           offending value plus the accepted set / range
           (Requirements 4.4).
        2. Normalize ``symbol`` via :func:`normalize_symbol`.
        3. Read-through cache keyed by
           ``(symbol, report_type, periods)`` with TTL_FROZEN.
        4. On a cache miss, acquire one rate-limit token and call
           :func:`fetch_with_fallback` against the adapter's
           ``financial_report`` endpoint. ``DataNotFoundError`` from
           the adapter (insufficient periods, Requirement 4.5) is not
           caught here so it propagates verbatim to the tool layer.
        5. Sort the returned ``periods`` ascending by ``period_end``
           and return a fresh :class:`FinancialReport` so the cached
           object stays stable (Requirement 4.6).

        Raises
        ------
        ValidationError
            If ``report_type`` is not in ``{"annual", "quarterly"}``
            or ``periods`` is not in ``[1, 12]``.
        DataNotFoundError
            If the adapter cannot satisfy the request (e.g. the symbol
            has fewer than ``periods`` of the requested type).
        ChinaStockMCPError
            Any other subclass raised by the adapter is propagated
            verbatim.
        """

        # 1) Strict input validation.
        if report_type not in _VALID_REPORT_TYPES:
            raise ValidationError(
                f"report_type 必须是 {sorted(_VALID_REPORT_TYPES)} 之一, "
                f"实际收到 {report_type!r}"
            )
        if (
            not isinstance(periods, int)
            or isinstance(periods, bool)
        ):
            # ``bool`` subclasses ``int`` in Python; reject it explicitly
            # so a stray ``True`` does not pass the bounds check.
            raise ValidationError(
                f"periods 必须是 int 类型, 实际收到 {type(periods).__name__}"
            )
        if periods < _MIN_PERIODS or periods > _MAX_PERIODS:
            raise ValidationError(
                f"periods 必须在 [{_MIN_PERIODS}, {_MAX_PERIODS}] 之间, "
                f"实际收到 {periods}"
            )

        # 2) Symbol normalization.
        std_symbol = normalize_symbol(symbol)

        # 3) Read-through cache.
        params: dict[str, object] = {
            "report_type": report_type,
            "periods": periods,
        }
        report: FinancialReport = cache_get_or_fetch(
            tool=_FIN_TOOL_NAMESPACE,
            symbol=std_symbol,
            params=params,
            ttl=TTL_FROZEN,
            fetcher=lambda: self._fetch_report(std_symbol, report_type, periods),
            schema_version=_FIN_SCHEMA_VERSION,
            cache=self._cache,
        )

        # 5) Stable ascending sort by period_end so the most recent
        #    period is at the tail, regardless of which adapter
        #    produced the cached payload.
        sorted_periods = sorted(report.periods, key=lambda p: p.period_end)

        return FinancialReport(
            symbol=report.symbol,
            report_type=cast(ReportType, report_type),
            periods=sorted_periods,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_report(
        self,
        std_symbol: str,
        report_type: str,
        periods: int,
    ) -> FinancialReport:
        """Acquire one rate-limit token and call ``adapter.financial_report``."""

        self._rate_limiter.acquire()

        primary = self._primary
        fallback = self._fallback

        def _primary_call() -> FinancialReport:
            return primary.financial_report(std_symbol, report_type, periods)

        def _fallback_call() -> FinancialReport:
            assert fallback is not None  # narrow for mypy --strict
            return fallback.financial_report(std_symbol, report_type, periods)

        return fetch_with_fallback(
            primary=_primary_call,
            fallback=_fallback_call if fallback is not None else None,
            primary_name=primary.name,
            fallback_name=fallback.name if fallback is not None else "none",
        )


__all__ = ["FinancialReportService"]
