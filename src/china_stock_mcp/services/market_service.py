"""MarketService -- 市场总览 (大盘指数 / 涨跌家数 / 北向 / 行业热度).

Implements the ``MarketService`` half of *design.md* Component 3
(Service Layer) for task 17.1. The service is intentionally tiny: a
single :meth:`overview` entry point that reads a fresh snapshot
through the adapter (currently :class:`AkshareAdapter`) and surfaces
it as :class:`MarketOverview`.

Caching uses :data:`TTL_COLD` (3 600s) which is the 基金净值 / 市场
总览 grade defined in Requirements 11.4 / Property 18 -- 大盘 breadth
moves slowly enough that an hour-long TTL still feels live without
hammering the upstream during off-hours queries. The cache key is
keyed solely on the schema version since :func:`overview` takes no
parameters; bump :data:`_OVERVIEW_SCHEMA_VERSION` whenever the
:class:`MarketOverview` shape changes (Requirement 11.3 / Property 4).

Rate limiting (Requirement 11.6 / Property 7) and
:func:`fetch_with_fallback` (Requirements 13.3 / 13.5, Property 6) are
applied identically to the other services so the upstream contract
stays uniform across the tool suite. :class:`DataNotFoundError` from
the adapter propagates verbatim (Property 5).

Requirement 9.4 ("非交易时段 SHALL 标注最近交易日快照") is honored
end-to-end: the adapter returns the most recently published spot
frame, the service hands it through unchanged, and the tool layer
adds the "非交易时段" banner when the rendered ``snapshot_at`` is
outside A 股 trading hours.
"""

from __future__ import annotations

from typing import Final

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.adapters.fallback import fetch_with_fallback
from china_stock_mcp.cache import (
    TTL_COLD,
    Cache,
    cache_get_or_fetch,
    get_default_cache,
)
from china_stock_mcp.models import MarketOverview
from china_stock_mcp.rate_limiter import RateLimiter, get_default_rate_limiter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Cache schema version for market-overview payloads. Bump whenever
#: the :class:`MarketOverview` shape changes (Requirement 11.3,
#: Property 4).
_OVERVIEW_SCHEMA_VERSION: Final[int] = 1

#: Cache "tool" namespace for market-overview payloads.
_OVERVIEW_TOOL_NAMESPACE: Final[str] = "market_service.overview"

#: Sentinel used as ``symbol`` in the cache key. Market overview is
#: process-wide and not symbol-scoped, so a fixed sentinel keeps the
#: key shape uniform with the other services without polluting the
#: symbol component.
_OVERVIEW_CACHE_SYMBOL: Final[str] = "_market"

#: Empty params dict reused for the cache key. The schema version
#: alone differentiates entries across releases.
_EMPTY_PARAMS: Final[dict[str, object]] = {}


class MarketService:
    """市场总览 service backed by an adapter pair.

    Parameters
    ----------
    primary:
        Primary :class:`BaseAdapter` used for upstream calls
        (typically :class:`AkshareAdapter`).
    fallback:
        Optional backup adapter; when ``None``, transient failures
        from ``primary`` propagate verbatim.
    cache:
        Optional :class:`Cache` injection. Defaults to the
        process-wide instance returned by :func:`get_default_cache`.
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

    def overview(self) -> MarketOverview:
        """Return the current :class:`MarketOverview` snapshot.

        Pipeline:

        1. Read-through cache keyed by the schema version with
           :data:`TTL_COLD` (Requirements 11.1 / 11.4).
        2. On a cache miss, acquire one rate-limit token
           (Requirement 11.6) and call :func:`fetch_with_fallback`
           against the adapter's ``market_overview`` endpoint
           (Requirements 13.3 / 13.5).

        Returns
        -------
        MarketOverview
            A fresh (or recently cached) snapshot of the A 股 market
            including indices, advance/decline counts, limit-up /
            limit-down stats, north flow, top-inflow industries and
            the ``heat_score`` (0..100).

        Raises
        ------
        ChinaStockMCPError
            Any subclass raised by the adapter is propagated
            verbatim. ``DataNotFoundError`` does not trigger fallback
            (Property 5).
        """

        return cache_get_or_fetch(
            tool=_OVERVIEW_TOOL_NAMESPACE,
            symbol=_OVERVIEW_CACHE_SYMBOL,
            params=_EMPTY_PARAMS,
            ttl=TTL_COLD,
            fetcher=self._fetch_overview,
            schema_version=_OVERVIEW_SCHEMA_VERSION,
            cache=self._cache,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_overview(self) -> MarketOverview:
        """Acquire one rate-limit token and call ``adapter.market_overview``."""

        self._rate_limiter.acquire()

        primary = self._primary
        fallback = self._fallback

        def _primary_call() -> MarketOverview:
            return primary.market_overview()

        def _fallback_call() -> MarketOverview:
            assert fallback is not None  # narrow for mypy --strict
            return fallback.market_overview()

        return fetch_with_fallback(
            primary=_primary_call,
            fallback=_fallback_call if fallback is not None else None,
            primary_name=primary.name,
            fallback_name=fallback.name if fallback is not None else "none",
        )


__all__ = ["MarketService"]
