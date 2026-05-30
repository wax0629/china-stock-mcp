"""FundamentalService -- 基本面快照 + 行业分位 (v1 stub).

Implements the ``FundamentalService`` half of *design.md* Component 3
(Service Layer) for task 11.1. The service composes:

1. :func:`normalize_symbol` so callers may pass either a bare 6-digit
   A-share code or an already-standardized symbol.
2. A read-through :func:`cache_get_or_fetch` keyed by the standardized
   symbol with :data:`TTL_FROZEN` (86 400s, the 财务数据 grade defined
   in Requirements 11.4 and Property 18) -- fundamentals only refresh
   on report dates so a one-day TTL is safe.
3. A single rate-limit token per upstream call
   (Requirements 11.6 / 11.7 / Property 7) and
   :func:`fetch_with_fallback` for transient failure transparency
   (Requirements 13.3 / 13.5, Property 6).

v1 limitation -- ``industry_percentile``
----------------------------------------

The 0-100 industry percentile defined in Requirement 4.2 requires a
cross-stock comparison across the symbol's industry universe, which is
the responsibility of the *industry peers* task (14.1) -- the
:class:`IndustryService` is what computes / caches the industry
universe. To keep task 11.1 self-contained the v1 implementation:

* Returns an **empty** ``industry_percentile`` dict from
  :meth:`FundamentalService.snapshot`. Empty is the well-defined
  signal the rendering layer (``tools/fundamental.py``) uses to skip
  the percentile column entirely, keeping the Markdown clean.
* Surfaces :meth:`FundamentalService.industry_percentile` as a stub
  that raises :class:`DataNotFoundError`. The contract is documented
  so callers (Prompts / future tools) get a clear "not yet" answer
  instead of a misleading number.

Once task 14.1 lands, :meth:`industry_percentile` will be promoted to
a real implementation (using :class:`IndustryService.peers`) and
:meth:`snapshot` will be updated to populate the dict for the four
key metrics (pe_ttm, pb, roe, revenue_yoy).

The service deliberately re-raises every
:class:`ChinaStockMCPError` subclass verbatim -- the unified hierarchy
is the contract relied on by the Tool layer.
"""

from __future__ import annotations

from typing import Final

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.adapters.fallback import fetch_with_fallback
from china_stock_mcp.cache import (
    TTL_FROZEN,
    Cache,
    cache_get_or_fetch,
    get_default_cache,
)
from china_stock_mcp.exceptions import DataNotFoundError
from china_stock_mcp.models import FundamentalSnapshot
from china_stock_mcp.normalizer import normalize_symbol
from china_stock_mcp.rate_limiter import RateLimiter, get_default_rate_limiter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Cache schema version for fundamental snapshots. Bump whenever the
#: :class:`FundamentalSnapshot` shape or bucket semantics change so old
#: entries are invalidated (Requirement 11.3, Property 4).
_FUND_SCHEMA_VERSION: Final[int] = 1

#: Cache "tool" namespace for fundamental payloads.
_FUND_TOOL_NAMESPACE: Final[str] = "fundamental_service.snapshot"

#: Empty params dict reused for every per-symbol cache key. The
#: standardized symbol itself is the variable component of the key, so
#: ``params`` carries no further information.
_EMPTY_PARAMS: Final[dict[str, object]] = {}


class FundamentalService:
    """Fundamental snapshot service backed by an adapter pair.

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

    def snapshot(self, symbol: str) -> FundamentalSnapshot:
        """Return the latest :class:`FundamentalSnapshot` for ``symbol``.

        Pipeline:

        1. Normalize ``symbol`` via :func:`normalize_symbol`.
        2. Read through cache with :data:`TTL_FROZEN` and the empty
           params dict -- the standardized symbol is the only variable
           component of the cache key.
        3. On a cache miss, acquire one rate-limit token and call
           :func:`fetch_with_fallback` against the adapter's
           ``fundamentals`` endpoint.

        Raises
        ------
        ChinaStockMCPError
            Any subclass raised by the adapter (e.g. ``SymbolError``
            from :func:`normalize_symbol`, ``DataNotFoundError`` from
            HK / fund codes) is propagated verbatim.
        """

        std_symbol = normalize_symbol(symbol)

        return cache_get_or_fetch(
            tool=_FUND_TOOL_NAMESPACE,
            symbol=std_symbol,
            params=_EMPTY_PARAMS,
            ttl=TTL_FROZEN,
            fetcher=lambda: self._fetch_snapshot(std_symbol),
            schema_version=_FUND_SCHEMA_VERSION,
            cache=self._cache,
        )

    def industry_percentile(self, symbol: str, metric: str) -> float:
        """Return the cross-stock industry percentile for ``metric``.

        Not yet implemented in v1; the cross-stock comparison is the
        responsibility of task 14.1 (industry peers). Always raises
        :class:`DataNotFoundError` so callers fail fast with a clear
        diagnostic instead of receiving a misleading default value.
        """

        # Reference both arguments so static analysers do not flag
        # this stub as having unused parameters; the message includes
        # them for clearer debuggability.
        _ = symbol, metric
        raise DataNotFoundError(
            "v1 暂未实现行业分位计算; 请等待 v2 (industry peers) 后续实现"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_snapshot(self, std_symbol: str) -> FundamentalSnapshot:
        """Acquire one rate-limit token and call ``adapter.fundamentals``."""

        self._rate_limiter.acquire()

        primary = self._primary
        fallback = self._fallback

        def _primary_call() -> FundamentalSnapshot:
            return primary.fundamentals(std_symbol)

        def _fallback_call() -> FundamentalSnapshot:
            assert fallback is not None  # narrow for mypy --strict
            return fallback.fundamentals(std_symbol)

        return fetch_with_fallback(
            primary=_primary_call,
            fallback=_fallback_call if fallback is not None else None,
            primary_name=primary.name,
            fallback_name=fallback.name if fallback is not None else "none",
        )


__all__ = ["FundamentalService"]
