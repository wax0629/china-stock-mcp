"""FundService -- 公募基金信息.

Implements the ``FundService`` half of *design.md* Component 3
(Service Layer) for task 15.1. The service composes:

1. Strict input validation for ``fund_code``: must be a 6-digit
   string. Non-conforming values raise :class:`SymbolError`
   (Requirement 7.3) -- funds use bare 6-digit codes with no
   exchange suffix, so we do **not** route through
   :func:`normalize_symbol`.
2. A read-through :func:`cache_get_or_fetch` keyed by ``fund_code``
   with :data:`TTL_COLD` (3 600s, the 基金净值 / 市场总览 grade
   defined in Requirements 11.4 and Property 18) -- 单位净值 only
   refreshes once per trading day so an hour-long TTL is safe and
   keeps NAV-derived returns fresh enough.
3. A single rate-limit token per upstream call
   (Requirements 11.6 / 11.7 / Property 7) and
   :func:`fetch_with_fallback` for transient failure transparency
   (Requirements 13.3 / 13.5, Property 6); :class:`DataNotFoundError`
   from the adapter (e.g. unknown fund code) propagates verbatim
   (Property 5).

The service deliberately re-raises every
:class:`ChinaStockMCPError` subclass verbatim -- the unified hierarchy
is the contract relied on by the Tool layer.
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
from china_stock_mcp.exceptions import SymbolError
from china_stock_mcp.models import FundInfo
from china_stock_mcp.rate_limiter import RateLimiter, get_default_rate_limiter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Cache schema version for fund-info payloads. Bump whenever the
#: :class:`FundInfo` shape changes so old entries are invalidated
#: (Requirement 11.3, Property 4).
_FUND_SCHEMA_VERSION: Final[int] = 1

#: Cache "tool" namespace for fund-info payloads.
_FUND_TOOL_NAMESPACE: Final[str] = "fund_service.info"

#: Empty params dict reused for every per-symbol cache key. The
#: standardized fund code itself is the only variable component of
#: the cache key, so ``params`` carries no further information.
_EMPTY_PARAMS: Final[dict[str, object]] = {}

#: Required length of a 公募基金 6-digit code.
_FUND_CODE_LEN: Final[int] = 6


class FundService:
    """公募基金 service backed by an adapter pair.

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

    def info(self, fund_code: str) -> FundInfo:
        """Return :class:`FundInfo` for ``fund_code``.

        Pipeline:

        1. Validate ``fund_code`` is a 6-digit string. Failures raise
           :class:`SymbolError` (Requirement 7.3).
        2. Read-through cache keyed by the bare fund code with
           :data:`TTL_COLD`.
        3. On a cache miss, acquire one rate-limit token and call
           :func:`fetch_with_fallback` against the adapter's
           ``fund_info`` endpoint.

        Raises
        ------
        SymbolError
            If ``fund_code`` is not a 6-digit string
            (Requirement 7.3).
        DataNotFoundError
            If the adapter cannot resolve the fund code.
        ChinaStockMCPError
            Any other subclass raised by the adapter is propagated
            verbatim.
        """

        # 1) Validate (Requirement 7.3).
        if not isinstance(fund_code, str):
            raise SymbolError(
                f"非法基金代码: {fund_code!r}, 必须是 6 位数字字符串"
            )
        code = fund_code.strip()
        if len(code) != _FUND_CODE_LEN or not code.isdigit():
            raise SymbolError(
                f"非法基金代码: {fund_code!r}, 必须是 6 位数字"
            )

        # 2) Read-through cache.
        return cache_get_or_fetch(
            tool=_FUND_TOOL_NAMESPACE,
            symbol=code,
            params=_EMPTY_PARAMS,
            ttl=TTL_COLD,
            fetcher=lambda: self._fetch_info(code),
            schema_version=_FUND_SCHEMA_VERSION,
            cache=self._cache,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_info(self, code: str) -> FundInfo:
        """Acquire one rate-limit token and call ``adapter.fund_info``."""

        self._rate_limiter.acquire()

        primary = self._primary
        fallback = self._fallback

        def _primary_call() -> FundInfo:
            return primary.fund_info(code)

        def _fallback_call() -> FundInfo:
            assert fallback is not None  # narrow for mypy --strict
            return fallback.fund_info(code)

        return fetch_with_fallback(
            primary=_primary_call,
            fallback=_fallback_call if fallback is not None else None,
            primary_name=primary.name,
            fallback_name=fallback.name if fallback is not None else "none",
        )


__all__ = ["FundService"]
