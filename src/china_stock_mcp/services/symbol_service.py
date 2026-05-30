"""SymbolService -- normalize + search composition.

Implements the ``SymbolService`` half of *design.md* Component 3
(Service Layer). The service stitches together four responsibilities:

1. :func:`normalize_symbol` for the public :meth:`SymbolService.normalize`
   helper (Requirement 1.4 / Property 1).
2. :func:`cache_get_or_fetch` with :data:`TTL_STATIC` so search results
   are cached for ~1 week (Requirement 11.1, design TTL grade).
3. :class:`RateLimiter` so every upstream search call passes through the
   global Token Bucket (Requirements 11.6 / 11.7 / Property 7).
4. :func:`fetch_with_fallback` so a transient ``NetworkError`` /
   ``RateLimitError`` from the primary adapter switches to the
   fallback (Requirements 13.3 / 13.5, Property 6).

The service deliberately re-raises every
:class:`ChinaStockMCPError` subclass verbatim -- the unified hierarchy
is the contract relied on by the Tool layer.
"""

from __future__ import annotations

from typing import Final

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.adapters.fallback import fetch_with_fallback
from china_stock_mcp.cache import (
    TTL_STATIC,
    Cache,
    cache_get_or_fetch,
    get_default_cache,
)
from china_stock_mcp.exceptions import ValidationError
from china_stock_mcp.models import SymbolHit
from china_stock_mcp.normalizer import (
    SymbolIndex,
    normalize_symbol,
    set_symbol_index,
)
from china_stock_mcp.rate_limiter import RateLimiter, get_default_rate_limiter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Markets accepted by :meth:`SymbolService.search` (Requirement 1.7).
_VALID_SEARCH_MARKETS: Final[frozenset[str]] = frozenset(
    {"a_stock", "hk_stock", "fund", "all"}
)

#: Cache schema version for ``search`` payloads. Bump when the
#: :class:`SymbolHit` shape changes so old entries are invalidated
#: (Requirement 11.3, Property 4).
_SEARCH_SCHEMA_VERSION: Final[int] = 1

#: Cache "tool" namespace for search payloads.
_SEARCH_TOOL_NAMESPACE: Final[str] = "symbol_service.search"

#: Sentinel "symbol" passed to :func:`make_key` for searches that are
#: not bound to a single standardized symbol. The actual disambiguation
#: comes from the ``params`` dict (``query`` + ``market``).
_SEARCH_SYMBOL_SENTINEL: Final[str] = "_search_"


class SymbolService:
    """Normalize symbols and search the standardized universe.

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
        instance returned by :func:`get_default_cache`. Tests can pass
        a stub here to keep them hermetic.
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
    # normalize
    # ------------------------------------------------------------------

    def normalize(self, raw: str, market: str | None = None) -> str:
        """Thin wrapper around :func:`normalize_symbol`.

        ``market`` is forwarded so an unrecognized input surfaces a
        :class:`SymbolError` whose ``to_user_message()`` lists the
        active market hint (Requirement 1.6).
        """

        return normalize_symbol(raw, market=market)

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(self, query: str, market: str = "all") -> list[SymbolHit]:
        """Search standardized symbols by code / Chinese name / pinyin.

        Pipeline:

        1. Validate ``market`` (Requirement 1.7).
        2. Build a cache key from ``(query.casefolded, market)`` so
           equivalent queries share a single cached payload.
        3. On a cache miss, acquire one rate-limit token (Requirement
           11.6 / Property 7) before invoking
           :func:`fetch_with_fallback` against the adapter's
           ``search`` endpoint.

        Returns
        -------
        list[SymbolHit]
            Possibly empty list; an empty result is cached so a popular
            misspelling does not re-hit the upstream every minute.

        Raises
        ------
        ValidationError
            If ``market`` is not one of ``a_stock`` / ``hk_stock`` /
            ``fund`` / ``all``.
        ChinaStockMCPError
            Any subclass raised by the adapter is propagated verbatim.
        """

        if market not in _VALID_SEARCH_MARKETS:
            raise ValidationError(
                f"market 必须是 {sorted(_VALID_SEARCH_MARKETS)} 之一, "
                f"实际收到 {market!r}"
            )

        # Canonicalize the query for cache-key purposes only; the
        # adapter still receives the original ``query`` so its
        # case-insensitive matching can be tuned independently.
        cache_query = (query or "").strip().casefold()
        params: dict[str, object] = {
            "query": cache_query,
            "market": market,
        }

        return cache_get_or_fetch(
            tool=_SEARCH_TOOL_NAMESPACE,
            symbol=_SEARCH_SYMBOL_SENTINEL,
            params=params,
            ttl=TTL_STATIC,
            fetcher=lambda: self._fetch_search(query, market),
            schema_version=_SEARCH_SCHEMA_VERSION,
            cache=self._cache,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_search(self, query: str, market: str) -> list[SymbolHit]:
        """Rate-limit + primary/fallback wrapper around adapter.search."""

        self._rate_limiter.acquire()

        primary = self._primary
        fallback = self._fallback

        def _primary_call() -> list[SymbolHit]:
            return primary.search(query, market)

        def _fallback_call() -> list[SymbolHit]:
            assert fallback is not None  # narrow for mypy --strict
            return fallback.search(query, market)

        return fetch_with_fallback(
            primary=_primary_call,
            fallback=_fallback_call if fallback is not None else None,
            primary_name=primary.name,
            fallback_name=fallback.name if fallback is not None else "none",
        )

    # ------------------------------------------------------------------
    # Index registration
    # ------------------------------------------------------------------

    @staticmethod
    def register_symbol_index(index: SymbolIndex) -> None:
        """Wire a :class:`SymbolIndex` into the global normalizer.

        Convenience passthrough to :func:`set_symbol_index` so callers
        can register a real lookup table at startup without importing
        the normalizer module directly.
        """

        set_symbol_index(index)


__all__ = ["SymbolService"]
