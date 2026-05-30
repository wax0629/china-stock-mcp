"""QuoteService -- batched quote snapshots with per-symbol caching.

Implements the ``QuoteService`` half of *design.md* Component 3
(Service Layer). The service composes:

1. :func:`normalize_symbol` for every input (Requirement 2.1, 2.2).
2. Per-symbol cache entries with :data:`TTL_HOT` so 19 cache hits + 1
   miss only consume one upstream call (Requirement 11.1).
3. A single rate-limit token per upstream batch call
   (Requirements 11.6 / 11.7 / Property 7).
4. :func:`fetch_with_fallback` for transient failure transparency
   (Requirements 13.3 / 13.5, Property 6).
5. A 20-symbol batch ceiling (Requirement 2.3).

Design notes
------------

* The single-key read-through helper :func:`cache_get_or_fetch` is
  *only* used for ``SymbolService.search``. Quote requests perform
  their own read-through manually so a partial-miss batch issues
  exactly one upstream call covering all missing symbols (rather than
  N independent calls from N independent ``cache_get_or_fetch``
  invocations).
* Caller order, including duplicates, is preserved in the returned
  list; deduplication is internal to the batch path.
"""

from __future__ import annotations

from typing import Final

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.adapters.fallback import fetch_with_fallback
from china_stock_mcp.cache import TTL_HOT, Cache, get_default_cache
from china_stock_mcp.exceptions import DataNotFoundError, ValidationError
from china_stock_mcp.models import Quote
from china_stock_mcp.normalizer import normalize_symbol
from china_stock_mcp.rate_limiter import RateLimiter, get_default_rate_limiter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Cache schema version for quote payloads. Bump when the
#: :class:`Quote` shape changes so old entries are invalidated
#: (Requirement 11.3, Property 4).
_QUOTE_SCHEMA_VERSION: Final[int] = 1

#: Cache "tool" namespace for per-symbol quote payloads.
_QUOTE_TOOL_NAMESPACE: Final[str] = "quote_service.snapshot"

#: Empty params dict reused for every per-symbol cache key. The
#: standardized symbol itself is the variable component of the key, so
#: ``params`` carries no further information.
_EMPTY_PARAMS: Final[dict[str, object]] = {}


class QuoteService:
    """Batched quote snapshots with per-symbol caching.

    Parameters
    ----------
    primary:
        Primary :class:`BaseAdapter` used for upstream calls.
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

    #: Maximum number of symbols accepted in a single
    #: :meth:`get_snapshot` call (Requirement 2.3).
    MAX_BATCH_SIZE: Final[int] = 20

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
    # get_snapshot
    # ------------------------------------------------------------------

    def get_snapshot(self, symbols: str | list[str]) -> list[Quote]:
        """Return a :class:`Quote` list mirroring caller-order ``symbols``.

        Pipeline:

        1. Coerce a single string into a one-element list.
        2. Validate batch size (1..20, Requirement 2.3).
        3. Normalize each symbol via :func:`normalize_symbol`.
        4. Read every symbol through the per-symbol cache; collect the
           subset that missed.
        5. If any missed: acquire one rate-limit token, call
           ``primary.quote(missing)`` (with fallback) once, write each
           returned :class:`Quote` into its own cache entry.
        6. Replay caller order (duplicates allowed) from the merged
           ``cache hits + fresh fetches`` map.

        Raises
        ------
        ValidationError
            If ``symbols`` is empty or has more than 20 entries.
        ChinaStockMCPError
            Any subclass raised by the adapter is propagated verbatim.
        DataNotFoundError
            If the upstream batch call did not return data for one or
            more requested symbols.
        """

        # 1) Coerce to list, preserving caller order.
        if isinstance(symbols, str):
            raw_list: list[str] = [symbols]
        else:
            raw_list = list(symbols)

        # 2) Batch size validation.
        if len(raw_list) == 0:
            raise ValidationError("symbol 列表不能为空")
        if len(raw_list) > self.MAX_BATCH_SIZE:
            raise ValidationError(
                f"单次请求最多 {self.MAX_BATCH_SIZE} 个标的, "
                f"实际收到 {len(raw_list)} 个"
            )

        # 3) Normalize each input. ``SymbolError`` raised here
        #    propagates verbatim per the unified-error contract.
        std_symbols: list[str] = [normalize_symbol(s) for s in raw_list]

        # 4) Read-through: collect cache hits and the deduplicated set
        #    of misses while preserving first-seen order so the
        #    upstream batch call has a deterministic shape.
        quote_by_symbol: dict[str, Quote] = {}
        missing: list[str] = []
        seen_missing: set[str] = set()

        for std in std_symbols:
            if std in quote_by_symbol or std in seen_missing:
                # Already classified as hit or miss; skip duplicate
                # cache lookups for the same standardized symbol.
                continue
            cached = self._cache_get(std)
            if cached is not None:
                quote_by_symbol[std] = cached
            else:
                missing.append(std)
                seen_missing.add(std)

        # 5) Single upstream batch call for all misses.
        if missing:
            fresh = self._fetch_batch(missing)
            for std in missing:
                quote_dto = fresh.get(std)
                if quote_dto is None:
                    # Upstream silently dropped this symbol -- surface
                    # as DataNotFoundError so fallback is *not*
                    # triggered (Requirement 13.4 / Property 5).
                    raise DataNotFoundError(
                        f"未找到行情数据: {std}"
                    )
                self._cache_set(std, quote_dto)
                quote_by_symbol[std] = quote_dto

        # 6) Replay caller order, including duplicates.
        return [quote_by_symbol[std] for std in std_symbols]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cache_key(self, std_symbol: str) -> str:
        """Build the canonical per-symbol cache key."""

        return self._cache.make_key(
            tool=_QUOTE_TOOL_NAMESPACE,
            symbol=std_symbol,
            params=_EMPTY_PARAMS,
            schema_version=_QUOTE_SCHEMA_VERSION,
        )

    def _cache_get(self, std_symbol: str) -> Quote | None:
        cached = self._cache.get(self._cache_key(std_symbol))
        if cached is None:
            return None
        # ``Cache`` is a :class:`Protocol`; narrow the ``Any`` return to
        # :class:`Quote` defensively. A type mismatch indicates a stale
        # entry from an older schema, which we treat as a miss rather
        # than letting an unrelated payload leak into the result list.
        if isinstance(cached, Quote):
            return cached
        return None

    def _cache_set(self, std_symbol: str, value: Quote) -> None:
        self._cache.set(self._cache_key(std_symbol), value, TTL_HOT)

    def _fetch_batch(self, missing: list[str]) -> dict[str, Quote]:
        """Acquire one rate-limit token and call ``adapter.quote`` once.

        ``missing`` is already deduplicated by :meth:`get_snapshot`, so
        this method does not re-deduplicate. The result map is keyed by
        the :class:`Quote.symbol` field so callers can replay arbitrary
        caller orders without an additional pass.
        """

        self._rate_limiter.acquire()

        primary = self._primary
        fallback = self._fallback

        def _primary_call() -> list[Quote]:
            return primary.quote(missing)

        def _fallback_call() -> list[Quote]:
            assert fallback is not None  # narrow for mypy --strict
            return fallback.quote(missing)

        quotes: list[Quote] = fetch_with_fallback(
            primary=_primary_call,
            fallback=_fallback_call if fallback is not None else None,
            primary_name=primary.name,
            fallback_name=fallback.name if fallback is not None else "none",
        )

        return {q.symbol: q for q in quotes}


__all__ = ["QuoteService"]
