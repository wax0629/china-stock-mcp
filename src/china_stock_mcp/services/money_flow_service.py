"""MoneyFlowService -- 资金流向 (north / main / dragon_tiger).

Implements the ``MoneyFlowService`` half of *design.md* Component 3
(Service Layer) for task 13.1. The service composes:

1. Strict input validation for ``flow_type`` and ``top_n``
   (Requirements 5.4 / 5.5) -- failures surface as
   :class:`ValidationError` listing the offending value plus the
   accepted set / range.
2. :func:`normalize_symbol` for ``main`` (where ``symbol`` is required,
   Requirement 5.2) and ``dragon_tiger`` (where ``symbol`` is optional,
   Requirement 5.3). For ``north`` the ``symbol`` is ignored entirely
   (Requirement 5.1) -- the caller-provided value, if any, is dropped
   before reaching the cache key so two callers querying north flow
   share the same cache slot.
3. A read-through :func:`cache_get_or_fetch` keyed by
   ``(symbol_or_sentinel, flow_type, top_n)`` with :data:`TTL_WARM`
   (300s, the K 线 / 资金流 grade defined in Requirements 11.4 and
   Property 18) -- 资金流向 refreshes intra-day so a 5-minute TTL
   strikes the right balance between staleness and upstream load.
4. A single rate-limit token per upstream call
   (Requirements 11.6 / 11.7 / Property 7) and
   :func:`fetch_with_fallback` for transient failure transparency
   (Requirements 13.3 / 13.5, Property 6); ``DataNotFoundError`` from
   the adapter (e.g. an unfilled trading-day-not-yet-open frame)
   propagates verbatim because :func:`fetch_with_fallback` only
   switches sources on transient errors (Property 5).

The service deliberately re-raises every
:class:`ChinaStockMCPError` subclass verbatim -- the unified hierarchy
is the contract relied on by the Tool layer.
"""

from __future__ import annotations

from typing import Final, cast

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.adapters.fallback import fetch_with_fallback
from china_stock_mcp.cache import (
    TTL_WARM,
    Cache,
    cache_get_or_fetch,
    get_default_cache,
)
from china_stock_mcp.exceptions import ValidationError
from china_stock_mcp.models import FlowType, MoneyFlow
from china_stock_mcp.normalizer import normalize_symbol
from china_stock_mcp.rate_limiter import RateLimiter, get_default_rate_limiter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Cache schema version for money-flow payloads. Bump whenever the
#: :class:`MoneyFlow` shape (or the per-``flow_type`` row dict layout)
#: changes so old entries are invalidated (Requirement 11.3,
#: Property 4).
_FLOW_SCHEMA_VERSION: Final[int] = 1

#: Cache "tool" namespace for money-flow payloads.
_FLOW_TOOL_NAMESPACE: Final[str] = "money_flow_service.get"

#: Accepted ``flow_type`` values (Requirement 5.5).
_VALID_FLOW_TYPES: Final[frozenset[str]] = frozenset(
    {"north", "main", "dragon_tiger"}
)

#: ``top_n`` lower / upper bounds (Requirement 5.4).
_MIN_TOP_N: Final[int] = 1
_MAX_TOP_N: Final[int] = 100

#: Sentinel used for the cache-key ``symbol`` slot when no symbol is
#: provided (north flow always; dragon_tiger when caller asked for the
#: aggregate board). Picked to be visually distinct from a valid
#: standardized symbol so cache dumps remain debuggable.
_AGGREGATE_SYMBOL_SENTINEL: Final[str] = "_aggregate_"


class MoneyFlowService:
    """资金流向 service backed by an adapter pair.

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

    def get(
        self,
        symbol: str | None,
        flow_type: str,
        top_n: int,
    ) -> MoneyFlow:
        """Return :class:`MoneyFlow` rows for the given ``flow_type``.

        Pipeline:

        1. Validate ``flow_type`` and ``top_n``. Each failure raises
           :class:`ValidationError` whose message lists the offending
           value plus the accepted set / range (Requirements 5.4 /
           5.5).
        2. For ``main`` ensure ``symbol`` is provided
           (Requirement 5.2); for ``north`` drop any caller-supplied
           ``symbol`` so two clients share a cache slot
           (Requirement 5.1); for ``dragon_tiger`` normalize ``symbol``
           when provided, otherwise treat as aggregate
           (Requirement 5.3).
        3. Read-through cache keyed by
           ``(symbol_or_sentinel, flow_type, top_n)`` with TTL_WARM.
        4. On a cache miss, acquire one rate-limit token and call
           :func:`fetch_with_fallback` against the adapter's
           ``money_flow`` endpoint. ``DataNotFoundError`` from the
           adapter is not caught here so it propagates verbatim to
           the tool layer.

        Raises
        ------
        ValidationError
            If ``flow_type`` is not in
            ``{"north", "main", "dragon_tiger"}`` or ``top_n`` is not
            in ``[1, 100]``.
        DataNotFoundError
            If the adapter cannot satisfy the request (e.g. ``main``
            called without a symbol, or the upstream returned an
            empty frame).
        ChinaStockMCPError
            Any other subclass raised by the adapter is propagated
            verbatim.
        """

        # 1) Strict input validation.
        if flow_type not in _VALID_FLOW_TYPES:
            raise ValidationError(
                f"flow_type 必须是 {sorted(_VALID_FLOW_TYPES)} 之一, "
                f"实际收到 {flow_type!r}"
            )
        if (
            not isinstance(top_n, int)
            or isinstance(top_n, bool)
        ):
            # ``bool`` subclasses ``int`` in Python; reject it explicitly
            # so a stray ``True`` does not pass the bounds check.
            raise ValidationError(
                f"top_n 必须是 int 类型, 实际收到 {type(top_n).__name__}"
            )
        if top_n < _MIN_TOP_N or top_n > _MAX_TOP_N:
            raise ValidationError(
                f"top_n 必须在 [{_MIN_TOP_N}, {_MAX_TOP_N}] 之间, "
                f"实际收到 {top_n}"
            )

        # 2) Symbol normalization / requirement.
        std_symbol: str | None
        cache_symbol: str
        if flow_type == "north":
            # Requirement 5.1: ``symbol`` is irrelevant for north flow;
            # drop it so two callers share the same cache entry.
            std_symbol = None
            cache_symbol = _AGGREGATE_SYMBOL_SENTINEL
        elif flow_type == "main":
            # Requirement 5.2: main flow needs a specific symbol.
            if symbol is None or not str(symbol).strip():
                raise ValidationError(
                    "main 资金流需要 symbol 参数, 请提供具体股票代码"
                )
            std_symbol = normalize_symbol(symbol)
            cache_symbol = std_symbol
        else:
            # Requirement 5.3: dragon_tiger optionally narrows by
            # symbol; missing symbol returns the full latest board.
            if symbol is not None and str(symbol).strip():
                std_symbol = normalize_symbol(symbol)
                cache_symbol = std_symbol
            else:
                std_symbol = None
                cache_symbol = _AGGREGATE_SYMBOL_SENTINEL

        # 3) Read-through cache.
        params: dict[str, object] = {
            "flow_type": flow_type,
            "top_n": top_n,
        }
        flow: MoneyFlow = cache_get_or_fetch(
            tool=_FLOW_TOOL_NAMESPACE,
            symbol=cache_symbol,
            params=params,
            ttl=TTL_WARM,
            fetcher=lambda: self._fetch_flow(std_symbol, flow_type, top_n),
            schema_version=_FLOW_SCHEMA_VERSION,
            cache=self._cache,
        )

        # Re-cast ``flow_type`` so the return value's annotation matches
        # the validated literal regardless of how the adapter built it.
        return MoneyFlow(
            flow_type=cast(FlowType, flow_type),
            rows=list(flow.rows),
            snapshot_at=flow.snapshot_at,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_flow(
        self,
        std_symbol: str | None,
        flow_type: str,
        top_n: int,
    ) -> MoneyFlow:
        """Acquire one rate-limit token and call ``adapter.money_flow``."""

        self._rate_limiter.acquire()

        primary = self._primary
        fallback = self._fallback

        def _primary_call() -> MoneyFlow:
            return primary.money_flow(std_symbol, flow_type, top_n)

        def _fallback_call() -> MoneyFlow:
            assert fallback is not None  # narrow for mypy --strict
            return fallback.money_flow(std_symbol, flow_type, top_n)

        return fetch_with_fallback(
            primary=_primary_call,
            fallback=_fallback_call if fallback is not None else None,
            primary_name=primary.name,
            fallback_name=fallback.name if fallback is not None else "none",
        )


__all__ = ["MoneyFlowService"]
