"""IndustryService -- 同行业可比公司对比 + 行业分位标注.

Implements the ``IndustryService`` half of *design.md* Component 3
(Service Layer) for task 14.1. The service composes:

1. Strict input validation for ``metrics`` (subset of ``{pe, pb, roe,
   revenue_growth}``) and ``top_n`` (``[1, 50]``)
   (Requirements 6.2 / 6.4) -- failures surface as
   :class:`ValidationError` listing the offending value plus the
   accepted set / range. The error message lists the *full* set of
   accepted metrics (Requirement 6.5).
2. :func:`normalize_symbol` so callers may pass either a bare 6-digit
   A-share code or an already-standardized symbol.
3. A read-through :func:`cache_get_or_fetch` keyed by
   ``(symbol, sorted_metrics, top_n)`` with :data:`TTL_FROZEN`
   (86 400s, the 行业分类 / 公司信息 grade defined in Requirements
   11.4 / 11.5 and Property 18) -- industry constituents and their
   valuation metrics refresh slowly enough for a one-day TTL.
4. A single rate-limit token per upstream call
   (Requirements 11.6 / 11.7 / Property 7) and
   :func:`fetch_with_fallback` for transient failure transparency
   (Requirements 13.3 / 13.5, Property 6); ``DataNotFoundError`` from
   the adapter (e.g. an unknown industry, or an empty constituent
   list) propagates verbatim (Property 5).
5. After fetch, computes the **行业分位 (industry percentile)** of each
   numeric metric across the returned rows and attaches it as a
   ``{metric}_percentile`` key alongside the raw value
   (Requirement 6.3). The percentile is the 0-100 rank of the row's
   value within the rows that have a non-null value for that metric.

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
from china_stock_mcp.exceptions import ValidationError
from china_stock_mcp.models import PeerTable
from china_stock_mcp.normalizer import normalize_symbol
from china_stock_mcp.rate_limiter import RateLimiter, get_default_rate_limiter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Cache schema version for industry-peer payloads. Bump whenever the
#: :class:`PeerTable` shape or per-row dict layout changes so old
#: entries are invalidated (Requirement 11.3, Property 4).
_PEER_SCHEMA_VERSION: Final[int] = 1

#: Cache "tool" namespace for industry-peer payloads.
_PEER_TOOL_NAMESPACE: Final[str] = "industry_service.peers"

#: Accepted ``metrics`` values (Requirement 6.2). The set is
#: surfaced verbatim in :class:`ValidationError` messages so callers
#: see every supported name on a violation (Requirement 6.5).
_VALID_METRICS: Final[frozenset[str]] = frozenset(
    {"pe", "pb", "roe", "revenue_growth"}
)

#: ``top_n`` lower / upper bounds (Requirement 6.4).
_MIN_TOP_N: Final[int] = 1
_MAX_TOP_N: Final[int] = 50

#: Suffix used for percentile annotation keys -- e.g. ``pe`` raw value
#: lives at ``row["pe"]`` and its percentile at ``row["pe_percentile"]``.
_PERCENTILE_SUFFIX: Final[str] = "_percentile"


class IndustryService:
    """同行业对比 service backed by an adapter pair.

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

    def peers(
        self,
        symbol: str,
        metrics: list[str],
        top_n: int,
    ) -> PeerTable:
        """Return :class:`PeerTable` for the symbol's industry peers.

        Pipeline:

        1. Validate ``metrics`` and ``top_n``. Each failure raises
           :class:`ValidationError` whose message lists the offending
           value plus the accepted set / range (Requirements 6.2 /
           6.4 / 6.5).
        2. Normalize ``symbol`` via :func:`normalize_symbol`.
        3. Read-through cache keyed by
           ``(std_symbol, sorted_metrics, top_n)`` with TTL_FROZEN.
        4. On a cache miss, acquire one rate-limit token and call
           :func:`fetch_with_fallback` against the adapter's
           ``industry_peers`` endpoint. The adapter raises
           :class:`DataNotFoundError` for unknown industries / empty
           constituent lists; the error propagates verbatim to the
           tool layer.
        5. Compute the 0-100 行业分位 of each numeric metric across the
           returned rows and attach it as ``{metric}_percentile``
           (Requirement 6.3).

        Raises
        ------
        ValidationError
            If ``metrics`` is not a subset of ``{pe, pb, roe,
            revenue_growth}`` or ``top_n`` is not in ``[1, 50]``.
        DataNotFoundError
            If the adapter cannot resolve the symbol's industry or
            the industry has no constituents.
        ChinaStockMCPError
            Any other subclass raised by the adapter is propagated
            verbatim.
        """

        # 1) Strict input validation (Requirement 6.2 / 6.5).
        if not isinstance(metrics, list) or not metrics:
            raise ValidationError(
                "metrics 必须是非空列表, "
                f"支持的取值: {sorted(_VALID_METRICS)}"
            )
        invalid_metrics = sorted(set(metrics) - _VALID_METRICS)
        if invalid_metrics:
            raise ValidationError(
                f"metrics 包含不支持的取值: {invalid_metrics}, "
                f"支持的取值: {sorted(_VALID_METRICS)}"
            )

        # ``bool`` subclasses ``int`` in Python; reject explicitly so
        # a stray ``True`` does not sneak past the bounds check.
        if not isinstance(top_n, int) or isinstance(top_n, bool):
            raise ValidationError(
                f"top_n 必须是 int 类型, 实际收到 {type(top_n).__name__}"
            )
        if top_n < _MIN_TOP_N or top_n > _MAX_TOP_N:
            raise ValidationError(
                f"top_n 必须在 [{_MIN_TOP_N}, {_MAX_TOP_N}] 之间, "
                f"实际收到 {top_n}"
            )

        # 2) Normalize symbol.
        std_symbol = normalize_symbol(symbol)

        # De-duplicate while preserving caller order so the rendered
        # column order matches the request. The cache key uses a
        # *sorted* metric list so two callers requesting
        # ``["pe", "pb"]`` and ``["pb", "pe"]`` share a slot.
        seen_metrics: set[str] = set()
        ordered_metrics: list[str] = []
        for m in metrics:
            if m in seen_metrics:
                continue
            seen_metrics.add(m)
            ordered_metrics.append(m)

        params: dict[str, object] = {
            "metrics": sorted(ordered_metrics),
            "top_n": top_n,
        }

        # 3) Read-through cache.
        peers: PeerTable = cache_get_or_fetch(
            tool=_PEER_TOOL_NAMESPACE,
            symbol=std_symbol,
            params=params,
            ttl=TTL_FROZEN,
            fetcher=lambda: self._fetch_peers(
                std_symbol, ordered_metrics, top_n
            ),
            schema_version=_PEER_SCHEMA_VERSION,
            cache=self._cache,
        )

        # 5) Annotate rows with industry percentile per numeric metric.
        annotated_rows = _annotate_with_percentile(peers.rows, ordered_metrics)

        return PeerTable(
            base_symbol=peers.base_symbol,
            industry=peers.industry,
            metrics=list(ordered_metrics),
            rows=annotated_rows,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_peers(
        self,
        std_symbol: str,
        metrics: list[str],
        top_n: int,
    ) -> PeerTable:
        """Acquire one rate-limit token and call ``adapter.industry_peers``."""

        self._rate_limiter.acquire()

        primary = self._primary
        fallback = self._fallback

        def _primary_call() -> PeerTable:
            return primary.industry_peers(std_symbol, metrics, top_n)

        def _fallback_call() -> PeerTable:
            assert fallback is not None  # narrow for mypy --strict
            return fallback.industry_peers(std_symbol, metrics, top_n)

        return fetch_with_fallback(
            primary=_primary_call,
            fallback=_fallback_call if fallback is not None else None,
            primary_name=primary.name,
            fallback_name=fallback.name if fallback is not None else "none",
        )


# ---------------------------------------------------------------------------
# Pure helpers (annotation)
# ---------------------------------------------------------------------------


def _annotate_with_percentile(
    rows: list[dict[str, object]],
    metrics: list[str],
) -> list[dict[str, object]]:
    """Attach ``{metric}_percentile`` to each row for every numeric metric.

    Implementation
    --------------
    For each metric:

    * Collect ``(row_index, numeric_value)`` pairs across rows whose
      value is a non-NaN number; rows whose value is ``None`` / NaN /
      non-numeric are skipped (their ``{metric}_percentile`` stays
      absent so the formatter renders ``"-"``).
    * Compute each row's percentile rank as the fraction of peers with
      a strictly lower value plus half the count of ties, scaled to
      ``[0, 100]``. This is the standard "fractional rank" definition
      that lies in ``[0, 100]`` even with ties and degenerates
      gracefully when only one peer has a value (percentile = 50).

    Higher ``value`` ⇒ higher percentile. The convention matches design
    Component 3 §FundamentalService.industry_percentile (which targets
    metrics like ROE / revenue_growth where larger is better). For
    metrics where smaller is "better" (e.g. PE / PB), the renderer
    layer is responsible for any inversion -- the service surfaces a
    raw rank so callers stay in control.

    Returns a new list; the input is not mutated.
    """

    # Build new row dicts so callers cannot accidentally observe
    # half-annotated state.
    new_rows: list[dict[str, object]] = [dict(row) for row in rows]

    for metric in metrics:
        # Collect (index, value) pairs for rows with a numeric value.
        numeric: list[tuple[int, float]] = []
        for idx, row in enumerate(new_rows):
            value = row.get(metric)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            fvalue = float(value)
            if fvalue != fvalue:  # NaN check without importing math
                continue
            numeric.append((idx, fvalue))

        n = len(numeric)
        if n == 0:
            # No numeric values -- skip the percentile annotation
            # entirely for this metric.
            continue

        if n == 1:
            # Only one peer has a value; assign a neutral 50.0 so the
            # renderer can still display a percentile column.
            only_idx = numeric[0][0]
            new_rows[only_idx][f"{metric}{_PERCENTILE_SUFFIX}"] = 50.0
            continue

        for idx, value in numeric:
            lower = sum(1 for _, v in numeric if v < value)
            ties = sum(1 for _, v in numeric if v == value)
            # Fractional rank: count strictly-lower values plus half
            # the ties (excluding self -- (ties - 1)). Result lies in
            # ``[0, n - 1]``; rescale to ``[0, 100]``.
            rank = lower + (ties - 1) / 2.0
            percentile = (rank / (n - 1)) * 100.0
            # Clamp to [0, 100] defensively in case of floating-point
            # drift on extreme values.
            if percentile < 0.0:
                percentile = 0.0
            elif percentile > 100.0:
                percentile = 100.0
            new_rows[idx][f"{metric}{_PERCENTILE_SUFFIX}"] = round(
                percentile, 1
            )

    return new_rows


__all__ = ["IndustryService"]
