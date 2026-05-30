"""ScreenService -- 多因子选股 (design Algorithm 5).

Implements the ``ScreenService`` half of *design.md* Component 3
(Service Layer) for task 16.1. The pipeline mirrors design Algorithm
5:

1. **Universe selection** -- 行业候选 ∩ A股全集. The full A股 spot
   snapshot is pulled once via ``ak.stock_zh_a_spot_em``; when
   ``criteria.industry`` is set, the universe is intersected with
   the union of constituents reported by
   ``ak.stock_board_industry_cons_em`` for each industry name.
2. **Range filtering** -- every row is checked against
   ``criteria.{pe_ttm, pb, roe, market_cap, revenue_growth}``.
   Rows whose value for an *active* filter is ``None`` / NaN fail
   that filter (Property 10 -- closure: every output passes all
   criteria).
3. **Stable sort** -- ``sort_by`` ∈ :data:`SUPPORTED_FIELDS`; equal
   values keep universe order (Property 12 -- stable sort).
4. **Truncate** -- ``len(result) ≤ limit`` (Property 11).

Caching uses :data:`TTL_WARM` (300s) so screen results refresh
intra-day without thrashing the upstream. Rate limiting and
fallback follow the same pattern as the other services.

v1 limitation
-------------

The spot endpoint exposes ``市盈率-动态``, ``市净率`` and
``总市值`` directly, so ``pe_ttm`` / ``pb`` / ``market_cap`` are
populated. ``roe`` and ``revenue_growth`` are *not* on the spot
payload -- per-symbol indicator calls would balloon the request
budget for a 5 000-stock universe. v1 leaves those metrics as
``None`` in the resulting :class:`ScreenHit.fields`; this means a
caller that filters by ``roe`` / ``revenue_growth`` will receive an
empty result (rows with ``None`` for an active filter fail the
filter by construction). A future iteration may join a per-symbol
financial-indicator pull or a precomputed bulk dataset.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Final, TypeVar, cast

import pandas as pd

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.adapters.fallback import fetch_with_fallback
from china_stock_mcp.cache import (
    TTL_WARM,
    Cache,
    cache_get_or_fetch,
    get_default_cache,
)
from china_stock_mcp.exceptions import (
    ChinaStockMCPError,
    DataSourceError,
    NetworkError,
    RateLimitError,
    ValidationError,
)
from china_stock_mcp.models import RangeFilter, ScreenCriteria, ScreenHit
from china_stock_mcp.rate_limiter import RateLimiter, get_default_rate_limiter

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Cache schema version for screen payloads. Bump whenever the
#: :class:`ScreenHit` shape or universe-construction logic changes so
#: old entries are invalidated (Requirement 11.3, Property 4).
_SCREEN_SCHEMA_VERSION: Final[int] = 1

#: Cache "tool" namespace for screen payloads.
_SCREEN_TOOL_NAMESPACE: Final[str] = "screen_service.filter"

#: Sentinel used as ``symbol`` in the cache key. The screen result
#: depends on the criteria + sort + limit, not on a single symbol, so
#: a fixed sentinel keeps the cache key shape uniform with other
#: services without polluting the symbol component.
_SCREEN_CACHE_SYMBOL: Final[str] = "_universe"

#: ``sort_by`` keys recognised by the service. Must align with
#: :class:`ScreenHit.fields` keys populated by the universe builder.
SUPPORTED_FIELDS: Final[frozenset[str]] = frozenset(
    {"pe_ttm", "pb", "roe", "market_cap", "revenue_growth"}
)

#: ``limit`` lower / upper bounds (Requirement 8.2).
_MIN_LIMIT: Final[int] = 1
_MAX_LIMIT: Final[int] = 200

#: A 股 6-digit code prefixes → exchange suffix. Matches the table in
#: :mod:`china_stock_mcp.adapters.akshare_adapter` so screened codes
#: render as :class:`ScreenHit.code` (which validates the standardized
#: shape via the model regex).
_A_STOCK_SH_PREFIXES: Final[frozenset[str]] = frozenset({"60", "68", "90"})
_A_STOCK_SZ_PREFIXES: Final[frozenset[str]] = frozenset({"00", "30", "20"})
_A_STOCK_BJ_FIRST_CHAR: Final[str] = "8"

#: Substrings inside an exception ``str`` that indicate rate limiting.
#: Mirrors the akshare adapter's local helper so the screen service
#: stays self-contained (no cross-module import of private helpers).
_RATE_LIMIT_PATTERNS: Final[tuple[str, ...]] = (
    "too many requests",
    "rate limit",
    "rate-limit",
    "429",
    "请求过于频繁",
    "访问频率过快",
)

#: Type names that indicate a transient network failure.
_NETWORK_EXC_NAMES: Final[frozenset[str]] = frozenset(
    {
        "ConnectionError",
        "ConnectTimeout",
        "ReadTimeout",
        "Timeout",
        "TimeoutException",
        "ConnectError",
        "URLError",
        "HTTPError",
        "RemoteDisconnected",
        "ChunkedEncodingError",
        "ProxyError",
    }
)


def _is_network_exception(exc: BaseException) -> bool:
    return any(base.__name__ in _NETWORK_EXC_NAMES for base in type(exc).__mro__)


def _is_rate_limit_exception(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(pattern.lower() in msg for pattern in _RATE_LIMIT_PATTERNS)


def _call_akshare_safe(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Call ``func`` and translate exceptions to :class:`ChinaStockMCPError`.

    Mirrors :func:`china_stock_mcp.adapters.akshare_adapter._call_akshare`
    so the screen service can call ``akshare`` directly without depending
    on an adapter helper that may not exist on every adapter
    implementation.
    """

    try:
        return func(*args, **kwargs)
    except ChinaStockMCPError:
        raise
    except Exception as exc:
        if _is_rate_limit_exception(exc):
            raise RateLimitError(f"akshare 调用频率受限: {exc}") from exc
        if _is_network_exception(exc):
            raise NetworkError(f"akshare 网络错误: {exc}") from exc
        raise DataSourceError(f"akshare error: {exc}") from exc


def _a_stock_suffix(code: str) -> str | None:
    """Return ``SH`` / ``SZ`` / ``BJ`` for a bare 6-digit A 股 code."""

    if len(code) != 6 or not code.isdigit():
        return None
    prefix = code[:2]
    if prefix in _A_STOCK_SH_PREFIXES:
        return "SH"
    if prefix in _A_STOCK_SZ_PREFIXES:
        return "SZ"
    if code[0] == _A_STOCK_BJ_FIRST_CHAR:
        return "BJ"
    return None


def _safe_optional_float(value: Any) -> float | None:
    """Convert ``value`` to ``float`` or ``None`` on NaN / failure."""

    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


# ---------------------------------------------------------------------------
# ScreenService
# ---------------------------------------------------------------------------


class ScreenService:
    """多因子选股 service.

    Parameters
    ----------
    primary:
        Primary :class:`BaseAdapter`; carried for symmetry with other
        services and for ``adapter.name`` logging on fallback. The
        spot data is pulled directly via ``akshare`` (not through the
        adapter) because :class:`BaseAdapter` does not expose a
        full-universe screen endpoint.
    fallback:
        Optional backup adapter. Currently unused (the screen path
        does not flow through ``fetch_with_fallback`` for the spot
        pull) but accepted for forward compatibility with task 19.3.
    cache:
        Optional :class:`Cache` injection. Defaults to the process-wide
        instance returned by :func:`get_default_cache`.
    rate_limiter:
        Optional :class:`RateLimiter` injection. Defaults to the
        process-wide instance from :func:`get_default_rate_limiter`.
    """

    __slots__ = ("_ak", "_cache", "_fallback", "_primary", "_rate_limiter")

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
        # Lazy ``akshare`` import bound on the instance so tests can
        # monkeypatch ``service._ak`` without monkeypatching the module.
        self._ak: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self,
        criteria: ScreenCriteria,
        sort_by: str,
        order: str,
        limit: int,
    ) -> list[ScreenHit]:
        """Return ranked :class:`ScreenHit` rows matching ``criteria``.

        Pipeline (design Algorithm 5):

        1. Validate ``limit`` ∈ ``[1, 200]``, ``order`` ∈
           ``{"asc", "desc"}``, ``sort_by`` ∈ :data:`SUPPORTED_FIELDS`.
        2. Read-through cache keyed by the full criteria + sort +
           limit with :data:`TTL_WARM`.
        3. On a miss, acquire one rate-limit token, build the universe
           via :func:`fetch_with_fallback`, apply range filters, sort
           stably and truncate to ``limit``.

        Raises
        ------
        ValidationError
            On any boundary violation.
        ChinaStockMCPError
            Any subclass raised while pulling spot / industry data is
            propagated verbatim.
        """

        # 1) Validate inputs (Requirements 8.2 / 8.3 / 8.5).
        self._validate_inputs(sort_by, order, limit)

        # 2) Build cache params -- include every screening parameter so
        #    different invocations get different keys and the cache key
        #    is invariant under criteria-dict key ordering (the
        #    canonical JSON serializer in ``cache.make_key`` handles
        #    this; we emit a stable shape here).
        params: dict[str, Any] = {
            "criteria": _criteria_to_params(criteria),
            "sort_by": sort_by,
            "order": order,
            "limit": limit,
        }

        return cache_get_or_fetch(
            tool=_SCREEN_TOOL_NAMESPACE,
            symbol=_SCREEN_CACHE_SYMBOL,
            params=params,
            ttl=TTL_WARM,
            fetcher=lambda: self._build_screen(criteria, sort_by, order, limit),
            schema_version=_SCREEN_SCHEMA_VERSION,
            cache=self._cache,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(sort_by: str, order: str, limit: int) -> None:
        # ``bool`` is a subclass of ``int`` in Python; reject explicitly.
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValidationError(
                f"limit 必须是 int 类型, 实际收到 {type(limit).__name__}"
            )
        if limit < _MIN_LIMIT or limit > _MAX_LIMIT:
            raise ValidationError(
                f"limit 必须在 [{_MIN_LIMIT}, {_MAX_LIMIT}] 之间, "
                f"实际收到 {limit}"
            )
        if order not in {"asc", "desc"}:
            raise ValidationError(
                f"order 必须是 'asc' 或 'desc', 实际收到 {order!r}"
            )
        if sort_by not in SUPPORTED_FIELDS:
            raise ValidationError(
                f"sort_by 不是支持的字段: {sort_by!r}, "
                f"支持的字段: {sorted(SUPPORTED_FIELDS)}"
            )

    def _build_screen(
        self,
        criteria: ScreenCriteria,
        sort_by: str,
        order: str,
        limit: int,
    ) -> list[ScreenHit]:
        """Universe → filter → sort → truncate (Algorithm 5)."""

        self._rate_limiter.acquire()

        # Pull universe via fetch_with_fallback so transient failures
        # follow the same Properties 5 / 6 / 13.x contract as other
        # services. The fallback path simply re-runs the same builder
        # against the (currently unused) ``self._fallback`` adapter so
        # we keep the API uniform when task 19.3 wires a real backup.
        primary_name = self._primary.name
        fallback_name = (
            self._fallback.name if self._fallback is not None else "none"
        )

        def _primary_call() -> list[ScreenHit]:
            return self._fetch_universe(criteria)

        def _fallback_call() -> list[ScreenHit]:
            # The screen service does not currently route through a
            # backup adapter; re-running the same builder is a safe
            # placeholder until task 19.3 introduces a real backup.
            return self._fetch_universe(criteria)

        universe: list[ScreenHit] = fetch_with_fallback(
            primary=_primary_call,
            fallback=_fallback_call if self._fallback is not None else None,
            primary_name=primary_name,
            fallback_name=fallback_name,
        )

        # 2) Range filtering (Property 10 -- closure).
        filtered = [hit for hit in universe if _match_range(hit, criteria)]

        # 3) Stable sort by ``sort_by`` (Property 12).
        reverse = order == "desc"
        # Rows whose ``sort_by`` value is missing sink to the end of
        # the result regardless of direction, keeping the comparable
        # rows in their natural order at the front.
        filtered.sort(
            key=lambda h: _sort_key(h.fields.get(sort_by), reverse),
        )

        # 4) Truncate (Property 11).
        if len(filtered) > limit:
            filtered = filtered[:limit]

        return filtered

    def _fetch_universe(self, criteria: ScreenCriteria) -> list[ScreenHit]:
        """Pull A股 spot snapshot ∩ optional industry filter."""

        ak = self._get_ak()
        df = _call_akshare_safe(ak.stock_zh_a_spot_em)
        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty:
            return []

        # Restrict to the requested industries (if any) before any
        # row-level work. A single missing industry should not abort
        # the screen -- log + continue so the caller still gets a
        # partial answer when the upstream classification is patchy.
        allowed_codes: set[str] | None = None
        if criteria.industry:
            allowed_codes = self._fetch_industry_constituent_codes(
                ak, criteria.industry
            )
            if not allowed_codes:
                return []

        return _project_universe(df, allowed_codes)

    def _fetch_industry_constituent_codes(
        self,
        ak: Any,
        industries: list[str],
    ) -> set[str]:
        """Pull constituent bare-codes for every requested industry.

        Industries that fail upstream (empty / missing) are silently
        skipped so a single bad industry name does not abort the
        whole screen. The returned set is the *union* of constituent
        codes across all requested industries.
        """

        codes: set[str] = set()
        for industry_name in industries:
            name = (industry_name or "").strip()
            if not name:
                continue
            try:
                df = _call_akshare_safe(
                    ak.stock_board_industry_cons_em, symbol=name
                )
            except (NetworkError, RateLimitError):
                # Transient failures are surfaced -- the outer
                # fetch_with_fallback handles the retry policy.
                raise
            except DataSourceError:
                # An unknown industry name (or an upstream payload
                # change) yields a DataSourceError; skip and continue.
                continue
            df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
            if df.empty or "代码" not in df.columns:
                continue
            for raw in df["代码"].astype(str):
                bare = raw.strip()
                if bare.isdigit() and len(bare) == 6:
                    codes.add(bare)
        return codes

    def _get_ak(self) -> Any:
        """Lazy-import ``akshare`` and cache it on the instance."""

        if self._ak is None:
            import akshare as ak

            self._ak = ak
        return self._ak


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _criteria_to_params(criteria: ScreenCriteria) -> dict[str, Any]:
    """Serialize ``criteria`` to a JSON-safe dict for cache keying."""

    out: dict[str, Any] = {}
    for field in ("pe_ttm", "pb", "roe", "market_cap", "revenue_growth"):
        rng: RangeFilter | None = getattr(criteria, field)
        if rng is None:
            continue
        out[field] = {"min": rng.min, "max": rng.max}
    if criteria.industry:
        out["industry"] = list(criteria.industry)
    # Surface any caller-supplied extras so two distinct extras maps
    # do not share a cache slot. ``ScreenCriteria`` allows extras via
    # ``extra="allow"``; ``model_extra`` exposes them on Pydantic v2.
    extras = getattr(criteria, "model_extra", None) or {}
    if extras:
        out["_extras"] = dict(extras)
    return out


def _project_universe(
    df: pd.DataFrame,
    allowed_codes: set[str] | None,
) -> list[ScreenHit]:
    """Project the spot dataframe into :class:`ScreenHit` rows.

    Universe order matches the upstream order, which is what
    :class:`ScreenService.filter` relies on for stable sorting
    (Property 12).
    """

    code_col = _pick_column(df, ("代码", "code", "symbol"))
    name_col = _pick_column(df, ("名称", "name"))
    if code_col is None or name_col is None:
        return []

    pe_col = _pick_column(df, ("市盈率-动态", "市盈率(动态)", "pe_dynamic"))
    pb_col = _pick_column(df, ("市净率", "pb"))
    mcap_col = _pick_column(df, ("总市值", "market_cap"))

    rows: list[ScreenHit] = []
    for record in df.to_dict(orient="records"):
        bare = str(record.get(code_col, "")).strip()
        if not bare:
            continue
        suffix = _a_stock_suffix(bare)
        if suffix is None:
            continue
        if allowed_codes is not None and bare not in allowed_codes:
            continue

        name = str(record.get(name_col, "")).strip()
        if not name:
            continue

        fields: dict[str, float] = {}
        pe_val = (
            _safe_optional_float(record.get(pe_col)) if pe_col else None
        )
        if pe_val is not None:
            fields["pe_ttm"] = pe_val
        pb_val = (
            _safe_optional_float(record.get(pb_col)) if pb_col else None
        )
        if pb_val is not None:
            fields["pb"] = pb_val
        mcap_val = (
            _safe_optional_float(record.get(mcap_col)) if mcap_col else None
        )
        if mcap_val is not None:
            fields["market_cap"] = mcap_val

        # ``roe`` / ``revenue_growth`` are not on the spot endpoint;
        # see module docstring for the v1 limitation.

        rows.append(
            ScreenHit(
                code=f"{bare}.{suffix}",
                name=name,
                industry="",  # spot endpoint does not surface industry
                fields=fields,
            )
        )

    return rows


def _pick_column(
    df: pd.DataFrame, candidates: Iterable[str]
) -> str | None:
    """Return the first column from *candidates* present in ``df``."""

    for name in candidates:
        if name in df.columns:
            return name
    return None


def _match_range(hit: ScreenHit, criteria: ScreenCriteria) -> bool:
    """Return ``True`` if ``hit`` passes every active range filter.

    Property 10 (closure): a row whose value is missing for an
    *active* filter fails that filter -- we do not let nullable
    fields sneak past a constraint the caller explicitly set.
    """

    for field in ("pe_ttm", "pb", "roe", "market_cap", "revenue_growth"):
        rng: RangeFilter | None = getattr(criteria, field)
        if rng is None:
            continue
        value = hit.fields.get(field)
        if value is None:
            return False
        if rng.min is not None and value < rng.min:
            return False
        if rng.max is not None and value > rng.max:
            return False
    return True


def _sort_key(value: float | None, reverse: bool) -> tuple[int, float]:
    """Build a stable sort key that sinks ``None`` to the end.

    Returns a ``(missing_flag, ordered_value)`` tuple. ``missing_flag``
    is ``0`` for present values and ``1`` for missing, so missing
    rows always sort *after* present ones. The ``ordered_value``
    component is negated when ``reverse`` is true so a single
    ``list.sort`` call (which is stable) implements both ascending
    and descending order without the ``reverse=True`` flag, which
    would otherwise bubble missing rows to the *front* under
    descending order.
    """

    if value is None:
        return (1, 0.0)
    return (0, -float(value) if reverse else float(value))


__all__ = ["SUPPORTED_FIELDS", "ScreenService"]
