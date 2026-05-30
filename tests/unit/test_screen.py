"""Property tests for :mod:`china_stock_mcp.services.screen_service`.

Covers tasks 16.2, 16.3 and 16.4 of the china-stock-mcp spec:

- 16.2 Property 10 -- 选股闭包 (Validates: Requirements 8.1)
- 16.3 Property 11 -- 选股上限 (Validates: Requirements 8.2)
- 16.4 Property 12 -- 排序稳定性 (Validates: Requirements 8.3)

Each test injects a stub :class:`ScreenService` whose
``_fetch_universe`` method returns a hand-rolled list of
:class:`ScreenHit` rows so the upstream ``akshare`` call is never
exercised. The cache and rate limiter are replaced with hermetic
in-memory stand-ins so every hypothesis example runs in microseconds
without polluting the process-wide cache.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.cache import Cache, make_key, reset_default_cache
from china_stock_mcp.models import (
    FinancialReport,
    FundamentalSnapshot,
    FundInfo,
    KLineSeries,
    MarketOverview,
    MoneyFlow,
    PeerTable,
    Quote,
    RangeFilter,
    ScreenCriteria,
    ScreenHit,
    SymbolHit,
)
from china_stock_mcp.rate_limiter import RateLimiter
from china_stock_mcp.services.screen_service import (
    ScreenService,
    _match_range,
)

# ---------------------------------------------------------------------------
# In-memory cache + stub adapter (StubAdapter pattern from
# tests/integration/test_search_quote_flow.py)
# ---------------------------------------------------------------------------


class _InMemoryCache:
    """Hermetic :class:`Cache` implementation backed by a plain dict.

    Avoids the disk-cache I/O cost so each hypothesis example runs in
    microseconds. TTL is recorded but not enforced because every test
    method starts with a fresh cache instance.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self._store.get(key)

    def set(self, key: str, value: Any, ttl: int) -> None:
        if ttl <= 0:
            raise ValueError(f"ttl must be > 0, got {ttl}")
        self._store[key] = value

    def make_key(
        self,
        tool: str,
        symbol: str,
        params: Any,
        schema_version: int,
    ) -> str:
        return make_key(tool, symbol, params, schema_version)

    def close(self) -> None:
        self._store.clear()


class _StubAdapter(BaseAdapter):
    """:class:`BaseAdapter` whose methods raise :class:`NotImplementedError`.

    The screen service only reads ``primary.name`` for fallback logging
    and pulls universe data via its own ``akshare`` import that we
    monkeypatch on the subclass below; no adapter method is actually
    invoked during these tests.
    """

    name: str = "stub"

    def search(self, query: str, market: str) -> list[SymbolHit]:
        raise NotImplementedError

    def quote(self, symbols: list[str]) -> list[Quote]:
        raise NotImplementedError

    def kline(
        self, symbol: str, period: str, count: int, adjust: str
    ) -> KLineSeries:
        raise NotImplementedError

    def fundamentals(self, symbol: str) -> FundamentalSnapshot:
        raise NotImplementedError

    def financial_report(
        self, symbol: str, report_type: str, periods: int
    ) -> FinancialReport:
        raise NotImplementedError

    def money_flow(
        self, symbol: str | None, flow_type: str, top_n: int
    ) -> MoneyFlow:
        raise NotImplementedError

    def industry_peers(
        self, symbol: str, metrics: list[str], top_n: int
    ) -> PeerTable:
        raise NotImplementedError

    def fund_info(self, fund_code: str) -> FundInfo:
        raise NotImplementedError

    def market_overview(self) -> MarketOverview:
        raise NotImplementedError


class _StubScreenService(ScreenService):
    """:class:`ScreenService` whose universe is a fixed list.

    Subclassing :class:`ScreenService` (which declares ``__slots__``)
    without re-declaring ``__slots__`` gives the subclass a regular
    ``__dict__``, which is exactly what we need to attach the canned
    universe without modifying the production class.
    """

    def __init__(
        self,
        universe: list[ScreenHit],
        *,
        cache: Cache,
        rate_limiter: RateLimiter,
    ) -> None:
        super().__init__(
            primary=_StubAdapter(),
            cache=cache,
            rate_limiter=rate_limiter,
        )
        # Subclass without ``__slots__`` is allowed to use ``__dict__``.
        self._stub_universe: list[ScreenHit] = list(universe)

    def _fetch_universe(self, criteria: ScreenCriteria) -> list[ScreenHit]:
        # Mirror the production "industry filter" semantics: when the
        # caller specifies industries, only keep rows whose industry
        # appears in the requested list. We intentionally do not match
        # constituent-code unions because the stub universe carries
        # explicit ``industry`` strings -- enough to exercise both
        # branches of the universe builder.
        if not criteria.industry:
            return list(self._stub_universe)
        wanted = {name for name in criteria.industry if name}
        return [hit for hit in self._stub_universe if hit.industry in wanted]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_default_cache() -> Iterator[None]:
    """Each test starts and ends with a fresh process-wide cache."""

    reset_default_cache()
    try:
        yield
    finally:
        reset_default_cache()


@pytest.fixture()
def cache() -> _InMemoryCache:
    return _InMemoryCache()


@pytest.fixture()
def rate_limiter() -> RateLimiter:
    """Generous limiter so hypothesis runs never exhaust the budget."""

    return RateLimiter(rate_per_minute=1_000_000)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Use a short, controlled value range so generated criteria intersect
# the universe non-trivially: half the rows pass, half fail.
_FIELD_NAMES = ("pe_ttm", "pb", "roe", "market_cap", "revenue_growth")
_VALUE_STRATEGY = st.floats(
    min_value=-100.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


def _bare_code(index: int) -> str:
    """Generate a deterministic bare 6-digit A 股 code (SZ).

    The standardized symbol pattern enforced by :class:`ScreenHit`
    requires ``\\d{6}\\.(SH|SZ|BJ)`` for A 股 codes. Using a stable
    deterministic generator keeps the universe ordering predictable
    and the codes unique across the universe.
    """

    return f"{index % 1_000_000:06d}.SZ"


@st.composite
def _screen_hit(
    draw: st.DrawFn,
    *,
    index: int,
    industry: str | None = None,
    fixed_value: tuple[str, float] | None = None,
) -> ScreenHit:
    """Generate a :class:`ScreenHit` whose ``fields`` cover all metrics.

    ``fixed_value`` lets a caller pin a single field to a specific
    value -- used by Property 12 to construct rows with equal
    ``sort_by`` values.
    """

    fields: dict[str, float] = {}
    for name in _FIELD_NAMES:
        # Every field is populated so range filters always have data
        # to compare against (Property 10 closure semantics include
        # the "missing value fails an active filter" branch, but we
        # exercise that branch separately via _match_range unit
        # checks; here we focus on the through-pipeline behavior).
        fields[name] = draw(_VALUE_STRATEGY)
    if fixed_value is not None:
        field, value = fixed_value
        fields[field] = value
    return ScreenHit(
        code=_bare_code(index),
        name=f"S{index:06d}",
        industry=industry if industry is not None else "测试行业",
        fields=fields,
    )


@st.composite
def _universe(draw: st.DrawFn, *, max_size: int = 25) -> list[ScreenHit]:
    """Generate a :class:`ScreenHit` universe with unique codes."""

    size = draw(st.integers(min_value=0, max_value=max_size))
    rows: list[ScreenHit] = []
    for i in range(size):
        rows.append(draw(_screen_hit(index=i)))
    return rows


@st.composite
def _range_filter(draw: st.DrawFn) -> RangeFilter:
    """Generate a non-empty :class:`RangeFilter` (at least one bound)."""

    has_min = draw(st.booleans())
    has_max = draw(st.booleans())
    if not has_min and not has_max:
        # Force at least one bound so the filter is "active".
        has_max = True
    lo = draw(_VALUE_STRATEGY) if has_min else None
    hi = draw(_VALUE_STRATEGY) if has_max else None
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    return RangeFilter(min=lo, max=hi)


@st.composite
def _criteria(draw: st.DrawFn) -> ScreenCriteria:
    """Generate a :class:`ScreenCriteria` with 0..3 active range filters."""

    selected = draw(
        st.lists(
            st.sampled_from(_FIELD_NAMES),
            min_size=0,
            max_size=3,
            unique=True,
        )
    )
    payload: dict[str, Any] = {}
    for name in selected:
        payload[name] = draw(_range_filter())
    return ScreenCriteria(**payload)


# ---------------------------------------------------------------------------
# Property 10 -- 选股闭包 (task 16.2, Validates: Requirements 8.1)
# ---------------------------------------------------------------------------


class TestScreenClosure:
    """**Validates: Requirements 8.1** (design Property 10).

    For every hit returned by :meth:`ScreenService.filter`, every
    *active* range filter declared in ``criteria`` SHALL pass when
    re-checked via :func:`_match_range`.
    """

    @given(universe=_universe(), criteria=_criteria())
    @settings(
        max_examples=80,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_every_hit_passes_every_active_filter(
        self,
        universe: list[ScreenHit],
        criteria: ScreenCriteria,
    ) -> None:
        cache = _InMemoryCache()
        rate_limiter = RateLimiter(rate_per_minute=1_000_000)
        service = _StubScreenService(
            universe, cache=cache, rate_limiter=rate_limiter
        )

        hits = service.filter(
            criteria=criteria,
            sort_by="pe_ttm",
            order="desc",
            limit=200,
        )

        # Every returned hit must satisfy every active range filter.
        for hit in hits:
            assert _match_range(hit, criteria), (
                f"hit {hit.code} fields={hit.fields} violates criteria "
                f"{criteria.model_dump(exclude_none=True)}"
            )


# ---------------------------------------------------------------------------
# Property 11 -- 选股上限 (task 16.3, Validates: Requirements 8.2)
# ---------------------------------------------------------------------------


class TestScreenLimit:
    """**Validates: Requirements 8.2** (design Property 11).

    For any ``limit ∈ [1, 200]`` and any universe size, the
    :meth:`ScreenService.filter` result SHALL satisfy
    ``len(result) ≤ limit``.
    """

    @given(
        universe_size=st.integers(min_value=0, max_value=80),
        limit=st.integers(min_value=1, max_value=200),
    )
    @settings(
        max_examples=80,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_result_length_never_exceeds_limit(
        self, universe_size: int, limit: int
    ) -> None:
        # Build a deterministic universe of the requested size; values
        # are uniform so no row is filtered out. With an empty
        # criteria the entire universe is eligible, so the truncation
        # branch is exercised whenever ``universe_size > limit``.
        universe = [
            ScreenHit(
                code=_bare_code(i),
                name=f"S{i:06d}",
                industry="x",
                fields={
                    "pe_ttm": float(i),
                    "pb": float(i),
                    "roe": float(i),
                    "market_cap": float(i),
                    "revenue_growth": float(i),
                },
            )
            for i in range(universe_size)
        ]

        cache = _InMemoryCache()
        rate_limiter = RateLimiter(rate_per_minute=1_000_000)
        service = _StubScreenService(
            universe, cache=cache, rate_limiter=rate_limiter
        )

        hits = service.filter(
            criteria=ScreenCriteria(),
            sort_by="market_cap",
            order="desc",
            limit=limit,
        )

        assert len(hits) <= limit
        # Sanity: when the universe fits inside the limit, every row
        # is returned (no spurious dropping).
        if universe_size <= limit:
            assert len(hits) == universe_size


# ---------------------------------------------------------------------------
# Property 12 -- 排序稳定性 (task 16.4, Validates: Requirements 8.3)
# ---------------------------------------------------------------------------


class TestScreenSortStability:
    """**Validates: Requirements 8.3** (design Property 12).

    Two rows with equal ``sort_by`` values SHALL preserve their
    relative order from the input universe, regardless of ``order``.
    The universe is constructed deterministically from a hypothesis-
    generated *grouping*: each item is assigned a "bucket" value drawn
    from a tiny set so equal-valued runs are guaranteed.
    """

    @given(
        # ``buckets`` indexes into a small fixed value table; each
        # entry seeds one row in the universe. Drawing from a small
        # set forces frequent ties on the sort field, which is the
        # whole point of Property 12.
        buckets=st.lists(
            st.integers(min_value=0, max_value=2),
            min_size=2,
            max_size=15,
        ),
        order=st.sampled_from(["asc", "desc"]),
    )
    @settings(
        max_examples=80,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_equal_sort_values_preserve_input_order(
        self,
        buckets: list[int],
        order: str,
    ) -> None:
        # Tiny value table -- three buckets ensure 3-way ties.
        bucket_values = (10.0, 20.0, 30.0)

        universe: list[ScreenHit] = []
        for i, bucket in enumerate(buckets):
            value = bucket_values[bucket]
            universe.append(
                ScreenHit(
                    code=_bare_code(i),
                    name=f"S{i:06d}",
                    industry="x",
                    fields={
                        "pe_ttm": value,
                        "pb": float(i),
                        "roe": float(i),
                        "market_cap": float(i),
                        "revenue_growth": float(i),
                    },
                )
            )

        cache = _InMemoryCache()
        rate_limiter = RateLimiter(rate_per_minute=1_000_000)
        service = _StubScreenService(
            universe, cache=cache, rate_limiter=rate_limiter
        )

        hits = service.filter(
            criteria=ScreenCriteria(),
            sort_by="pe_ttm",
            order=order,
            limit=200,
        )

        # Closure sanity: every input row must come back (no row is
        # dropped, since the criteria is empty).
        assert len(hits) == len(universe)

        # Group hits by their ``pe_ttm`` value and assert that within
        # each group the codes appear in the same order they were
        # inserted into the universe.
        input_index = {hit.code: i for i, hit in enumerate(universe)}
        per_bucket_indexes: dict[float, list[int]] = {}
        for hit in hits:
            value = hit.fields["pe_ttm"]
            per_bucket_indexes.setdefault(value, []).append(
                input_index[hit.code]
            )

        for value, idxs in per_bucket_indexes.items():
            assert idxs == sorted(idxs), (
                f"bucket {value} lost stability: indexes={idxs}"
            )

        # Cross-bucket ordering follows ``order`` -- ascending puts
        # the smallest bucket first; descending puts the largest first.
        seen_buckets: list[float] = []
        for hit in hits:
            value = hit.fields["pe_ttm"]
            if not seen_buckets or seen_buckets[-1] != value:
                seen_buckets.append(value)
        if order == "asc":
            assert seen_buckets == sorted(seen_buckets)
        else:
            assert seen_buckets == sorted(seen_buckets, reverse=True)
