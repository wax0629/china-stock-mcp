"""Unit tests for :mod:`china_stock_mcp.services.fundamental_service`.

Covers task 11.2 in ``.kiro/specs/china-stock-mcp/tasks.md``:

- Requirement 4.1 -- 估值 / 盈利 / 成长 / 健康 四组指标完整性: each
  bucket on the returned :class:`FundamentalSnapshot` matches the
  upstream payload field-for-field after the service round-trip.
- Requirement 4.2 -- ``industry_percentile`` 取值范围 ``[0, 100]``:
  the service preserves boundary values exactly as the DTO enforces
  them.
- v1 limitation -- :meth:`FundamentalService.industry_percentile` is
  a stub that always raises :class:`DataNotFoundError`.
- Read-through cache: a cache miss invokes the adapter exactly once;
  a follow-up call hits the cache and never re-invokes the adapter.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.cache import Cache, build_cache, reset_default_cache
from china_stock_mcp.config import Settings
from china_stock_mcp.exceptions import DataNotFoundError
from china_stock_mcp.models import (
    FinancialReport,
    FundamentalSnapshot,
    FundInfo,
    KLineSeries,
    MarketOverview,
    MoneyFlow,
    PeerTable,
    Quote,
    SymbolHit,
)
from china_stock_mcp.rate_limiter import RateLimiter
from china_stock_mcp.services.fundamental_service import FundamentalService

# ---------------------------------------------------------------------------
# Stub adapter
# ---------------------------------------------------------------------------


class StubAdapter(BaseAdapter):
    """Hermetic :class:`BaseAdapter` exposing only ``fundamentals``.

    Records call counts so tests can assert that caching short-circuits
    a second invocation. Methods unrelated to ``fundamentals`` raise
    :class:`NotImplementedError` because task 11.2 only exercises the
    fundamental snapshot pipeline.
    """

    name: str = "stub"

    def __init__(self, snapshot: FundamentalSnapshot) -> None:
        self._snapshot = snapshot
        self.fundamentals_call_count: int = 0
        self.last_fundamentals_call: str | None = None

    # ----- exercised by tests --------------------------------------------

    def fundamentals(self, symbol: str) -> FundamentalSnapshot:
        self.fundamentals_call_count += 1
        self.last_fundamentals_call = symbol
        return self._snapshot

    # ----- not implemented in 11.2 ---------------------------------------

    def search(self, query: str, market: str) -> list[SymbolHit]:
        raise NotImplementedError("StubAdapter does not implement search")

    def quote(self, symbols: list[str]) -> list[Quote]:
        raise NotImplementedError("StubAdapter does not implement quote")

    def kline(
        self,
        symbol: str,
        period: str,
        count: int,
        adjust: str,
    ) -> KLineSeries:
        raise NotImplementedError("StubAdapter does not implement kline")

    def financial_report(
        self,
        symbol: str,
        report_type: str,
        periods: int,
    ) -> FinancialReport:
        raise NotImplementedError("StubAdapter does not implement financial_report")

    def money_flow(
        self,
        symbol: str | None,
        flow_type: str,
        top_n: int,
    ) -> MoneyFlow:
        raise NotImplementedError("StubAdapter does not implement money_flow")

    def industry_peers(
        self,
        symbol: str,
        metrics: list[str],
        top_n: int,
    ) -> PeerTable:
        raise NotImplementedError("StubAdapter does not implement industry_peers")

    def fund_info(self, fund_code: str) -> FundInfo:
        raise NotImplementedError("StubAdapter does not implement fund_info")

    def market_overview(self) -> MarketOverview:
        raise NotImplementedError("StubAdapter does not implement market_overview")


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
def cache(tmp_path: Path) -> Iterator[Cache]:
    """Disk-backed :class:`Cache` rooted at ``tmp_path``."""

    backend = build_cache(Settings(cache_backend="disk", cache_dir=tmp_path))
    try:
        yield backend
    finally:
        backend.close()


@pytest.fixture()
def rate_limiter() -> RateLimiter:
    """Generous limiter so tests never trip the budget accidentally."""

    return RateLimiter(rate_per_minute=10_000)


# ---------------------------------------------------------------------------
# Snapshot helper
# ---------------------------------------------------------------------------


def _make_full_snapshot(
    *,
    symbol: str = "300750.SZ",
    industry_percentile: dict[str, float] | None = None,
) -> FundamentalSnapshot:
    """Build a :class:`FundamentalSnapshot` with all four buckets populated."""

    return FundamentalSnapshot(
        symbol=symbol,
        valuation={
            "pe_ttm": 18.5,
            "pe_dynamic": 17.2,
            "pb": 3.4,
            "ps": 2.1,
            "peg": 1.0,
        },
        profitability={
            "roe": 22.5,
            "roa": 12.4,
            "gross_margin": 35.6,
            "net_margin": 18.7,
        },
        growth={
            "revenue_yoy": 30.5,
            "net_profit_yoy": 45.6,
            "qoq": 5.2,
        },
        health={
            "debt_ratio": 45.0,
            "current_ratio": 1.8,
            "ocf_to_net_profit": 1.05,
        },
        industry_percentile=(
            industry_percentile
            if industry_percentile is not None
            else {"pe_ttm": 60.0, "roe": 85.0}
        ),
    )


# ---------------------------------------------------------------------------
# Snapshot bucket completeness (Requirement 4.1)
# ---------------------------------------------------------------------------


class TestSnapshotBuckets:
    """**Validates: Requirements 4.1**."""

    def test_snapshot_returns_all_four_buckets(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        upstream = _make_full_snapshot()
        adapter = StubAdapter(upstream)
        service = FundamentalService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        result = service.snapshot("300750.SZ")

        # Bucket-level shape preserved.
        assert result.valuation == upstream.valuation
        assert result.profitability == upstream.profitability
        assert result.growth == upstream.growth
        assert result.health == upstream.health
        assert result.symbol == "300750.SZ"

        # Adapter received the standardized symbol.
        assert adapter.last_fundamentals_call == "300750.SZ"
        assert adapter.fundamentals_call_count == 1

    def test_snapshot_normalizes_bare_six_digit_input(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Bare ``300750`` resolves to ``300750.SZ`` before reaching the adapter."""

        upstream = _make_full_snapshot()
        adapter = StubAdapter(upstream)
        service = FundamentalService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        result = service.snapshot("300750")

        assert result.symbol == "300750.SZ"
        assert adapter.last_fundamentals_call == "300750.SZ"


# ---------------------------------------------------------------------------
# Industry percentile range (Requirement 4.2)
# ---------------------------------------------------------------------------


class TestIndustryPercentileRange:
    """**Validates: Requirements 4.2**."""

    def test_boundary_percentile_values_preserved(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """0.0 and 100.0 round-trip through the service unchanged."""

        upstream = _make_full_snapshot(
            industry_percentile={
                "pe_ttm": 0.0,
                "pb": 50.0,
                "roe": 100.0,
            }
        )
        adapter = StubAdapter(upstream)
        service = FundamentalService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        result = service.snapshot("300750.SZ")

        assert result.industry_percentile == {
            "pe_ttm": 0.0,
            "pb": 50.0,
            "roe": 100.0,
        }
        # Each value still satisfies the DTO range invariant.
        for metric, value in result.industry_percentile.items():
            assert 0.0 <= value <= 100.0, f"{metric}: {value}"

    def test_empty_percentile_dict_is_preserved(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """v1 default -- no industry percentile data, empty dict preserved."""

        upstream = _make_full_snapshot(industry_percentile={})
        adapter = StubAdapter(upstream)
        service = FundamentalService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        result = service.snapshot("300750.SZ")

        assert result.industry_percentile == {}


# ---------------------------------------------------------------------------
# industry_percentile() v1 stub
# ---------------------------------------------------------------------------


class TestIndustryPercentileStub:
    """v1 limitation -- :meth:`industry_percentile` always raises."""

    def test_industry_percentile_raises_data_not_found(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        adapter = StubAdapter(_make_full_snapshot())
        service = FundamentalService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        with pytest.raises(DataNotFoundError, match="v1 暂未实现"):
            service.industry_percentile("300750.SZ", "pe_ttm")

        # No upstream call should leak through the stub method.
        assert adapter.fundamentals_call_count == 0


# ---------------------------------------------------------------------------
# Cache hit / miss observability
# ---------------------------------------------------------------------------


class TestSnapshotCaching:
    """Read-through cache short-circuits a second adapter call."""

    def test_second_call_served_from_cache(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        upstream = _make_full_snapshot()
        adapter = StubAdapter(upstream)
        service = FundamentalService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        first = service.snapshot("300750.SZ")
        second = service.snapshot("300750.SZ")

        # Adapter was hit exactly once across both calls.
        assert adapter.fundamentals_call_count == 1
        # Cached payload returns the same bucket data.
        assert first.valuation == second.valuation
        assert first.profitability == second.profitability
        assert first.growth == second.growth
        assert first.health == second.health
        assert first.industry_percentile == second.industry_percentile

    def test_distinct_symbols_each_invoke_adapter_once(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Per-symbol cache key isolation -- different symbols = different misses."""

        # Adapter returns the same payload regardless of input; the
        # test asserts the *call count* rather than payload identity.
        upstream = _make_full_snapshot()
        adapter = StubAdapter(upstream)
        service = FundamentalService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        service.snapshot("300750.SZ")
        service.snapshot("600519.SH")
        # Repeat each to verify both warm slots short-circuit the adapter.
        service.snapshot("300750.SZ")
        service.snapshot("600519.SH")

        assert adapter.fundamentals_call_count == 2
