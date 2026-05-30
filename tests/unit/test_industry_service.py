"""Unit tests for :mod:`china_stock_mcp.services.industry_service`.

Covers task 14.2 in ``.kiro/specs/china-stock-mcp/tasks.md``:

- Requirement 6.2 / 6.5 -- ``metrics`` ⊆ ``{pe, pb, roe,
  revenue_growth}``; an empty list or an unsupported value raises
  :class:`ValidationError` whose message lists the full accepted
  set.
- Requirement 6.4 -- ``top_n`` ∈ ``[1, 50]``; out-of-range values
  raise :class:`ValidationError`.
- Requirement 6.3 -- :func:`_annotate_with_percentile` assigns the
  correct fractional ranks for known PE values, including the case
  where some rows have ``None`` (skipped from percentile
  computation).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.cache import Cache, build_cache, reset_default_cache
from china_stock_mcp.config import Settings
from china_stock_mcp.exceptions import ValidationError
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
from china_stock_mcp.services.industry_service import (
    IndustryService,
    _annotate_with_percentile,
)

# ---------------------------------------------------------------------------
# Stub adapter
# ---------------------------------------------------------------------------


class StubAdapter(BaseAdapter):
    """Hermetic :class:`BaseAdapter` exposing only ``industry_peers``."""

    name: str = "stub"

    def __init__(self, peers: PeerTable) -> None:
        self._peers = peers
        self.industry_peers_call_count: int = 0

    def industry_peers(
        self,
        symbol: str,
        metrics: list[str],
        top_n: int,
    ) -> PeerTable:
        self.industry_peers_call_count += 1
        return self._peers

    # ----- not implemented in 14.2 ---------------------------------------

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

    def fundamentals(self, symbol: str) -> FundamentalSnapshot:
        raise NotImplementedError("StubAdapter does not implement fundamentals")

    def financial_report(
        self,
        symbol: str,
        report_type: str,
        periods: int,
    ) -> FinancialReport:
        raise NotImplementedError(
            "StubAdapter does not implement financial_report"
        )

    def money_flow(
        self,
        symbol: str | None,
        flow_type: str,
        top_n: int,
    ) -> MoneyFlow:
        raise NotImplementedError("StubAdapter does not implement money_flow")

    def fund_info(self, fund_code: str) -> FundInfo:
        raise NotImplementedError("StubAdapter does not implement fund_info")

    def market_overview(self) -> MarketOverview:
        raise NotImplementedError("StubAdapter does not implement market_overview")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_default_cache() -> Iterator[None]:
    reset_default_cache()
    try:
        yield
    finally:
        reset_default_cache()


@pytest.fixture()
def cache(tmp_path: Path) -> Iterator[Cache]:
    backend = build_cache(Settings(cache_backend="disk", cache_dir=tmp_path))
    try:
        yield backend
    finally:
        backend.close()


@pytest.fixture()
def rate_limiter() -> RateLimiter:
    return RateLimiter(rate_per_minute=10_000)


def _make_peer_table() -> PeerTable:
    """Empty :class:`PeerTable` -- input validation fires before adapter."""

    return PeerTable(
        base_symbol="300750.SZ",
        industry="电池",
        metrics=[],
        rows=[],
    )


# ---------------------------------------------------------------------------
# metrics validation (Requirement 6.2 / 6.5)
# ---------------------------------------------------------------------------


class TestMetricsValidation:
    """**Validates: Requirements 6.2, 6.5**."""

    def test_unsupported_metric_lists_full_accepted_set(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        adapter = StubAdapter(_make_peer_table())
        service = IndustryService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        with pytest.raises(ValidationError) as exc_info:
            service.peers(
                symbol="300750.SZ",
                metrics=["roe", "unknown"],
                top_n=10,
            )

        message = str(exc_info.value)
        # The full accepted set must appear (Requirement 6.5).
        for accepted in ("pe", "pb", "roe", "revenue_growth"):
            assert accepted in message
        # And the offending value too.
        assert "unknown" in message
        assert adapter.industry_peers_call_count == 0

    def test_empty_metrics_list_raises_validation_error(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        adapter = StubAdapter(_make_peer_table())
        service = IndustryService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        with pytest.raises(ValidationError) as exc_info:
            service.peers(symbol="300750.SZ", metrics=[], top_n=10)

        message = str(exc_info.value)
        for accepted in ("pe", "pb", "roe", "revenue_growth"):
            assert accepted in message
        assert adapter.industry_peers_call_count == 0


# ---------------------------------------------------------------------------
# top_n validation (Requirement 6.4)
# ---------------------------------------------------------------------------


class TestTopNValidation:
    """**Validates: Requirements 6.4**."""

    def test_top_n_zero_raises_validation_error(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        adapter = StubAdapter(_make_peer_table())
        service = IndustryService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        with pytest.raises(ValidationError) as exc_info:
            service.peers(symbol="300750.SZ", metrics=["pe"], top_n=0)

        message = str(exc_info.value)
        assert "1" in message
        assert "50" in message
        assert adapter.industry_peers_call_count == 0

    def test_top_n_above_upper_bound_raises_validation_error(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        adapter = StubAdapter(_make_peer_table())
        service = IndustryService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        with pytest.raises(ValidationError) as exc_info:
            service.peers(symbol="300750.SZ", metrics=["pe"], top_n=51)

        message = str(exc_info.value)
        assert "1" in message
        assert "50" in message


# ---------------------------------------------------------------------------
# Percentile annotation (Requirement 6.3)
# ---------------------------------------------------------------------------


class TestAnnotateWithPercentile:
    """**Validates: Requirements 6.3** -- correct fractional ranks."""

    def test_three_distinct_pe_values_yield_0_50_100(self) -> None:
        """Three rows with PE ∈ {10, 20, 30} → ranks {0, 50, 100}."""

        rows: list[dict[str, object]] = [
            {"代码": "A", "pe": 10.0},
            {"代码": "B", "pe": 20.0},
            {"代码": "C", "pe": 30.0},
        ]

        annotated = _annotate_with_percentile(rows, ["pe"])

        # Original input is not mutated.
        for original, original_value in zip(rows, [10.0, 20.0, 30.0]):
            assert original["pe"] == original_value
            assert "pe_percentile" not in original

        # Ranks reflect "higher is higher percentile".
        pe_a = annotated[0]["pe_percentile"]
        pe_b = annotated[1]["pe_percentile"]
        pe_c = annotated[2]["pe_percentile"]
        assert pe_a == pytest.approx(0.0)
        assert pe_b == pytest.approx(50.0)
        assert pe_c == pytest.approx(100.0)

    def test_none_values_are_skipped_from_percentile_computation(self) -> None:
        """Rows with ``None`` for the metric have no percentile key."""

        rows: list[dict[str, object]] = [
            {"代码": "A", "pe": 10.0},
            {"代码": "B", "pe": None},
            {"代码": "C", "pe": 30.0},
        ]

        annotated = _annotate_with_percentile(rows, ["pe"])

        # Numeric peers compute against each other only -- two values,
        # 10 and 30, so they end up at 0 and 100.
        assert annotated[0]["pe_percentile"] == pytest.approx(0.0)
        assert annotated[2]["pe_percentile"] == pytest.approx(100.0)

        # The middle row stays without a percentile annotation; the
        # formatter renders ``"-"`` when the key is absent.
        assert "pe_percentile" not in annotated[1]

    def test_all_none_values_skip_metric_entirely(self) -> None:
        """No numeric values → no rows gain the percentile key."""

        rows: list[dict[str, object]] = [
            {"代码": "A", "pe": None},
            {"代码": "B", "pe": None},
        ]

        annotated = _annotate_with_percentile(rows, ["pe"])

        for row in annotated:
            assert "pe_percentile" not in row

    def test_single_numeric_value_assigns_neutral_50(self) -> None:
        """One numeric row gets a neutral 50.0 so the column still renders."""

        rows: list[dict[str, object]] = [
            {"代码": "A", "pe": None},
            {"代码": "B", "pe": 25.0},
        ]

        annotated = _annotate_with_percentile(rows, ["pe"])

        assert annotated[1]["pe_percentile"] == pytest.approx(50.0)
        assert "pe_percentile" not in annotated[0]
