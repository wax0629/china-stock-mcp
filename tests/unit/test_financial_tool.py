"""Unit tests for :mod:`china_stock_mcp.tools.financial` (task 12.2).

Covers tool-layer rendering and validation:

- Requirement 4.4 -- ``report_type`` ∈ ``{"annual", "quarterly"}`` and
  ``periods`` ∈ ``[1, 12]``; out-of-range values raise
  :class:`ValidationError`.
- Requirement 4.5 -- when the adapter returns a
  :class:`DataNotFoundError` (e.g. fewer periods than requested), it
  propagates verbatim through the tool.
- Property 5 / Requirement 13.4 -- :class:`DataNotFoundError` from the
  primary adapter does **not** trigger the fallback adapter inside
  :func:`fetch_with_fallback`.
- Requirement 4.3 -- the rendered Markdown carries one row for each
  of the eight metrics plus one column for each ``period_end``.
- Requirement 12.1 / Property 14 -- Markdown ends with the canonical
  disclaimer.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from typing import Any

import pytest

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.cache import Cache, make_key, reset_default_cache
from china_stock_mcp.exceptions import DataNotFoundError, ValidationError
from china_stock_mcp.formatters import DISCLAIMER
from china_stock_mcp.models import (
    FinancialPeriod,
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
from china_stock_mcp.services.financial_report_service import (
    FinancialReportService,
)
from china_stock_mcp.tools.financial import get_financial_report

# ---------------------------------------------------------------------------
# Hermetic cache + stub adapter
# ---------------------------------------------------------------------------


class _StubCache:
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
    """Adapter exposing only ``financial_report``."""

    name: str = "stub"

    def __init__(
        self,
        *,
        report: FinancialReport | None = None,
        raises: Exception | None = None,
    ) -> None:
        self._report = report
        self._raises = raises
        self.financial_report_call_count: int = 0

    def financial_report(
        self,
        symbol: str,
        report_type: str,
        periods: int,
    ) -> FinancialReport:
        self.financial_report_call_count += 1
        if self._raises is not None:
            raise self._raises
        assert self._report is not None
        return self._report

    # ----- not implemented ------------------------------------------------

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
def cache() -> _StubCache:
    return _StubCache()


@pytest.fixture()
def rate_limiter() -> RateLimiter:
    return RateLimiter(rate_per_minute=10_000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _period(year: int) -> FinancialPeriod:
    return FinancialPeriod(
        period_end=date(year, 12, 31),
        revenue=1.0e10 + year,
        net_profit=1.0e9 + year,
        net_profit_excl_nrgl=9.5e8 + year,
        gross_profit=3.0e9 + year,
        operating_cash_flow=1.5e9 + year,
        total_assets=5.0e10 + year,
        total_liabilities=2.0e10 + year,
        equity=3.0e10 + year,
    )


def _report(years: list[int]) -> FinancialReport:
    return FinancialReport(
        symbol="300750.SZ",
        report_type="annual",
        periods=[_period(y) for y in years],
    )


def _make_service(
    adapter: _StubAdapter,
    cache: Cache,
    rate_limiter: RateLimiter,
    *,
    fallback: _StubAdapter | None = None,
) -> FinancialReportService:
    return FinancialReportService(
        adapter,
        fallback=fallback,
        cache=cache,
        rate_limiter=rate_limiter,
    )


# ---------------------------------------------------------------------------
# report_type validation (Requirement 4.4)
# ---------------------------------------------------------------------------


class TestReportTypeValidation:
    """**Validates: Requirements 4.4**."""

    @pytest.mark.parametrize("report_type", ["annual", "quarterly"])
    def test_valid_report_types_accepted(
        self,
        cache: _StubCache,
        rate_limiter: RateLimiter,
        report_type: str,
    ) -> None:
        adapter = _StubAdapter(report=_report([2024]))
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_financial_report(
            service, "300750.SZ", report_type=report_type, periods=1
        )
        assert markdown.rstrip().endswith(DISCLAIMER)

    def test_invalid_report_type_raises_validation_error(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        adapter = _StubAdapter(report=_report([2024]))
        service = _make_service(adapter, cache, rate_limiter)

        with pytest.raises(ValidationError):
            get_financial_report(
                service, "300750.SZ", report_type="monthly", periods=4
            )

        # Adapter was never called because validation fails at the
        # tool boundary.
        assert adapter.financial_report_call_count == 0


# ---------------------------------------------------------------------------
# periods validation (Requirement 4.4)
# ---------------------------------------------------------------------------


class TestPeriodsValidation:
    """**Validates: Requirements 4.4**."""

    @pytest.mark.parametrize("periods", [1, 6, 12])
    def test_periods_in_range_accepted(
        self,
        cache: _StubCache,
        rate_limiter: RateLimiter,
        periods: int,
    ) -> None:
        years = list(range(2024 - periods + 1, 2025))
        adapter = _StubAdapter(report=_report(years))
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_financial_report(
            service, "300750.SZ", report_type="annual", periods=periods
        )
        assert markdown.rstrip().endswith(DISCLAIMER)

    @pytest.mark.parametrize("periods", [0, 13, -1, 100])
    def test_periods_out_of_range_raises_validation_error(
        self,
        cache: _StubCache,
        rate_limiter: RateLimiter,
        periods: int,
    ) -> None:
        adapter = _StubAdapter(report=_report([2024]))
        service = _make_service(adapter, cache, rate_limiter)

        with pytest.raises(ValidationError):
            get_financial_report(
                service, "300750.SZ", report_type="annual", periods=periods
            )

        assert adapter.financial_report_call_count == 0


# ---------------------------------------------------------------------------
# DataNotFoundError on insufficient periods (Requirement 4.5)
# ---------------------------------------------------------------------------


class TestDataNotFoundOnInsufficientPeriods:
    """**Validates: Requirements 4.5** -- adapter NotFound propagates."""

    def test_adapter_data_not_found_propagates(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        not_found = DataNotFoundError(
            "300750.SZ 暂无 12 期年报, 请缩小 periods"
        )
        adapter = _StubAdapter(raises=not_found)
        service = _make_service(adapter, cache, rate_limiter)

        with pytest.raises(DataNotFoundError) as exc_info:
            get_financial_report(
                service, "300750.SZ", report_type="annual", periods=12
            )

        assert exc_info.value is not_found
        assert adapter.financial_report_call_count == 1


# ---------------------------------------------------------------------------
# fetch_with_fallback does NOT switch on DataNotFoundError
# (Requirement 13.4 / Property 5)
# ---------------------------------------------------------------------------


class TestNotFoundBypassesFallback:
    """**Validates: Requirements 13.4** (Property 5)."""

    def test_data_not_found_does_not_invoke_fallback(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        not_found = DataNotFoundError("新股期数不足")

        primary = _StubAdapter(raises=not_found)
        # Fallback would happily return data; the test asserts it never
        # gets called.
        fallback = _StubAdapter(report=_report([2024]))

        service = _make_service(
            primary, cache, rate_limiter, fallback=fallback
        )

        with pytest.raises(DataNotFoundError) as exc_info:
            get_financial_report(
                service, "300750.SZ", report_type="annual", periods=12
            )

        assert exc_info.value is not_found
        assert primary.financial_report_call_count == 1
        # Critical assertion: fallback was never invoked.
        assert fallback.financial_report_call_count == 0


# ---------------------------------------------------------------------------
# Markdown contains 8 metrics + period_end columns (Requirement 4.3)
# ---------------------------------------------------------------------------


class TestMarkdownStructure:
    """**Validates: Requirements 4.3, 4.6**."""

    def test_markdown_contains_all_eight_metric_rows(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        adapter = _StubAdapter(report=_report([2022, 2023, 2024]))
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_financial_report(
            service, "300750.SZ", report_type="annual", periods=3
        )

        # The eight metric labels rendered as row headers.
        for label in (
            "营业总收入",
            "归母净利润",
            "扣非净利润",
            "毛利",
            "经营性现金流",
            "总资产",
            "总负债",
            "所有者权益",
        ):
            assert label in markdown

    def test_markdown_contains_period_end_columns(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        years = [2022, 2023, 2024]
        adapter = _StubAdapter(report=_report(years))
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_financial_report(
            service, "300750.SZ", report_type="annual", periods=3
        )

        # Each period_end appears as a column header in ISO format.
        for year in years:
            assert f"{year}-12-31" in markdown

        assert markdown.rstrip().endswith(DISCLAIMER)
