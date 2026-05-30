"""Unit tests for :mod:`china_stock_mcp.services.financial_report_service`.

Covers task 12.2 in ``.kiro/specs/china-stock-mcp/tasks.md``:

- Requirement 4.4 -- ``report_type`` ∈ ``{annual, quarterly}`` and
  ``periods`` ∈ ``[1, 12]``; violations raise
  :class:`ValidationError` with a message that lists the offending
  value plus the accepted set / range. ``bool`` periods are rejected
  too (``bool`` is a subclass of ``int`` in Python).
- Requirement 4.5 -- adapter-raised :class:`DataNotFoundError`
  (insufficient periods) propagates verbatim.
- Requirement 4.6 -- the service re-sorts ``periods`` ascending by
  ``period_end`` even when the upstream returns descending order.
- Requirement 13.4 / Property 5 -- :func:`fetch_with_fallback` does
  **not** invoke the fallback adapter on
  :class:`DataNotFoundError`.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.cache import Cache, build_cache, reset_default_cache
from china_stock_mcp.config import Settings
from china_stock_mcp.exceptions import DataNotFoundError, ValidationError
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
from china_stock_mcp.services.financial_report_service import FinancialReportService

# ---------------------------------------------------------------------------
# Stub adapter
# ---------------------------------------------------------------------------


class StubAdapter(BaseAdapter):
    """Hermetic :class:`BaseAdapter` exposing only ``financial_report``.

    Records call counts so tests can assert that fallbacks are
    triggered (or not) as expected. Methods unrelated to
    ``financial_report`` raise :class:`NotImplementedError`.
    """

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

    # ----- not implemented in 12.2 ---------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_period(year: int) -> FinancialPeriod:
    """Build a :class:`FinancialPeriod` whose ``period_end`` is Dec 31 of ``year``."""

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


# ---------------------------------------------------------------------------
# Input validation (Requirement 4.4)
# ---------------------------------------------------------------------------


class TestReportTypeValidation:
    """**Validates: Requirements 4.4** -- report_type whitelist."""

    def test_invalid_report_type_raises_validation_error(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        adapter = StubAdapter(report=FinancialReport(
            symbol="300750.SZ", report_type="annual", periods=[]
        ))
        service = FinancialReportService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        with pytest.raises(ValidationError) as exc_info:
            service.report("300750.SZ", report_type="monthly", periods=4)

        message = str(exc_info.value)
        # Message must list both accepted values + the offending one.
        assert "annual" in message
        assert "quarterly" in message
        assert "monthly" in message
        # No upstream call should be made on validation failure.
        assert adapter.financial_report_call_count == 0


class TestPeriodsValidation:
    """**Validates: Requirements 4.4** -- periods range + type."""

    def test_periods_zero_raises_validation_error(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        adapter = StubAdapter(report=FinancialReport(
            symbol="300750.SZ", report_type="annual", periods=[]
        ))
        service = FinancialReportService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        with pytest.raises(ValidationError) as exc_info:
            service.report("300750.SZ", report_type="annual", periods=0)

        message = str(exc_info.value)
        assert "1" in message
        assert "12" in message
        assert adapter.financial_report_call_count == 0

    def test_periods_above_upper_bound_raises_validation_error(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        adapter = StubAdapter(report=FinancialReport(
            symbol="300750.SZ", report_type="annual", periods=[]
        ))
        service = FinancialReportService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        with pytest.raises(ValidationError) as exc_info:
            service.report("300750.SZ", report_type="annual", periods=13)

        message = str(exc_info.value)
        assert "1" in message
        assert "12" in message

    def test_periods_bool_input_rejected(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """``True`` is technically ``int`` in Python; service rejects it."""

        adapter = StubAdapter(report=FinancialReport(
            symbol="300750.SZ", report_type="annual", periods=[]
        ))
        service = FinancialReportService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        with pytest.raises(ValidationError) as exc_info:
            service.report("300750.SZ", report_type="annual", periods=True)  # type: ignore[arg-type]

        assert "bool" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Stable ascending sort (Requirement 4.6)
# ---------------------------------------------------------------------------


class TestStableAscendingSort:
    """**Validates: Requirements 4.6** -- service re-sorts upstream output."""

    def test_descending_upstream_resorted_ascending(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        # Upstream returns 2024, 2023, 2022, 2021 (descending).
        descending = FinancialReport(
            symbol="300750.SZ",
            report_type="annual",
            periods=[
                _make_period(2024),
                _make_period(2023),
                _make_period(2022),
                _make_period(2021),
            ],
        )
        adapter = StubAdapter(report=descending)
        service = FinancialReportService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        result = service.report("300750.SZ", report_type="annual", periods=4)

        # Service must produce ascending order regardless of source.
        years = [p.period_end.year for p in result.periods]
        assert years == [2021, 2022, 2023, 2024]


# ---------------------------------------------------------------------------
# DataNotFoundError + fallback non-switch (Requirement 4.5 / 13.4 / Property 5)
# ---------------------------------------------------------------------------


class TestDataNotFoundDoesNotSwitchFallback:
    """**Validates: Requirements 4.5, 13.4** (Property 5)."""

    def test_data_not_found_propagates_verbatim_without_fallback_call(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        not_found = DataNotFoundError(
            "300750.SZ 暂无 12 期年报, 请缩小 periods 或切换 report_type"
        )
        primary = StubAdapter(raises=not_found)

        # Fallback flips a flag if it gets called -- it must not.
        fallback = StubAdapter(report=FinancialReport(
            symbol="300750.SZ", report_type="annual", periods=[
                _make_period(2024),
            ],
        ))

        service = FinancialReportService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        with pytest.raises(DataNotFoundError) as exc_info:
            service.report("300750.SZ", report_type="annual", periods=12)

        # Verbatim propagation: same instance escapes the helper.
        assert exc_info.value is not_found
        # Primary called exactly once; fallback never called (P5).
        assert primary.financial_report_call_count == 1
        assert fallback.financial_report_call_count == 0
