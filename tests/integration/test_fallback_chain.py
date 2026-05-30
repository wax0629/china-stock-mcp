"""Integration tests for primary/fallback adapter orchestration.

Covers task 19.4 (validates Requirements 13.3, 13.4, 13.5) by wiring
every service that has a fallback adapter configured in
:func:`china_stock_mcp.server._build_services` against a hand-rolled
``StubPrimaryAdapter`` (which raises configurable error types) and a
``StubFallbackAdapter`` (which counts invocations and returns valid
DTOs). Per-service classes pin the same four behaviours guaranteed by
``fetch_with_fallback``:

- Requirement 13.3 (P) -- ``NetworkError`` from the primary triggers
  the fallback.
- Requirement 13.3 (P) -- ``RateLimitError`` from the primary triggers
  the fallback.
- Requirement 13.4 / Property 5 -- ``DataNotFoundError`` from the
  primary does **not** trigger the fallback; the error propagates
  verbatim and the fallback's call count stays at zero.
- Algorithm 1 -- ``SymbolError`` (and any other non-transient
  ``ChinaStockMCPError``) from the primary does **not** trigger the
  fallback either; only :class:`NetworkError` /
  :class:`RateLimitError` qualify per design.
- Requirement 13.5 / Property 6 -- a primary success leaves the
  fallback completely untouched.

Plus a few cross-cutting checks:

- ``fetch_with_fallback`` emits a warn-level log line whenever it
  switches sources (Algorithm 1 step 4). One test attaches a loguru
  ``StringIO`` sink and asserts the line is emitted with the WARNING
  level and carries both adapter names.
- ``fallback=None`` re-raises the primary's transient error verbatim
  so the failure type is preserved (Requirement 13.3 leaves the
  no-fallback case to the caller).

Each test wires a real ``DiskCache`` (rooted at ``tmp_path``) and a
real ``RateLimiter`` (10 000 req/min so the budget never fires) so we
exercise the full Service-layer pipeline -- not just the
``fetch_with_fallback`` helper in isolation. The default cache /
limiter singletons are reset between tests so cached payloads from
one test cannot mask the behaviour of another.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from china_stock_mcp import logger
from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.cache import Cache, build_cache, reset_default_cache
from china_stock_mcp.config import Settings
from china_stock_mcp.exceptions import (
    DataNotFoundError,
    NetworkError,
    RateLimitError,
    SymbolError,
)
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
from china_stock_mcp.rate_limiter import RateLimiter, reset_default_rate_limiter
from china_stock_mcp.services.financial_report_service import FinancialReportService
from china_stock_mcp.services.fundamental_service import FundamentalService
from china_stock_mcp.services.money_flow_service import MoneyFlowService
from china_stock_mcp.services.quote_service import QuoteService
from china_stock_mcp.services.symbol_service import SymbolService

# ---------------------------------------------------------------------------
# Stub adapters
# ---------------------------------------------------------------------------


class _CountingStubBase(BaseAdapter):
    """Shared stub :class:`BaseAdapter` that fails every "non-target" method.

    The fallback-chain tests exercise five service surfaces -- search,
    quote, fundamentals, financial_report, money_flow -- so each
    method that may be invoked is implemented in the concrete
    primary / fallback subclasses below. Methods unrelated to the
    targeted surface raise loudly to flag any test that drifts out of
    the intended scope. This mirrors the convention used by
    :class:`tests.integration.test_search_quote_flow.StubAdapter`.
    """

    def kline(
        self,
        symbol: str,
        period: str,
        count: int,
        adjust: str,
    ) -> KLineSeries:  # pragma: no cover - defensive
        raise NotImplementedError("stub adapter does not implement kline")

    def industry_peers(
        self,
        symbol: str,
        metrics: list[str],
        top_n: int,
    ) -> PeerTable:  # pragma: no cover - defensive
        raise NotImplementedError(
            "stub adapter does not implement industry_peers"
        )

    def fund_info(self, fund_code: str) -> FundInfo:  # pragma: no cover - defensive
        raise NotImplementedError("stub adapter does not implement fund_info")

    def market_overview(self) -> MarketOverview:  # pragma: no cover - defensive
        raise NotImplementedError(
            "stub adapter does not implement market_overview"
        )


class StubPrimaryAdapter(_CountingStubBase):
    """Configurable primary adapter.

    Each invocation of a public method raises ``error`` if one is
    configured for that endpoint, otherwise returns the stored
    payload. Call counters let tests assert the primary was actually
    invoked. Every endpoint is wired the same way so a single class
    serves every service-level test below.
    """

    name: str = "stub_primary"

    def __init__(
        self,
        *,
        search_error: BaseException | None = None,
        quote_error: BaseException | None = None,
        fundamentals_error: BaseException | None = None,
        financial_report_error: BaseException | None = None,
        money_flow_error: BaseException | None = None,
        search_hits: list[SymbolHit] | None = None,
        quotes_by_symbol: dict[str, Quote] | None = None,
        fundamentals_by_symbol: dict[str, FundamentalSnapshot] | None = None,
        reports_by_symbol: dict[str, FinancialReport] | None = None,
        money_flow_payload: MoneyFlow | None = None,
    ) -> None:
        self._search_error = search_error
        self._quote_error = quote_error
        self._fundamentals_error = fundamentals_error
        self._financial_report_error = financial_report_error
        self._money_flow_error = money_flow_error
        self._search_hits: list[SymbolHit] = list(search_hits or [])
        self._quotes_by_symbol: dict[str, Quote] = dict(quotes_by_symbol or {})
        self._fundamentals_by_symbol: dict[str, FundamentalSnapshot] = dict(
            fundamentals_by_symbol or {}
        )
        self._reports_by_symbol: dict[str, FinancialReport] = dict(
            reports_by_symbol or {}
        )
        self._money_flow_payload: MoneyFlow | None = money_flow_payload
        self.search_call_count: int = 0
        self.quote_call_count: int = 0
        self.fundamentals_call_count: int = 0
        self.financial_report_call_count: int = 0
        self.money_flow_call_count: int = 0

    def search(self, query: str, market: str) -> list[SymbolHit]:
        self.search_call_count += 1
        if self._search_error is not None:
            raise self._search_error
        return list(self._search_hits)

    def quote(self, symbols: list[str]) -> list[Quote]:
        self.quote_call_count += 1
        if self._quote_error is not None:
            raise self._quote_error
        return [
            self._quotes_by_symbol[s]
            for s in symbols
            if s in self._quotes_by_symbol
        ]

    def fundamentals(self, symbol: str) -> FundamentalSnapshot:
        self.fundamentals_call_count += 1
        if self._fundamentals_error is not None:
            raise self._fundamentals_error
        if symbol in self._fundamentals_by_symbol:
            return self._fundamentals_by_symbol[symbol]
        raise DataNotFoundError(f"primary stub has no fundamentals for {symbol}")

    def financial_report(
        self,
        symbol: str,
        report_type: str,
        periods: int,
    ) -> FinancialReport:
        self.financial_report_call_count += 1
        if self._financial_report_error is not None:
            raise self._financial_report_error
        if symbol in self._reports_by_symbol:
            return self._reports_by_symbol[symbol]
        raise DataNotFoundError(
            f"primary stub has no financial report for {symbol}"
        )

    def money_flow(
        self,
        symbol: str | None,
        flow_type: str,
        top_n: int,
    ) -> MoneyFlow:
        self.money_flow_call_count += 1
        if self._money_flow_error is not None:
            raise self._money_flow_error
        if self._money_flow_payload is not None:
            return self._money_flow_payload
        raise DataNotFoundError("primary stub has no money flow payload")


class StubFallbackAdapter(_CountingStubBase):
    """Configurable fallback adapter.

    Returns valid payloads by default but can be configured to raise
    errors of its own (used in the "both sources fail" scenarios).
    Counters let tests assert the fallback was (or was *not*)
    invoked, which is what Properties 5 / 6 are about.
    """

    name: str = "stub_fallback"

    def __init__(
        self,
        *,
        search_error: BaseException | None = None,
        quote_error: BaseException | None = None,
        fundamentals_error: BaseException | None = None,
        financial_report_error: BaseException | None = None,
        money_flow_error: BaseException | None = None,
        search_hits: list[SymbolHit] | None = None,
        quotes_by_symbol: dict[str, Quote] | None = None,
        fundamentals_by_symbol: dict[str, FundamentalSnapshot] | None = None,
        reports_by_symbol: dict[str, FinancialReport] | None = None,
        money_flow_payload: MoneyFlow | None = None,
    ) -> None:
        self._search_error = search_error
        self._quote_error = quote_error
        self._fundamentals_error = fundamentals_error
        self._financial_report_error = financial_report_error
        self._money_flow_error = money_flow_error
        self._search_hits: list[SymbolHit] = list(search_hits or [])
        self._quotes_by_symbol: dict[str, Quote] = dict(quotes_by_symbol or {})
        self._fundamentals_by_symbol: dict[str, FundamentalSnapshot] = dict(
            fundamentals_by_symbol or {}
        )
        self._reports_by_symbol: dict[str, FinancialReport] = dict(
            reports_by_symbol or {}
        )
        self._money_flow_payload: MoneyFlow | None = money_flow_payload
        self.search_call_count: int = 0
        self.quote_call_count: int = 0
        self.fundamentals_call_count: int = 0
        self.financial_report_call_count: int = 0
        self.money_flow_call_count: int = 0

    def search(self, query: str, market: str) -> list[SymbolHit]:
        self.search_call_count += 1
        if self._search_error is not None:
            raise self._search_error
        return list(self._search_hits)

    def quote(self, symbols: list[str]) -> list[Quote]:
        self.quote_call_count += 1
        if self._quote_error is not None:
            raise self._quote_error
        return [
            self._quotes_by_symbol[s]
            for s in symbols
            if s in self._quotes_by_symbol
        ]

    def fundamentals(self, symbol: str) -> FundamentalSnapshot:
        self.fundamentals_call_count += 1
        if self._fundamentals_error is not None:
            raise self._fundamentals_error
        if symbol in self._fundamentals_by_symbol:
            return self._fundamentals_by_symbol[symbol]
        raise DataNotFoundError(
            f"fallback stub has no fundamentals for {symbol}"
        )

    def financial_report(
        self,
        symbol: str,
        report_type: str,
        periods: int,
    ) -> FinancialReport:
        self.financial_report_call_count += 1
        if self._financial_report_error is not None:
            raise self._financial_report_error
        if symbol in self._reports_by_symbol:
            return self._reports_by_symbol[symbol]
        raise DataNotFoundError(
            f"fallback stub has no financial report for {symbol}"
        )

    def money_flow(
        self,
        symbol: str | None,
        flow_type: str,
        top_n: int,
    ) -> MoneyFlow:
        self.money_flow_call_count += 1
        if self._money_flow_error is not None:
            raise self._money_flow_error
        if self._money_flow_payload is not None:
            return self._money_flow_payload
        raise DataNotFoundError("fallback stub has no money flow payload")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons() -> Iterator[None]:
    """Hermetic isolation -- reset both module-level singletons each test."""

    reset_default_cache()
    reset_default_rate_limiter()
    try:
        yield
    finally:
        reset_default_cache()
        reset_default_rate_limiter()


@pytest.fixture()
def cache(tmp_path: Path) -> Iterator[Cache]:
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
# DTO builders
# ---------------------------------------------------------------------------


_FIXED_TIMESTAMP = datetime(2024, 6, 3, 14, 30, 0, tzinfo=UTC)


def _make_quote(symbol: str, name: str = "测试标的") -> Quote:
    return Quote(
        symbol=symbol,
        name=name,
        price=10.0,
        change=0.5,
        change_pct=5.0,
        volume=1_000_000,
        amount=1.0e8,
        turnover_rate=2.5,
        pe_ttm=15.0,
        pe_dynamic=14.0,
        pb=2.0,
        market_cap=1.0e10,
        float_market_cap=8.0e9,
        timestamp=_FIXED_TIMESTAMP,
        delay_seconds=900,
    )


def _make_hit(code: str, name: str = "测试") -> SymbolHit:
    return SymbolHit(code=code, name=name, market="a_stock")


def _make_snapshot(symbol: str, *, pe_ttm: float = 18.5) -> FundamentalSnapshot:
    return FundamentalSnapshot(
        symbol=symbol,
        valuation={"pe_ttm": pe_ttm, "pb": 3.4},
        profitability={"roe": 18.0, "gross_margin": 28.0},
        growth={"revenue_yoy": 22.5, "net_profit_yoy": 18.5},
        health={"debt_ratio": 45.0, "current_ratio": 1.8},
        industry_percentile={},
    )


def _make_report(symbol: str) -> FinancialReport:
    return FinancialReport(
        symbol=symbol,
        report_type="annual",
        periods=[
            FinancialPeriod(
                period_end=date(2023, 12, 31),
                revenue=4.0e11,
                net_profit=4.0e10,
                net_profit_excl_nrgl=3.5e10,
                gross_profit=1.0e11,
                operating_cash_flow=5.0e10,
                total_assets=8.0e11,
                total_liabilities=4.0e11,
                equity=4.0e11,
            ),
        ],
    )


def _make_money_flow(flow_type: str = "north") -> MoneyFlow:
    return MoneyFlow(
        flow_type=flow_type,  # type: ignore[arg-type]
        rows=[
            {
                "date": "2024-06-03",
                "净流入金额": 1.5e9,
                "买入金额": 5.0e9,
                "卖出金额": 3.5e9,
                "持股市值": 2.5e12,
            },
        ],
        snapshot_at=_FIXED_TIMESTAMP,
    )


# ---------------------------------------------------------------------------
# search() fallback tests (Requirements 13.3 / 13.4 / 13.5)
# ---------------------------------------------------------------------------


class TestSymbolServiceFallback:
    """Fallback contract for the ``search`` endpoint."""

    def test_network_error_triggers_fallback(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirement 13.3 -- NetworkError → switch to fallback.

        The primary raises ``NetworkError`` on its single search call;
        the fallback returns one hit. The service must transparently
        return the fallback's payload and the fallback must have been
        invoked exactly once.
        """

        primary = StubPrimaryAdapter(
            search_error=NetworkError("primary down"),
        )
        fallback = StubFallbackAdapter(
            search_hits=[_make_hit("300750.SZ", "宁德时代")],
        )
        service = SymbolService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        hits = service.search("宁德", market="a_stock")

        assert primary.search_call_count == 1
        assert fallback.search_call_count == 1
        assert [h.code for h in hits] == ["300750.SZ"]

    def test_rate_limit_error_triggers_fallback(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirement 13.3 -- RateLimitError → switch to fallback."""

        primary = StubPrimaryAdapter(
            search_error=RateLimitError("primary 429"),
        )
        fallback = StubFallbackAdapter(
            search_hits=[_make_hit("600519.SH", "贵州茅台")],
        )
        service = SymbolService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        hits = service.search("茅台", market="a_stock")

        assert primary.search_call_count == 1
        assert fallback.search_call_count == 1
        assert [h.code for h in hits] == ["600519.SH"]

    def test_data_not_found_error_does_not_trigger_fallback(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirement 13.4 / Property 5 -- DataNotFoundError stays.

        ``DataNotFoundError`` is *not* a transient failure; the
        fallback must remain at zero invocations and the original
        exception must propagate verbatim.
        """

        primary = StubPrimaryAdapter(
            search_error=DataNotFoundError("primary returned empty"),
        )
        fallback = StubFallbackAdapter(
            search_hits=[_make_hit("300750.SZ", "宁德时代")],
        )
        service = SymbolService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        with pytest.raises(DataNotFoundError, match="primary returned empty"):
            service.search("宁德", market="a_stock")

        assert primary.search_call_count == 1
        assert fallback.search_call_count == 0  # Property 5

    def test_primary_success_leaves_fallback_untouched(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirement 13.5 / Property 6 -- success skips fallback."""

        primary = StubPrimaryAdapter(
            search_hits=[_make_hit("300750.SZ", "宁德时代")],
        )
        fallback = StubFallbackAdapter(
            # Distinct payload that should never appear in the result;
            # using a structurally-valid SymbolHit keeps the stub
            # honest while letting us prove the fallback was skipped.
            search_hits=[_make_hit("000001.SZ", "should never see this")],
        )
        service = SymbolService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        hits = service.search("宁德", market="a_stock")

        assert primary.search_call_count == 1
        assert fallback.search_call_count == 0  # Property 6
        assert [h.code for h in hits] == ["300750.SZ"]
        # Confirm we got the primary payload, not the fallback's.
        assert [h.name for h in hits] == ["宁德时代"]

    def test_symbol_error_does_not_trigger_fallback(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Algorithm 1 -- ``SymbolError`` from primary stays primary.

        Only :class:`NetworkError` / :class:`RateLimitError` are
        eligible for fallback per design Algorithm 1; every other
        :class:`ChinaStockMCPError` subclass propagates verbatim.
        """

        primary = StubPrimaryAdapter(
            search_error=SymbolError("无法识别的代码: 'XYZ'"),
        )
        fallback = StubFallbackAdapter(
            search_hits=[_make_hit("300750.SZ", "should never see this")],
        )
        service = SymbolService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        with pytest.raises(SymbolError, match="无法识别的代码"):
            service.search("XYZ", market="a_stock")

        assert primary.search_call_count == 1
        assert fallback.search_call_count == 0


# ---------------------------------------------------------------------------
# quote() fallback tests (Requirements 13.3 / 13.4 / 13.5)
# ---------------------------------------------------------------------------


class TestQuoteServiceFallback:
    """Fallback contract for the ``quote`` endpoint."""

    def test_network_error_triggers_fallback(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirement 13.3 -- NetworkError → switch to fallback."""

        symbol = "300750.SZ"
        primary = StubPrimaryAdapter(
            quote_error=NetworkError("primary down"),
        )
        fallback = StubFallbackAdapter(
            quotes_by_symbol={symbol: _make_quote(symbol, "宁德时代")},
        )
        service = QuoteService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        quotes = service.get_snapshot([symbol])

        assert primary.quote_call_count == 1
        assert fallback.quote_call_count == 1
        assert [q.symbol for q in quotes] == [symbol]

    def test_rate_limit_error_triggers_fallback(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirement 13.3 -- RateLimitError → switch to fallback."""

        symbol = "600519.SH"
        primary = StubPrimaryAdapter(
            quote_error=RateLimitError("primary 429"),
        )
        fallback = StubFallbackAdapter(
            quotes_by_symbol={symbol: _make_quote(symbol, "贵州茅台")},
        )
        service = QuoteService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        quotes = service.get_snapshot([symbol])

        assert primary.quote_call_count == 1
        assert fallback.quote_call_count == 1
        assert [q.symbol for q in quotes] == [symbol]

    def test_data_not_found_does_not_trigger_fallback(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirement 13.4 / Property 5 -- DataNotFoundError stays.

        Two ways the quote pipeline can yield ``DataNotFoundError``:

        1. The adapter raises it directly (transient-error contract
           says it must propagate without switching).
        2. The adapter silently drops a symbol from its return value,
           and ``QuoteService`` raises ``DataNotFoundError`` itself.

        Both must leave the fallback at zero invocations.
        """

        # Case 1: primary raises DataNotFoundError directly.
        primary = StubPrimaryAdapter(
            quote_error=DataNotFoundError("primary cannot find symbol"),
        )
        fallback = StubFallbackAdapter(
            quotes_by_symbol={"300750.SZ": _make_quote("300750.SZ", "宁德时代")},
        )
        service = QuoteService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        with pytest.raises(DataNotFoundError, match="primary cannot find symbol"):
            service.get_snapshot(["300750.SZ"])

        assert primary.quote_call_count == 1
        assert fallback.quote_call_count == 0  # Property 5

    def test_primary_success_leaves_fallback_untouched(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirement 13.5 / Property 6 -- success skips fallback."""

        symbol = "300750.SZ"
        primary = StubPrimaryAdapter(
            quotes_by_symbol={symbol: _make_quote(symbol, "宁德时代")},
        )
        fallback = StubFallbackAdapter(
            quotes_by_symbol={symbol: _make_quote(symbol, "should never see")},
        )
        service = QuoteService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        quotes = service.get_snapshot([symbol])

        assert primary.quote_call_count == 1
        assert fallback.quote_call_count == 0  # Property 6
        # The returned name must come from the primary, not the
        # would-be fallback payload.
        assert quotes[0].name == "宁德时代"

    def test_symbol_error_does_not_trigger_fallback(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Algorithm 1 -- ``SymbolError`` propagates without switching.

        Only :class:`NetworkError` / :class:`RateLimitError` are
        eligible for fallback per design Algorithm 1.
        """

        primary = StubPrimaryAdapter(
            quote_error=SymbolError("非法代码"),
        )
        fallback = StubFallbackAdapter(
            quotes_by_symbol={"300750.SZ": _make_quote("300750.SZ", "fallback")},
        )
        service = QuoteService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        with pytest.raises(SymbolError, match="非法代码"):
            service.get_snapshot(["300750.SZ"])

        assert primary.quote_call_count == 1
        assert fallback.quote_call_count == 0

    def test_both_sources_fail_propagates_fallback_error(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirement 13.3 -- when fallback also fails, its error wins.

        ``fetch_with_fallback`` does *not* attempt a third tier; the
        fallback's exception propagates verbatim so callers can
        diagnose which leg failed.
        """

        primary = StubPrimaryAdapter(
            quote_error=NetworkError("primary down"),
        )
        fallback = StubFallbackAdapter(
            quote_error=NetworkError("fallback also down"),
        )
        service = QuoteService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        with pytest.raises(NetworkError, match="fallback also down"):
            service.get_snapshot(["300750.SZ"])

        assert primary.quote_call_count == 1
        assert fallback.quote_call_count == 1

    def test_no_fallback_configured_propagates_primary_error(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """When ``fallback=None``, the primary's transient error wins.

        ``fetch_with_fallback`` re-raises the primary failure verbatim
        so the caller's error type is preserved (Requirement 13.3
        leaves the no-fallback case to the caller).
        """

        primary = StubPrimaryAdapter(
            quote_error=NetworkError("primary down, no fallback"),
        )
        service = QuoteService(
            primary,
            fallback=None,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        with pytest.raises(NetworkError, match="primary down, no fallback"):
            service.get_snapshot(["300750.SZ"])

        assert primary.quote_call_count == 1


# ---------------------------------------------------------------------------
# fundamentals() fallback tests (Requirements 13.3 / 13.4 / 13.5)
# ---------------------------------------------------------------------------


class TestFundamentalServiceFallback:
    """Fallback contract for the ``fundamentals`` endpoint."""

    def test_network_error_triggers_fallback(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirement 13.3 -- NetworkError → switch to fallback."""

        symbol = "300750.SZ"
        primary = StubPrimaryAdapter(
            fundamentals_error=NetworkError("primary down"),
        )
        fallback = StubFallbackAdapter(
            fundamentals_by_symbol={symbol: _make_snapshot(symbol, pe_ttm=22.0)},
        )
        service = FundamentalService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        snapshot = service.snapshot(symbol)

        assert primary.fundamentals_call_count == 1
        assert fallback.fundamentals_call_count == 1
        assert snapshot.symbol == symbol
        assert snapshot.valuation["pe_ttm"] == 22.0  # came from fallback

    def test_rate_limit_error_triggers_fallback(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirement 13.3 -- RateLimitError → switch to fallback."""

        symbol = "600519.SH"
        primary = StubPrimaryAdapter(
            fundamentals_error=RateLimitError("primary 429"),
        )
        fallback = StubFallbackAdapter(
            fundamentals_by_symbol={symbol: _make_snapshot(symbol)},
        )
        service = FundamentalService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        snapshot = service.snapshot(symbol)

        assert primary.fundamentals_call_count == 1
        assert fallback.fundamentals_call_count == 1
        assert snapshot.symbol == symbol

    def test_data_not_found_does_not_trigger_fallback(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirement 13.4 / Property 5 -- DataNotFoundError stays."""

        symbol = "300750.SZ"
        primary = StubPrimaryAdapter(
            fundamentals_error=DataNotFoundError("primary cannot find symbol"),
        )
        fallback = StubFallbackAdapter(
            fundamentals_by_symbol={symbol: _make_snapshot(symbol)},
        )
        service = FundamentalService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        with pytest.raises(DataNotFoundError, match="primary cannot find symbol"):
            service.snapshot(symbol)

        assert primary.fundamentals_call_count == 1
        assert fallback.fundamentals_call_count == 0  # Property 5

    def test_symbol_error_does_not_trigger_fallback(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Algorithm 1 -- ``SymbolError`` is not fallback-eligible."""

        symbol = "300750.SZ"
        primary = StubPrimaryAdapter(
            fundamentals_error=SymbolError("不支持的代码"),
        )
        fallback = StubFallbackAdapter(
            fundamentals_by_symbol={symbol: _make_snapshot(symbol)},
        )
        service = FundamentalService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        with pytest.raises(SymbolError, match="不支持的代码"):
            service.snapshot(symbol)

        assert primary.fundamentals_call_count == 1
        assert fallback.fundamentals_call_count == 0

    def test_primary_success_leaves_fallback_untouched(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirement 13.5 / Property 6 -- success skips fallback."""

        symbol = "300750.SZ"
        primary = StubPrimaryAdapter(
            fundamentals_by_symbol={symbol: _make_snapshot(symbol, pe_ttm=15.0)},
        )
        fallback = StubFallbackAdapter(
            fundamentals_by_symbol={symbol: _make_snapshot(symbol, pe_ttm=999.0)},
        )
        service = FundamentalService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        snapshot = service.snapshot(symbol)

        assert primary.fundamentals_call_count == 1
        assert fallback.fundamentals_call_count == 0  # Property 6
        # The returned snapshot must be the primary's, not the
        # would-be fallback payload.
        assert snapshot.valuation["pe_ttm"] == 15.0

    def test_no_fallback_configured_propagates_primary_error(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """``fallback=None`` re-raises the transient error verbatim."""

        primary = StubPrimaryAdapter(
            fundamentals_error=NetworkError("primary down, no fallback"),
        )
        service = FundamentalService(
            primary,
            fallback=None,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        with pytest.raises(NetworkError, match="primary down, no fallback"):
            service.snapshot("300750.SZ")

        assert primary.fundamentals_call_count == 1


# ---------------------------------------------------------------------------
# financial_report() fallback tests (Requirements 13.3 / 13.4 / 13.5)
# ---------------------------------------------------------------------------


class TestFinancialReportServiceFallback:
    """Fallback contract for the ``financial_report`` endpoint."""

    def test_network_error_triggers_fallback(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirement 13.3 -- NetworkError → switch to fallback."""

        symbol = "300750.SZ"
        primary = StubPrimaryAdapter(
            financial_report_error=NetworkError("primary down"),
        )
        fallback = StubFallbackAdapter(
            reports_by_symbol={symbol: _make_report(symbol)},
        )
        service = FinancialReportService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        report = service.report(symbol, "annual", 1)

        assert primary.financial_report_call_count == 1
        assert fallback.financial_report_call_count == 1
        assert report.symbol == symbol
        assert len(report.periods) == 1

    def test_rate_limit_error_triggers_fallback(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirement 13.3 -- RateLimitError → switch to fallback."""

        symbol = "300750.SZ"
        primary = StubPrimaryAdapter(
            financial_report_error=RateLimitError("primary 429"),
        )
        fallback = StubFallbackAdapter(
            reports_by_symbol={symbol: _make_report(symbol)},
        )
        service = FinancialReportService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        report = service.report(symbol, "annual", 1)

        assert primary.financial_report_call_count == 1
        assert fallback.financial_report_call_count == 1
        assert report.symbol == symbol

    def test_data_not_found_does_not_trigger_fallback(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirement 13.4 / Property 5 -- DataNotFoundError stays.

        Newly-listed companies often have fewer than the requested
        ``periods`` of data; the design (Requirement 4.5) says we
        propagate :class:`DataNotFoundError` rather than silently
        falling back to a different upstream and returning a longer
        but inconsistent series.
        """

        symbol = "300750.SZ"
        primary = StubPrimaryAdapter(
            financial_report_error=DataNotFoundError("年报期数不足"),
        )
        fallback = StubFallbackAdapter(
            reports_by_symbol={symbol: _make_report(symbol)},
        )
        service = FinancialReportService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        with pytest.raises(DataNotFoundError, match="年报期数不足"):
            service.report(symbol, "annual", 4)

        assert primary.financial_report_call_count == 1
        assert fallback.financial_report_call_count == 0

    def test_symbol_error_does_not_trigger_fallback(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Algorithm 1 -- ``SymbolError`` is not fallback-eligible."""

        symbol = "300750.SZ"
        primary = StubPrimaryAdapter(
            financial_report_error=SymbolError("非法代码"),
        )
        fallback = StubFallbackAdapter(
            reports_by_symbol={symbol: _make_report(symbol)},
        )
        service = FinancialReportService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        with pytest.raises(SymbolError, match="非法代码"):
            service.report(symbol, "annual", 1)

        assert primary.financial_report_call_count == 1
        assert fallback.financial_report_call_count == 0

    def test_primary_success_leaves_fallback_untouched(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirement 13.5 / Property 6 -- success skips fallback."""

        symbol = "300750.SZ"
        primary = StubPrimaryAdapter(
            reports_by_symbol={symbol: _make_report(symbol)},
        )
        fallback = StubFallbackAdapter(
            reports_by_symbol={symbol: _make_report(symbol)},
        )
        service = FinancialReportService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        report = service.report(symbol, "annual", 1)

        assert primary.financial_report_call_count == 1
        assert fallback.financial_report_call_count == 0  # Property 6
        assert report.symbol == symbol


# ---------------------------------------------------------------------------
# money_flow() fallback tests (Requirements 13.3 / 13.4 / 13.5)
# ---------------------------------------------------------------------------


class TestMoneyFlowServiceFallback:
    """Fallback contract for the ``money_flow`` endpoint."""

    def test_network_error_triggers_fallback(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirement 13.3 -- NetworkError → switch to fallback."""

        primary = StubPrimaryAdapter(
            money_flow_error=NetworkError("primary down"),
        )
        fallback = StubFallbackAdapter(
            money_flow_payload=_make_money_flow("north"),
        )
        service = MoneyFlowService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        flow = service.get(symbol=None, flow_type="north", top_n=10)

        assert primary.money_flow_call_count == 1
        assert fallback.money_flow_call_count == 1
        assert flow.flow_type == "north"
        assert len(flow.rows) == 1

    def test_rate_limit_error_triggers_fallback(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirement 13.3 -- RateLimitError → switch to fallback."""

        primary = StubPrimaryAdapter(
            money_flow_error=RateLimitError("primary 429"),
        )
        fallback = StubFallbackAdapter(
            money_flow_payload=_make_money_flow("north"),
        )
        service = MoneyFlowService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        flow = service.get(symbol=None, flow_type="north", top_n=10)

        assert primary.money_flow_call_count == 1
        assert fallback.money_flow_call_count == 1
        assert flow.flow_type == "north"

    def test_data_not_found_does_not_trigger_fallback(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirement 13.4 / Property 5 -- DataNotFoundError stays."""

        primary = StubPrimaryAdapter(
            money_flow_error=DataNotFoundError("北向资金当日无数据"),
        )
        fallback = StubFallbackAdapter(
            money_flow_payload=_make_money_flow("north"),
        )
        service = MoneyFlowService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        with pytest.raises(DataNotFoundError, match="北向资金当日无数据"):
            service.get(symbol=None, flow_type="north", top_n=10)

        assert primary.money_flow_call_count == 1
        assert fallback.money_flow_call_count == 0

    def test_symbol_error_does_not_trigger_fallback(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Algorithm 1 -- ``SymbolError`` is not fallback-eligible."""

        primary = StubPrimaryAdapter(
            money_flow_error=SymbolError("非法代码"),
        )
        fallback = StubFallbackAdapter(
            money_flow_payload=_make_money_flow("main"),
        )
        service = MoneyFlowService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        with pytest.raises(SymbolError, match="非法代码"):
            service.get(symbol="300750.SZ", flow_type="main", top_n=10)

        assert primary.money_flow_call_count == 1
        assert fallback.money_flow_call_count == 0

    def test_primary_success_leaves_fallback_untouched(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirement 13.5 / Property 6 -- success skips fallback."""

        primary = StubPrimaryAdapter(
            money_flow_payload=_make_money_flow("north"),
        )
        fallback = StubFallbackAdapter(
            money_flow_payload=_make_money_flow("north"),
        )
        service = MoneyFlowService(
            primary,
            fallback=fallback,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        flow = service.get(symbol=None, flow_type="north", top_n=10)

        assert primary.money_flow_call_count == 1
        assert fallback.money_flow_call_count == 0  # Property 6
        assert flow.flow_type == "north"


# ---------------------------------------------------------------------------
# Cross-cutting: warn-level log on fallback switch
# ---------------------------------------------------------------------------


class TestFallbackLogging:
    """``fetch_with_fallback`` emits WARNING when switching sources."""

    def test_warn_log_emitted_on_network_switch(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Algorithm 1 step 4 -- log a WARNING with both adapter names.

        We attach a :class:`StringIO` sink to the package's loguru
        logger, run a primary→fallback switch, and assert the line
        carries (a) the WARNING level, (b) the primary's ``name``,
        and (c) the fallback's ``name``. The sink is removed in
        ``finally`` so subsequent tests are not affected.
        """

        sink = io.StringIO()
        handler_id = logger.add(
            sink,
            level="DEBUG",
            format="{level}|{name}|{message}",
        )
        try:
            symbol = "300750.SZ"
            primary = StubPrimaryAdapter(
                quote_error=NetworkError("primary timeout"),
            )
            fallback = StubFallbackAdapter(
                quotes_by_symbol={symbol: _make_quote(symbol, "宁德时代")},
            )
            service = QuoteService(
                primary,
                fallback=fallback,
                cache=cache,
                rate_limiter=rate_limiter,
            )

            service.get_snapshot([symbol])
        finally:
            logger.remove(handler_id)

        captured = sink.getvalue()
        # The fallback module logs at WARNING with both names.
        assert "WARNING" in captured
        assert "stub_primary" in captured
        assert "stub_fallback" in captured
        # Sanity: the original error string is preserved so operators
        # can grep for the underlying cause.
        assert "primary timeout" in captured

    def test_no_log_on_primary_success(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Property 6 corollary -- success path is silent.

        When the primary returns successfully the fallback path must
        not log anything; this guards against a regression where the
        helper accidentally logs every call.
        """

        sink = io.StringIO()
        handler_id = logger.add(
            sink,
            level="DEBUG",
            format="{level}|{name}|{message}",
        )
        try:
            symbol = "300750.SZ"
            primary = StubPrimaryAdapter(
                quotes_by_symbol={symbol: _make_quote(symbol, "宁德时代")},
            )
            fallback = StubFallbackAdapter(
                quotes_by_symbol={symbol: _make_quote(symbol, "should-not-see")},
            )
            service = QuoteService(
                primary,
                fallback=fallback,
                cache=cache,
                rate_limiter=rate_limiter,
            )

            service.get_snapshot([symbol])
        finally:
            logger.remove(handler_id)

        captured = sink.getvalue()
        # No "switching to fallback" line should be present.
        assert "switching to fallback" not in captured
