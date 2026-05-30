"""Property 14 — every tool / prompt / resource Markdown ends with DISCLAIMER.

Covers task 23.3 of the china-stock-mcp spec:

- 23.3 Property 14 — 免责声明 (**Validates: Requirements 12.1**)

For every tool function in :mod:`china_stock_mcp.tools`, every prompt
in :mod:`china_stock_mcp.prompts`, and every resource in
:mod:`china_stock_mcp.resources`, this integration test asserts:

    output.rstrip().endswith(formatters.DISCLAIMER)

So a single regression in any tool's exit pipeline (missing
``finalize_tool_output``, accidentally re-renderering after the
disclaimer was appended, etc.) is caught by exactly one test failure.

Hermetic stubs
--------------

Every test composes a ``StubAdapter`` (mirroring the pattern in
``tests/integration/test_search_quote_flow.py``) plus an in-memory
cache + a generous rate limiter so no upstream is reached and no
disk I/O happens. ``ScreenService`` additionally calls ``akshare``
directly via ``self._ak``; we inject a ``_StubAk`` instance through
``service._ak = StubAk()`` so the universe builder sees a small
deterministic dataframe instead of attempting a real HTTP call.

The :class:`MarketService` reaches the upstream through the adapter
boundary (``BaseAdapter.market_overview``), so for it we just supply
a stub ``market_overview`` payload on the :class:`StubAdapter`.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.cache import Cache, build_cache, reset_default_cache
from china_stock_mcp.config import Settings
from china_stock_mcp.formatters import DISCLAIMER
from china_stock_mcp.models import (
    FinancialPeriod,
    FinancialReport,
    FundamentalSnapshot,
    FundInfo,
    KLineBar,
    KLineSeries,
    MarketOverview,
    MoneyFlow,
    PeerTable,
    Quote,
    SymbolHit,
)
from china_stock_mcp.prompts.research_report import research_report
from china_stock_mcp.prompts.valuation_compare import valuation_compare
from china_stock_mcp.prompts.weekly_review import weekly_review
from china_stock_mcp.rate_limiter import RateLimiter
from china_stock_mcp.resources.market_overview import market_overview_resource
from china_stock_mcp.resources.north_flow import north_flow_resource
from china_stock_mcp.resources.symbol_profile import symbol_profile_resource
from china_stock_mcp.services.financial_report_service import FinancialReportService
from china_stock_mcp.services.fund_service import FundService
from china_stock_mcp.services.fundamental_service import FundamentalService
from china_stock_mcp.services.industry_service import IndustryService
from china_stock_mcp.services.kline_service import KLineService
from china_stock_mcp.services.market_service import MarketService
from china_stock_mcp.services.money_flow_service import MoneyFlowService
from china_stock_mcp.services.quote_service import QuoteService
from china_stock_mcp.services.screen_service import ScreenService
from china_stock_mcp.services.symbol_service import SymbolService
from china_stock_mcp.tools.financial import get_financial_report
from china_stock_mcp.tools.fund import get_fund_info
from china_stock_mcp.tools.fundamental import get_fundamentals
from china_stock_mcp.tools.industry import get_industry_peers
from china_stock_mcp.tools.kline import get_kline
from china_stock_mcp.tools.market_overview import get_market_overview
from china_stock_mcp.tools.money_flow import get_money_flow
from china_stock_mcp.tools.quote import get_quote
from china_stock_mcp.tools.screen import screen_stocks
from china_stock_mcp.tools.search import search_symbol

# ---------------------------------------------------------------------------
# Constants used by the canned DTO builders
# ---------------------------------------------------------------------------

_FIXED_TIMESTAMP = datetime(2024, 6, 3, 14, 30, 0, tzinfo=UTC)
_DEFAULT_SYMBOL = "300750.SZ"
_DEFAULT_BARE = "300750"
_DEFAULT_NAME = "宁德时代"
_DEFAULT_INDUSTRY = "电池"
_DEFAULT_FUND_CODE = "000001"


# ---------------------------------------------------------------------------
# Stub adapter — pre-loads sample data for every endpoint
# ---------------------------------------------------------------------------


def _make_quote(symbol: str = _DEFAULT_SYMBOL, name: str = _DEFAULT_NAME) -> Quote:
    return Quote(
        symbol=symbol,
        name=name,
        price=180.5,
        change=2.5,
        change_pct=1.4,
        volume=12_000_000,
        amount=2.16e9,
        turnover_rate=1.5,
        pe_ttm=18.5,
        pe_dynamic=17.2,
        pb=3.4,
        market_cap=8.0e11,
        float_market_cap=6.5e11,
        timestamp=_FIXED_TIMESTAMP,
        delay_seconds=900,
    )


def _make_kline_series(
    symbol: str = _DEFAULT_SYMBOL,
    period: str = "daily",
    adjust: str = "qfq",
    *,
    bar_count: int = 5,
) -> KLineSeries:
    bars: list[KLineBar] = []
    base_date = date(2024, 5, 1)
    base_price = 180.0
    for i in range(bar_count):
        bars.append(
            KLineBar(
                date=date(base_date.year, base_date.month, max(1, (base_date.day + i) % 28 or 1)),
                open=base_price + i,
                high=base_price + i + 2,
                low=base_price + i - 1,
                close=base_price + i + 1,
                volume=1_000_000 + i * 10_000,
                amount=1.8e8 + i * 1.0e6,
            )
        )
    # ``period`` / ``adjust`` are ``Literal``s on the model; ``cast`` via
    # plain str is fine because we always pass valid values from the
    # service.
    return KLineSeries(
        symbol=symbol,
        period=period,  # type: ignore[arg-type]
        adjust=adjust,  # type: ignore[arg-type]
        bars=bars,
        indicators={},
        pattern_note=None,
    )


def _make_snapshot(symbol: str = _DEFAULT_SYMBOL) -> FundamentalSnapshot:
    return FundamentalSnapshot(
        symbol=symbol,
        valuation={"pe_ttm": 18.5, "pe_dynamic": 17.2, "pb": 3.4},
        profitability={"roe": 18.0, "gross_margin": 28.0, "net_margin": 12.0},
        growth={"revenue_yoy": 22.5, "net_profit_yoy": 18.5},
        health={"debt_ratio": 45.0, "current_ratio": 1.8},
        industry_percentile={},
    )


def _make_financial_report(
    symbol: str = _DEFAULT_SYMBOL,
    report_type: str = "annual",
    periods: int = 4,
) -> FinancialReport:
    out: list[FinancialPeriod] = []
    for i in range(periods):
        # Use distinct ``period_end`` values so the model's stable-sort
        # invariant has something to anchor on.
        year = 2020 + i
        out.append(
            FinancialPeriod(
                period_end=date(year, 12, 31),
                revenue=4.0e11 + i * 1e10,
                net_profit=4.0e10 + i * 1e9,
                net_profit_excl_nrgl=3.5e10 + i * 1e9,
                gross_profit=1.0e11 + i * 1e9,
                operating_cash_flow=5.0e10 + i * 1e9,
                total_assets=8.0e11 + i * 1e10,
                total_liabilities=4.0e11 + i * 5e9,
                equity=4.0e11 + i * 5e9,
            )
        )
    return FinancialReport(
        symbol=symbol,
        report_type=report_type,  # type: ignore[arg-type]
        periods=out,
    )


def _make_money_flow(flow_type: str = "north", row_count: int = 5) -> MoneyFlow:
    rows: list[dict[str, object]] = []
    for i in range(row_count):
        rows.append(
            {
                "date": f"2024-06-{(i % 28) + 1:02d}",
                "净流入金额": 1.0e9 + i * 1e8,
                "买入金额": 5.0e9 + i * 1e8,
                "卖出金额": 4.0e9 + i * 1e8,
                "持股市值": 2.5e12,
            }
        )
    return MoneyFlow(
        flow_type=flow_type,  # type: ignore[arg-type]
        rows=rows,
        snapshot_at=_FIXED_TIMESTAMP,
    )


def _make_peer_table(
    base_symbol: str = _DEFAULT_SYMBOL,
    metrics: list[str] | None = None,
    row_count: int = 8,
) -> PeerTable:
    metrics = metrics if metrics is not None else ["pe", "pb", "roe", "revenue_growth"]
    rows: list[dict[str, object]] = []
    for i in range(row_count):
        row: dict[str, object] = {
            "代码": f"{300000 + i:06d}",
            "名称": f"对标{i:02d}",
        }
        if "pe" in metrics:
            row["pe"] = 15.0 + i
        if "pb" in metrics:
            row["pb"] = 2.0 + i * 0.5
        if "roe" in metrics:
            row["roe"] = 10.0 + i
        if "revenue_growth" in metrics:
            row["revenue_growth"] = 5.0 + i * 2
        rows.append(row)
    return PeerTable(
        base_symbol=base_symbol,
        industry=_DEFAULT_INDUSTRY,
        metrics=list(metrics),
        rows=rows,
    )


def _make_fund_info(code: str = _DEFAULT_FUND_CODE) -> FundInfo:
    return FundInfo(
        code=code,
        name=f"测试基金{code}",
        manager="张三",
        inception_date=date(2018, 1, 1),
        aum=5.0e9,
        return_1m=1.5,
        return_3m=3.5,
        return_6m=8.0,
        return_12m=12.0,
        max_drawdown=-15.0,
        sharpe=1.8,
        rank_in_category="42/345",
        top_holdings=[
            {"symbol": "300750", "name": "宁德时代", "weight": 8.5},
            {"symbol": "002594", "name": "比亚迪", "weight": 7.0},
        ],
        industry_distribution=[
            {"industry": "电池", "weight": 30.0},
            {"industry": "汽车", "weight": 20.0},
        ],
    )


def _make_market_overview() -> MarketOverview:
    return MarketOverview(
        indices=[
            {"name": "上证指数", "code": "000001", "last": 3050.0, "change_pct": 0.85},
            {"name": "深证成指", "code": "399001", "last": 9500.0, "change_pct": 1.20},
            {"name": "创业板指", "code": "399006", "last": 1900.0, "change_pct": 1.50},
        ],
        advance_decline={"advance": 3000, "decline": 1500, "flat": 200},
        limit_stats={"limit_up": 50, "limit_down": 10},
        north_net_inflow=1.5e9,
        top_inflow_industries=[
            {"name": "电池", "net_inflow": 5e8},
            {"name": "汽车", "net_inflow": 3e8},
        ],
        heat_score=72.5,
        snapshot_at=_FIXED_TIMESTAMP,
    )


class StubAdapter(BaseAdapter):
    """In-memory :class:`BaseAdapter` covering every endpoint.

    Mirrors the StubAdapter pattern from
    ``tests/integration/test_search_quote_flow.py`` but with sample
    data preloaded for every adapter method. Each method records its
    invocation count so tests can assert the cache short-circuited a
    second call when needed.
    """

    name: str = "stub"

    def __init__(self) -> None:
        self.search_calls = 0
        self.quote_calls = 0
        self.kline_calls = 0
        self.fundamentals_calls = 0
        self.financial_calls = 0
        self.money_flow_calls = 0
        self.peers_calls = 0
        self.fund_info_calls = 0
        self.market_calls = 0

    def search(self, query: str, market: str) -> list[SymbolHit]:
        self.search_calls += 1
        return [
            SymbolHit(
                code=_DEFAULT_SYMBOL,
                name=_DEFAULT_NAME,
                market="a_stock",
                industry=_DEFAULT_INDUSTRY,
            ),
            SymbolHit(
                code="002594.SZ",
                name="比亚迪",
                market="a_stock",
                industry="新能源车",
            ),
        ]

    def quote(self, symbols: list[str]) -> list[Quote]:
        self.quote_calls += 1
        return [_make_quote(symbol=s, name=f"标的{s}") for s in symbols]

    def kline(
        self,
        symbol: str,
        period: str,
        count: int,
        adjust: str,
    ) -> KLineSeries:
        self.kline_calls += 1
        # Hand back enough bars to exercise the rendering paths but
        # well below the 250-bar cap so the test stays fast.
        return _make_kline_series(
            symbol=symbol,
            period=period,
            adjust=adjust,
            bar_count=min(count, 60),
        )

    def fundamentals(self, symbol: str) -> FundamentalSnapshot:
        self.fundamentals_calls += 1
        return _make_snapshot(symbol)

    def financial_report(
        self,
        symbol: str,
        report_type: str,
        periods: int,
    ) -> FinancialReport:
        self.financial_calls += 1
        return _make_financial_report(
            symbol=symbol,
            report_type=report_type,
            periods=periods,
        )

    def money_flow(
        self,
        symbol: str | None,
        flow_type: str,
        top_n: int,
    ) -> MoneyFlow:
        self.money_flow_calls += 1
        return _make_money_flow(flow_type=flow_type, row_count=min(top_n, 5))

    def industry_peers(
        self,
        symbol: str,
        metrics: list[str],
        top_n: int,
    ) -> PeerTable:
        self.peers_calls += 1
        return _make_peer_table(
            base_symbol=symbol,
            metrics=list(metrics),
            row_count=min(top_n, 8),
        )

    def fund_info(self, fund_code: str) -> FundInfo:
        self.fund_info_calls += 1
        return _make_fund_info(code=fund_code)

    def market_overview(self) -> MarketOverview:
        self.market_calls += 1
        return _make_market_overview()


# ---------------------------------------------------------------------------
# Stub akshare module for ScreenService (which calls ak directly)
# ---------------------------------------------------------------------------


class _StubAk:
    """Minimal ``akshare`` stand-in for :class:`ScreenService`.

    ``ScreenService._fetch_universe`` calls
    ``ak.stock_zh_a_spot_em()`` (and optionally
    ``ak.stock_board_industry_cons_em(symbol=...)``); it does not
    touch any other module attribute. Returning a small dataframe
    here keeps the test hermetic.
    """

    def stock_zh_a_spot_em(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "代码": "300750",
                    "名称": "宁德时代",
                    "市盈率-动态": 18.5,
                    "市净率": 3.4,
                    "总市值": 8.0e11,
                },
                {
                    "代码": "002594",
                    "名称": "比亚迪",
                    "市盈率-动态": 22.0,
                    "市净率": 4.0,
                    "总市值": 6.5e11,
                },
                {
                    "代码": "600519",
                    "名称": "贵州茅台",
                    "市盈率-动态": 28.0,
                    "市净率": 9.5,
                    "总市值": 2.1e12,
                },
            ]
        )

    def stock_board_industry_cons_em(self, symbol: str) -> pd.DataFrame:
        # Only invoked when the caller passes ``criteria.industry``;
        # tests in this module never set ``industry`` so this method
        # should not be reached. Returning an empty frame keeps the
        # universe builder safe against future test additions.
        return pd.DataFrame()


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
    backend = build_cache(Settings(cache_backend="disk", cache_dir=tmp_path))
    try:
        yield backend
    finally:
        backend.close()


@pytest.fixture()
def rate_limiter() -> RateLimiter:
    """Generous limiter so tests never trip the budget accidentally."""

    return RateLimiter(rate_per_minute=10_000)


@pytest.fixture()
def adapter() -> StubAdapter:
    return StubAdapter()


# ---------------------------------------------------------------------------
# Helper assertions
# ---------------------------------------------------------------------------


def _assert_disclaimer_tail(markdown: str) -> None:
    """Single trailing-disclaimer invariant (Property 14)."""

    assert markdown.rstrip().endswith(DISCLAIMER), (
        f"output must end with DISCLAIMER, got tail: {markdown[-200:]!r}"
    )
    # ``finalize_tool_output`` is idempotent and the prompts re-strip
    # the per-tool disclaimer before re-applying their own; the
    # cross-tool invariant we care about is "ends with DISCLAIMER",
    # which the assertion above already enforces.


# ---------------------------------------------------------------------------
# Tools — every Markdown payload ends with DISCLAIMER
# ---------------------------------------------------------------------------


class TestToolsDisclaimer:
    """Property 14 / Requirement 12.1 — every tool's output ends with DISCLAIMER."""

    def test_search_symbol(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        service = SymbolService(adapter, cache=cache, rate_limiter=rate_limiter)
        out = search_symbol(service, query="宁德", market="all")
        _assert_disclaimer_tail(out)

    def test_get_quote_single(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        service = QuoteService(adapter, cache=cache, rate_limiter=rate_limiter)
        out = get_quote(
            service,
            _DEFAULT_SYMBOL,
            settings=Settings(
                cache_backend="disk",
                cache_dir=Path("."),
                data_delay_notice=False,
            ),
        )
        _assert_disclaimer_tail(out)

    def test_get_quote_batch(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        service = QuoteService(adapter, cache=cache, rate_limiter=rate_limiter)
        out = get_quote(
            service,
            [_DEFAULT_SYMBOL, "002594.SZ"],
            settings=Settings(
                cache_backend="disk",
                cache_dir=Path("."),
                data_delay_notice=False,
            ),
        )
        _assert_disclaimer_tail(out)

    def test_get_kline(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        service = KLineService(adapter, cache=cache, rate_limiter=rate_limiter)
        out = get_kline(
            service,
            symbol=_DEFAULT_SYMBOL,
            period="daily",
            count=60,
            adjust="qfq",
            indicators=["MA20", "MA60", "MACD"],
        )
        _assert_disclaimer_tail(out)

    def test_get_fundamentals(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        service = FundamentalService(adapter, cache=cache, rate_limiter=rate_limiter)
        out = get_fundamentals(service, _DEFAULT_SYMBOL)
        _assert_disclaimer_tail(out)

    def test_get_financial_report(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        service = FinancialReportService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )
        out = get_financial_report(
            service,
            symbol=_DEFAULT_SYMBOL,
            report_type="annual",
            periods=4,
        )
        _assert_disclaimer_tail(out)

    def test_get_money_flow_north(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        service = MoneyFlowService(adapter, cache=cache, rate_limiter=rate_limiter)
        out = get_money_flow(service, symbol=None, flow_type="north", top_n=10)
        _assert_disclaimer_tail(out)

    def test_get_money_flow_main(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        service = MoneyFlowService(adapter, cache=cache, rate_limiter=rate_limiter)
        out = get_money_flow(
            service,
            symbol=_DEFAULT_SYMBOL,
            flow_type="main",
            top_n=10,
        )
        _assert_disclaimer_tail(out)

    def test_get_industry_peers(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        service = IndustryService(adapter, cache=cache, rate_limiter=rate_limiter)
        out = get_industry_peers(
            service,
            symbol=_DEFAULT_SYMBOL,
            metrics=["pe", "pb", "roe", "revenue_growth"],
            top_n=10,
        )
        _assert_disclaimer_tail(out)

    def test_get_fund_info(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        service = FundService(adapter, cache=cache, rate_limiter=rate_limiter)
        out = get_fund_info(service, _DEFAULT_FUND_CODE)
        _assert_disclaimer_tail(out)

    def test_screen_stocks(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        # ``ScreenService`` calls ``akshare`` directly via
        # ``self._ak``; inject a stub so the test stays hermetic.
        service = ScreenService(adapter, cache=cache, rate_limiter=rate_limiter)
        service._ak = _StubAk()
        out = screen_stocks(
            service,
            criteria={"pe_ttm": {"min": 0, "max": 50}},
            sort_by="market_cap",
            order="desc",
            limit=30,
        )
        _assert_disclaimer_tail(out)

    def test_screen_stocks_no_criteria(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Empty criteria still surface the disclaimer (the universe is
        small but non-empty thanks to the stub)."""

        service = ScreenService(adapter, cache=cache, rate_limiter=rate_limiter)
        service._ak = _StubAk()
        out = screen_stocks(service)
        _assert_disclaimer_tail(out)

    def test_get_market_overview(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        service = MarketService(adapter, cache=cache, rate_limiter=rate_limiter)
        out = get_market_overview(service)
        _assert_disclaimer_tail(out)


# ---------------------------------------------------------------------------
# Prompts — every prompt's Markdown ends with DISCLAIMER
# ---------------------------------------------------------------------------


def _build_prompt_services(
    adapter: StubAdapter,
    cache: Cache,
    rate_limiter: RateLimiter,
) -> dict[str, Any]:
    """Compose every service the three prompts may need."""

    return {
        "fundamental": FundamentalService(
            adapter, cache=cache, rate_limiter=rate_limiter
        ),
        "financial_report": FinancialReportService(
            adapter, cache=cache, rate_limiter=rate_limiter
        ),
        "industry": IndustryService(
            adapter, cache=cache, rate_limiter=rate_limiter
        ),
        "money_flow": MoneyFlowService(
            adapter, cache=cache, rate_limiter=rate_limiter
        ),
        "kline": KLineService(adapter, cache=cache, rate_limiter=rate_limiter),
        "quote": QuoteService(adapter, cache=cache, rate_limiter=rate_limiter),
        "market": MarketService(adapter, cache=cache, rate_limiter=rate_limiter),
        "symbol": SymbolService(adapter, cache=cache, rate_limiter=rate_limiter),
    }


class TestPromptsDisclaimer:
    """Property 14 / Requirement 10.6 — every prompt ends with DISCLAIMER."""

    def test_research_report(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        services = _build_prompt_services(adapter, cache, rate_limiter)
        out = research_report(
            _DEFAULT_SYMBOL,
            report_length="standard",
            services={
                "fundamental": services["fundamental"],
                "financial_report": services["financial_report"],
                "industry": services["industry"],
                "money_flow": services["money_flow"],
                "kline": services["kline"],
            },  # type: ignore[arg-type]
        )
        _assert_disclaimer_tail(out)

    def test_valuation_compare(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        services = _build_prompt_services(adapter, cache, rate_limiter)
        out = valuation_compare(
            [_DEFAULT_SYMBOL, "002594.SZ"],
            services={
                "quote": services["quote"],
                "fundamental": services["fundamental"],
                "industry": services["industry"],
            },  # type: ignore[arg-type]
        )
        _assert_disclaimer_tail(out)

    def test_weekly_review(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        services = _build_prompt_services(adapter, cache, rate_limiter)
        out = weekly_review(
            services={
                "market": services["market"],
                "money_flow": services["money_flow"],
            },  # type: ignore[arg-type]
        )
        _assert_disclaimer_tail(out)


# ---------------------------------------------------------------------------
# Resources — every resource's Markdown ends with DISCLAIMER
# ---------------------------------------------------------------------------


class TestResourcesDisclaimer:
    """Property 14 / Requirement 12.1 — every resource ends with DISCLAIMER."""

    def test_market_overview_resource(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        market = MarketService(adapter, cache=cache, rate_limiter=rate_limiter)
        out = market_overview_resource(services={"market": market})
        _assert_disclaimer_tail(out)

    def test_north_flow_resource(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        flow = MoneyFlowService(adapter, cache=cache, rate_limiter=rate_limiter)
        out = north_flow_resource(services={"money_flow": flow})
        _assert_disclaimer_tail(out)

    def test_symbol_profile_resource(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        symbol_service = SymbolService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )
        fundamental_service = FundamentalService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )
        out = symbol_profile_resource(
            _DEFAULT_SYMBOL,
            services={
                "symbol": symbol_service,
                "fundamental": fundamental_service,
            },
        )
        _assert_disclaimer_tail(out)
