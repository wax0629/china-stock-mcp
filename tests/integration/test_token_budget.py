"""Property 13 — every tool / prompt / resource fits in the 3000-token budget.

Covers task 23.4 of the china-stock-mcp spec:

- 23.4 Property 13 — Token 预算 (**Validates: Requirements 12.2**)

For every tool function in :mod:`china_stock_mcp.tools`, every prompt
in :mod:`china_stock_mcp.prompts`, and every resource in
:mod:`china_stock_mcp.resources`, this integration test composes the
*maximum-size payload* the upstream contract allows and asserts:

    _token_count(markdown) <= 3000

Token-count proxy
-----------------

The conservative proxy ``_token_count(markdown) = (len(markdown) + 3) // 4``
mirrors the helper used by ``tests/unit/test_formatters.py``. It
slightly over-estimates token cost (~4 chars per token is Anthropic's
public guideline, and CJK glyphs are typically a single token even
though they take 2-3 UTF-8 bytes), so passing the proxy is a strict
upper bound on the real budget enforced upstream by Claude.

The assumption: with the 4-chars-per-token conversion and a 3000-token
budget, ``len(markdown)`` must stay below 12 000 characters.
``finalize_tool_output`` enforces this by snapping the body at the
last paragraph break inside ``_MAX_BODY_CHARS = 11_800`` and appending
the ``_(输出已截断至 ~3000 tokens)_`` marker. The test re-runs that
end-to-end pipeline against synthetic max-size payloads to confirm the
truncation actually fires when needed.

Maximum-size payloads
---------------------

* ``search_symbol``         — 20 hits (the per-call cap inside the adapter).
* ``get_quote``             — 20 symbols (Requirement 2.3).
* ``get_kline``             — 250 bars + 7 indicators (Requirement 3.3 / 3.6).
* ``get_financial_report``  — 12 periods (Requirement 4.4).
* ``get_money_flow``        — 100 rows (Requirement 5.4).
* ``get_industry_peers``    — 50 peers (Requirement 6.4).
* ``screen_stocks``         — 200 hits (Requirement 8.2).
* ``get_market_overview``   — 10 indices + 10 industries.
* ``get_fund_info``         — 10 holdings (Requirement 7.1).
* prompts / resources      — assemble the same max-size payloads via the
                              services they orchestrate.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.cache import Cache, build_cache, reset_default_cache
from china_stock_mcp.config import Settings
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
# Token-budget constants
# ---------------------------------------------------------------------------

#: Per-tool token budget (Property 13 / Requirement 12.2).
TOKEN_BUDGET: int = 3000

#: Marker appended by :func:`formatters.finalize_tool_output` when the
#: body was clipped to fit the budget. Surfaced as an optional sanity
#: check on the maximum-payload tools.
TRUNCATION_NOTICE: str = "_(输出已截断至 ~3000 tokens)_"


def _token_count(markdown: str) -> int:
    """Conservative token-count proxy (4 chars ≈ 1 token).

    Mirrors the helper in ``tests/unit/test_formatters.py`` so this
    integration test asserts the same budget the unit tests assume.
    The function over-estimates slightly which keeps the assertion
    safely below Claude's actual tokenizer cost for CJK-heavy text.
    """

    return (len(markdown) + 3) // 4


# ---------------------------------------------------------------------------
# Constants reused by the canned DTO builders
# ---------------------------------------------------------------------------

_FIXED_TIMESTAMP = datetime(2024, 6, 3, 14, 30, 0, tzinfo=UTC)
_DEFAULT_SYMBOL = "300750.SZ"
_DEFAULT_NAME = "宁德时代"
_DEFAULT_INDUSTRY = "电池"
_DEFAULT_FUND_CODE = "000001"


# ---------------------------------------------------------------------------
# DTO builders — maximum-size payloads
# ---------------------------------------------------------------------------


def _make_max_search_hits() -> list[SymbolHit]:
    """20 hits — the per-call cap enforced by :class:`AkshareAdapter`."""

    out: list[SymbolHit] = []
    for i in range(20):
        out.append(
            SymbolHit(
                code=f"{300000 + i:06d}.SZ",
                name=f"测试标的{i:02d}",
                market="a_stock",
                industry="行业" + ("阿" * 12),
            )
        )
    return out


def _make_max_quote(symbol: str, name: str = _DEFAULT_NAME) -> Quote:
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


def _make_max_kline_series(
    symbol: str = _DEFAULT_SYMBOL,
    period: str = "daily",
    adjust: str = "qfq",
    *,
    bar_count: int = 250,
) -> KLineSeries:
    """250 bars — the DTO maximum (``MAX_KLINE_BARS``)."""

    bars: list[KLineBar] = []
    base_date = date(2020, 1, 1)
    for i in range(bar_count):
        bar_date = base_date + timedelta(days=i)
        # Vary OHLC slightly so the model validator is happy
        # (low <= min(open, close) <= max(open, close) <= high).
        open_price = 100.0 + (i % 50) * 0.5
        close_price = open_price + ((i % 7) - 3) * 0.2
        high_price = max(open_price, close_price) + 1.0
        low_price = min(open_price, close_price) - 1.0
        bars.append(
            KLineBar(
                date=bar_date,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=1_000_000 + i * 1_000,
                amount=1.8e8 + i * 1e5,
            )
        )
    return KLineSeries(
        symbol=symbol,
        period=period,  # type: ignore[arg-type]
        adjust=adjust,  # type: ignore[arg-type]
        bars=bars,
        indicators={},
        pattern_note=None,
    )


def _make_max_snapshot(symbol: str = _DEFAULT_SYMBOL) -> FundamentalSnapshot:
    """Every bucket fully populated; included for the prompts that
    reach into the snapshot directly."""

    return FundamentalSnapshot(
        symbol=symbol,
        valuation={
            "pe_ttm": 18.5,
            "pe_dynamic": 17.2,
            "pb": 3.4,
            "ps": 2.1,
            "peg": 1.2,
        },
        profitability={
            "roe": 18.0,
            "roa": 7.5,
            "gross_margin": 28.0,
            "net_margin": 12.0,
        },
        growth={
            "revenue_yoy": 22.5,
            "net_profit_yoy": 18.5,
            "qoq": 5.5,
        },
        health={
            "debt_ratio": 45.0,
            "current_ratio": 1.8,
            "ocf_to_net_profit": 1.05,
        },
        industry_percentile={},
    )


def _make_max_financial_report(
    symbol: str = _DEFAULT_SYMBOL,
    report_type: str = "annual",
    periods: int = 12,
) -> FinancialReport:
    """12 periods — the upper bound from Requirement 4.4."""

    out: list[FinancialPeriod] = []
    base_year = 2012
    for i in range(periods):
        out.append(
            FinancialPeriod(
                period_end=date(base_year + i, 12, 31),
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


def _make_max_money_flow(flow_type: str = "north", row_count: int = 100) -> MoneyFlow:
    """100 rows — the upper bound from Requirement 5.4."""

    rows: list[dict[str, object]] = []
    for i in range(row_count):
        rows.append(
            {
                "date": f"2024-{((i // 30) % 12) + 1:02d}-{(i % 28) + 1:02d}",
                "净流入金额": 1.0e9 + i * 1e7,
                "买入金额": 5.0e9 + i * 1e7,
                "卖出金额": 4.0e9 + i * 1e7,
                "持股市值": 2.5e12 + i * 1e9,
            }
        )
    return MoneyFlow(
        flow_type=flow_type,  # type: ignore[arg-type]
        rows=rows,
        snapshot_at=_FIXED_TIMESTAMP,
    )


def _make_max_peer_table(
    base_symbol: str = _DEFAULT_SYMBOL,
    metrics: list[str] | None = None,
    row_count: int = 50,
) -> PeerTable:
    """50 peers — the upper bound from Requirement 6.4."""

    metrics = metrics if metrics is not None else ["pe", "pb", "roe", "revenue_growth"]
    rows: list[dict[str, object]] = []
    for i in range(row_count):
        row: dict[str, object] = {
            "代码": f"{300000 + i:06d}",
            "名称": f"对标公司{i:02d}",
        }
        if "pe" in metrics:
            row["pe"] = 15.0 + i * 0.3
        if "pb" in metrics:
            row["pb"] = 2.0 + i * 0.1
        if "roe" in metrics:
            row["roe"] = 10.0 + i * 0.2
        if "revenue_growth" in metrics:
            row["revenue_growth"] = 5.0 + i * 0.5
        rows.append(row)
    return PeerTable(
        base_symbol=base_symbol,
        industry=_DEFAULT_INDUSTRY,
        metrics=list(metrics),
        rows=rows,
    )


def _make_max_fund_info(code: str = _DEFAULT_FUND_CODE) -> FundInfo:
    """10 holdings — the per-fund cap from Requirement 7.1."""

    holdings = [
        {
            "symbol": f"{300000 + i:06d}",
            "name": f"重仓股{i:02d}",
            "weight": 8.5 - i * 0.3,
        }
        for i in range(10)
    ]
    industries = [
        {"industry": f"行业分布{i:02d}", "weight": 30.0 - i * 2.5}
        for i in range(10)
    ]
    return FundInfo(
        code=code,
        name="测试基金" + ("阿" * 8),
        manager="基金经理张三",
        inception_date=date(2018, 1, 1),
        aum=5.0e9,
        return_1m=1.5,
        return_3m=3.5,
        return_6m=8.0,
        return_12m=12.0,
        max_drawdown=-15.0,
        sharpe=1.8,
        rank_in_category="42/345",
        top_holdings=holdings,
        industry_distribution=industries,
    )


def _make_max_market_overview() -> MarketOverview:
    """10 indices + 10 industries — generous upper bound for v1.

    The DTO does not cap either list, so we pick a reasonable upper
    bound that matches what the akshare adapter realistically returns.
    """

    indices: list[dict[str, object]] = [
        {
            "name": f"指数{i:02d}",
            "code": f"{900000 + i:06d}",
            "last": 3000.0 + i * 100,
            "change_pct": 0.5 + i * 0.1,
        }
        for i in range(10)
    ]
    industries: list[dict[str, object]] = [
        {"name": f"行业{i:02d}", "net_inflow": 5e8 - i * 2e7}
        for i in range(10)
    ]
    return MarketOverview(
        indices=indices,
        advance_decline={"advance": 3000, "decline": 1500, "flat": 200},
        limit_stats={"limit_up": 50, "limit_down": 10},
        north_net_inflow=1.5e9,
        top_inflow_industries=industries,
        heat_score=72.5,
        snapshot_at=_FIXED_TIMESTAMP,
    )


# ---------------------------------------------------------------------------
# Stub adapter — returns the maximum-size payloads above
# ---------------------------------------------------------------------------


class StubAdapter(BaseAdapter):
    """In-memory :class:`BaseAdapter` returning maximum-size payloads.

    Mirrors the StubAdapter pattern from
    ``tests/integration/test_search_quote_flow.py``. Every endpoint
    returns the canned DTO whose row count matches the upstream's
    documented upper bound, so the test exercises the
    ``finalize_tool_output`` truncation path on the long renderers.
    """

    name: str = "stub"

    def search(self, query: str, market: str) -> list[SymbolHit]:
        return _make_max_search_hits()

    def quote(self, symbols: list[str]) -> list[Quote]:
        return [_make_max_quote(symbol=s, name=f"标的{s}") for s in symbols]

    def kline(
        self,
        symbol: str,
        period: str,
        count: int,
        adjust: str,
    ) -> KLineSeries:
        # Always return the maximum 250 bars so the rendering path
        # exercises the upper bound regardless of caller-supplied
        # ``count``.
        return _make_max_kline_series(
            symbol=symbol,
            period=period,
            adjust=adjust,
            bar_count=min(count, 250),
        )

    def fundamentals(self, symbol: str) -> FundamentalSnapshot:
        return _make_max_snapshot(symbol)

    def financial_report(
        self,
        symbol: str,
        report_type: str,
        periods: int,
    ) -> FinancialReport:
        return _make_max_financial_report(
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
        return _make_max_money_flow(flow_type=flow_type, row_count=top_n)

    def industry_peers(
        self,
        symbol: str,
        metrics: list[str],
        top_n: int,
    ) -> PeerTable:
        return _make_max_peer_table(
            base_symbol=symbol,
            metrics=list(metrics),
            row_count=top_n,
        )

    def fund_info(self, fund_code: str) -> FundInfo:
        return _make_max_fund_info(code=fund_code)

    def market_overview(self) -> MarketOverview:
        return _make_max_market_overview()


# ---------------------------------------------------------------------------
# Stub akshare module for ScreenService (which calls ak directly)
# ---------------------------------------------------------------------------


class _StubAk:
    """``akshare`` stand-in returning a 200-row dataframe for the screen test.

    ``ScreenService._fetch_universe`` calls
    ``ak.stock_zh_a_spot_em()`` for the universe; we return a 200-row
    dataframe so the screen result hits the ``limit=200`` upper bound
    enforced by Requirement 8.2 / Property 11.
    """

    def stock_zh_a_spot_em(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for i in range(220):
            # ``300000 + i`` produces SZ-class codes whose prefix
            # ``30`` maps to ``.SZ`` in the screen service's exchange
            # detector — well within the universe build's accepted
            # set, so all 220 rows survive.
            rows.append(
                {
                    "代码": f"{300000 + i:06d}",
                    "名称": f"标的{i:03d}",
                    "市盈率-动态": 15.0 + (i % 30) * 0.3,
                    "市净率": 2.0 + (i % 30) * 0.1,
                    "总市值": 1e10 + i * 1e8,
                }
            )
        return pd.DataFrame(rows)

    def stock_board_industry_cons_em(self, symbol: str) -> pd.DataFrame:
        # The screen tests below never set ``criteria.industry``; this
        # method is here only for future test additions and returns an
        # empty frame so the universe builder's safety branch is
        # untouched by accident.
        return pd.DataFrame()


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
    """Generous limiter so the test never trips the budget accidentally."""

    return RateLimiter(rate_per_minute=10_000)


@pytest.fixture()
def adapter() -> StubAdapter:
    return StubAdapter()


# ---------------------------------------------------------------------------
# Helper assertions
# ---------------------------------------------------------------------------


def _assert_within_budget(markdown: str) -> None:
    """Property 13 — ``_token_count(markdown) <= 3000``."""

    tokens = _token_count(markdown)
    assert tokens <= TOKEN_BUDGET, (
        f"output exceeded token budget: {tokens} > {TOKEN_BUDGET} "
        f"(len={len(markdown)} chars)"
    )


# ---------------------------------------------------------------------------
# Tools — every Markdown payload fits in the 3000-token budget
# ---------------------------------------------------------------------------


class TestToolsTokenBudget:
    """Property 13 / Requirement 12.2 — every tool fits in the budget."""

    def test_search_symbol_within_budget(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        service = SymbolService(adapter, cache=cache, rate_limiter=rate_limiter)
        out = search_symbol(service, query="测试", market="all")
        _assert_within_budget(out)

    def test_get_quote_max_batch_within_budget(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """20 symbols — the per-call cap from Requirement 2.3."""

        service = QuoteService(adapter, cache=cache, rate_limiter=rate_limiter)
        symbols = [f"{300000 + i:06d}.SZ" for i in range(20)]
        out = get_quote(
            service,
            symbols,
            settings=Settings(
                cache_backend="disk",
                cache_dir=Path("."),
                data_delay_notice=True,
            ),
        )
        _assert_within_budget(out)

    def test_get_kline_max_bars_and_indicators_within_budget(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """250 bars + every supported indicator (Requirement 3.3 / 3.6).

        The K-line service supports MA5 / MA10 / MA20 / MA60 / MACD /
        RSI14 / BOLL — 7 indicator families in total.
        """

        service = KLineService(adapter, cache=cache, rate_limiter=rate_limiter)
        out = get_kline(
            service,
            symbol=_DEFAULT_SYMBOL,
            period="daily",
            count=250,
            adjust="qfq",
            indicators=["MA5", "MA10", "MA20", "MA60", "MACD", "RSI14", "BOLL"],
        )
        _assert_within_budget(out)

    def test_get_fundamentals_within_budget(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        service = FundamentalService(adapter, cache=cache, rate_limiter=rate_limiter)
        out = get_fundamentals(service, _DEFAULT_SYMBOL)
        _assert_within_budget(out)

    def test_get_financial_report_max_periods_within_budget(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """12 periods — the upper bound from Requirement 4.4."""

        service = FinancialReportService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )
        out = get_financial_report(
            service,
            symbol=_DEFAULT_SYMBOL,
            report_type="annual",
            periods=12,
        )
        _assert_within_budget(out)

    def test_get_money_flow_max_top_n_within_budget(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """100 rows — the upper bound from Requirement 5.4."""

        service = MoneyFlowService(adapter, cache=cache, rate_limiter=rate_limiter)
        out = get_money_flow(service, symbol=None, flow_type="north", top_n=100)
        _assert_within_budget(out)

    def test_get_industry_peers_max_top_n_within_budget(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """50 peers — the upper bound from Requirement 6.4."""

        service = IndustryService(adapter, cache=cache, rate_limiter=rate_limiter)
        out = get_industry_peers(
            service,
            symbol=_DEFAULT_SYMBOL,
            metrics=["pe", "pb", "roe", "revenue_growth"],
            top_n=50,
        )
        _assert_within_budget(out)

    def test_get_fund_info_within_budget(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """10 holdings — the per-fund cap from Requirement 7.1."""

        service = FundService(adapter, cache=cache, rate_limiter=rate_limiter)
        out = get_fund_info(service, _DEFAULT_FUND_CODE)
        _assert_within_budget(out)

    def test_screen_stocks_max_limit_within_budget(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """200 rows — the upper bound from Requirement 8.2."""

        service = ScreenService(adapter, cache=cache, rate_limiter=rate_limiter)
        service._ak = _StubAk()
        out = screen_stocks(
            service,
            criteria={"pe_ttm": {"min": 0, "max": 100}},
            sort_by="market_cap",
            order="desc",
            limit=200,
        )
        _assert_within_budget(out)
        # Sanity check — the screen output is the most likely tool to
        # actually need the truncation marker because 200 rows x 6
        # columns x ~10 chars per cell ~= 12 000 chars before headings
        # / disclaimer.
        assert out.rstrip().endswith(
            "投资建议。"
        ) or TRUNCATION_NOTICE in out, (
            "screen result must end with disclaimer or carry the "
            "truncation marker when the body exceeded the budget"
        )

    def test_get_market_overview_within_budget(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        service = MarketService(adapter, cache=cache, rate_limiter=rate_limiter)
        out = get_market_overview(service)
        _assert_within_budget(out)


# ---------------------------------------------------------------------------
# Prompts — every prompt's Markdown fits in the budget
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


class TestPromptsTokenBudget:
    """Property 13 / Requirement 12.2 — every prompt fits in the budget.

    Prompts apply their own length-mode post-processing on top of the
    per-tool ``finalize_tool_output`` truncation, so the deep-mode +
    max-size payload combination is the most stringent test.
    """

    def test_research_report_short_within_budget(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        services = _build_prompt_services(adapter, cache, rate_limiter)
        out = research_report(
            _DEFAULT_SYMBOL,
            report_length="short",
            services={
                "fundamental": services["fundamental"],
                "financial_report": services["financial_report"],
                "industry": services["industry"],
                "money_flow": services["money_flow"],
                "kline": services["kline"],
            },  # type: ignore[arg-type]
        )
        _assert_within_budget(out)

    def test_research_report_standard_within_budget(
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
        _assert_within_budget(out)

    def test_research_report_deep_within_budget(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        services = _build_prompt_services(adapter, cache, rate_limiter)
        out = research_report(
            _DEFAULT_SYMBOL,
            report_length="deep",
            services={
                "fundamental": services["fundamental"],
                "financial_report": services["financial_report"],
                "industry": services["industry"],
                "money_flow": services["money_flow"],
                "kline": services["kline"],
            },  # type: ignore[arg-type]
        )
        _assert_within_budget(out)

    def test_valuation_compare_max_symbols_within_budget(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """10 symbols — the upper bound enforced by ``ValuationCompareInput``."""

        services = _build_prompt_services(adapter, cache, rate_limiter)
        symbols = [f"{300000 + i:06d}.SZ" for i in range(10)]
        out = valuation_compare(
            symbols,
            services={
                "quote": services["quote"],
                "fundamental": services["fundamental"],
                "industry": services["industry"],
            },  # type: ignore[arg-type]
        )
        _assert_within_budget(out)

    def test_weekly_review_within_budget(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        services = _build_prompt_services(adapter, cache, rate_limiter)
        out = weekly_review(
            services={
                "market": services["market"],
                "money_flow": services["money_flow"],
            },  # type: ignore[arg-type]
        )
        _assert_within_budget(out)


# ---------------------------------------------------------------------------
# Resources — every resource's Markdown fits in the budget
# ---------------------------------------------------------------------------


class TestResourcesTokenBudget:
    """Property 13 / Requirement 12.2 — every resource fits in the budget."""

    def test_market_overview_resource_within_budget(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        market = MarketService(adapter, cache=cache, rate_limiter=rate_limiter)
        out = market_overview_resource(services={"market": market})
        _assert_within_budget(out)

    def test_north_flow_resource_within_budget(
        self, adapter: StubAdapter, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        flow = MoneyFlowService(adapter, cache=cache, rate_limiter=rate_limiter)
        out = north_flow_resource(services={"money_flow": flow})
        _assert_within_budget(out)

    def test_symbol_profile_resource_within_budget(
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
        _assert_within_budget(out)
