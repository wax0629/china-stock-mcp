"""Unit tests for prompt degradation, length-mode and disclaimer.

Covers task 20.4 (validates Requirements 10.2, 10.5, 10.6) for all
three prompts. The integration suite at
``tests/integration/test_prompt_degradation.py`` exercises end-to-end
section composition with stub services; this file complements it
with **unit-level** assertions on the post-processing pipeline:

- ``research_report(report_length="short")`` truncates the body to
  the documented 1500-token (~6000-char) budget and appends the
  ``_(short 模式: 内容已截断)_`` marker (Requirement 10.2).
- ``research_report(report_length="deep")`` appends a sixth
  ``## 深度分析`` section (Requirement 10.2).
- ``research_report(report_length="standard")`` emits the five base
  sections without the truncation marker or the deep-analysis
  heading.
- A happy-path render of every prompt has **no** ``⚠️ ... 数据不可用``
  banner (Requirement 10.5 corollary -- degradation is opt-in,
  triggered only by upstream failures).
- Every rendered prompt -- happy-path and degraded paths alike --
  ends with exactly one canonical :data:`DISCLAIMER` (Requirement
  10.6 / Property 14).
- Per-prompt degradation: when a single sub-call raises
  ``SymbolError`` / ``DataNotFoundError``, the failing section
  carries the ``⚠️ 该子模块数据不可用`` (or per-symbol equivalent)
  notice while the other sections still render normally.

The tests rely on lightweight in-memory stub services (no FastMCP /
akshare / cache / rate-limiter dependencies) so the assertions stay
focused on the prompt's post-processing logic.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from china_stock_mcp.exceptions import (
    DataNotFoundError,
    NetworkError,
    SymbolError,
)
from china_stock_mcp.formatters import DISCLAIMER
from china_stock_mcp.models import (
    FinancialPeriod,
    FinancialReport,
    FundamentalSnapshot,
    KLineBar,
    KLineSeries,
    MarketOverview,
    MoneyFlow,
    PeerTable,
    Quote,
)
from china_stock_mcp.prompts.research_report import research_report
from china_stock_mcp.prompts.valuation_compare import valuation_compare
from china_stock_mcp.prompts.weekly_review import weekly_review

# ---------------------------------------------------------------------------
# DTO builders
# ---------------------------------------------------------------------------

_FIXED_TIMESTAMP = datetime(2024, 6, 3, 14, 30, 0, tzinfo=UTC)


def _make_snapshot(symbol: str = "300750.SZ") -> FundamentalSnapshot:
    return FundamentalSnapshot(
        symbol=symbol,
        valuation={"pe_ttm": 18.5, "pb": 3.4, "pe_dynamic": 17.2},
        profitability={"roe": 18.0, "gross_margin": 28.0, "net_margin": 12.0},
        growth={"revenue_yoy": 22.5, "net_profit_yoy": 18.5},
        health={"debt_ratio": 45.0, "current_ratio": 1.8},
        industry_percentile={},
    )


def _make_report(symbol: str = "300750.SZ") -> FinancialReport:
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


def _make_peer_table(symbol: str = "300750.SZ") -> PeerTable:
    return PeerTable(
        base_symbol=symbol,
        industry="电池",
        metrics=["pe", "pb", "roe", "revenue_growth"],
        rows=[
            {
                "代码": "300750",
                "名称": "宁德时代",
                "pe": 18.5,
                "pb": 3.4,
                "roe": 18.0,
                "revenue_growth": 22.5,
            },
            {
                "代码": "002594",
                "名称": "比亚迪",
                "pe": 22.0,
                "pb": 4.5,
                "roe": 16.0,
                "revenue_growth": 30.0,
            },
        ],
    )


def _make_money_flow() -> MoneyFlow:
    return MoneyFlow(
        flow_type="north",
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


def _make_kline(symbol: str = "300750.SZ") -> KLineSeries:
    bars = [
        KLineBar(
            date=date(2024, 5, 1),
            open=180.0,
            high=185.0,
            low=178.0,
            close=183.0,
            volume=1_000_000,
            amount=1.83e8,
        ),
        KLineBar(
            date=date(2024, 5, 2),
            open=183.0,
            high=188.0,
            low=182.0,
            close=187.0,
            volume=1_200_000,
            amount=2.20e8,
        ),
    ]
    return KLineSeries(
        symbol=symbol,
        period="daily",
        adjust="qfq",
        bars=bars,
        indicators={},
        pattern_note=None,
    )


def _make_quote(symbol: str, name: str = "测试") -> Quote:
    return Quote(
        symbol=symbol,
        name=name,
        price=100.0,
        change=1.5,
        change_pct=1.5,
        volume=1_000_000,
        amount=1.0e8,
        turnover_rate=1.5,
        pe_ttm=18.5,
        pe_dynamic=17.2,
        pb=3.4,
        market_cap=5e10,
        float_market_cap=4e10,
        timestamp=_FIXED_TIMESTAMP,
        delay_seconds=900,
    )


def _make_market_overview() -> MarketOverview:
    return MarketOverview(
        indices=[
            {
                "name": "上证指数",
                "code": "000001.SH",
                "last": 3050.0,
                "change_pct": 0.85,
            },
            {
                "name": "深证成指",
                "code": "399001.SZ",
                "last": 9500.0,
                "change_pct": 1.20,
            },
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


# ---------------------------------------------------------------------------
# Stub services
# ---------------------------------------------------------------------------


class _StubFundamentalService:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self._error = error

    def snapshot(self, symbol: str) -> FundamentalSnapshot:
        if self._error is not None:
            raise self._error
        return _make_snapshot(symbol)


class _StubFinancialReportService:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self._error = error

    def report(
        self,
        symbol: str,
        report_type: str,
        periods: int,
    ) -> FinancialReport:
        if self._error is not None:
            raise self._error
        return _make_report(symbol)


class _StubIndustryService:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self._error = error

    def peers(
        self,
        symbol: str,
        metrics: list[str],
        top_n: int,
    ) -> PeerTable:
        if self._error is not None:
            raise self._error
        return _make_peer_table(symbol)


class _StubMoneyFlowService:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self._error = error

    def get(
        self,
        symbol: str | None,
        flow_type: str,
        top_n: int,
    ) -> MoneyFlow:
        if self._error is not None:
            raise self._error
        return _make_money_flow()


class _StubKLineService:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self._error = error

    def get_series(
        self,
        symbol: str,
        period: str,
        count: int,
        adjust: str,
        indicators: list[str],
    ) -> KLineSeries:
        if self._error is not None:
            raise self._error
        return _make_kline(symbol)


class _StubQuoteService:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self._error = error

    def get_snapshot(self, symbols: str | list[str]) -> list[Quote]:
        if self._error is not None:
            raise self._error
        normalized = [symbols] if isinstance(symbols, str) else list(symbols)
        return [_make_quote(s, name=s) for s in normalized]


class _StubMarketService:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self._error = error

    def overview(self) -> MarketOverview:
        if self._error is not None:
            raise self._error
        return _make_market_overview()


# ---------------------------------------------------------------------------
# Service-bundle factories
# ---------------------------------------------------------------------------


def _make_research_services(**overrides: object) -> dict[str, object]:
    """Build the bundle expected by :func:`research_report`."""

    services: dict[str, object] = {
        "fundamental": _StubFundamentalService(),
        "financial_report": _StubFinancialReportService(),
        "industry": _StubIndustryService(),
        "money_flow": _StubMoneyFlowService(),
        "kline": _StubKLineService(),
    }
    services.update(overrides)
    return services


def _make_valuation_services(**overrides: object) -> dict[str, object]:
    """Build the bundle expected by :func:`valuation_compare`."""

    services: dict[str, object] = {
        "quote": _StubQuoteService(),
        "fundamental": _StubFundamentalService(),
        "industry": _StubIndustryService(),
    }
    services.update(overrides)
    return services


def _make_weekly_services(**overrides: object) -> dict[str, object]:
    """Build the bundle expected by :func:`weekly_review`."""

    services: dict[str, object] = {
        "market": _StubMarketService(),
        "money_flow": _StubMoneyFlowService(),
    }
    services.update(overrides)
    return services


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_DEGRADATION_BANNER = "⚠️ 该子模块数据不可用"
_SHORT_TRUNCATION_MARKER = "_(short 模式: 内容已截断)_"


def _assert_disclaimer_tail(markdown: str) -> None:
    """Single-disclaimer suffix invariant (Property 14)."""

    assert markdown.rstrip().endswith(DISCLAIMER), (
        "rendered prompt must end with DISCLAIMER"
    )
    assert markdown.count(DISCLAIMER) == 1, (
        f"DISCLAIMER must appear exactly once, found {markdown.count(DISCLAIMER)}"
    )


# ---------------------------------------------------------------------------
# research_report length-mode tests (Requirement 10.2)
# ---------------------------------------------------------------------------


class TestResearchReportLengthModes:
    """Length-mode contract for ``research_report``."""

    def test_standard_mode_emits_five_sections_without_markers(self) -> None:
        """``standard`` (default) -- five sections, no truncation, no deep heading."""

        out = research_report(
            "300750.SZ",
            report_length="standard",
            services=_make_research_services(),  # type: ignore[arg-type]
        )

        # Five base sections.
        for heading in (
            "## 基本面",
            "## 财务报告",
            "## 行业对比",
            "## 资金流向",
            "## 技术形态",
        ):
            assert heading in out

        # No degradation banner on the happy path.
        assert _DEGRADATION_BANNER not in out

        # Neither the short marker nor the deep heading appear.
        assert _SHORT_TRUNCATION_MARKER not in out
        assert "## 深度分析" not in out

        _assert_disclaimer_tail(out)

    def test_short_mode_truncates_body_and_appends_marker(self) -> None:
        """``short`` -- body fits the budget and ends with the marker.

        Requirement 10.2 says ``short`` mode truncates the body to
        ~1500 tokens; the prompt encodes that as ~6000 characters
        (the conservative 4-chars-per-token proxy used elsewhere).
        We deliberately inflate the peer table so the joined body
        exceeds the budget and the truncation branch actually fires.
        We then assert:

        - The marker line ``_(short 模式: 内容已截断)_`` appears.
        - The body (everything before the disclaimer) is at most
          a small constant over the 6000-char budget. The exact
          ceiling depends on the paragraph the prompt snapped to
          plus the marker length itself, so we allow a generous
          1000-char buffer.
        - The standard disclaimer is still appended exactly once.
        """

        # Build an oversized peer table: many rows x wide string
        # cells push the rendered Markdown well over the 6000-char
        # short-mode budget so the truncation branch is exercised.
        big_rows = [
            {
                "代码": f"00{i:04d}",
                "名称": "宁德时代" * 5,  # padded to bulk up the row width
                "pe": 18.5 + i,
                "pb": 3.4 + i / 10,
                "roe": 18.0 - i / 5,
                "revenue_growth": 22.5 + i,
            }
            for i in range(80)
        ]
        big_peer = PeerTable(
            base_symbol="300750.SZ",
            industry="电池产业链",
            metrics=["pe", "pb", "roe", "revenue_growth"],
            rows=big_rows,
        )

        class _BigIndustryService(_StubIndustryService):
            def peers(
                self,
                symbol: str,
                metrics: list[str],
                top_n: int,
            ) -> PeerTable:
                return big_peer

        services = _make_research_services(industry=_BigIndustryService())

        out = research_report(
            "300750.SZ",
            report_length="short",
            services=services,  # type: ignore[arg-type]
        )

        # The truncation marker is the explicit signal that ``short``
        # post-processing fired.
        assert _SHORT_TRUNCATION_MARKER in out

        # Strip the trailing disclaimer to check the body budget.
        body = out.rstrip()
        assert body.endswith(DISCLAIMER)
        body_without_disclaimer = body[: -len(DISCLAIMER)].rstrip()

        # 6000-char budget + truncation marker + paragraph snap
        # overhead. A 7000-char ceiling gives generous headroom.
        assert len(body_without_disclaimer) <= 7000, (
            f"short mode body too long: {len(body_without_disclaimer)} chars"
        )

        # ``deep`` heading must not appear in ``short`` mode.
        assert "## 深度分析" not in out

        _assert_disclaimer_tail(out)

    def test_deep_mode_appends_deep_analysis_section(self) -> None:
        """``deep`` -- sixth ``## 深度分析`` section appended verbatim.

        The deep-analysis section reuses snapshot + peer-table data
        already gathered, so it should appear after the five base
        sections without changing them.
        """

        out = research_report(
            "300750.SZ",
            report_length="deep",
            services=_make_research_services(),  # type: ignore[arg-type]
        )

        # Five base sections still present.
        for heading in (
            "## 基本面",
            "## 财务报告",
            "## 行业对比",
            "## 资金流向",
            "## 技术形态",
        ):
            assert heading in out

        # Deep heading appended.
        deep_idx = out.index("## 深度分析")
        # Deep section must come AFTER the technical-pattern section.
        assert out.index("## 技术形态") < deep_idx

        # The short truncation marker must NOT appear in deep mode.
        assert _SHORT_TRUNCATION_MARKER not in out

        # The deep section carries actual analysis bullets sourced
        # from the snapshot + peer payloads we provided.
        deep_block = out[deep_idx:]
        assert "估值水平" in deep_block or "盈利质量" in deep_block

        _assert_disclaimer_tail(out)


# ---------------------------------------------------------------------------
# Per-prompt degradation tests (Requirement 10.5)
# ---------------------------------------------------------------------------


class TestResearchReportDegradation:
    """Single sub-call failure → that section degrades; rest renders."""

    def test_symbol_error_in_fundamentals_degrades_only_that_section(self) -> None:
        """Requirement 10.5 -- one section's failure does not propagate."""

        services = _make_research_services(
            fundamental=_StubFundamentalService(
                error=SymbolError("无法识别的代码: 'XYZ'"),
            ),
        )

        out = research_report(
            "300750.SZ",
            report_length="standard",
            services=services,  # type: ignore[arg-type]
        )

        # The 基本面 section degrades.
        idx = out.index("## 基本面")
        next_idx = out.index("## ", idx + 5)
        fundamental_block = out[idx:next_idx]
        assert _DEGRADATION_BANNER in fundamental_block
        assert "无法识别的代码" in fundamental_block

        # Subsequent sections still render with their normal headings.
        for heading in (
            "## 财务报告",
            "## 行业对比",
            "## 资金流向",
            "## 技术形态",
        ):
            assert heading in out

        # Confirm the degradation banner appears in EXACTLY the one
        # section we asked to fail.
        assert out.count(_DEGRADATION_BANNER) == 1

        _assert_disclaimer_tail(out)

    def test_data_not_found_in_financial_report_degrades_only_that_section(
        self,
    ) -> None:
        """``DataNotFoundError`` from one sub-call → that section degrades."""

        services = _make_research_services(
            financial_report=_StubFinancialReportService(
                error=DataNotFoundError("年报期数不足"),
            ),
        )

        out = research_report(
            "300750.SZ",
            report_length="standard",
            services=services,  # type: ignore[arg-type]
        )

        idx = out.index("## 财务报告")
        next_idx = out.index("## ", idx + 5)
        financial_block = out[idx:next_idx]
        assert _DEGRADATION_BANNER in financial_block
        assert "年报期数不足" in financial_block

        # The 基本面 (preceding) and 行业对比 (following) sections
        # still render normally without the banner.
        fund_idx = out.index("## 基本面")
        fund_block = out[fund_idx:idx]
        assert _DEGRADATION_BANNER not in fund_block

        _assert_disclaimer_tail(out)


class TestValuationCompareDegradation:
    """Per-symbol degradation for ``valuation_compare``."""

    def test_per_symbol_fundamental_failure_renders_unavailable(self) -> None:
        """Requirement 10.5 -- one symbol fails, the other renders."""

        symbols = ["300750.SZ", "002594.SZ"]

        class _SelectiveFundamentalService(_StubFundamentalService):
            def snapshot(self, symbol: str) -> FundamentalSnapshot:
                if symbol == "002594.SZ":
                    raise DataNotFoundError("002594 暂无基本面数据")
                return _make_snapshot(symbol)

        services = _make_valuation_services(
            fundamental=_SelectiveFundamentalService(),
        )

        out = valuation_compare(symbols, services=services)  # type: ignore[arg-type]

        # The failing symbol's per-symbol notice mentions "数据不可用".
        assert "002594.SZ 数据不可用" in out
        assert "002594 暂无基本面数据" in out

        # The successful symbol still renders its valuation block
        # with the standard "PE(TTM)" / "ROE" labels.
        assert "300750.SZ" in out
        assert "PE(TTM)" in out
        assert "ROE" in out

        _assert_disclaimer_tail(out)

    def test_quote_batch_failure_degrades_only_quote_section(self) -> None:
        """A batched quote failure replaces the 行情对比 block.

        The 估值横向对比 / 行业横切 sections still render because
        they do not depend on the failed service.
        """

        symbols = ["300750.SZ", "002594.SZ"]
        services = _make_valuation_services(
            quote=_StubQuoteService(error=NetworkError("upstream quote down")),
        )

        out = valuation_compare(symbols, services=services)  # type: ignore[arg-type]

        # 行情对比 block degrades.
        idx = out.index("## 行情对比")
        next_idx = out.index("## ", idx + 5)
        quote_block = out[idx:next_idx]
        assert "数据不可用" in quote_block
        assert "upstream quote down" in quote_block

        # 估值横向对比 still renders both symbols.
        assert "## 估值横向对比" in out
        assert "300750.SZ" in out
        assert "002594.SZ" in out

        _assert_disclaimer_tail(out)


class TestWeeklyReviewDegradation:
    """Per-section degradation for ``weekly_review``."""

    def test_money_flow_failure_renders_unavailable_for_north_only(self) -> None:
        """Requirement 10.5 -- 北向资金 down → degrade only that section."""

        services = _make_weekly_services(
            money_flow=_StubMoneyFlowService(
                error=DataNotFoundError("北向资金当日无数据"),
            ),
        )

        out = weekly_review(services=services)  # type: ignore[arg-type]

        # 市场总览 still renders.
        idx = out.index("## 市场总览")
        next_idx = out.index("## ", idx + 5)
        market_block = out[idx:next_idx]
        assert _DEGRADATION_BANNER not in market_block
        assert "**指数行情**" in market_block

        # 北向资金近期走势 degrades.
        north_idx = out.index("## 北向资金近期走势")
        north_next = out.index("## ", north_idx + 5)
        north_block = out[north_idx:north_next]
        assert _DEGRADATION_BANNER in north_block
        assert "北向资金当日无数据" in north_block

        # 行业热度排行 (from working overview) still renders.
        industries_block = out[out.index("## 行业热度排行"):]
        assert _DEGRADATION_BANNER not in industries_block
        assert "电池" in industries_block

        _assert_disclaimer_tail(out)


# ---------------------------------------------------------------------------
# Cross-prompt happy-path invariant (Requirement 10.6 / Property 14)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt_caller",
    [
        pytest.param(
            lambda: research_report(
                "300750.SZ",
                report_length="standard",
                services=_make_research_services(),  # type: ignore[arg-type]
            ),
            id="research_report_standard",
        ),
        pytest.param(
            lambda: research_report(
                "300750.SZ",
                report_length="short",
                services=_make_research_services(),  # type: ignore[arg-type]
            ),
            id="research_report_short",
        ),
        pytest.param(
            lambda: research_report(
                "300750.SZ",
                report_length="deep",
                services=_make_research_services(),  # type: ignore[arg-type]
            ),
            id="research_report_deep",
        ),
        pytest.param(
            lambda: valuation_compare(
                ["300750.SZ", "002594.SZ"],
                services=_make_valuation_services(),  # type: ignore[arg-type]
            ),
            id="valuation_compare_happy",
        ),
        pytest.param(
            lambda: weekly_review(
                services=_make_weekly_services(),  # type: ignore[arg-type]
            ),
            id="weekly_review_happy",
        ),
    ],
)
def test_happy_path_has_no_degradation_banner_and_single_disclaimer(
    prompt_caller,  # type: ignore[no-untyped-def]
) -> None:
    """Requirement 10.6 / Property 14 + Requirement 10.5 corollary.

    On the happy path every prompt must:

    - end with exactly one canonical :data:`DISCLAIMER`; and
    - contain no ``⚠️ ... 数据不可用`` banner -- degradation is only
      triggered by upstream failures, never by a successful render.
    """

    out = prompt_caller()
    _assert_disclaimer_tail(out)
    assert _DEGRADATION_BANNER not in out
    # The per-symbol valuation_compare notice uses a different
    # phrasing -- guard against that too.
    assert "数据不可用" not in out
