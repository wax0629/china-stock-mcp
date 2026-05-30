"""Unit tests for :mod:`china_stock_mcp.tools.market_overview` (task 17.2).

Covers tool-layer rendering and validation:

- Requirement 9.1 -- Markdown contains 指数行情 / 涨跌家数 / 涨跌停 /
  北向 / 行业热度 / heat_score sections.
- Requirement 9.2 -- ``heat_score ∈ [0, 100]`` is enforced at DTO
  construction; out-of-range values raise pydantic ``ValidationError``.
- Requirement 9.3 -- ``snapshot_at`` is rendered in the
  ``> 数据时间: ...`` header line.
- Requirement 9.4 -- 非交易时段 banner appears when ``snapshot_at`` is
  outside A-share trading hours (weekend or non-09:30-15:00 CST), and
  is absent during regular Mon-Fri trading hours.
- Requirement 12.1 / Property 14 -- Markdown ends with the canonical
  disclaimer.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError as PydanticValidationError

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.cache import Cache, make_key, reset_default_cache
from china_stock_mcp.formatters import DISCLAIMER
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
from china_stock_mcp.services.market_service import MarketService
from china_stock_mcp.tools.market_overview import get_market_overview

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
    name: str = "stub"

    def __init__(self, overview: MarketOverview) -> None:
        self._overview = overview
        self.market_overview_call_count: int = 0

    def market_overview(self) -> MarketOverview:
        self.market_overview_call_count += 1
        return self._overview

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


_CST = timezone(timedelta(hours=8))


def _overview(
    *,
    snapshot_at: datetime,
    heat_score: float = 50.0,
) -> MarketOverview:
    return MarketOverview(
        indices=[
            {"name": "上证指数", "code": "000001", "last": 3000.5, "change_pct": 0.85},
            {
                "name": "深证成指",
                "code": "399001",
                "last": 9500.2,
                "change_pct": -0.42,
            },
            {
                "name": "创业板指",
                "code": "399006",
                "last": 1900.0,
                "change_pct": 1.23,
            },
        ],
        advance_decline={"advance": 2500, "decline": 1800, "flat": 200},
        limit_stats={"limit_up": 45, "limit_down": 12},
        north_net_inflow=1.5e9,
        top_inflow_industries=[
            {"name": "电池", "net_inflow": 5.5e8},
            {"name": "新能源车", "net_inflow": 3.2e8},
        ],
        heat_score=heat_score,
        snapshot_at=snapshot_at,
    )


def _make_service(
    adapter: _StubAdapter,
    cache: Cache,
    rate_limiter: RateLimiter,
) -> MarketService:
    return MarketService(adapter, cache=cache, rate_limiter=rate_limiter)


# ---------------------------------------------------------------------------
# heat_score range (Requirement 9.2)
# ---------------------------------------------------------------------------


class TestHeatScoreRange:
    """**Validates: Requirements 9.2**."""

    @pytest.mark.parametrize("score", [0.0, 50.0, 100.0])
    def test_boundary_heat_scores_accepted(self, score: float) -> None:
        # 2024-01-02 (Tuesday) at 10:30 CST = trading hours.
        snapshot_at = datetime(2024, 1, 2, 10, 30, 0, tzinfo=_CST)
        overview = _overview(snapshot_at=snapshot_at, heat_score=score)
        assert overview.heat_score == score

    @pytest.mark.parametrize("score", [-0.1, 100.01, 200.0])
    def test_out_of_range_heat_score_rejected_at_dto(
        self, score: float
    ) -> None:
        with pytest.raises(PydanticValidationError):
            _overview(
                snapshot_at=datetime(2024, 1, 2, 10, 30, 0, tzinfo=_CST),
                heat_score=score,
            )


# ---------------------------------------------------------------------------
# snapshot_at rendered (Requirement 9.3)
# ---------------------------------------------------------------------------


class TestSnapshotAtRendered:
    """**Validates: Requirements 9.3**."""

    def test_snapshot_at_renders_in_header_line(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        snapshot_at = datetime(2024, 1, 2, 10, 30, 0, tzinfo=_CST)
        adapter = _StubAdapter(_overview(snapshot_at=snapshot_at))
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_market_overview(service)

        # The renderer formats snapshot_at as ``YYYY-MM-DD HH:MM:SS``
        # inside a ``> 数据时间:`` blockquote.
        assert "> 数据时间:" in markdown
        assert "2024-01-02 10:30:00" in markdown


# ---------------------------------------------------------------------------
# 非交易时段 banner (Requirement 9.4)
# ---------------------------------------------------------------------------


class TestNonTradingBanner:
    """**Validates: Requirements 9.4**."""

    def test_banner_present_on_weekend(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        # 2024-01-06 is a Saturday.
        weekend_at = datetime(2024, 1, 6, 10, 30, 0, tzinfo=_CST)
        adapter = _StubAdapter(_overview(snapshot_at=weekend_at))
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_market_overview(service)

        assert "非交易时段" in markdown

    def test_banner_present_outside_trading_hours_on_weekday(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        # Monday 08:00 CST -- before market open.
        early_morning = datetime(2024, 1, 8, 8, 0, 0, tzinfo=_CST)
        adapter = _StubAdapter(_overview(snapshot_at=early_morning))
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_market_overview(service)

        assert "非交易时段" in markdown

    def test_banner_present_after_market_close(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        # Monday 16:00 CST -- after market close.
        evening = datetime(2024, 1, 8, 16, 0, 0, tzinfo=_CST)
        adapter = _StubAdapter(_overview(snapshot_at=evening))
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_market_overview(service)

        assert "非交易时段" in markdown

    def test_banner_absent_during_trading_hours(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        # Monday 10:00 CST -- inside the morning session.
        trading = datetime(2024, 1, 8, 10, 0, 0, tzinfo=_CST)
        adapter = _StubAdapter(_overview(snapshot_at=trading))
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_market_overview(service)

        assert "非交易时段" not in markdown


# ---------------------------------------------------------------------------
# Markdown structure (Requirement 9.1)
# ---------------------------------------------------------------------------


class TestMarkdownStructure:
    """**Validates: Requirements 9.1**."""

    def test_markdown_contains_all_six_sections(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        snapshot_at = datetime(2024, 1, 2, 10, 30, 0, tzinfo=_CST)
        adapter = _StubAdapter(_overview(snapshot_at=snapshot_at))
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_market_overview(service)

        # All six required sections are present.
        assert "指数行情" in markdown
        assert "涨跌家数" in markdown
        assert "涨跌停" in markdown
        assert "北向" in markdown
        assert "行业热度" in markdown
        # ``heat_score`` surfaces under the 市场热度评分 label.
        assert "市场热度评分" in markdown
        # Score formatted as ``XX.X / 100``.
        assert "/ 100" in markdown

        # Disclaimer terminator (Property 14).
        assert markdown.rstrip().endswith(DISCLAIMER)
