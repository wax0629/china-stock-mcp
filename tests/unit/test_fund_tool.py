"""Unit tests for :mod:`china_stock_mcp.tools.fund` (task 15.2).

Covers tool-layer rendering and validation:

- Requirement 7.3 -- ``fund_code`` must be 6-digit numeric;
  non-conforming inputs raise :class:`SymbolError`.
- Requirement 7.4 -- top-holdings weights render as 2-decimal
  percentages.
- Requirement 7.5 -- missing optional fields (e.g. ``sharpe=None``)
  render as ``"-"``; an empty ``industry_distribution`` omits the
  section entirely.
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
from china_stock_mcp.exceptions import SymbolError
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
from china_stock_mcp.services.fund_service import FundService
from china_stock_mcp.tools.fund import get_fund_info

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

    def __init__(self, info: FundInfo) -> None:
        self._info = info
        self.fund_info_call_count: int = 0

    def fund_info(self, fund_code: str) -> FundInfo:
        self.fund_info_call_count += 1
        return self._info

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


def _fund_info(
    *,
    code: str = "510300",
    sharpe: float | None = 1.25,
    top_holdings: list[dict[str, object]] | None = None,
    industry_distribution: list[dict[str, object]] | None = None,
) -> FundInfo:
    return FundInfo(
        code=code,
        name="沪深300ETF",
        manager="张三",
        inception_date=date(2012, 5, 4),
        aum=2.5e10,
        return_1m=2.5,
        return_3m=8.4,
        return_6m=15.2,
        return_12m=22.1,
        max_drawdown=-15.5,
        sharpe=sharpe,
        rank_in_category="12/345",
        top_holdings=(
            top_holdings
            if top_holdings is not None
            else [
                {"symbol": "600519.SH", "name": "贵州茅台", "weight": 8.2345},
                {"symbol": "300750.SZ", "name": "宁德时代", "weight": 4.5678},
            ]
        ),
        industry_distribution=(
            industry_distribution if industry_distribution is not None else []
        ),
    )


def _make_service(
    adapter: _StubAdapter,
    cache: Cache,
    rate_limiter: RateLimiter,
) -> FundService:
    return FundService(adapter, cache=cache, rate_limiter=rate_limiter)


# ---------------------------------------------------------------------------
# 6-digit fund_code accepted (Requirement 7.3)
# ---------------------------------------------------------------------------


class TestFundCodeAccepted:
    """**Validates: Requirements 7.3** -- happy path."""

    @pytest.mark.parametrize("code", ["510300", "000001", "159915"])
    def test_six_digit_fund_code_accepted(
        self,
        cache: _StubCache,
        rate_limiter: RateLimiter,
        code: str,
    ) -> None:
        adapter = _StubAdapter(_fund_info(code=code))
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_fund_info(service, code)

        assert code in markdown
        assert markdown.rstrip().endswith(DISCLAIMER)


# ---------------------------------------------------------------------------
# Non-6-digit fund_code raises SymbolError (Requirement 7.3)
# ---------------------------------------------------------------------------


class TestFundCodeRejected:
    """**Validates: Requirements 7.3** -- failure path."""

    @pytest.mark.parametrize(
        "bad_code",
        [
            "ABC123",  # contains letters
            "12345",  # 5 digits
            "1234567",  # 7 digits
            "",  # empty
            "510-300",  # punctuation
        ],
    )
    def test_non_six_digit_code_raises_symbol_error(
        self,
        cache: _StubCache,
        rate_limiter: RateLimiter,
        bad_code: str,
    ) -> None:
        adapter = _StubAdapter(_fund_info())
        service = _make_service(adapter, cache, rate_limiter)

        with pytest.raises(SymbolError):
            get_fund_info(service, bad_code)

        assert adapter.fund_info_call_count == 0


# ---------------------------------------------------------------------------
# Optional fields (Requirement 7.5)
# ---------------------------------------------------------------------------


class TestOptionalFieldRendering:
    """**Validates: Requirements 7.5**."""

    def test_sharpe_none_renders_as_dash(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        adapter = _StubAdapter(_fund_info(sharpe=None))
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_fund_info(service, "510300")

        # The 夏普比率 row exists and its value cell is the placeholder.
        assert "夏普比率" in markdown
        # ``"-"`` placeholder appears in a 夏普比率 row segment.
        sharpe_idx = markdown.index("夏普比率")
        # Examine the slice immediately after the label; the cell
        # delimiter ``|`` precedes the placeholder.
        tail = markdown[sharpe_idx : sharpe_idx + 60]
        assert "-" in tail

    def test_industry_distribution_omitted_when_empty(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        adapter = _StubAdapter(_fund_info(industry_distribution=[]))
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_fund_info(service, "510300")

        # The optional 行业分布 section is omitted entirely when empty.
        assert "行业分布" not in markdown

    def test_industry_distribution_rendered_when_present(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        adapter = _StubAdapter(
            _fund_info(
                industry_distribution=[
                    {"industry": "金融", "weight": 25.5},
                    {"industry": "消费", "weight": 18.3},
                ]
            )
        )
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_fund_info(service, "510300")

        assert "行业分布" in markdown
        assert "金融" in markdown
        assert "消费" in markdown


# ---------------------------------------------------------------------------
# Top-holdings weight formatting (Requirement 7.4)
# ---------------------------------------------------------------------------


class TestTopHoldingsWeightFormat:
    """**Validates: Requirements 7.4**."""

    def test_weights_render_as_two_decimal_percent(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        adapter = _StubAdapter(_fund_info())
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_fund_info(service, "510300")

        # Default fixture has weights 8.2345 -> 8.23% and 4.5678 -> 4.57%.
        assert "8.23%" in markdown
        assert "4.57%" in markdown
        # The 前十大持仓 heading is rendered.
        assert "前十大持仓" in markdown
        # Disclaimer footer.
        assert markdown.rstrip().endswith(DISCLAIMER)
