"""Validation unit tests for :mod:`china_stock_mcp.models`.

These tests cover task 1.4 (Requirements 2.7 + cross-cutting validation
rules captured in ``design.md`` §"Data Models"):

- ``Quote`` numeric bounds (``price``/``volume``/``amount`` non-negative,
  ``turnover_rate ∈ [0, 100]``, ``delay_seconds >= 0``).
- ``KLineBar`` OHLC inequality
  ``low <= min(open, close) <= max(open, close) <= high``
  (Property P8 / Requirements 3.4).
- ``KLineSeries`` indicator alignment + ``max_length=250``
  (Property P9 / Requirements 3.5).
- ``FundamentalSnapshot.industry_percentile`` range ``[0, 100]``.
- ``MarketOverview.heat_score`` range ``[0, 100]``.
- ``RangeFilter`` requires ``min <= max``.
- ``SymbolHit.code`` matches the standardized symbol pattern.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from china_stock_mcp.models import (
    DEFAULT_QUOTE_DELAY_SECONDS,
    MAX_KLINE_BARS,
    FundamentalSnapshot,
    KLineBar,
    KLineSeries,
    MarketOverview,
    Quote,
    RangeFilter,
    SymbolHit,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _valid_quote_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = dict(
        symbol="600519.SH",
        name="贵州茅台",
        price=1700.0,
        change=12.5,
        change_pct=0.74,
        volume=1_000_000,
        amount=1_700_000_000.0,
        turnover_rate=0.85,
        pe_ttm=30.0,
        pe_dynamic=28.0,
        pb=10.0,
        market_cap=2_100_000_000_000.0,
        float_market_cap=2_100_000_000_000.0,
        timestamp=datetime(2024, 1, 2, 15, 0, tzinfo=UTC),
    )
    base.update(overrides)
    return base


def _valid_bar_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = dict(
        date=date(2024, 1, 2),
        open=10.0,
        high=11.0,
        low=9.5,
        close=10.5,
        volume=100_000,
        amount=1_050_000.0,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Quote
# ---------------------------------------------------------------------------


class TestQuote:
    def test_valid_quote_uses_default_delay(self) -> None:
        q = Quote(**_valid_quote_kwargs())
        assert q.delay_seconds == DEFAULT_QUOTE_DELAY_SECONDS

    @pytest.mark.parametrize("price", [-0.01, -1.0, -1e-9])
    def test_negative_price_rejected(self, price: float) -> None:
        with pytest.raises(ValidationError, match="price"):
            Quote(**_valid_quote_kwargs(price=price))

    def test_zero_price_accepted(self) -> None:
        # 退市/停牌等特殊场景下允许 price == 0
        q = Quote(**_valid_quote_kwargs(price=0.0))
        assert q.price == 0.0

    def test_negative_volume_rejected(self) -> None:
        with pytest.raises(ValidationError, match="volume"):
            Quote(**_valid_quote_kwargs(volume=-1))

    def test_negative_amount_rejected(self) -> None:
        with pytest.raises(ValidationError, match="amount"):
            Quote(**_valid_quote_kwargs(amount=-0.01))

    @pytest.mark.parametrize("turnover_rate", [-0.1, 100.01, 250.0])
    def test_turnover_rate_out_of_range_rejected(
        self, turnover_rate: float
    ) -> None:
        with pytest.raises(ValidationError, match="turnover_rate"):
            Quote(**_valid_quote_kwargs(turnover_rate=turnover_rate))

    @pytest.mark.parametrize("turnover_rate", [0.0, 50.0, 100.0])
    def test_turnover_rate_boundaries_accepted(
        self, turnover_rate: float
    ) -> None:
        q = Quote(**_valid_quote_kwargs(turnover_rate=turnover_rate))
        assert q.turnover_rate == turnover_rate

    def test_negative_delay_seconds_rejected(self) -> None:
        with pytest.raises(ValidationError, match="delay_seconds"):
            Quote(**_valid_quote_kwargs(delay_seconds=-1))

    def test_zero_delay_seconds_accepted(self) -> None:
        q = Quote(**_valid_quote_kwargs(delay_seconds=0))
        assert q.delay_seconds == 0

    def test_change_can_be_negative(self) -> None:
        q = Quote(**_valid_quote_kwargs(change=-5.0, change_pct=-2.0))
        assert q.change == -5.0
        assert q.change_pct == -2.0

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Quote(**_valid_quote_kwargs(unknown_field="x"))


# ---------------------------------------------------------------------------
# KLineBar  (Property P8 / Requirements 3.4)
# ---------------------------------------------------------------------------


class TestKLineBar:
    def test_valid_bar(self) -> None:
        bar = KLineBar(**_valid_bar_kwargs())
        assert bar.low <= min(bar.open, bar.close)
        assert max(bar.open, bar.close) <= bar.high

    def test_doji_bar_with_equal_ohlc(self) -> None:
        bar = KLineBar(
            **_valid_bar_kwargs(open=10.0, high=10.0, low=10.0, close=10.0)
        )
        assert bar.high == bar.low == bar.open == bar.close

    def test_low_above_open_rejected(self) -> None:
        with pytest.raises(ValidationError, match="OHLC"):
            KLineBar(**_valid_bar_kwargs(low=10.5, open=10.0))

    def test_low_above_close_rejected(self) -> None:
        with pytest.raises(ValidationError, match="OHLC"):
            KLineBar(**_valid_bar_kwargs(low=10.6, close=10.5, open=10.7))

    def test_high_below_open_rejected(self) -> None:
        with pytest.raises(ValidationError, match="OHLC"):
            KLineBar(**_valid_bar_kwargs(open=12.0, high=11.5))

    def test_high_below_close_rejected(self) -> None:
        with pytest.raises(ValidationError, match="OHLC"):
            KLineBar(**_valid_bar_kwargs(close=12.0, high=11.5))

    def test_high_below_low_rejected(self) -> None:
        with pytest.raises(ValidationError, match="OHLC"):
            KLineBar(**_valid_bar_kwargs(low=12.0, high=11.0))

    def test_negative_volume_rejected(self) -> None:
        with pytest.raises(ValidationError, match="volume"):
            KLineBar(**_valid_bar_kwargs(volume=-1))

    def test_negative_amount_rejected(self) -> None:
        with pytest.raises(ValidationError, match="amount"):
            KLineBar(**_valid_bar_kwargs(amount=-0.01))


# ---------------------------------------------------------------------------
# KLineSeries  (Property P9 / Requirements 3.5)
# ---------------------------------------------------------------------------


def _make_bars(n: int) -> list[KLineBar]:
    return [
        KLineBar(**_valid_bar_kwargs(date=date(2024, 1, max(1, i + 1) % 28 + 1)))
        for i in range(n)
    ]


class TestKLineSeries:
    def test_empty_series_is_valid(self) -> None:
        series = KLineSeries(
            symbol="600519.SH", period="daily", adjust="qfq", bars=[]
        )
        assert series.bars == []
        assert series.indicators == {}

    def test_indicator_aligned_with_bars(self) -> None:
        bars = _make_bars(3)
        series = KLineSeries(
            symbol="600519.SH",
            period="daily",
            adjust="qfq",
            bars=bars,
            indicators={"MA20": [1.0, 2.0, 3.0]},
        )
        assert len(series.indicators["MA20"]) == len(series.bars)

    def test_indicator_length_mismatch_rejected(self) -> None:
        bars = _make_bars(3)
        with pytest.raises(ValidationError, match="MA20"):
            KLineSeries(
                symbol="600519.SH",
                period="daily",
                adjust="qfq",
                bars=bars,
                indicators={"MA20": [1.0, 2.0]},
            )

    def test_max_length_enforced(self) -> None:
        too_many = _make_bars(MAX_KLINE_BARS + 1)
        with pytest.raises(ValidationError):
            KLineSeries(
                symbol="600519.SH",
                period="daily",
                adjust="qfq",
                bars=too_many,
            )

    def test_invalid_period_rejected(self) -> None:
        with pytest.raises(ValidationError):
            KLineSeries(
                symbol="600519.SH",
                period="hourly",  # type: ignore[arg-type]
                adjust="qfq",
                bars=[],
            )

    def test_invalid_adjust_rejected(self) -> None:
        with pytest.raises(ValidationError):
            KLineSeries(
                symbol="600519.SH",
                period="daily",
                adjust="forward",  # type: ignore[arg-type]
                bars=[],
            )


# ---------------------------------------------------------------------------
# FundamentalSnapshot
# ---------------------------------------------------------------------------


class TestFundamentalSnapshot:
    def test_default_groups_are_empty_dicts(self) -> None:
        fs = FundamentalSnapshot(symbol="600519.SH")
        assert fs.valuation == {}
        assert fs.profitability == {}
        assert fs.growth == {}
        assert fs.health == {}
        assert fs.industry_percentile == {}

    @pytest.mark.parametrize("value", [0.0, 50.0, 100.0])
    def test_industry_percentile_boundaries_accepted(
        self, value: float
    ) -> None:
        fs = FundamentalSnapshot(
            symbol="600519.SH", industry_percentile={"pe_ttm": value}
        )
        assert fs.industry_percentile["pe_ttm"] == value

    @pytest.mark.parametrize("value", [-0.1, 100.01, 200.0])
    def test_industry_percentile_out_of_range_rejected(
        self, value: float
    ) -> None:
        with pytest.raises(ValidationError, match="industry_percentile"):
            FundamentalSnapshot(
                symbol="600519.SH", industry_percentile={"pe_ttm": value}
            )


# ---------------------------------------------------------------------------
# MarketOverview
# ---------------------------------------------------------------------------


def _make_overview(**overrides: object) -> MarketOverview:
    base: dict[str, object] = dict(
        north_net_inflow=12_345_678.0,
        heat_score=50.0,
        snapshot_at=datetime(2024, 1, 2, 15, 0, tzinfo=UTC),
    )
    base.update(overrides)
    return MarketOverview(**base)


class TestMarketOverview:
    @pytest.mark.parametrize("score", [0.0, 50.0, 100.0])
    def test_heat_score_boundaries_accepted(self, score: float) -> None:
        mo = _make_overview(heat_score=score)
        assert mo.heat_score == score

    @pytest.mark.parametrize("score", [-0.1, 100.01, 1000.0])
    def test_heat_score_out_of_range_rejected(self, score: float) -> None:
        with pytest.raises(ValidationError, match="heat_score"):
            _make_overview(heat_score=score)


# ---------------------------------------------------------------------------
# RangeFilter
# ---------------------------------------------------------------------------


class TestRangeFilter:
    def test_both_none_is_valid(self) -> None:
        rf = RangeFilter()
        assert rf.min is None
        assert rf.max is None

    def test_min_only_is_valid(self) -> None:
        rf = RangeFilter(min=5.0)
        assert rf.min == 5.0
        assert rf.max is None

    def test_max_only_is_valid(self) -> None:
        rf = RangeFilter(max=10.0)
        assert rf.max == 10.0
        assert rf.min is None

    def test_min_equals_max_is_valid(self) -> None:
        rf = RangeFilter(min=5.0, max=5.0)
        assert rf.min == rf.max == 5.0

    def test_min_less_than_max_is_valid(self) -> None:
        rf = RangeFilter(min=5.0, max=10.0)
        assert rf.min == 5.0 and rf.max == 10.0

    def test_min_greater_than_max_rejected(self) -> None:
        with pytest.raises(ValidationError, match="min"):
            RangeFilter(min=10.0, max=5.0)


# ---------------------------------------------------------------------------
# SymbolHit
# ---------------------------------------------------------------------------


class TestSymbolHit:
    @pytest.mark.parametrize(
        "code",
        [
            "600519.SH",
            "300750.SZ",
            "831175.BJ",
            "00700.HK",
            "510300",  # 6 位基金代码
        ],
    )
    def test_valid_codes_accepted(self, code: str) -> None:
        hit = SymbolHit(code=code, name="X", market="a_stock")
        assert hit.code == code

    @pytest.mark.parametrize(
        "code",
        [
            "600519.HK",  # 6 位 + .HK 不合法
            "00700.SH",  # 5 位港股加了 SH
            "ABCDEF",  # 字母
            "12345",  # 5 位无后缀
            "6005190.SH",  # 7 位
            "600519.sh",  # 小写后缀
            "600519.US",  # 不支持的市场后缀
            "",
        ],
    )
    def test_invalid_codes_rejected(self, code: str) -> None:
        with pytest.raises(ValidationError):
            SymbolHit(code=code, name="X", market="a_stock")

    def test_invalid_market_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SymbolHit(
                code="600519.SH", name="X", market="us_stock"  # type: ignore[arg-type]
            )

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SymbolHit(code="600519.SH", name="", market="a_stock")
