"""Pydantic v2 DTO models for the China Stock MCP server.

These data transfer objects are the single source of truth for the
shapes passed across the *Adapter → Service → Tool* boundary. They are
designed so that:

- Field names use ``snake_case``.
- Monetary amounts are denominated in 元 (CNY) unless explicitly stated.
- Dates and timestamps serialize as ISO8601.
- Validation rules from ``design.md`` (sections "Data Models" and
  "Correctness Properties") are enforced here, not in the calling layer.

Only model definitions live in this module; keep adapter, service and
formatter logic out so this file remains a pure schema reference.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    model_validator,
)

# ---------------------------------------------------------------------------
# Shared type aliases
# ---------------------------------------------------------------------------

# Standardized symbol matches A 股 / 港股 / 公募基金 6 位代码 之一.
# Patterns:
#   - "300750.SZ" / "600519.SH" / "831175.BJ"  -> 6-digit + .SH/.SZ/.BJ
#   - "00700.HK"                                -> 5-digit + .HK
#   - "510300"                                  -> 6-digit fund code
_SYMBOL_PATTERN: str = r"^(?:\d{6}\.(?:SH|SZ|BJ)|\d{5}\.HK|\d{6})$"
_FUND_CODE_PATTERN: str = r"^\d{6}$"

Market = Literal["a_stock", "hk_stock", "fund"]
KLinePeriod = Literal["daily", "weekly", "monthly", "60min", "30min"]
AdjustMode = Literal["qfq", "hfq", "none"]
ReportType = Literal["annual", "quarterly"]
FlowType = Literal["north", "main", "dragon_tiger"]

#: Maximum number of bars allowed in a ``KLineSeries`` (design §Data Models).
MAX_KLINE_BARS: int = 250

#: Default delay for delayed quote feeds (seconds, design §Data Models).
DEFAULT_QUOTE_DELAY_SECONDS: int = 900


# ---------------------------------------------------------------------------
# Symbol search
# ---------------------------------------------------------------------------


class SymbolHit(BaseModel):
    """A single symbol search result.

    Validation:
        - ``code`` must match one of the standardized symbol shapes
          (Requirements 1.5 / Property P2).
        - ``market`` must be one of ``a_stock``, ``hk_stock``, ``fund``.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    code: Annotated[str, Field(pattern=_SYMBOL_PATTERN)]
    name: Annotated[str, Field(min_length=1)]
    market: Market
    industry: str | None = None


# ---------------------------------------------------------------------------
# Real-time quote
# ---------------------------------------------------------------------------


class Quote(BaseModel):
    """Real-time quote snapshot.

    Validation (Requirements 2.7):
        - ``price >= 0``, ``volume >= 0``, ``amount >= 0``.
        - ``turnover_rate ∈ [0, 100]``.
        - ``delay_seconds >= 0`` (Property P15); defaults to 900s
          (~15 minute delay) per design.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    symbol: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(min_length=1)]
    price: NonNegativeFloat
    change: float  # 涨跌额, 可正可负
    change_pct: float  # 涨跌幅, 百分比, 可正可负
    volume: NonNegativeInt  # 成交量 (股)
    amount: NonNegativeFloat  # 成交额 (元)
    turnover_rate: Annotated[float, Field(ge=0.0, le=100.0)]
    pe_ttm: float | None = None
    pe_dynamic: float | None = None
    pb: float | None = None
    market_cap: float  # 总市值 (元)
    float_market_cap: float  # 流通市值 (元)
    timestamp: datetime
    delay_seconds: Annotated[int, Field(ge=0)] = DEFAULT_QUOTE_DELAY_SECONDS


# ---------------------------------------------------------------------------
# K 线
# ---------------------------------------------------------------------------


class KLineBar(BaseModel):
    """A single OHLCV bar.

    Validation (Requirements 3.4 / Property P8):
        ``low <= min(open, close) <= max(open, close) <= high``.
    """

    model_config = ConfigDict(extra="forbid")

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: NonNegativeInt
    amount: NonNegativeFloat

    @model_validator(mode="after")
    def _check_ohlc_inequality(self) -> KLineBar:
        body_low = min(self.open, self.close)
        body_high = max(self.open, self.close)
        if not (self.low <= body_low <= body_high <= self.high):
            raise ValueError(
                "OHLC inequality violated: expected "
                "low <= min(open, close) <= max(open, close) <= high, "
                f"got low={self.low}, open={self.open}, "
                f"close={self.close}, high={self.high}"
            )
        return self


class KLineSeries(BaseModel):
    """K 线序列 + 派生指标。

    Validation:
        - ``len(bars) <= 250`` (design §Data Models).
        - For each ``k``: ``len(indicators[k]) == len(bars)``
          (Requirements 3.5 / Property P9).
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    symbol: Annotated[str, Field(min_length=1)]
    period: KLinePeriod
    adjust: AdjustMode
    bars: Annotated[list[KLineBar], Field(max_length=MAX_KLINE_BARS)]
    indicators: dict[str, list[float]] = Field(default_factory=dict)
    pattern_note: str | None = None

    @model_validator(mode="after")
    def _check_indicator_alignment(self) -> KLineSeries:
        bar_count = len(self.bars)
        for name, values in self.indicators.items():
            if len(values) != bar_count:
                raise ValueError(
                    f"indicator {name!r} length mismatch: "
                    f"expected {bar_count}, got {len(values)}"
                )
        return self


# ---------------------------------------------------------------------------
# 基本面快照
# ---------------------------------------------------------------------------


class FundamentalSnapshot(BaseModel):
    """Fundamental snapshot grouped into 4 buckets + 行业分位.

    Validation (Requirements 4.1, 4.2):
        - All four metric groups are required dicts (may be empty if a
          datasource cannot provide a particular bucket).
        - Each ``industry_percentile`` value lies in ``[0, 100]``.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    symbol: Annotated[str, Field(min_length=1)]
    valuation: dict[str, float | None] = Field(default_factory=dict)
    profitability: dict[str, float | None] = Field(default_factory=dict)
    growth: dict[str, float | None] = Field(default_factory=dict)
    health: dict[str, float | None] = Field(default_factory=dict)
    industry_percentile: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_industry_percentile_range(self) -> FundamentalSnapshot:
        for metric, value in self.industry_percentile.items():
            if not (0.0 <= value <= 100.0):
                raise ValueError(
                    f"industry_percentile[{metric!r}] must be in [0, 100], "
                    f"got {value}"
                )
        return self


# ---------------------------------------------------------------------------
# 财务报告
# ---------------------------------------------------------------------------


class FinancialPeriod(BaseModel):
    """单期财务摘要 (年报或季报)."""

    model_config = ConfigDict(extra="forbid")

    period_end: date
    revenue: float  # 营业总收入
    net_profit: float  # 归母净利润
    net_profit_excl_nrgl: float  # 扣非净利润
    gross_profit: float  # 毛利
    operating_cash_flow: float  # 经营性现金流
    total_assets: float  # 总资产
    total_liabilities: float  # 总负债
    equity: float  # 所有者权益


class FinancialReport(BaseModel):
    """多期财务报告聚合."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    symbol: Annotated[str, Field(min_length=1)]
    report_type: ReportType
    periods: list[FinancialPeriod] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 资金流向
# ---------------------------------------------------------------------------


class MoneyFlow(BaseModel):
    """资金流向结果.

    ``rows`` 字段的具体 schema 因 ``flow_type`` 而异, 由 service 层文档约定,
    因此此处保留为 ``list[dict]``.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    flow_type: FlowType
    rows: list[dict[str, object]] = Field(default_factory=list)
    snapshot_at: datetime


# ---------------------------------------------------------------------------
# 行业对比
# ---------------------------------------------------------------------------


class PeerTable(BaseModel):
    """同行业可比公司对比表."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    base_symbol: Annotated[str, Field(min_length=1)]
    industry: Annotated[str, Field(min_length=1)]
    metrics: list[str] = Field(default_factory=list)
    rows: list[dict[str, object]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 公募基金信息
# ---------------------------------------------------------------------------


class FundInfo(BaseModel):
    """公募基金信息. ``code`` 为 6 位数字基金代码 (不带交易所后缀)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    code: Annotated[str, Field(pattern=_FUND_CODE_PATTERN)]
    name: Annotated[str, Field(min_length=1)]
    manager: str
    inception_date: date
    aum: NonNegativeFloat  # 规模 (元)
    return_1m: float
    return_3m: float
    return_6m: float
    return_12m: float
    max_drawdown: float
    sharpe: float | None = None
    rank_in_category: str  # 例: "12/345"
    top_holdings: list[dict[str, object]] = Field(default_factory=list)
    industry_distribution: list[dict[str, object]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 选股器
# ---------------------------------------------------------------------------


class RangeFilter(BaseModel):
    """``[min, max]`` 数值闭区间, 两端均可省略."""

    model_config = ConfigDict(extra="forbid")

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _check_min_le_max(self) -> RangeFilter:
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(
                f"RangeFilter.min ({self.min}) must be <= max ({self.max})"
            )
        return self


class ScreenCriteria(BaseModel):
    """多因子选股条件。

    Note:
        允许通过 ``extra='allow'`` 扩展额外字段 (design 注明 "可扩展字段..."),
        以便 service 层在不破坏 schema 的情况下新增过滤维度.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    pe_ttm: RangeFilter | None = None
    pb: RangeFilter | None = None
    roe: RangeFilter | None = None
    market_cap: RangeFilter | None = None
    revenue_growth: RangeFilter | None = None
    industry: list[str] | None = None


class ScreenHit(BaseModel):
    """选股命中结果的一行."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    code: Annotated[str, Field(pattern=_SYMBOL_PATTERN)]
    name: Annotated[str, Field(min_length=1)]
    industry: str
    fields: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 市场总览
# ---------------------------------------------------------------------------


class MarketOverview(BaseModel):
    """大盘指数 + 涨跌家数 + 涨跌停 + 北向 + 行业热度.

    Validation (Requirements 9.2):
        ``heat_score ∈ [0, 100]``.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    indices: list[dict[str, object]] = Field(default_factory=list)
    advance_decline: dict[str, int] = Field(default_factory=dict)
    limit_stats: dict[str, int] = Field(default_factory=dict)
    north_net_inflow: float
    top_inflow_industries: list[dict[str, object]] = Field(default_factory=list)
    heat_score: Annotated[float, Field(ge=0.0, le=100.0)]
    snapshot_at: datetime


__all__ = [
    "DEFAULT_QUOTE_DELAY_SECONDS",
    "MAX_KLINE_BARS",
    "AdjustMode",
    "FinancialPeriod",
    "FinancialReport",
    "FlowType",
    "FundInfo",
    "FundamentalSnapshot",
    "KLineBar",
    "KLinePeriod",
    "KLineSeries",
    "Market",
    "MarketOverview",
    "MoneyFlow",
    "PeerTable",
    "Quote",
    "RangeFilter",
    "ReportType",
    "ScreenCriteria",
    "ScreenHit",
    "SymbolHit",
]
