"""``get_kline`` tool -- K 线序列 + 技术指标 Markdown 渲染.

Implements *Component 2 (Tools Layer)* from ``design.md`` for the
``get_kline`` entry. The tool layer is intentionally thin:

1. Validate caller input via :class:`GetKlineInput` (pydantic v2),
   bridging any pydantic ``ValidationError`` to the unified
   :class:`china_stock_mcp.exceptions.ValidationError`
   (Requirements 13.7).
2. Delegate to :meth:`KLineService.get_series`, which performs symbol
   normalization, caching, rate-limit admission, adapter fallback,
   indicator computation and ``pattern_note`` derivation.
3. Render a Markdown summary block + indicator snapshot table + tail
   bar table using helpers from :mod:`china_stock_mcp.formatters`.
4. Append the canonical disclaimer via :func:`append_disclaimer`
   (Requirements 12.1, Property 14).

Acceptance criteria covered
---------------------------

- Requirement 3.1 -- Markdown 摘要含 OHLCV / 成交额 / 选定指标.
- Requirement 3.2 -- 入参 ``period`` / ``adjust`` 校验.
- Requirement 3.3 -- ``len(bars) <= min(count, 250)`` (enforced by
  service + adapter; the tool layer just renders what it receives).
- Requirement 3.6 -- 不支持的 indicator 抛 :class:`ValidationError`.
- Requirement 3.7 -- ``pattern_note`` 在数据足够时渲染于摘要块.
"""

from __future__ import annotations

import math
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from china_stock_mcp.error_mapping import bridge_pydantic_error
from china_stock_mcp.formatters import (
    NONE_PLACEHOLDER,
    finalize_tool_output,
    format_amount,
    render_change,
    render_table,
)
from china_stock_mcp.models import MAX_KLINE_BARS, KLineBar, KLineSeries
from china_stock_mcp.services.kline_service import KLineService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default indicators the tool requests when the caller does not pass
#: ``indicators`` explicitly (mirrors the design Component 2 default).
_DEFAULT_INDICATORS: Final[tuple[str, ...]] = ("MA20", "MA60", "MACD")

#: Number of trailing bars rendered in the OHLCV table; a hard cap so
#: the response stays within the 3000-token budget (Property 13) even
#: for the maximum 250-bar series.
_TAIL_BAR_LIMIT: Final[int] = 20

#: Headers for the trailing-bar table.
_BAR_HEADERS: Final[list[str]] = [
    "日期",
    "开盘",
    "最高",
    "最低",
    "收盘",
    "成交量",
    "成交额",
]

#: Headers for the indicator snapshot table.
_INDICATOR_HEADERS: Final[list[str]] = ["指标", "最新值"]


# ---------------------------------------------------------------------------
# Pydantic input model
# ---------------------------------------------------------------------------


class GetKlineInput(BaseModel):
    """Pydantic v2 input schema for :func:`get_kline`.

    Constraints:

    - ``symbol`` is required and bounded to ``[1, 64]`` characters.
    - ``period`` is restricted to the five supported intervals
      (Requirement 3.2).
    - ``count`` is bounded to ``[1, 250]`` (Requirement 3.3).
    - ``adjust`` is restricted to ``qfq`` / ``hfq`` / ``none``.
    - ``indicators`` defaults to MA20 / MA60 / MACD; the service layer
      is the single source of truth for the supported set, so this
      field is left unconstrained at the schema layer to keep error
      messages centralized (Requirement 3.6).
    - ``extra="forbid"`` rejects unknown keys.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    symbol: str = Field(..., min_length=1, max_length=64)
    period: Literal["daily", "weekly", "monthly", "60min", "30min"] = "daily"
    count: int = Field(60, ge=1, le=MAX_KLINE_BARS)
    adjust: Literal["qfq", "hfq", "none"] = "qfq"
    indicators: list[str] = Field(default_factory=lambda: list(_DEFAULT_INDICATORS))


# ---------------------------------------------------------------------------
# Public tool entry
# ---------------------------------------------------------------------------


def get_kline(
    service: KLineService,
    symbol: str,
    period: str = "daily",
    count: int = 60,
    adjust: str = "qfq",
    indicators: list[str] | None = None,
) -> str:
    """Return a Markdown K 线 summary for ``symbol``.

    Parameters
    ----------
    service:
        Pre-wired :class:`KLineService` instance.
    symbol:
        Standardized or bare symbol; whitespace is stripped before
        validation. Normalization happens inside the service.
    period:
        Bar interval -- one of ``"daily"`` / ``"weekly"`` /
        ``"monthly"`` / ``"60min"`` / ``"30min"``.
    count:
        Maximum number of bars to fetch, ``[1, 250]``.
    adjust:
        Price adjustment mode -- one of ``"qfq"`` / ``"hfq"`` /
        ``"none"``.
    indicators:
        Optional list of indicator names. ``None`` falls back to the
        default :data:`_DEFAULT_INDICATORS`. Unsupported names raise
        :class:`ValidationError` from the service layer
        (Requirement 3.6).

    Returns
    -------
    str
        Markdown document ending with the standard disclaimer.

    Raises
    ------
    ValidationError
        If any input fails Pydantic validation, or if the service
        layer rejects an unsupported indicator name.
    ChinaStockMCPError
        Any subclass raised by the service layer is propagated
        verbatim (Requirements 13.1).
    """

    resolved_indicators: list[str] = (
        list(indicators) if indicators is not None else list(_DEFAULT_INDICATORS)
    )

    # 1) Input validation -- pydantic failures bridge to the unified
    #    error tree.
    try:
        validated = GetKlineInput(
            symbol=symbol,
            period=period,
            count=count,
            adjust=adjust,
            indicators=resolved_indicators,
        )
    except PydanticValidationError as exc:
        raise bridge_pydantic_error(exc) from exc

    # 2) Service call -- all heavy lifting (cache + rate-limit +
    #    fallback + indicator computation + pattern_note) happens here.
    series: KLineSeries = service.get_series(
        symbol=validated.symbol,
        period=validated.period,
        count=validated.count,
        adjust=validated.adjust,
        indicators=validated.indicators,
    )

    # 3) Render Markdown.
    body = _render_series(series)

    # 4) Unified tool-exit pipeline (Property 13 / 14, Requirement 12.1 / 12.2).
    return finalize_tool_output(body)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_series(series: KLineSeries) -> str:
    """Build the Markdown body for a :class:`KLineSeries`."""

    title = f"### K 线 ({series.symbol}, {series.period}, {series.adjust})"

    summary = _render_summary(series)
    indicator_table = _render_indicator_snapshot(series.indicators)
    bar_table = _render_tail_bars(series.bars)

    sections: list[str] = [title, summary]
    if indicator_table:
        sections.append("#### 指标快照\n\n" + indicator_table)
    sections.append("#### 最近 K 线\n\n" + bar_table)
    return "\n\n".join(sections)


def _render_summary(series: KLineSeries) -> str:
    """Render the summary block above the data tables."""

    bars = series.bars
    bar_count = len(bars)
    if bar_count == 0:
        rows: list[dict[str, str]] = [
            {"指标": "数据条数", "值": "0"},
            {"指标": "起止日期", "值": NONE_PLACEHOLDER},
            {"指标": "最新收盘", "值": NONE_PLACEHOLDER},
            {"指标": "区间涨跌", "值": NONE_PLACEHOLDER},
        ]
        if series.pattern_note is not None:
            rows.append({"指标": "形态简评", "值": series.pattern_note})
        return render_table(rows, headers=["指标", "值"])

    first = bars[0]
    last = bars[-1]
    last_close = last.close

    # Compute change vs previous bar's close (the conventional "今日
    # 涨跌幅" reference). When only one bar is available, fall back to
    # the bar's open as the reference so the cell is still informative.
    reference = bars[-2].close if bar_count >= 2 else last.open
    if reference == 0:
        change_pct: float = 0.0
    else:
        change_pct = (last_close - reference) / reference * 100.0

    summary_rows: list[dict[str, str]] = [
        {"指标": "数据条数", "值": str(bar_count)},
        {
            "指标": "起止日期",
            "值": f"{first.date.isoformat()} → {last.date.isoformat()}",
        },
        {"指标": "最新收盘", "值": f"{last_close:.2f}"},
        {"指标": "最新涨跌幅", "值": render_change(change_pct)},
    ]
    if series.pattern_note is not None:
        summary_rows.append({"指标": "形态简评", "值": series.pattern_note})

    return render_table(summary_rows, headers=["指标", "值"])


def _render_indicator_snapshot(
    indicators: dict[str, list[float]],
) -> str:
    """Render each indicator's most-recent value.

    Returns an empty string when no indicators are present so callers
    can omit the entire section without an empty heading.
    """

    if not indicators:
        return ""

    rows: list[dict[str, str]] = []
    for name, series in indicators.items():
        last_value = series[-1] if series else math.nan
        rows.append({"指标": name, "最新值": _format_indicator_value(last_value)})
    return render_table(rows, headers=_INDICATOR_HEADERS)


def _render_tail_bars(bars: list[KLineBar]) -> str:
    """Render the trailing :data:`_TAIL_BAR_LIMIT` bars as a table."""

    tail = bars[-_TAIL_BAR_LIMIT:] if bars else []
    if not tail:
        # ``render_table`` still produces a usable header-only table so
        # downstream consumers can detect "no data" by the absence of
        # body lines rather than an empty string.
        return render_table([], headers=_BAR_HEADERS)

    rows: list[dict[str, str]] = [_bar_to_row(bar) for bar in tail]
    return render_table(rows, headers=_BAR_HEADERS)


def _bar_to_row(bar: KLineBar) -> dict[str, str]:
    """Project a :class:`KLineBar` into the OHLCV table columns."""

    return {
        "日期": bar.date.isoformat(),
        "开盘": f"{bar.open:.2f}",
        "最高": f"{bar.high:.2f}",
        "最低": f"{bar.low:.2f}",
        "收盘": f"{bar.close:.2f}",
        "成交量": _format_volume(bar.volume),
        "成交额": format_amount(bar.amount),
    }


def _format_volume(volume: int) -> str:
    """Render share volume with 万 / 亿 unit selection."""

    return format_amount(float(volume)) + " 股"


def _format_indicator_value(value: float) -> str:
    """Render an indicator value, mapping NaN to :data:`NONE_PLACEHOLDER`."""

    if isinstance(value, float) and math.isnan(value):
        return NONE_PLACEHOLDER
    return f"{float(value):.4f}"


__all__ = ["GetKlineInput", "get_kline"]
