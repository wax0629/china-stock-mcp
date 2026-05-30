"""``get_financial_report`` tool -- 多期财务报告 Markdown 渲染.

Implements *Component 2 (Tools Layer)* from ``design.md`` for the
``get_financial_report`` entry. The tool layer is intentionally thin:

1. Validate caller input via :class:`GetFinancialReportInput`
   (pydantic v2), bridging any pydantic ``ValidationError`` to the
   unified :class:`china_stock_mcp.exceptions.ValidationError`
   (Requirements 13.7).
2. Delegate to :meth:`FinancialReportService.report`, which performs
   symbol normalization, caching, rate-limit admission and adapter
   fallback.
3. Render a Markdown table with one column per period -- each period
   contributes one column so the table reads chronologically left to
   right with the most recent period on the right (Requirement 4.6).
4. Append the canonical disclaimer via :func:`append_disclaimer`
   (Requirements 12.1, Property 14).

Acceptance criteria covered
---------------------------

- Requirement 4.3 -- 营收 / 净利润 / 扣非净利润 / 毛利 / 经营性现金流 /
  总资产 / 总负债 / 所有者权益 全部出表.
- Requirement 4.4 -- ``report_type`` ∈ ``{annual, quarterly}``,
  ``periods`` ∈ ``[1, 12]``.
- Requirement 4.5 -- 期数不足时抛 :class:`DataNotFoundError`
  (来自 service / adapter 层, 由 server 层渲染为用户消息).
- Requirement 4.6 -- 报告期排序稳定 (service 层做升序, 这里按列渲染).
- Requirement 12.1 / Property 14 -- 末尾追加免责声明.
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
    render_table,
)
from china_stock_mcp.models import FinancialPeriod, FinancialReport
from china_stock_mcp.services.financial_report_service import FinancialReportService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: ``periods`` lower / upper bounds. Mirrors the service layer's
#: validation but exposed at the schema layer too so pydantic surfaces
#: a clearer per-field error message.
_MIN_PERIODS: Final[int] = 1
_MAX_PERIODS: Final[int] = 12

#: Default ``periods`` value -- 4 covers a full year of quarters or 4
#: years of annual reports, the common defaults for AI-driven research.
_DEFAULT_PERIODS: Final[int] = 4

#: Mapping from :class:`FinancialPeriod` field name to the row label
#: rendered in Markdown. Insertion order is preserved by Python dicts
#: so this is also the row order of the resulting table.
_FIELD_LABELS: Final[dict[str, str]] = {
    "revenue": "营业总收入",
    "net_profit": "归母净利润",
    "net_profit_excl_nrgl": "扣非净利润",
    "gross_profit": "毛利",
    "operating_cash_flow": "经营性现金流",
    "total_assets": "总资产",
    "total_liabilities": "总负债",
    "equity": "所有者权益",
}

#: Localized labels for ``report_type`` so the heading reads naturally.
_REPORT_TYPE_LABEL: Final[dict[str, str]] = {
    "annual": "年报",
    "quarterly": "季报",
}


# ---------------------------------------------------------------------------
# Pydantic input model
# ---------------------------------------------------------------------------


class GetFinancialReportInput(BaseModel):
    """Pydantic v2 input schema for :func:`get_financial_report`.

    Constraints:

    - ``symbol`` is required and bounded to ``[1, 64]`` characters.
    - ``report_type`` is restricted to ``"annual"`` / ``"quarterly"``
      (Requirement 4.4).
    - ``periods`` is bounded to ``[1, 12]`` (Requirement 4.4) and
      defaults to ``4``.
    - ``extra="forbid"`` rejects unknown keys.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    symbol: str = Field(..., min_length=1, max_length=64)
    report_type: Literal["annual", "quarterly"] = "annual"
    periods: int = Field(_DEFAULT_PERIODS, ge=_MIN_PERIODS, le=_MAX_PERIODS)


# ---------------------------------------------------------------------------
# Public tool entry
# ---------------------------------------------------------------------------


def get_financial_report(
    service: FinancialReportService,
    symbol: str,
    report_type: str = "annual",
    periods: int = _DEFAULT_PERIODS,
) -> str:
    """Return a Markdown 财务报告 for ``symbol``.

    Parameters
    ----------
    service:
        Pre-wired :class:`FinancialReportService` instance.
    symbol:
        Standardized or bare A-share symbol; whitespace is stripped
        before validation. Normalization happens inside the service.
    report_type:
        ``"annual"`` for 年报 only or ``"quarterly"`` for
        年报 + 中报 + 季报. Defaults to ``"annual"``.
    periods:
        Number of historical periods to fetch, ``[1, 12]``. Defaults
        to ``4``.

    Returns
    -------
    str
        Markdown document ending with the standard disclaimer.

    Raises
    ------
    ValidationError
        If any input fails Pydantic validation, or if the service
        layer rejects an out-of-range value.
    DataNotFoundError
        If the upstream cannot satisfy ``periods`` for the requested
        ``report_type`` (Requirement 4.5).
    ChinaStockMCPError
        Any other subclass raised by the service / adapter layer is
        propagated verbatim (Requirements 13.1).
    """

    # 1) Input validation -- pydantic failures bridge to the unified
    #    error tree.
    try:
        validated = GetFinancialReportInput(
            symbol=symbol,
            report_type=report_type,
            periods=periods,
        )
    except PydanticValidationError as exc:
        raise bridge_pydantic_error(exc) from exc

    # 2) Service call -- all heavy lifting (cache + rate-limit +
    #    fallback + ascending period_end sort) happens here.
    report: FinancialReport = service.report(
        symbol=validated.symbol,
        report_type=validated.report_type,
        periods=validated.periods,
    )

    # 3) Render Markdown.
    body = _render_report(report)

    # 4) Unified tool-exit pipeline (Property 13 / 14, Requirement 12.1 / 12.2).
    return finalize_tool_output(body)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_report(report: FinancialReport) -> str:
    """Build the Markdown body for a :class:`FinancialReport`."""

    type_label = _REPORT_TYPE_LABEL.get(report.report_type, report.report_type)
    actual = len(report.periods)
    title = (
        f"### 财务报告 ({report.symbol}, {type_label}, "
        f"{actual} 期)"
    )

    if not report.periods:
        # Should never happen in practice -- the service raises
        # ``DataNotFoundError`` before reaching the renderer when the
        # adapter returned zero periods -- but guard defensively so a
        # cached pre-validation payload never produces a broken table.
        return f"{title}\n\n_暂无可用财务数据_"

    table = _render_periods_table(report.periods)
    return f"{title}\n\n{table}"


def _render_periods_table(periods: list[FinancialPeriod]) -> str:
    """Render one row per metric, one column per reporting period.

    The first column is the metric label; each subsequent column is a
    reporting-period date in ``YYYY-MM-DD`` form. ``periods`` arrives
    sorted ascending by ``period_end`` (service-layer guarantee), so
    the leftmost data column is the oldest and the rightmost is the
    most recent.
    """

    headers: list[str] = ["指标"]
    for period in periods:
        headers.append(period.period_end.isoformat())

    rows: list[dict[str, str]] = []
    for field_name, label in _FIELD_LABELS.items():
        row: dict[str, str] = {"指标": label}
        for period in periods:
            value = getattr(period, field_name)
            row[period.period_end.isoformat()] = _format_amount_cell(value)
        rows.append(row)

    return render_table(rows, headers=headers)


def _format_amount_cell(value: float | None) -> str:
    """Render a 元-denominated value with 亿 / 万 unit selection.

    Empty placeholders for ``None`` / ``NaN`` so the table never shows
    a misleading ``0.00 亿`` for missing data.
    """

    if value is None:
        return NONE_PLACEHOLDER
    if isinstance(value, float) and math.isnan(value):
        return NONE_PLACEHOLDER
    return format_amount(float(value))


__all__ = ["GetFinancialReportInput", "get_financial_report"]
