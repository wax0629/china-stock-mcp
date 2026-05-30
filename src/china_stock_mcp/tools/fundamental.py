"""``get_fundamentals`` tool -- 基本面快照 Markdown 渲染.

Implements *Component 2 (Tools Layer)* from ``design.md`` for the
``get_fundamentals`` entry. The tool layer is intentionally thin:

1. Validate caller input via :class:`GetFundamentalsInput` (pydantic
   v2), bridging any pydantic ``ValidationError`` to the unified
   :class:`china_stock_mcp.exceptions.ValidationError`
   (Requirements 13.7).
2. Delegate to :meth:`FundamentalService.snapshot`, which performs
   symbol normalization, caching, rate-limit admission and adapter
   fallback.
3. Render four labelled sub-tables (估值 / 盈利 / 成长 / 健康) using
   :func:`render_table`. When the upstream populates an
   ``industry_percentile`` map, an extra "行业分位" column is appended
   to each table; when the map is empty (the v1 case) the column is
   omitted so the Markdown stays clean.
4. Append the canonical disclaimer via :func:`append_disclaimer`
   (Requirements 12.1, Property 14).

Acceptance criteria covered
---------------------------

- Requirement 4.1 -- 估值 / 盈利 / 成长 / 健康 四组指标分别成表.
- Requirement 4.2 -- 当存在 ``industry_percentile`` 时显示行业分位列.
- Requirement 12.1 / Property 14 -- 末尾追加免责声明.
"""

from __future__ import annotations

import math
from typing import Final

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from china_stock_mcp.error_mapping import bridge_pydantic_error
from china_stock_mcp.formatters import (
    NONE_PLACEHOLDER,
    finalize_tool_output,
    format_percent,
    render_table,
)
from china_stock_mcp.models import FundamentalSnapshot
from china_stock_mcp.services.fundamental_service import FundamentalService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Localized labels for each metric. Centralized here so a single
#: source of truth maps between the upstream-derived bucket key
#: (``roe``, ``revenue_yoy``, ...) and the human-readable Chinese
#: column label rendered in Markdown. Unknown keys fall back to
#: ``str(key)`` so future bucket additions degrade gracefully.
_METRIC_LABELS: Final[dict[str, str]] = {
    # valuation
    "pe_ttm": "市盈率(TTM)",
    "pe_dynamic": "市盈率(动态)",
    "pb": "市净率",
    "ps": "市销率",
    "peg": "PEG",
    # profitability
    "roe": "净资产收益率",
    "roa": "总资产收益率",
    "gross_margin": "毛利率",
    "net_margin": "净利率",
    # growth
    "revenue_yoy": "营收同比",
    "net_profit_yoy": "净利润同比",
    "qoq": "营收环比",
    # health
    "debt_ratio": "资产负债率",
    "current_ratio": "流动比率",
    "ocf_to_net_profit": "经营现金流/净利润",
}

#: Metrics whose value is denominated in *percent* and should be
#: rendered with :func:`format_percent` (which appends ``%``). Every
#: other metric is rendered as a 2-decimal float.
_PERCENT_METRICS: Final[frozenset[str]] = frozenset(
    {
        "roe",
        "roa",
        "gross_margin",
        "net_margin",
        "revenue_yoy",
        "net_profit_yoy",
        "qoq",
        "debt_ratio",
    }
)

#: Bucket display order plus the heading rendered above each table.
_BUCKETS: Final[tuple[tuple[str, str], ...]] = (
    ("valuation", "估值指标"),
    ("profitability", "盈利能力"),
    ("growth", "成长性"),
    ("health", "财务健康"),
)


# ---------------------------------------------------------------------------
# Pydantic input model
# ---------------------------------------------------------------------------


class GetFundamentalsInput(BaseModel):
    """Pydantic v2 input schema for :func:`get_fundamentals`.

    Constraints:

    - ``symbol`` is required and bounded to ``[1, 64]`` characters.
    - ``extra="forbid"`` rejects unknown keys.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    symbol: str = Field(..., min_length=1, max_length=64)


# ---------------------------------------------------------------------------
# Public tool entry
# ---------------------------------------------------------------------------


def get_fundamentals(service: FundamentalService, symbol: str) -> str:
    """Return a Markdown 基本面 snapshot for ``symbol``.

    Parameters
    ----------
    service:
        Pre-wired :class:`FundamentalService` instance.
    symbol:
        Standardized or bare symbol; whitespace is stripped before
        validation. Normalization happens inside the service.

    Returns
    -------
    str
        Markdown document ending with the standard disclaimer.

    Raises
    ------
    ValidationError
        If ``symbol`` fails Pydantic validation.
    ChinaStockMCPError
        Any subclass raised by the service layer is propagated
        verbatim (Requirements 13.1).
    """

    # 1) Input validation -- pydantic failures bridge to the unified
    #    error tree.
    try:
        validated = GetFundamentalsInput(symbol=symbol)
    except PydanticValidationError as exc:
        raise bridge_pydantic_error(exc) from exc

    # 2) Service call -- all heavy lifting (cache + rate-limit +
    #    fallback) happens here.
    snapshot: FundamentalSnapshot = service.snapshot(validated.symbol)

    # 3) Render Markdown.
    body = _render_snapshot(snapshot)

    # 4) Unified tool-exit pipeline (Property 13 / 14, Requirement 12.1 / 12.2).
    return finalize_tool_output(body)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_snapshot(snapshot: FundamentalSnapshot) -> str:
    """Build the Markdown body for a :class:`FundamentalSnapshot`."""

    title = f"### 基本面快照 ({snapshot.symbol})"
    sections: list[str] = [title]

    has_percentile = bool(snapshot.industry_percentile)

    for bucket_attr, bucket_heading in _BUCKETS:
        bucket = getattr(snapshot, bucket_attr)
        sub = _render_bucket(
            heading=bucket_heading,
            bucket=bucket,
            industry_percentile=snapshot.industry_percentile,
            include_percentile_column=has_percentile,
        )
        sections.append(sub)

    return "\n\n".join(sections)


def _render_bucket(
    heading: str,
    bucket: dict[str, float | None],
    industry_percentile: dict[str, float],
    include_percentile_column: bool,
) -> str:
    """Render one of the four metric buckets as a labelled sub-table.

    When the upstream provided no values for the bucket, render an
    explicit "数据暂无" line under the sub-heading so the consumer
    never sees an empty section that could be mistaken for a bug.
    """

    sub_heading = f"#### {heading}"

    if not bucket:
        return f"{sub_heading}\n\n_数据暂无_"

    headers: list[str] = ["指标", "数值"]
    if include_percentile_column:
        headers.append("行业分位")

    rows: list[dict[str, str]] = []
    for metric, value in bucket.items():
        row: dict[str, str] = {
            "指标": _METRIC_LABELS.get(metric, str(metric)),
            "数值": _format_metric_value(metric, value),
        }
        if include_percentile_column:
            row["行业分位"] = _format_percentile(industry_percentile.get(metric))
        rows.append(row)

    table = render_table(rows, headers=headers)
    return f"{sub_heading}\n\n{table}"


def _format_metric_value(metric: str, value: float | None) -> str:
    """Render a single metric value with the appropriate unit.

    - Percent-denominated metrics (ROE / margins / YoY / debt ratio)
      go through :func:`format_percent` (appends ``%``).
    - Every other metric (PE / PB / current ratio / ...) renders as a
      2-decimal float.
    - ``None`` and NaN map to :data:`NONE_PLACEHOLDER`.
    """

    if value is None:
        return NONE_PLACEHOLDER
    if not isinstance(value, (int, float)):
        return NONE_PLACEHOLDER
    if isinstance(value, float) and math.isnan(value):
        return NONE_PLACEHOLDER

    if metric in _PERCENT_METRICS:
        return format_percent(value)
    return f"{float(value):.2f}"


def _format_percentile(value: float | None) -> str:
    """Render a 0..100 industry percentile, or ``"-"`` when missing."""

    if value is None:
        return NONE_PLACEHOLDER
    if not isinstance(value, (int, float)):
        return NONE_PLACEHOLDER
    if isinstance(value, float) and math.isnan(value):
        return NONE_PLACEHOLDER
    return f"{float(value):.1f}"


__all__ = ["GetFundamentalsInput", "get_fundamentals"]
