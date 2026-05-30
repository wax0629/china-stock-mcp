"""``weekly_review`` prompt -- 周复盘 Markdown 编排.

Implements *design.md Algorithm 6 / Requirement 10.4* for task 20.3.
The prompt orchestrates two service-layer calls and stitches the
results into a single Markdown document with three labelled sections:

1. **市场总览** -- ``MarketService.overview`` (indices /
   advance-decline / 涨跌停 / heat_score).
2. **北向资金近期走势** -- ``MoneyFlowService.get(flow_type="north")``
   showing the most recent daily 净流入 / 买入 / 卖出 / 持股市值.
3. **行业热度排行** -- the ``top_inflow_industries`` rows already
   carried by :class:`MarketOverview`, surfaced as a dedicated table
   so the AI client can scan industry sentiment at a glance.

Behaviour
---------

- Per-section graceful degradation (Requirement 10.5): every
  sub-call is wrapped in a ``try/except ChinaStockMCPError`` and on
  failure the section body is replaced by a
  ``> ⚠️ 该子模块数据不可用`` block-quote that quotes the error's
  ``to_user_message()`` text. The remaining sections still render so
  the AI client always gets *something* useful.

- The disclaimer is appended exactly once at the end via
  :func:`append_disclaimer` (Requirement 10.6 / Property 14).

The prompt is **pure** with respect to the services it receives:
callers (e.g. :mod:`china_stock_mcp.server`) construct the service
instances and pass them in via the ``services`` keyword, which keeps
the function trivially testable without any FastMCP / akshare side
effects.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Final, TypedDict

from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError

from china_stock_mcp.error_mapping import bridge_pydantic_error
from china_stock_mcp.exceptions import ChinaStockMCPError
from china_stock_mcp.formatters import (
    DISCLAIMER,
    NONE_PLACEHOLDER,
    append_disclaimer,
    format_amount,
    format_percent,
    render_change,
    render_table,
)
from china_stock_mcp.models import MarketOverview, MoneyFlow
from china_stock_mcp.services.market_service import MarketService
from china_stock_mcp.services.money_flow_service import MoneyFlowService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of recent north-bound trading days to surface. 20 keeps the
#: section comparable to the ``get_money_flow`` tool default and fits
#: comfortably inside the per-prompt token budget alongside the
#: market-overview tables.
_NORTH_FLOW_TOP_N: Final[int] = 20

#: Substrings that mark a column as monetary (元-denominated). Cells
#: matching one of these tokens are rendered through
#: :func:`format_amount`. Mirrors the heuristics used by
#: :mod:`china_stock_mcp.tools.money_flow` so the prompt and the tool
#: produce visually identical north-flow tables.
_AMOUNT_KEY_TOKENS: Final[tuple[str, ...]] = (
    "金额",
    "净流入",
    "买入",
    "卖出",
    "净买额",
    "持股市值",
    "累计",
)

#: Substrings that mark a column as percentage-denominated.
_PERCENT_KEY_TOKENS: Final[tuple[str, ...]] = (
    "涨跌幅",
    "换手率",
    "净占比",
    "占比",
)


class _Services(TypedDict):
    """Typed bundle of pre-wired service instances."""

    market: MarketService
    money_flow: MoneyFlowService


# ---------------------------------------------------------------------------
# Pydantic input model
# ---------------------------------------------------------------------------


class WeeklyReviewInput(BaseModel):
    """Pydantic v2 input schema for :func:`weekly_review`.

    The prompt currently accepts no arguments; the schema enforces
    this explicitly so callers that pass extra keys see a
    :class:`ValidationError` (Requirement 13.7) rather than having
    the value silently ignored. Declared for symmetry with the rest
    of the prompt suite.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# Public prompt entry
# ---------------------------------------------------------------------------


def weekly_review(*, services: _Services) -> str:
    """Return a 3-section 周复盘 Markdown document.

    Parameters
    ----------
    services:
        Bundle of pre-wired :class:`MarketService` and
        :class:`MoneyFlowService` instances. The prompt is pure with
        respect to its services so unit tests can supply lightweight
        stubs.

    Returns
    -------
    str
        Markdown document ending with the standard disclaimer.

    Raises
    ------
    ValidationError
        Reserved for future arguments; the current prompt surface
        accepts no parameters but unknown keyword bleed is rejected.
    """

    # 1) Input validation -- the model has no fields yet, but
    #    instantiating it rejects unexpected keyword bleed that may
    #    appear in future overloads.
    try:
        WeeklyReviewInput()
    except PydanticValidationError as exc:
        raise bridge_pydantic_error(exc) from exc

    # 2) Run each sub-call independently so a failure in one does
    #    not deny the others their slot in the rendered document
    #    (Requirement 10.5).
    overview, overview_body = _section_market_overview(services["market"])
    north_body = _section_north_flow(services["money_flow"])
    industries_body = _section_top_industries(overview)

    # 3) Title carries the snapshot date so callers who skim the
    #    rendered Markdown immediately see how fresh the review is.
    snapshot_label = _format_snapshot_label(overview)
    title = (
        f"# 周复盘 ({snapshot_label})"
        if snapshot_label is not None
        else "# 周复盘"
    )

    body = "\n\n".join(
        [
            title,
            "## 市场总览",
            overview_body,
            "## 北向资金近期走势",
            north_body,
            "## 行业热度排行",
            industries_body,
        ]
    )

    # 4) Disclaimer (Requirement 10.6, Property 14).
    return append_disclaimer(body)


# ---------------------------------------------------------------------------
# Per-section helpers
# ---------------------------------------------------------------------------


def _section_market_overview(
    service: MarketService,
) -> tuple[MarketOverview | None, str]:
    """Render the 市场总览 section, degrading on adapter errors."""

    try:
        overview = service.overview()
    except ChinaStockMCPError as exc:
        return None, _format_unavailable(exc)
    return overview, _render_overview_body(overview)


def _section_north_flow(service: MoneyFlowService) -> str:
    """Render the 北向资金近期走势 section, degrading on adapter errors."""

    try:
        flow = service.get(
            symbol=None,
            flow_type="north",
            top_n=_NORTH_FLOW_TOP_N,
        )
    except ChinaStockMCPError as exc:
        return _format_unavailable(exc)
    return _render_north_flow_body(flow)


def _section_top_industries(overview: MarketOverview | None) -> str:
    """Render the 行业热度排行 section using the overview snapshot.

    The data is already inside :class:`MarketOverview`, so this
    section degrades together with the 市场总览 section. When the
    overview call failed we still emit a structural placeholder so
    callers see a consistent three-section layout regardless of
    upstream availability.
    """

    if overview is None:
        return "> ⚠️ 该子模块数据不可用：依赖市场总览数据"  # noqa: RUF001
    if not overview.top_inflow_industries:
        return "_暂无行业资金流向数据_"

    rows: list[dict[str, str]] = []
    for raw in overview.top_inflow_industries:
        name = _stringify(raw.get("name"))
        net_value = _to_float(raw.get("net_inflow"))
        net_cell = (
            format_amount(net_value)
            if net_value is not None
            else NONE_PLACEHOLDER
        )
        rows.append({"行业": name, "主力净流入": net_cell})

    return render_table(rows, headers=["行业", "主力净流入"])


# ---------------------------------------------------------------------------
# Body renderers
# ---------------------------------------------------------------------------


def _render_overview_body(overview: MarketOverview) -> str:
    """Build the Markdown body for the 市场总览 section.

    Combines indices + advance/decline counts + 涨跌停 stats + 北向
    净流入 + heat_score in a compact set of sub-tables. The format
    mirrors :mod:`china_stock_mcp.tools.market_overview` so the AI
    client experiences the same visual language whether the data is
    surfaced directly or through this prompt.
    """

    sections: list[str] = []

    # 指数行情
    indices_section = _render_indices_section(overview)
    sections.append(indices_section)

    # 涨跌家数
    advance = overview.advance_decline.get("advance", 0)
    decline = overview.advance_decline.get("decline", 0)
    flat = overview.advance_decline.get("flat", 0)
    breadth_table = render_table(
        [{"上涨": str(advance), "下跌": str(decline), "平": str(flat)}],
        headers=["上涨", "下跌", "平"],
    )
    sections.append(f"**涨跌家数**\n\n{breadth_table}")

    # 涨跌停
    limit_up = overview.limit_stats.get("limit_up", 0)
    limit_down = overview.limit_stats.get("limit_down", 0)
    limit_table = render_table(
        [{"涨停数": str(limit_up), "跌停数": str(limit_down)}],
        headers=["涨停数", "跌停数"],
    )
    sections.append(f"**涨跌停**\n\n{limit_table}")

    # 北向资金净流入
    sections.append(
        "**北向资金净流入**\n\n"
        f"{format_amount(overview.north_net_inflow)}"
    )

    # 市场热度评分 (DTO clamps to [0, 100], so a fixed-precision
    # render is safe).
    sections.append(f"**市场热度评分**: {overview.heat_score:.1f} / 100")

    return "\n\n".join(sections)


def _render_indices_section(overview: MarketOverview) -> str:
    """Render the 指数行情 sub-table inside the 市场总览 section."""

    header = "**指数行情**"
    if not overview.indices:
        return f"{header}\n\n_暂无指数数据_"

    rows: list[dict[str, str]] = []
    for raw in overview.indices:
        name = _stringify(raw.get("name"))
        code = _stringify(raw.get("code"))
        last_value = _to_float(raw.get("last"))
        change_value = _to_float(raw.get("change_pct"))
        last_cell = (
            f"{last_value:.2f}" if last_value is not None else NONE_PLACEHOLDER
        )
        change_cell = (
            render_change(change_value)
            if change_value is not None
            else NONE_PLACEHOLDER
        )
        rows.append(
            {
                "名称": name,
                "代码": code,
                "最新": last_cell,
                "涨跌幅": change_cell,
            }
        )

    table = render_table(rows, headers=["名称", "代码", "最新", "涨跌幅"])
    return f"{header}\n\n{table}"


def _render_north_flow_body(flow: MoneyFlow) -> str:
    """Build the Markdown body for the 北向资金近期走势 section."""

    snapshot_line = (
        f"> 数据时间: {flow.snapshot_at.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    if not flow.rows:
        return f"{snapshot_line}\n\n_暂无可用资金流向数据_"

    # Derive headers from the union of keys, preserving the order in
    # which keys first appear -- defensive against per-row holes
    # while staying aligned with the per-row schema documented by
    # :class:`MoneyFlow`.
    headers: list[str] = []
    seen: set[str] = set()
    for row in flow.rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            headers.append(key)

    rendered_rows: list[dict[str, str]] = [
        {key: _format_flow_cell(key, row.get(key)) for key in headers}
        for row in flow.rows
    ]

    # Map the internal "date" key to a friendlier "日期" label so the
    # rendered column header reads naturally.
    display_headers = [_display_header(h) for h in headers]
    if display_headers != headers:
        rendered_rows = [
            {_display_header(key): cell for key, cell in row.items()}
            for row in rendered_rows
        ]

    table = render_table(rendered_rows, headers=display_headers)
    return f"{snapshot_line}\n\n{table}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_snapshot_label(overview: MarketOverview | None) -> str | None:
    """Return ``snapshot_at`` as ``YYYY-MM-DD`` or ``None`` when missing."""

    if overview is None:
        return None
    snapshot: datetime = overview.snapshot_at
    return snapshot.strftime("%Y-%m-%d")


def _display_header(key: str) -> str:
    """Map an internal row key to its rendered Chinese column label."""

    if key == "date":
        return "日期"
    return key


def _format_flow_cell(key: str, value: Any) -> str:
    """Render a single north-flow cell, picking units from the column key.

    Mirrors :func:`china_stock_mcp.tools.money_flow._format_cell` so the
    prompt-rendered table looks identical to the tool output:

    - Monetary keys (``净流入`` / ``买入`` / ``持股市值`` / ...) go
      through :func:`format_amount` (亿 / 万 unit selection).
    - Percentage keys (``涨跌幅`` / ``占比``) go through
      :func:`format_percent`.
    - ``None`` and NaN map to :data:`NONE_PLACEHOLDER`.
    - Strings, dates and other free-form values render verbatim.
    """

    if value is None:
        return NONE_PLACEHOLDER
    if isinstance(value, float) and math.isnan(value):
        return NONE_PLACEHOLDER

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if any(token in key for token in _AMOUNT_KEY_TOKENS):
            return format_amount(float(value))
        if any(token in key for token in _PERCENT_KEY_TOKENS):
            return format_percent(float(value))
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    return str(value)


def _stringify(value: Any) -> str:
    """Render an arbitrary cell value to a stripped string or placeholder."""

    if value is None:
        return NONE_PLACEHOLDER
    text = str(value).strip()
    return text if text else NONE_PLACEHOLDER


def _to_float(value: Any) -> float | None:
    """Coerce a value to ``float`` or return ``None`` on NaN / failure."""

    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def _format_unavailable(error: ChinaStockMCPError | None) -> str:
    """Render the Requirement 10.5 graceful-degradation block."""

    message = error.to_user_message() if error is not None else "未知错误"
    # Collapse newlines so the rendered block stays a single Markdown
    # block-quote line.
    flat = " ".join(line.strip() for line in message.splitlines() if line.strip())
    return f"> ⚠️ 该子模块数据不可用：{flat}"  # noqa: RUF001


# Reference :data:`DISCLAIMER` so future iterations can introspect
# the canonical disclaimer without re-importing; the helper is wired
# into :func:`append_disclaimer` above.
_ = DISCLAIMER


__all__ = ["WeeklyReviewInput", "weekly_review"]
