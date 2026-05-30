"""``get_market_overview`` tool -- 市场总览 Markdown 渲染.

Implements *Component 2 (Tools Layer)* from ``design.md`` for the
``get_market_overview`` entry. The tool layer is intentionally thin:

1. Validate caller input via :class:`GetMarketOverviewInput`
   (pydantic v2). Currently ``get_market_overview`` takes no
   arguments, but the schema is declared for symmetry with the rest
   of the tool suite and to make future extension (e.g. a
   ``include_top_industries`` toggle) a no-op for callers that bind
   to keyword arguments.
2. Delegate to :meth:`MarketService.overview`, which performs caching,
   rate-limit admission and adapter fallback.
3. Render a Markdown summary covering:

   - 数据时间 + optional "非交易时段" banner (Requirement 9.4).
   - 指数行情 (上证 / 深证成指 / 创业板) (Requirement 9.1).
   - 涨跌家数 (advance / decline / flat).
   - 涨跌停 counts (limit_up / limit_down).
   - 北向资金净流入 (元 → 亿 / 万 unit selection).
   - 行业热度排行 top-N (with 主力净流入).
   - heat_score (``XX.X / 100``; Requirement 9.2).

4. Append the canonical disclaimer via :func:`append_disclaimer`
   (Requirement 12.1, Property 14).

Acceptance criteria covered
---------------------------

- Requirement 9.1 -- 指数 / 涨跌家数 / 涨跌停 / 北向 / 行业热度 / heat_score.
- Requirement 9.2 -- ``heat_score ∈ [0, 100]`` (DTO-enforced; surfaced).
- Requirement 9.3 -- ``snapshot_at`` rendered at the top of Markdown.
- Requirement 9.4 -- 非交易时段 banner.
- Requirement 12.1 / Property 14 -- 末尾追加免责声明.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Final

from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError

from china_stock_mcp.error_mapping import bridge_pydantic_error
from china_stock_mcp.formatters import (
    NONE_PLACEHOLDER,
    finalize_tool_output,
    format_amount,
    render_change,
    render_table,
)
from china_stock_mcp.models import MarketOverview
from china_stock_mcp.services.market_service import MarketService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: A 股 trading session reference -- used for the 非交易时段 banner.
#: Sessions: morning 09:30-11:30 and afternoon 13:00-15:00 (China
#: Standard Time, UTC+8). The banner triggers outside both windows
#: or on weekends. Holiday calendar handling is out of scope for v1;
#: the banner errs on the side of "show the warning" so users on a
#: bank holiday still see "非交易时段".
_CST: Final[timezone] = timezone(timedelta(hours=8))
_MORNING_OPEN: Final[tuple[int, int]] = (9, 30)
_MORNING_CLOSE: Final[tuple[int, int]] = (11, 30)
_AFTERNOON_OPEN: Final[tuple[int, int]] = (13, 0)
_AFTERNOON_CLOSE: Final[tuple[int, int]] = (15, 0)


# ---------------------------------------------------------------------------
# Pydantic input model
# ---------------------------------------------------------------------------


class GetMarketOverviewInput(BaseModel):
    """Pydantic v2 input schema for :func:`get_market_overview`.

    The tool currently accepts no arguments; the schema enforces this
    explicitly so callers that pass extra keys see a :class:`ValidationError`
    (Requirement 13.7) rather than having the value silently ignored.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# Public tool entry
# ---------------------------------------------------------------------------


def get_market_overview(service: MarketService) -> str:
    """Return a Markdown 市场总览 document.

    Parameters
    ----------
    service:
        Pre-wired :class:`MarketService` instance.

    Returns
    -------
    str
        Markdown document covering 指数 / 涨跌家数 / 涨跌停 / 北向资金 /
        行业热度排行 / heat_score, ending with the standard disclaimer.

    Raises
    ------
    ValidationError
        Reserved for future arguments; the current tool surface
        accepts no parameters.
    ChinaStockMCPError
        Any subclass raised by the service / adapter layer is
        propagated verbatim (Requirement 13.1).
    """

    # 1) Tool-boundary input validation -- the model has no fields
    #    yet, but instantiating it rejects unexpected keyword bleed
    #    that may appear in future overloads.
    try:
        GetMarketOverviewInput()
    except PydanticValidationError as exc:
        raise bridge_pydantic_error(exc) from exc

    # 2) Service call -- caching, rate limiting and adapter fallback
    #    happen here.
    overview: MarketOverview = service.overview()

    # 3) Render Markdown.
    body = _render_overview(overview)

    # 4) Unified tool-exit pipeline (Property 13 / 14, Requirement 12.1 / 12.2).
    return finalize_tool_output(body)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_overview(overview: MarketOverview) -> str:
    """Build the Markdown body for a :class:`MarketOverview`."""

    title = "### 市场总览"
    snapshot_line = (
        f"> 数据时间: {overview.snapshot_at.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    sections: list[str] = [title, snapshot_line]

    if _is_non_trading_hour(overview.snapshot_at):
        sections.append("> ⚠️ 非交易时段, 显示最近交易日快照")

    sections.append(_render_indices_section(overview))
    sections.append(_render_breadth_section(overview))
    sections.append(_render_limit_section(overview))
    sections.append(_render_north_flow_section(overview))
    sections.append(_render_top_industries_section(overview))
    sections.append(_render_heat_score_section(overview))

    return "\n\n".join(sections)


def _render_indices_section(overview: MarketOverview) -> str:
    """Render the 指数行情 sub-table."""

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


def _render_breadth_section(overview: MarketOverview) -> str:
    """Render the 涨跌家数 sub-table."""

    header = "**涨跌家数**"
    advance = overview.advance_decline.get("advance", 0)
    decline = overview.advance_decline.get("decline", 0)
    flat = overview.advance_decline.get("flat", 0)
    rows = [{"上涨": str(advance), "下跌": str(decline), "平": str(flat)}]
    table = render_table(rows, headers=["上涨", "下跌", "平"])
    return f"{header}\n\n{table}"


def _render_limit_section(overview: MarketOverview) -> str:
    """Render the 涨跌停 sub-table."""

    header = "**涨跌停**"
    limit_up = overview.limit_stats.get("limit_up", 0)
    limit_down = overview.limit_stats.get("limit_down", 0)
    rows = [{"涨停数": str(limit_up), "跌停数": str(limit_down)}]
    table = render_table(rows, headers=["涨停数", "跌停数"])
    return f"{header}\n\n{table}"


def _render_north_flow_section(overview: MarketOverview) -> str:
    """Render the 北向资金净流入 single-line section."""

    return (
        "**北向资金净流入**\n\n"
        f"{format_amount(overview.north_net_inflow)}"
    )


def _render_top_industries_section(overview: MarketOverview) -> str:
    """Render the 行业热度排行 sub-table (top 5)."""

    header = "**行业热度排行**"
    if not overview.top_inflow_industries:
        return f"{header}\n\n_暂无行业资金流向数据_"

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

    table = render_table(rows, headers=["行业", "主力净流入"])
    return f"{header}\n\n{table}"


def _render_heat_score_section(overview: MarketOverview) -> str:
    """Render the heat_score line."""

    # ``heat_score`` is constrained to ``[0, 100]`` by the DTO so we
    # can format it as a fixed-precision decimal without re-clamping.
    return f"**市场热度评分**: {overview.heat_score:.1f} / 100"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_non_trading_hour(snapshot_at: datetime) -> bool:
    """Return ``True`` when ``snapshot_at`` lies outside A 股 trading hours.

    The check converts ``snapshot_at`` to China Standard Time (UTC+8)
    so naive timestamps and timestamps from any other zone collapse
    to the canonical exchange clock. Weekends always count as
    non-trading; holiday handling is out of scope for v1.
    """

    # Treat naive timestamps as already in CST -- akshare returns
    # 时间 columns in local exchange time, so this matches the
    # underlying convention without an extra conversion.
    local = (
        snapshot_at if snapshot_at.tzinfo is None else snapshot_at.astimezone(_CST)
    )

    if local.weekday() >= 5:  # Saturday / Sunday
        return True

    minutes = local.hour * 60 + local.minute
    morning_open = _MORNING_OPEN[0] * 60 + _MORNING_OPEN[1]
    morning_close = _MORNING_CLOSE[0] * 60 + _MORNING_CLOSE[1]
    afternoon_open = _AFTERNOON_OPEN[0] * 60 + _AFTERNOON_OPEN[1]
    afternoon_close = _AFTERNOON_CLOSE[0] * 60 + _AFTERNOON_CLOSE[1]

    in_morning = morning_open <= minutes <= morning_close
    in_afternoon = afternoon_open <= minutes <= afternoon_close
    return not (in_morning or in_afternoon)


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


__all__ = ["GetMarketOverviewInput", "get_market_overview"]
