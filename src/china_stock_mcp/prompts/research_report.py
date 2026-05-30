"""``research_report`` prompt -- 投研报告 Markdown 编排.

Implements *design.md Algorithm 6* (research_report Prompt) for
task 20.1. The prompt orchestrates five service-layer calls and
stitches the results into a single Markdown document with five
labelled sections:

1. **基本面** -- ``FundamentalService.snapshot``
2. **财务报告** -- ``FinancialReportService.report(annual, 4)``
3. **行业对比** -- ``IndustryService.peers(default metrics, top 8)``
4. **资金流向** -- ``MoneyFlowService.get(north or main, top 10)``
5. **技术形态** -- ``KLineService.get_series(daily, 120, qfq, MA20/MA60/MACD)``

Behaviour
---------

- ``report_length`` selects the post-processing strategy
  (Requirement 10.2):

  * ``short``    -- truncate the joined body to ~1 500 tokens
                    (~6 000 characters using the conservative
                    4-chars-per-token proxy used by the formatters
                    test suite). Truncation snaps to the last
                    paragraph boundary so half-tables are never
                    left on screen, then appends a small
                    ``_(short 模式: 内容已截断)_`` marker.
  * ``standard`` -- emit the five sections verbatim.
  * ``deep``     -- append a sixth ``## 深度分析`` section that
                    cross-references the snapshot + peer-table
                    data already gathered (no extra upstream
                    calls, so the prompt remains within the
                    rate-limit budget).

- Per-section graceful degradation (Requirement 10.5):
  every sub-call is wrapped in a ``try/except ChinaStockMCPError``
  and on failure the section body is replaced by a
  ``> ⚠️ 该子模块数据不可用`` block-quote that quotes the error's
  ``to_user_message()`` text. The remaining sections still render so
  the AI client always gets *something* useful.

- The disclaimer is appended exactly once at the end via
  :func:`append_disclaimer` (Requirement 10.6, Property 14).

The prompt is **pure** with respect to the services it receives:
callers (e.g. :mod:`china_stock_mcp.server`) construct the service
instances and pass them in via the ``services`` keyword, which keeps
the function trivially testable without any FastMCP / akshare side
effects.
"""

from __future__ import annotations

from typing import Final, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from china_stock_mcp.error_mapping import bridge_pydantic_error
from china_stock_mcp.exceptions import ChinaStockMCPError
from china_stock_mcp.formatters import DISCLAIMER, append_disclaimer
from china_stock_mcp.models import (
    FundamentalSnapshot,
    KLineSeries,
    MoneyFlow,
    PeerTable,
)
from china_stock_mcp.services.financial_report_service import FinancialReportService
from china_stock_mcp.services.fundamental_service import FundamentalService
from china_stock_mcp.services.industry_service import IndustryService
from china_stock_mcp.services.kline_service import KLineService
from china_stock_mcp.services.money_flow_service import MoneyFlowService
from china_stock_mcp.tools.financial import get_financial_report as _render_financial
from china_stock_mcp.tools.fundamental import get_fundamentals as _render_fundamentals
from china_stock_mcp.tools.industry import get_industry_peers as _render_peers
from china_stock_mcp.tools.kline import get_kline as _render_kline
from china_stock_mcp.tools.money_flow import get_money_flow as _render_money_flow

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Defaults for the orchestrated sub-calls. Pinned here (not exposed
#: to the prompt caller) because design Algorithm 6 fixes them and
#: changing them would alter the report shape.
_FINANCIAL_REPORT_TYPE: Final[str] = "annual"
_FINANCIAL_PERIODS: Final[int] = 4
_PEER_METRICS: Final[tuple[str, ...]] = ("pe", "pb", "roe", "revenue_growth")
_PEER_TOP_N: Final[int] = 8
_MONEY_FLOW_TOP_N: Final[int] = 10
_KLINE_PERIOD: Final[str] = "daily"
_KLINE_COUNT: Final[int] = 120
_KLINE_ADJUST: Final[str] = "qfq"
_KLINE_INDICATORS: Final[tuple[str, ...]] = ("MA20", "MA60", "MACD")

#: Token budget for ``short`` mode (Requirement 10.2). Translated to
#: a character budget via the conservative 4-chars-per-token proxy
#: used by :func:`tests.unit.test_formatters._token_count` so the
#: budget here matches the budget the test suite asserts for
#: rendered tools (Property 13).
_SHORT_TOKEN_BUDGET: Final[int] = 1500
_SHORT_CHAR_BUDGET: Final[int] = _SHORT_TOKEN_BUDGET * 4

#: Marker appended after a ``short`` truncation so callers can detect
#: the truncation downstream.
_SHORT_TRUNCATION_NOTICE: Final[str] = "_(short 模式: 内容已截断)_"

#: Section heading order. Each tuple is ``(heading, key)``; the key
#: is what appears in the ``services`` dict expected by
#: :func:`research_report`.
_SECTION_ORDER: Final[tuple[tuple[str, str], ...]] = (
    ("## 基本面", "fundamental"),
    ("## 财务报告", "financial_report"),
    ("## 行业对比", "industry"),
    ("## 资金流向", "money_flow"),
    ("## 技术形态", "kline"),
)


class _Services(TypedDict):
    """Typed bundle of pre-wired service instances."""

    fundamental: FundamentalService
    financial_report: FinancialReportService
    industry: IndustryService
    money_flow: MoneyFlowService
    kline: KLineService


# ---------------------------------------------------------------------------
# Pydantic input model
# ---------------------------------------------------------------------------


class ResearchReportInput(BaseModel):
    """Pydantic v2 input schema for :func:`research_report`.

    Constraints:

    - ``symbol`` is required and bounded to ``[1, 64]`` characters.
    - ``report_length`` is restricted to ``"short"`` / ``"standard"``
      / ``"deep"`` (Requirement 10.2).
    - ``extra="forbid"`` rejects unknown keys.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    symbol: str = Field(..., min_length=1, max_length=64)
    report_length: Literal["short", "standard", "deep"] = "standard"


# ---------------------------------------------------------------------------
# Public prompt entry
# ---------------------------------------------------------------------------


def research_report(
    symbol: str,
    report_length: str = "standard",
    *,
    services: _Services,
) -> str:
    """Return a 5-section 投研报告 Markdown document for ``symbol``.

    Parameters
    ----------
    symbol:
        Standardized or bare A-share symbol (Chinese name / pinyin
        also accepted; the underlying services normalize it).
    report_length:
        ``"short"`` truncates the body to ~1 500 tokens; ``"deep"``
        appends a sixth 深度分析 section; ``"standard"`` (default)
        emits the five base sections verbatim.
    services:
        Bundle of pre-wired :class:`FundamentalService`,
        :class:`FinancialReportService`, :class:`IndustryService`,
        :class:`MoneyFlowService` and :class:`KLineService`
        instances. The prompt is pure with respect to its services so
        unit tests can supply lightweight stubs.

    Returns
    -------
    str
        Markdown document ending with the standard disclaimer.

    Raises
    ------
    ValidationError
        If any input fails Pydantic validation.
    """

    # 1) Input validation -- pydantic failures bridge to the unified
    #    error tree (Requirement 13.7).
    try:
        validated = ResearchReportInput(
            symbol=symbol,
            report_length=report_length,
        )
    except PydanticValidationError as exc:
        raise bridge_pydantic_error(exc) from exc

    sym = validated.symbol
    length = validated.report_length

    # 2) Run each sub-call, capturing both the raw DTO (when the
    #    section is needed for ``deep`` post-processing) and the
    #    rendered Markdown body. Each section degrades gracefully on
    #    ``ChinaStockMCPError`` per Requirement 10.5.
    fundamental_snapshot, fundamental_body = _section_fundamentals(
        services["fundamental"], sym
    )
    _, financial_body = _section_financial(services["financial_report"], sym)
    peers, peers_body = _section_industry(services["industry"], sym)
    _, flow_body = _section_money_flow(services["money_flow"], sym)
    _, kline_body = _section_kline(services["kline"], sym)

    section_bodies: dict[str, str] = {
        "fundamental": fundamental_body,
        "financial_report": financial_body,
        "industry": peers_body,
        "money_flow": flow_body,
        "kline": kline_body,
    }

    # 3) Compose the five base sections.
    title = f"# 投研报告 ({sym})"
    parts: list[str] = [title]
    for heading, key in _SECTION_ORDER:
        parts.append(heading)
        parts.append(section_bodies[key])
    body = "\n\n".join(parts)

    # 4) Apply the report_length post-processing.
    if length == "short":
        body = _truncate_short(body)
    elif length == "deep":
        body = _append_deep_analysis(body, fundamental_snapshot, peers)

    # 5) Disclaimer (Requirement 10.6, Property 14).
    return append_disclaimer(body)


# ---------------------------------------------------------------------------
# Per-section helpers
# ---------------------------------------------------------------------------


def _section_fundamentals(
    service: FundamentalService, symbol: str
) -> tuple[FundamentalSnapshot | None, str]:
    """Render the 基本面 section, degrading on adapter errors."""

    try:
        # Capture the raw DTO so ``deep`` mode can inspect it without
        # round-tripping through Markdown.
        snapshot = service.snapshot(symbol)
    except ChinaStockMCPError as exc:
        return None, _format_unavailable(exc)
    body = _strip_disclaimer(_render_fundamentals(service, symbol))
    return snapshot, body


def _section_financial(
    service: FinancialReportService, symbol: str
) -> tuple[None, str]:
    """Render the 财务报告 section, degrading on adapter errors."""

    try:
        rendered = _render_financial(
            service,
            symbol=symbol,
            report_type=_FINANCIAL_REPORT_TYPE,
            periods=_FINANCIAL_PERIODS,
        )
    except ChinaStockMCPError as exc:
        return None, _format_unavailable(exc)
    return None, _strip_disclaimer(rendered)


def _section_industry(
    service: IndustryService, symbol: str
) -> tuple[PeerTable | None, str]:
    """Render the 行业对比 section, degrading on adapter errors."""

    try:
        peers = service.peers(
            symbol=symbol,
            metrics=list(_PEER_METRICS),
            top_n=_PEER_TOP_N,
        )
    except ChinaStockMCPError as exc:
        return None, _format_unavailable(exc)
    body = _strip_disclaimer(
        _render_peers(
            service,
            symbol=symbol,
            metrics=list(_PEER_METRICS),
            top_n=_PEER_TOP_N,
        )
    )
    return peers, body


def _section_money_flow(
    service: MoneyFlowService, symbol: str
) -> tuple[MoneyFlow | None, str]:
    """Render the 资金流向 section, degrading on adapter errors.

    For an A-share symbol we ask the upstream for ``main`` flow first
    -- that gives the AI the per-symbol 主力 / 超大单 / 大单 detail
    that a 投研报告 actually wants. If ``main`` is unavailable (e.g.
    HK listing or upstream outage) we transparently fall back to
    ``north`` flow so the section still carries something useful.
    """

    main_error: ChinaStockMCPError | None = None
    try:
        rendered = _render_money_flow(
            service,
            symbol=symbol,
            flow_type="main",
            top_n=_MONEY_FLOW_TOP_N,
        )
        return None, _strip_disclaimer(rendered)
    except ChinaStockMCPError as exc:
        main_error = exc

    try:
        rendered = _render_money_flow(
            service,
            symbol=None,
            flow_type="north",
            top_n=_MONEY_FLOW_TOP_N,
        )
        return None, _strip_disclaimer(rendered)
    except ChinaStockMCPError:
        # Surface the original ``main`` failure -- it is the more
        # informative one for an A-share research report.
        return None, _format_unavailable(main_error)


def _section_kline(
    service: KLineService, symbol: str
) -> tuple[KLineSeries | None, str]:
    """Render the 技术形态 section, degrading on adapter errors."""

    try:
        rendered = _render_kline(
            service,
            symbol=symbol,
            period=_KLINE_PERIOD,
            count=_KLINE_COUNT,
            adjust=_KLINE_ADJUST,
            indicators=list(_KLINE_INDICATORS),
        )
    except ChinaStockMCPError as exc:
        return None, _format_unavailable(exc)
    return None, _strip_disclaimer(rendered)


# ---------------------------------------------------------------------------
# Length-mode post-processing
# ---------------------------------------------------------------------------


def _truncate_short(body: str) -> str:
    """Truncate ``body`` to ``_SHORT_CHAR_BUDGET`` chars at a paragraph.

    The truncation snaps to the last ``\\n\\n`` boundary that fits
    inside the budget so we never leave a half-rendered Markdown
    table on screen. If no such boundary exists (the first paragraph
    alone is already over budget), we hard-cut at the budget and
    still append the truncation marker so the consumer can detect it.
    """

    if len(body) <= _SHORT_CHAR_BUDGET:
        return body

    cutoff = body.rfind("\n\n", 0, _SHORT_CHAR_BUDGET)
    if cutoff <= 0:
        cutoff = _SHORT_CHAR_BUDGET
    head = body[:cutoff].rstrip()
    return f"{head}\n\n{_SHORT_TRUNCATION_NOTICE}"


def _append_deep_analysis(
    body: str,
    fundamental_snapshot: FundamentalSnapshot | None,
    peers: PeerTable | None,
) -> str:
    """Append a ``## 深度分析`` section using the data already gathered.

    The deep analysis is intentionally low-effort: it cross-references
    the snapshot + peer rows we already have, so adding it does not
    incur extra upstream calls. When neither source is available we
    still append the heading + a "数据不足" note so the section
    structure stays consistent across runs.
    """

    bullets: list[str] = []

    if fundamental_snapshot is not None:
        valuation = fundamental_snapshot.valuation
        profitability = fundamental_snapshot.profitability
        growth = fundamental_snapshot.growth

        pe_ttm = valuation.get("pe_ttm")
        pb = valuation.get("pb")
        if pe_ttm is not None or pb is not None:
            bullets.append(
                "- 估值水平: "
                + ", ".join(
                    f"{label}={_fmt(value)}"
                    for label, value in (
                        ("PE_TTM", pe_ttm),
                        ("PB", pb),
                    )
                    if value is not None
                )
            )

        roe = profitability.get("roe")
        gross_margin = profitability.get("gross_margin")
        if roe is not None or gross_margin is not None:
            parts = []
            if roe is not None:
                parts.append(f"ROE={_fmt(roe)}%")
            if gross_margin is not None:
                parts.append(f"毛利率={_fmt(gross_margin)}%")
            bullets.append("- 盈利质量: " + ", ".join(parts))

        revenue_yoy = growth.get("revenue_yoy")
        net_profit_yoy = growth.get("net_profit_yoy")
        if revenue_yoy is not None or net_profit_yoy is not None:
            parts = []
            if revenue_yoy is not None:
                parts.append(f"营收同比={_fmt(revenue_yoy)}%")
            if net_profit_yoy is not None:
                parts.append(f"净利同比={_fmt(net_profit_yoy)}%")
            bullets.append("- 成长动能: " + ", ".join(parts))

    if peers is not None and peers.rows:
        bullets.append(
            f"- 行业坐标: {peers.industry} 共采样 {len(peers.rows)} 家可比公司, "
            f"详见上方行业对比表的分位标注。"
        )

    if not bullets:
        bullets.append("- 数据不足: 未能从上游聚合到可用于深度分析的数值。")

    deep_section = "## 深度分析\n\n" + "\n".join(bullets)
    return f"{body}\n\n{deep_section}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_disclaimer(rendered: str) -> str:
    """Remove the trailing disclaimer + surrounding whitespace.

    Each tool-level renderer appends the disclaimer as its final
    operation; the prompt re-applies it once at the document level so
    the output ends with exactly one disclaimer regardless of how
    many sub-tools contributed (Property 14).
    """

    stripped = rendered.rstrip()
    if stripped.endswith(DISCLAIMER):
        stripped = stripped[: -len(DISCLAIMER)].rstrip()
    return stripped


def _format_unavailable(error: ChinaStockMCPError | None) -> str:
    """Render the Requirement 10.5 graceful-degradation block."""

    message = error.to_user_message() if error is not None else "未知错误"
    # Collapse newlines so the rendered block stays a single Markdown
    # block-quote line (newlines inside a ``> `` paragraph would split
    # the block in some renderers).
    flat = " ".join(line.strip() for line in message.splitlines() if line.strip())
    return f"> ⚠️ 该子模块数据不可用：{flat}"  # noqa: RUF001


def _fmt(value: float | int) -> str:
    """Render a numeric metric value with 2 decimals."""

    return f"{float(value):.2f}"


__all__ = ["ResearchReportInput", "research_report"]
