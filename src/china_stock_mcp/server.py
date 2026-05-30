"""FastMCP server entrypoint for china-stock-mcp.

Implements *Component 1 (FastMCP Server)* from ``design.md``:

- Instantiates the module-level :data:`mcp` :class:`FastMCP` server.
- Wires a process-wide :class:`SymbolService` / :class:`QuoteService`
  pair lazily on first tool invocation, so importing this module is
  side-effect free (no upstream network calls, no ``akshare`` import
  side effects until a tool actually runs).
- Registers two MCP tools (``search_symbol`` / ``get_quote``) that
  delegate to :mod:`china_stock_mcp.tools.search` and
  :mod:`china_stock_mcp.tools.quote`, translating every
  :class:`ChinaStockMCPError` into the user-facing message returned by
  :meth:`ChinaStockMCPError.to_user_message` (Requirement 13.2 /
  Property 17). Any unexpected exception is logged with traceback to
  stderr and surfaced to the AI client as a generic "服务内部错误"
  message so no Python frame leaks across the protocol boundary.
- Exposes :func:`main` as the script entrypoint declared by
  ``[project.scripts]`` in ``pyproject.toml``. ``main`` reads
  :class:`Settings` once and dispatches on
  ``settings.transport`` -- ``"stdio"`` keeps the MCP channel on
  ``stdout`` while logs go to ``stderr``; ``"streamable-http"``
  exposes the HTTP variant for hosted deployments
  (Requirement 12.6).

Backup adapters (``TushareAdapter`` / ``EfinanceAdapter``) are wired
into the service constructors below per task 19.3 -- when an optional
dependency or token is missing, the adapter constructor raises
``ImportError`` / ``RuntimeError`` and we fall back to ``None`` while
logging a warning so the primary path remains usable.

Prompts (task 20.x) and resources (task 21.x) are also registered
here. Each resource handler reuses the corresponding tool's renderer
so the AI client sees byte-identical Markdown whether it pulls
``market://overview`` as a resource or invokes ``get_market_overview``
as a tool. The same ``ChinaStockMCPError`` → ``to_user_message``
translation pattern is applied at every entry point so no Python
frame leaks across the protocol boundary.
"""

from __future__ import annotations

from typing import Final, Literal

from fastmcp import FastMCP

from china_stock_mcp import logger, reconfigure_logger
from china_stock_mcp.adapters.akshare_adapter import AkshareAdapter
from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.adapters.efinance_adapter import EfinanceAdapter
from china_stock_mcp.adapters.tushare_adapter import TushareAdapter
from china_stock_mcp.config import Settings, load_settings
from china_stock_mcp.exceptions import ChinaStockMCPError
from china_stock_mcp.prompts.research_report import (
    research_report as _build_research_report,
)
from china_stock_mcp.prompts.valuation_compare import (
    valuation_compare as _build_valuation_compare,
)
from china_stock_mcp.prompts.weekly_review import (
    weekly_review as _build_weekly_review,
)
from china_stock_mcp.resources.market_overview import (
    market_overview_resource as _build_market_overview_resource,
)
from china_stock_mcp.resources.north_flow import (
    north_flow_resource as _build_north_flow_resource,
)
from china_stock_mcp.resources.symbol_profile import (
    symbol_profile_resource as _build_symbol_profile_resource,
)
from china_stock_mcp.services import (
    FinancialReportService,
    FundamentalService,
    FundService,
    IndustryService,
    KLineService,
    MarketService,
    MoneyFlowService,
    QuoteService,
    ScreenService,
    SymbolService,
)
from china_stock_mcp.tools.financial import (
    get_financial_report as _render_financial_report,
)
from china_stock_mcp.tools.fund import get_fund_info as _render_fund_info
from china_stock_mcp.tools.fundamental import get_fundamentals as _render_fundamentals
from china_stock_mcp.tools.industry import (
    get_industry_peers as _render_industry_peers,
)
from china_stock_mcp.tools.kline import get_kline as _render_kline
from china_stock_mcp.tools.market_overview import (
    get_market_overview as _render_market_overview,
)
from china_stock_mcp.tools.money_flow import get_money_flow as _render_money_flow
from china_stock_mcp.tools.quote import get_quote as _render_quote
from china_stock_mcp.tools.screen import screen_stocks as _render_screen_stocks
from china_stock_mcp.tools.search import search_symbol as _render_search

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: MCP server name advertised to clients (Claude Desktop / Cursor /
#: Cline). Stable so configurations referencing this name keep working
#: across releases.
_SERVER_NAME: Final[str] = "china-stock-mcp"

#: Generic message returned to the MCP client when a non-domain
#: exception escapes a tool body. The original exception is logged
#: with traceback at ERROR level so operators can investigate.
_INTERNAL_ERROR_MESSAGE: Final[str] = (
    "服务内部错误: 请求未能完成. 请稍后重试, 或检查服务端日志."
)


# ---------------------------------------------------------------------------
# Module-level FastMCP server
# ---------------------------------------------------------------------------

#: Process-wide :class:`FastMCP` instance. Tools are registered at
#: import time via the ``@mcp.tool()`` decorator below.
mcp: FastMCP = FastMCP(_SERVER_NAME)


# ---------------------------------------------------------------------------
# Lazy service singleton
# ---------------------------------------------------------------------------

#: Cache of the ``(SymbolService, QuoteService, KLineService,
#: FundamentalService, FinancialReportService, MoneyFlowService,
#: IndustryService, FundService, ScreenService, MarketService)``
#: tuple, built on first tool call. Kept module-level (not on the
#: FastMCP instance) so unit tests can swap it via
#: :func:`_reset_services` in the future without touching the
#: protocol layer. Fallback adapters are wired per service type
#: based on which backup source actually implements the relevant
#: endpoint -- see :func:`_build_services` for the mapping.
_services_singleton: (
    tuple[
        SymbolService,
        QuoteService,
        KLineService,
        FundamentalService,
        FinancialReportService,
        MoneyFlowService,
        IndustryService,
        FundService,
        ScreenService,
        MarketService,
    ]
    | None
) = None


def _build_services() -> (
    tuple[
        SymbolService,
        QuoteService,
        KLineService,
        FundamentalService,
        FinancialReportService,
        MoneyFlowService,
        IndustryService,
        FundService,
        ScreenService,
        MarketService,
    ]
):
    """Construct the shared service tuple (memoized).

    A single :class:`AkshareAdapter` instance backs all services as the
    *primary* source so they share the underlying ``akshare`` module
    import (which is expensive on first use). The cache and rate-limiter
    come from the process-wide singletons defined in
    :mod:`china_stock_mcp.cache` / :mod:`china_stock_mcp.rate_limiter`
    via the service defaults.

    Fallback wiring rationale (Requirements 13.3 / 13.4 / 13.5)
    -----------------------------------------------------------
    Each service receives the backup adapter that *actually implements*
    its endpoint, so :func:`fetch_with_fallback` only switches to a
    source that can answer:

    * ``SymbolService`` -- prefers tushare (clean ``stock_basic``
      universe) and falls back to efinance.
    * ``QuoteService`` -- prefers efinance (real-time, ~15 min delay)
      and falls back to tushare (end-of-day).
    * ``FundamentalService`` / ``FinancialReportService`` -- only
      tushare exposes the equivalent ``fina_indicator`` /
      ``income`` / ``balancesheet`` / ``cashflow`` endpoints on a
      free tier, so efinance is skipped here.
    * ``MoneyFlowService`` -- only efinance exposes 主力 /
      龙虎榜 detail rows in v1.
    * ``KLineService`` / ``IndustryService`` / ``FundService`` /
      ``ScreenService`` / ``MarketService`` -- neither fallback ships
      an equivalent endpoint that matches the v1 DTO contract, so
      ``fallback=None`` is intentional and outages on these tools
      surface to the caller verbatim.

    Construction of each backup adapter is wrapped in a try/except
    that catches:

    * :class:`ImportError` -- the optional dependency
      (``tushare`` / ``efinance``) is not installed; the adapter
      lazy-imports it inside ``__init__``.
    * :class:`RuntimeError` -- :class:`TushareAdapter` raises this
      when no token is configured (``CSM_TUSHARE_TOKEN``).

    On either error we log a warning and set the adapter reference to
    ``None`` so the primary path keeps working.
    """

    global _services_singleton
    if _services_singleton is None:
        adapter = AkshareAdapter()

        tushare_adapter: BaseAdapter | None
        try:
            tushare_adapter = TushareAdapter()
        except (ImportError, RuntimeError) as exc:
            logger.warning(
                "TushareAdapter 不可用 ({}); 相关服务将不启用 tushare 备用源",
                exc,
            )
            tushare_adapter = None

        efinance_adapter: BaseAdapter | None
        try:
            efinance_adapter = EfinanceAdapter()
        except ImportError as exc:
            logger.warning(
                "EfinanceAdapter 不可用 ({}); 相关服务将不启用 efinance 备用源",
                exc,
            )
            efinance_adapter = None

        logger.info(
            "services wired: tushare={}, efinance={}",
            tushare_adapter is not None,
            efinance_adapter is not None,
        )

        # Per-service fallback selection (see docstring above).
        symbol_fallback: BaseAdapter | None = tushare_adapter or efinance_adapter
        quote_fallback: BaseAdapter | None = efinance_adapter or tushare_adapter
        fundamental_fallback: BaseAdapter | None = tushare_adapter
        financial_report_fallback: BaseAdapter | None = tushare_adapter
        money_flow_fallback: BaseAdapter | None = efinance_adapter

        symbol_service = SymbolService(primary=adapter, fallback=symbol_fallback)
        quote_service = QuoteService(primary=adapter, fallback=quote_fallback)
        kline_service = KLineService(primary=adapter, fallback=None)
        fundamental_service = FundamentalService(
            primary=adapter, fallback=fundamental_fallback
        )
        financial_report_service = FinancialReportService(
            primary=adapter, fallback=financial_report_fallback
        )
        money_flow_service = MoneyFlowService(
            primary=adapter, fallback=money_flow_fallback
        )
        industry_service = IndustryService(primary=adapter, fallback=None)
        fund_service = FundService(primary=adapter, fallback=None)
        screen_service = ScreenService(primary=adapter, fallback=None)
        market_service = MarketService(primary=adapter, fallback=None)
        _services_singleton = (
            symbol_service,
            quote_service,
            kline_service,
            fundamental_service,
            financial_report_service,
            money_flow_service,
            industry_service,
            fund_service,
            screen_service,
            market_service,
        )
    return _services_singleton


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def search_symbol(query: str, market: str = "all") -> str:
    """Search A-share / HK-share / public fund symbols.

    Use this tool first when the user gives a Chinese company name,
    pinyin fragment, or partial code, to obtain the standardized symbol
    other tools expect (e.g. ``300750.SZ``).

    Parameters
    ----------
    query:
        Search keyword. Accepts a 6-digit A-share code (auto-suffixed
        with ``.SH`` / ``.SZ`` / ``.BJ``), a 5-digit HK code
        (auto-suffixed with ``.HK``), a 6-digit fund code, a Chinese
        company name, or a pinyin fragment. Empty strings are rejected.
    market:
        Market filter; one of ``"a_stock"`` / ``"hk_stock"`` /
        ``"fund"`` / ``"all"``. Defaults to ``"all"`` (no filter).

    Returns
    -------
    str
        Markdown table with columns: standardized code, name, market,
        industry. Returns "未找到匹配的标的" when nothing matches.
        Always ends with the disclaimer.
    """

    symbol_service, _, _, _, _, _, _, _, _, _ = _build_services()
    try:
        return _render_search(symbol_service, query, market)
    except ChinaStockMCPError as exc:
        return exc.to_user_message()
    except Exception:
        logger.exception("search_symbol failed unexpectedly")
        return _INTERNAL_ERROR_MESSAGE


@mcp.tool()
def get_quote(symbol: str | list[str]) -> str:
    """Fetch a near-real-time (~15 min delay) quote snapshot.

    Accepts a single standardized symbol or a list of up to 20
    symbols. Inputs are normalized server-side, so callers may mix
    standardized codes (``300750.SZ``), bare 6-digit A-share codes,
    bare 5-digit HK codes, Chinese names, or pinyin.

    Parameters
    ----------
    symbol:
        A single symbol string or a list of 1..20 symbols.

    Returns
    -------
    str
        - Single symbol: a Markdown card with price, change pct,
          volume / amount, turnover rate, PE_TTM, PE dynamic, PB,
          total market cap, and float market cap.
        - Multiple symbols: a multi-column Markdown table comparing
          the same key fields across symbols.

        When ``CSM_DATA_DELAY_NOTICE`` is enabled, a "数据延迟约 15 分钟"
        notice is prepended. The disclaimer is always appended last.
    """

    _, quote_service, _, _, _, _, _, _, _, _ = _build_services()
    try:
        return _render_quote(quote_service, symbol)
    except ChinaStockMCPError as exc:
        return exc.to_user_message()
    except Exception:
        logger.exception("get_quote failed unexpectedly")
        return _INTERNAL_ERROR_MESSAGE


@mcp.tool()
def get_kline(
    symbol: str,
    period: str = "daily",
    count: int = 60,
    adjust: str = "qfq",
    indicators: list[str] | None = None,
) -> str:
    """Fetch K 线 bars + 技术指标 + 形态简评 for an A-share symbol.

    Returns a Markdown summary covering: bar count, date range, last
    close, last-bar percent change, optional ``pattern_note``, an
    indicator snapshot table (latest value per requested indicator),
    and the most recent up-to-20 OHLCV bars.

    Parameters
    ----------
    symbol:
        Standardized or bare A-share symbol. HK-share / fund codes are
        not supported here -- the upstream akshare endpoints used for
        K 线 only cover A-share sufficiently in v1.
    period:
        Bar interval -- one of ``"daily"`` / ``"weekly"`` /
        ``"monthly"`` / ``"60min"`` / ``"30min"``. Defaults to
        ``"daily"``.
    count:
        Maximum number of bars (1..250). Defaults to ``60``.
    adjust:
        Price adjustment mode -- one of ``"qfq"`` (前复权) /
        ``"hfq"`` (后复权) / ``"none"``. Defaults to ``"qfq"``.
    indicators:
        Optional list of indicator names; supported names are
        ``MA5`` / ``MA10`` / ``MA20`` / ``MA60`` / ``MACD`` /
        ``RSI14`` / ``BOLL``. Defaults to ``["MA20", "MA60", "MACD"]``.

    Returns
    -------
    str
        Markdown document ending with the standard disclaimer.
    """

    _, _, kline_service, _, _, _, _, _, _, _ = _build_services()
    try:
        return _render_kline(
            kline_service,
            symbol=symbol,
            period=period,
            count=count,
            adjust=adjust,
            indicators=indicators,
        )
    except ChinaStockMCPError as exc:
        return exc.to_user_message()
    except Exception:
        logger.exception("get_kline failed unexpectedly")
        return _INTERNAL_ERROR_MESSAGE


@mcp.tool()
def get_fundamentals(symbol: str) -> str:
    """Fetch a 基本面快照 (估值 / 盈利 / 成长 / 健康) for an A-share symbol.

    Returns a Markdown document containing four labelled sub-tables:

    - 估值指标 (PE_TTM / PE_动 / PB / ...)
    - 盈利能力 (ROE / ROA / 毛利率 / 净利率)
    - 成长性 (营收同比 / 净利润同比 / 单季环比)
    - 财务健康 (资产负债率 / 流动比率 / ...)

    Industry percentile (行业分位) is **not** populated in v1; that
    cross-stock comparison is the responsibility of a future release
    (industry peers tool, task 14.1). When percentile data is present
    a fifth column is appended to each sub-table; when absent, the
    column is omitted entirely so the Markdown stays compact.

    Parameters
    ----------
    symbol:
        Standardized or bare A-share symbol (e.g. ``"600519.SH"`` or
        ``"600519"``). HK-share / fund codes are not supported in v1
        and surface as :class:`DataNotFoundError`.

    Returns
    -------
    str
        Markdown document ending with the standard disclaimer.
    """

    _, _, _, fundamental_service, _, _, _, _, _, _ = _build_services()
    try:
        return _render_fundamentals(fundamental_service, symbol)
    except ChinaStockMCPError as exc:
        return exc.to_user_message()
    except Exception:
        logger.exception("get_fundamentals failed unexpectedly")
        return _INTERNAL_ERROR_MESSAGE


@mcp.tool()
def get_financial_report(
    symbol: str,
    report_type: str = "annual",
    periods: int = 4,
) -> str:
    """Fetch 多期财务报告 (年报 or 季报) for an A-share symbol.

    Returns a Markdown table with one row per metric and one column
    per reporting period. Periods are sorted ascending by report date,
    so the leftmost data column is the oldest and the rightmost is
    the most recent.

    Reported metrics: 营业总收入, 归母净利润, 扣非净利润, 毛利,
    经营性现金流, 总资产, 总负债, 所有者权益.

    Parameters
    ----------
    symbol:
        Standardized or bare A-share symbol (e.g. ``"600519.SH"`` or
        ``"600519"``). HK-share / fund codes are not supported in v1
        and surface as :class:`DataNotFoundError`.
    report_type:
        ``"annual"`` for 年报 only, or ``"quarterly"`` for 年报 + 中报
        + 季报. Defaults to ``"annual"``.
    periods:
        Number of historical periods to fetch, ``[1, 12]``. Defaults
        to ``4``. When the upstream returns fewer rows than requested
        (e.g. a freshly listed company without 4 annual reports), the
        tool surfaces a :class:`DataNotFoundError` advising the
        caller to reduce ``periods`` or switch ``report_type``.

    Returns
    -------
    str
        Markdown document ending with the standard disclaimer.
    """

    _, _, _, _, financial_report_service, _, _, _, _, _ = _build_services()
    try:
        return _render_financial_report(
            financial_report_service,
            symbol=symbol,
            report_type=report_type,
            periods=periods,
        )
    except ChinaStockMCPError as exc:
        return exc.to_user_message()
    except Exception:
        logger.exception("get_financial_report failed unexpectedly")
        return _INTERNAL_ERROR_MESSAGE


@mcp.tool()
def get_money_flow(
    symbol: str | None = None,
    flow_type: str = "north",
    top_n: int = 20,
) -> str:
    """Fetch 资金流向 rows -- 北向 / 主力 / 龙虎榜.

    Returns a Markdown header + ``snapshot_at`` line + table whose
    columns adapt to the requested ``flow_type``:

    - ``north`` -- daily 北向资金 净流入 / 买入 / 卖出 / 持股市值;
      ``symbol`` is ignored.
    - ``main`` -- per-day 主力 / 超大单 / 大单 / 中单 / 小单 净流入
      for ``symbol``; ``symbol`` is required.
    - ``dragon_tiger`` -- recent 龙虎榜 rows (净买额 / 买入额 /
      卖出额 / 换手率 / 上榜原因). When ``symbol`` is provided rows
      are filtered to that code, otherwise the full latest board is
      returned.

    Parameters
    ----------
    symbol:
        Standardized or bare A-share symbol. Required for ``main``;
        optional for ``dragon_tiger``; ignored for ``north``.
    flow_type:
        One of ``"north"`` / ``"main"`` / ``"dragon_tiger"``.
        Defaults to ``"north"``.
    top_n:
        Maximum number of rows to render, ``[1, 100]``. Defaults to
        ``20``.

    Returns
    -------
    str
        Markdown document ending with the standard disclaimer.
    """

    _, _, _, _, _, money_flow_service, _, _, _, _ = _build_services()
    try:
        return _render_money_flow(
            money_flow_service,
            symbol=symbol,
            flow_type=flow_type,
            top_n=top_n,
        )
    except ChinaStockMCPError as exc:
        return exc.to_user_message()
    except Exception:
        logger.exception("get_money_flow failed unexpectedly")
        return _INTERNAL_ERROR_MESSAGE


@mcp.tool()
def get_industry_peers(
    symbol: str,
    metrics: list[str] | None = None,
    top_n: int = 10,
) -> str:
    """Fetch 同行业可比公司 + 行业分位 for an A-share symbol.

    Returns a Markdown 行业对比 table whose columns reflect the
    caller-supplied ``metrics`` order, prefixed by 代码 and 名称.
    Each row carries the requested metric values, and per-metric
    footnotes describe the symbol's own 行业分位 (industry
    percentile) within the comparable peer set.

    Parameters
    ----------
    symbol:
        Standardized or bare A-share symbol (e.g. ``"600519.SH"`` or
        ``"600519"``). HK-share / fund codes are not supported in v1
        and surface as :class:`DataNotFoundError`.
    metrics:
        Optional list of metric names; supported names are
        ``"pe"`` / ``"pb"`` / ``"roe"`` / ``"revenue_growth"``.
        Defaults to all four supported metrics when omitted.
    top_n:
        Maximum number of peer rows to render, ``[1, 50]``. Defaults
        to ``10``. The peer set is sorted by 成交额 descending so
        the most liquid constituents surface first.

    Returns
    -------
    str
        Markdown document ending with the standard disclaimer.
    """

    _, _, _, _, _, _, industry_service, _, _, _ = _build_services()
    try:
        return _render_industry_peers(
            industry_service,
            symbol=symbol,
            metrics=metrics,
            top_n=top_n,
        )
    except ChinaStockMCPError as exc:
        return exc.to_user_message()
    except Exception:
        logger.exception("get_industry_peers failed unexpectedly")
        return _INTERNAL_ERROR_MESSAGE


@mcp.tool()
def get_fund_info(fund_code: str) -> str:
    """Fetch 公募基金 metadata, returns and top holdings.

    Returns a Markdown document with:

    - 基本信息 -- 名称 / 基金经理 / 成立日期 / 规模 (亿/万 单位) /
      近 1/3/6/12 月收益率 / 最大回撤 / 夏普比率 / 同类排名.
    - 前十大持仓 -- 代码 / 名称 / 权重 (2 位小数百分比).
    - 行业分布 (when populated by the upstream).

    Missing optional fields render as ``"-"`` so a sparse upstream
    payload still produces a complete table.

    Parameters
    ----------
    fund_code:
        6-digit bare fund code (no exchange suffix; e.g. ``"510300"``).
        Invalid codes raise :class:`SymbolError` per Requirement 7.3.

    Returns
    -------
    str
        Markdown document ending with the standard disclaimer.
    """

    _, _, _, _, _, _, _, fund_service, _, _ = _build_services()
    try:
        return _render_fund_info(fund_service, fund_code)
    except ChinaStockMCPError as exc:
        return exc.to_user_message()
    except Exception:
        logger.exception("get_fund_info failed unexpectedly")
        return _INTERNAL_ERROR_MESSAGE


@mcp.tool()
def screen_stocks(
    criteria: dict[str, object] | None = None,
    sort_by: str = "market_cap",
    order: str = "desc",
    limit: int = 30,
) -> str:
    """Screen A 股 by multi-factor criteria (PE / PB / ROE / 市值 / 行业).

    Returns a Markdown 选股结果 table whose columns adapt to the
    criteria the caller actually filtered by, prefixed by 代码 / 名称 /
    行业.

    Parameters
    ----------
    criteria:
        Mapping of criterion name → constraint. Supported keys:

        - ``pe_ttm`` / ``pb`` / ``roe`` / ``market_cap`` /
          ``revenue_growth`` -- ``{"min": ..., "max": ...}`` (either
          bound is optional).
        - ``industry`` -- list of industry names; the universe is
          intersected with the union of constituents across listed
          industries.

        ``None`` is treated as the empty mapping (returns the
        unconstrained universe truncated to ``limit``).
    sort_by:
        One of ``"pe_ttm"`` / ``"pb"`` / ``"roe"`` / ``"market_cap"``
        / ``"revenue_growth"``. Defaults to ``"market_cap"``.
    order:
        Either ``"asc"`` or ``"desc"``. Defaults to ``"desc"``.
    limit:
        Maximum number of rows, ``[1, 200]``. Defaults to ``30``.

    Notes
    -----
    v1 limitation: ``roe`` and ``revenue_growth`` are not surfaced by
    the spot universe endpoint, so filtering or sorting by them
    returns an empty result. Filter by ``pe_ttm`` / ``pb`` /
    ``market_cap`` for now.

    Returns
    -------
    str
        Markdown document ending with the standard disclaimer.
    """

    _, _, _, _, _, _, _, _, screen_service, _ = _build_services()
    try:
        return _render_screen_stocks(
            screen_service,
            criteria=dict(criteria) if criteria is not None else {},
            sort_by=sort_by,
            order=order,
            limit=limit,
        )
    except ChinaStockMCPError as exc:
        return exc.to_user_message()
    except Exception:
        logger.exception("screen_stocks failed unexpectedly")
        return _INTERNAL_ERROR_MESSAGE


@mcp.tool()
def get_market_overview() -> str:
    """Fetch a snapshot of the overall A 股 market.

    Returns a Markdown summary covering:

    - 数据时间 + optional "非交易时段" banner (Requirement 9.4).
    - 指数行情 (上证 / 深证成指 / 创业板) -- 名称 / 代码 / 最新 / 涨跌幅.
    - 涨跌家数 (上涨 / 下跌 / 平).
    - 涨跌停 counts (涨停数 / 跌停数, approximated by 涨跌幅 ≥ 9.9
      / ≤ -9.9).
    - 北向资金净流入 (元 → 亿 / 万 unit selection).
    - 行业热度排行 top-5 by 主力净流入.
    - 市场热度评分 (``XX.X / 100``; Requirement 9.2).

    The tool takes no arguments. The result is cached for the
    ``市场总览`` TTL grade (3 600s) so off-hours refreshes do not hit
    the upstream every time. Requests outside trading hours surface
    the most recent published spot frame plus a "非交易时段" banner.

    Returns
    -------
    str
        Markdown document ending with the standard disclaimer.
    """

    _, _, _, _, _, _, _, _, _, market_service = _build_services()
    try:
        return _render_market_overview(market_service)
    except ChinaStockMCPError as exc:
        return exc.to_user_message()
    except Exception:
        logger.exception("get_market_overview failed unexpectedly")
        return _INTERNAL_ERROR_MESSAGE


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


@mcp.prompt()
def research_report(symbol: str, report_length: str = "standard") -> str:
    """Generate a 投研报告 Markdown document for an A-share symbol.

    Orchestrates five service calls (基本面 / 财务报告 / 行业对比 /
    资金流向 / 技术形态) and stitches the results into a single
    research deliverable. Sub-call failures degrade gracefully -- the
    failing section is replaced by a "⚠️ 该子模块数据不可用" block
    while the remaining sections still render so the AI client always
    receives something usable.

    Parameters
    ----------
    symbol:
        Standardized or bare A-share symbol (e.g. ``"600519.SH"`` or
        ``"600519"``). Chinese names / pinyin are also accepted; the
        underlying services normalize them.
    report_length:
        - ``"short"`` truncates the body to ~1 500 tokens for quick
          consumption.
        - ``"standard"`` (default) emits all five base sections.
        - ``"deep"`` appends a sixth 深度分析 section that
          cross-references the snapshot + peer-table data already
          gathered (no extra upstream calls).

    Returns
    -------
    str
        Markdown document ending with the standard disclaimer.
    """

    (
        _,
        _,
        kline_service,
        fundamental_service,
        financial_report_service,
        money_flow_service,
        industry_service,
        _,
        _,
        _,
    ) = _build_services()
    try:
        return _build_research_report(
            symbol,
            report_length,
            services={
                "fundamental": fundamental_service,
                "financial_report": financial_report_service,
                "industry": industry_service,
                "money_flow": money_flow_service,
                "kline": kline_service,
            },
        )
    except ChinaStockMCPError as exc:
        return exc.to_user_message()
    except Exception:
        logger.exception("research_report prompt failed unexpectedly")
        return _INTERNAL_ERROR_MESSAGE


@mcp.prompt()
def valuation_compare(symbols: list[str]) -> str:
    """Generate a 估值对比 Markdown document for 2..10 标的.

    Orchestrates :func:`get_quote` (one batched call covering every
    symbol), :func:`get_fundamentals` (per-symbol valuation /
    profitability / growth metrics) and :func:`get_industry_peers`
    (industry context) into three sections:

    1. 行情对比 -- multi-row table comparing 现价 / 涨跌幅 / PE_TTM /
       PB / 总市值 across every symbol.
    2. 估值横向对比 -- per-symbol sub-table with PE(TTM) / PB / ROE /
       营收增速 so the AI can see each symbol's valuation profile in
       isolation.
    3. 行业横切 -- per-symbol bullet listing the symbol's industry +
       sampled peer count for quick cross-industry context.

    Per-symbol failures degrade gracefully -- the failing row /
    section is replaced by a "⚠️ {symbol} 数据不可用" block while the
    remaining symbols still render (Requirement 10.5).

    Parameters
    ----------
    symbols:
        List of 2..10 standardized or bare symbols (Chinese names /
        pinyin also accepted; the underlying services normalize them).

    Returns
    -------
    str
        Markdown document ending with the standard disclaimer.
    """

    (
        _,
        quote_service,
        _,
        fundamental_service,
        _,
        _,
        industry_service,
        _,
        _,
        _,
    ) = _build_services()
    try:
        return _build_valuation_compare(
            symbols,
            services={
                "quote": quote_service,
                "fundamental": fundamental_service,
                "industry": industry_service,
            },
        )
    except ChinaStockMCPError as exc:
        return exc.to_user_message()
    except Exception:
        logger.exception("valuation_compare prompt failed unexpectedly")
        return _INTERNAL_ERROR_MESSAGE


@mcp.prompt()
def weekly_review() -> str:
    """Generate a 周复盘 Markdown document covering the A 股 market.

    Orchestrates :func:`get_market_overview` and
    :func:`get_money_flow` (北向资金) plus the 行业热度排行 already
    carried by :class:`MarketOverview` to produce a three-section
    weekly review:

    1. 市场总览 -- 指数 / 涨跌家数 / 涨跌停 / 北向 / heat_score.
    2. 北向资金近期走势 -- 最近 20 个交易日的 净流入 / 买入 / 卖出 /
       持股市值.
    3. 行业热度排行 -- 主力净流入排名靠前的行业。

    Sub-call failures degrade gracefully -- the failing section is
    replaced by a "⚠️ 该子模块数据不可用" block while the remaining
    sections still render so the AI client always receives something
    usable (Requirement 10.5).

    Returns
    -------
    str
        Markdown document ending with the standard disclaimer.
    """

    (
        _,
        _,
        _,
        _,
        _,
        money_flow_service,
        _,
        _,
        _,
        market_service,
    ) = _build_services()
    try:
        return _build_weekly_review(
            services={
                "market": market_service,
                "money_flow": money_flow_service,
            },
        )
    except ChinaStockMCPError as exc:
        return exc.to_user_message()
    except Exception:
        logger.exception("weekly_review prompt failed unexpectedly")
        return _INTERNAL_ERROR_MESSAGE


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource("market://overview")
def market_overview_resource() -> str:
    """Live ``市场总览`` Markdown resource.

    Re-uses the same renderer as :func:`get_market_overview` so the
    resource and tool surfaces stay byte-identical. AI clients that
    subscribe to ``market://overview`` receive 指数 / 涨跌家数 /
    涨跌停 / 北向 / 行业热度 / heat_score sections plus the standard
    disclaimer (Requirement 9.1, 12.1, Property 14).
    """

    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        market_service,
    ) = _build_services()
    try:
        return _build_market_overview_resource(
            services={"market": market_service},
        )
    except ChinaStockMCPError as exc:
        return exc.to_user_message()
    except Exception:
        logger.exception("market_overview resource failed unexpectedly")
        return _INTERNAL_ERROR_MESSAGE


@mcp.resource("flow://north")
def north_flow_resource() -> str:
    """Live ``北向资金`` Markdown resource.

    Re-uses the same renderer as
    :func:`get_money_flow` invoked with ``flow_type="north"`` and
    ``top_n=20`` so the resource and tool surfaces stay
    byte-identical. AI clients that subscribe to ``flow://north``
    receive recent 北向资金 净流入 / 买入 / 卖出 / 持股市值 rows
    plus the standard disclaimer (Requirement 5.1, 12.1, Property 14).
    """

    (
        _,
        _,
        _,
        _,
        _,
        money_flow_service,
        _,
        _,
        _,
        _,
    ) = _build_services()
    try:
        return _build_north_flow_resource(
            services={"money_flow": money_flow_service},
        )
    except ChinaStockMCPError as exc:
        return exc.to_user_message()
    except Exception:
        logger.exception("north_flow resource failed unexpectedly")
        return _INTERNAL_ERROR_MESSAGE


@mcp.resource("symbol://{code}/profile")
def symbol_profile_resource(code: str) -> str:
    """Live ``标的概览`` Markdown resource.

    URI parameters:

    - ``code`` -- standardized symbol (``300750.SZ``), bare 6-digit
      A-share code, bare 5-digit HK code, 6-digit fund code, Chinese
      name, or pinyin. The :class:`SymbolService` normalizes the
      caller-supplied value before any downstream lookup so all
      shapes converge on the same resource (Requirement 1.4 / Property
      1).

    Returns a compact 标的概览 Markdown document combining the search
    hit metadata (中文名 / 市场 / 行业) with a 估值 + 盈利 sub-table
    sourced from :class:`FundamentalService.snapshot`. Failures from
    the snapshot call (e.g. HK / fund codes that the v1 fundamentals
    adapter does not cover) are absorbed into a "数据不可用" notice so
    the resource still renders the basic-info section. The standard
    disclaimer is appended at the end (Requirement 12.1, Property 14).
    """

    (
        symbol_service,
        _,
        _,
        fundamental_service,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = _build_services()
    try:
        return _build_symbol_profile_resource(
            code,
            services={
                "symbol": symbol_service,
                "fundamental": fundamental_service,
            },
        )
    except ChinaStockMCPError as exc:
        return exc.to_user_message()
    except Exception:
        logger.exception("symbol_profile resource failed unexpectedly")
        return _INTERNAL_ERROR_MESSAGE


# ---------------------------------------------------------------------------
# Transport dispatch
# ---------------------------------------------------------------------------

#: Transports we expose to MCP clients. The literal type narrows the
#: argument we pass to :meth:`FastMCP.run` so ``mypy --strict`` is
#: happy without a ``cast``. ``Settings.__post_init__`` already
#: validates the inbound string against the same set, so the dict
#: lookup in :func:`_resolve_transport` cannot raise in practice; the
#: explicit ``RuntimeError`` is a defensive net for future
#: ``TRANSPORTS`` additions in :mod:`china_stock_mcp.config`.
_FastMCPTransport = Literal["stdio", "streamable-http"]

_TRANSPORT_MAP: Final[dict[str, _FastMCPTransport]] = {
    "stdio": "stdio",
    "streamable-http": "streamable-http",
}


def _resolve_transport(transport: str) -> _FastMCPTransport:
    """Translate a config transport value to a FastMCP transport id."""

    try:
        return _TRANSPORT_MAP[transport]
    except KeyError as exc:
        supported = ", ".join(sorted(_TRANSPORT_MAP))
        raise RuntimeError(
            f"Unsupported CSM_TRANSPORT={transport!r}; expected one of: {supported}"
        ) from exc


# ---------------------------------------------------------------------------
# Script entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the MCP server using settings from environment variables.

    Reads :class:`Settings` once via :func:`load_settings`, re-applies
    the package logger configuration so ``CSM_LOG_LEVEL`` takes effect
    even if the env var was mutated after import, then runs the
    :data:`mcp` server with the configured transport. ``stdio`` is
    safe to use as the script entrypoint declared in
    ``pyproject.toml``; ``streamable-http`` is the hosted variant.

    The function does not return until the underlying transport loop
    exits (e.g. on stdin EOF for ``stdio``, or on SIGINT for HTTP).
    """

    settings: Settings = load_settings()
    reconfigure_logger(settings)

    transport = _resolve_transport(settings.transport)
    logger.info(
        "starting china-stock-mcp server (transport={}, log_level={})",
        transport,
        settings.log_level,
    )
    mcp.run(transport=transport)


__all__ = ["main", "mcp"]
