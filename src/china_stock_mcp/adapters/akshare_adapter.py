"""AkshareAdapter: primary data source backed by the ``akshare`` library.

This module implements the *primary* adapter referenced by
``design.md`` Component 4 (Adapter Layer). For task 8.1 only
:meth:`AkshareAdapter.search` and :meth:`AkshareAdapter.quote` are
materialized; the remaining nine abstract methods raise
:class:`NotImplementedError` and will be filled in by subsequent
tasks (10.1 / 11.1 / 12.1 / ... / 17.1) so the class stays
instantiable for the search → quote integration tests in 8.7.

References
----------
- ``design.md`` Component 4: Adapter Layer
- Requirement 1.1: ``search_symbol`` returns standardized hits
- Requirement 2.1 / 2.2: single + batch ``get_quote`` snapshots
- Requirement 13.1: every adapter SHALL raise
  :class:`~china_stock_mcp.exceptions.ChinaStockMCPError` subclasses
  rather than letting third-party exceptions escape.

Exception translation
---------------------
``akshare`` re-uses ``requests`` / ``urllib3`` / ``httpx`` under the
hood. Any of the following surfaces are mapped to the unified
hierarchy by :func:`_call_akshare`:

* connection / DNS / timeout / 5xx           → :class:`NetworkError`
* HTTP 429 or "请求过于频繁" wording          → :class:`RateLimitError`
* every other third-party exception          → :class:`DataSourceError`

:class:`~china_stock_mcp.exceptions.DataNotFoundError` is reserved
for "the symbol exists but akshare returned no data" and is raised
by :meth:`AkshareAdapter.quote` when one of the requested symbols is
absent from the spot dataframe.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any, Final, TypeVar, cast

import pandas as pd

from china_stock_mcp import logger
from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.exceptions import (
    ChinaStockMCPError,
    DataNotFoundError,
    DataSourceError,
    NetworkError,
    RateLimitError,
    SymbolError,
)
from china_stock_mcp.models import (
    DEFAULT_QUOTE_DELAY_SECONDS,
    MAX_KLINE_BARS,
    FinancialPeriod,
    FinancialReport,
    FundamentalSnapshot,
    FundInfo,
    KLineBar,
    KLineSeries,
    MarketOverview,
    MoneyFlow,
    PeerTable,
    Quote,
    SymbolHit,
)
from china_stock_mcp.normalizer import detect_market, normalize_symbol

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum number of hits returned by :meth:`AkshareAdapter.search`.
#: Bounded so a wide-net query (e.g. ``"宁"``) cannot blow the
#: per-tool token budget enforced upstream by the formatter layer.
_MAX_SEARCH_HITS: Final[int] = 20

#: Markets accepted by :meth:`AkshareAdapter.search` (Requirement 1.7).
_VALID_SEARCH_MARKETS: Final[frozenset[str]] = frozenset(
    {"a_stock", "hk_stock", "fund", "all"}
)

#: A 股 6-digit code prefixes → exchange suffix.
_A_STOCK_SH_PREFIXES: Final[frozenset[str]] = frozenset({"60", "68", "90"})
_A_STOCK_SZ_PREFIXES: Final[frozenset[str]] = frozenset({"00", "30", "20"})
_A_STOCK_BJ_FIRST_CHAR: Final[str] = "8"

#: Type names that indicate a transient network failure. We compare by
#: name rather than by ``isinstance`` so we do not need to import
#: ``requests`` / ``httpx`` / ``urllib`` at module scope (akshare may
#: be the only place that pulls them in).
_NETWORK_EXC_NAMES: Final[frozenset[str]] = frozenset(
    {
        "ConnectionError",
        "ConnectTimeout",
        "ReadTimeout",
        "Timeout",
        "TimeoutException",
        "ConnectError",
        "URLError",
        "HTTPError",
        "RemoteDisconnected",
        "ChunkedEncodingError",
        "ProxyError",
    }
)

#: Substrings inside an exception ``str`` that indicate rate limiting.
_RATE_LIMIT_PATTERNS: Final[tuple[str, ...]] = (
    "too many requests",
    "rate limit",
    "rate-limit",
    "429",
    "请求过于频繁",
    "访问频率过快",
)

#: Placeholder message for not-yet-implemented adapter methods.
_NOT_IMPLEMENTED_MSG: Final[str] = "implemented in task 10.1+"


# ---------------------------------------------------------------------------
# K-line constants
# ---------------------------------------------------------------------------

#: Mapping from public ``period`` values to akshare ``period`` strings
#: for the daily-grade endpoint :func:`ak.stock_zh_a_hist`.
_AK_DAILY_PERIODS: Final[frozenset[str]] = frozenset(
    {"daily", "weekly", "monthly"}
)

#: Mapping from public ``period`` values to the integer-string accepted
#: by the minute-grade endpoint :func:`ak.stock_zh_a_hist_min_em`.
_AK_MIN_PERIODS: Final[dict[str, str]] = {
    "60min": "60",
    "30min": "30",
}

#: Mapping from public ``adjust`` values to akshare's convention.
_AK_ADJUST_MAP: Final[dict[str, str]] = {
    "qfq": "qfq",
    "hfq": "hfq",
    "none": "",
}

#: Conservative day multiplier per period when computing ``start_date``
#: from ``count``. Each unit of ``count`` corresponds to roughly this
#: many calendar days back. Weekly/monthly need a wider window because
#: trading-week / trading-month density is much sparser than calendar
#: days. The result is clamped at the lower bound below.
_AK_DAY_LOOKBACK: Final[dict[str, int]] = {
    "daily": 2,      # ~1.4 trading days per calendar day; 2x is safe
    "weekly": 10,    # ~1 week per 5 trading days, 10x to be safe
    "monthly": 45,   # ~1 month per 22 trading days, 45x to be safe
}

#: Lower bound for the lookback window so very small ``count`` still
#: pulls a usable amount of history.
_AK_DAY_LOOKBACK_MIN: Final[int] = 90


# ---------------------------------------------------------------------------
# Fundamentals constants
# ---------------------------------------------------------------------------

#: Mapping from :class:`FundamentalSnapshot` bucket field name to the
#: column name returned by ``ak.stock_financial_analysis_indicator_em``.
#: Each row reports values in *percent* (e.g. ``ROEJQ = 12.34`` means
#: 12.34%) which matches the convention used by the rest of the
#: pipeline (``format_percent`` does not rescale).
#:
#: Mappings are intentionally conservative -- only metrics whose
#: meaning is unambiguous from the column code are surfaced. Fields
#: like ``ps`` / ``peg`` / ``ocf_to_net_profit`` are *not* in the
#: upstream payload and therefore are omitted from the resulting
#: bucket entirely (per task spec: "don't insert None for absent
#: keys; only insert None when akshare returns null").
_AK_FUND_PROFITABILITY: Final[dict[str, str]] = {
    "roe": "ROEJQ",        # 净资产收益率(加权)
    "roa": "ZZCJLL",       # 总资产净利率
    "gross_margin": "XSMLL",  # 销售毛利率
    "net_margin": "XSJLL",    # 销售净利率
}
_AK_FUND_GROWTH: Final[dict[str, str]] = {
    "revenue_yoy": "TOTALOPERATEREVETZ",   # 营业总收入同比
    "net_profit_yoy": "PARENTNETPROFITTZ",  # 归母净利润同比
    "qoq": "DJD_TOI_QOQ",                   # 单季营业总收入环比
}
_AK_FUND_HEALTH: Final[dict[str, str]] = {
    "debt_ratio": "ZCFZL",     # 资产负债率
    "current_ratio": "LD",     # 流动比率
}

#: Spot-quote columns reused for the valuation bucket. Sourced from
#: :func:`ak.stock_zh_a_spot_em`, the same endpoint that powers
#: :meth:`AkshareAdapter.quote`. ``ps`` / ``peg`` are not present
#: there and therefore omitted from the bucket entirely (see above).
_AK_FUND_VALUATION_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "pe_ttm": ("市盈率-TTM", "pe_ttm"),
    "pe_dynamic": ("市盈率-动态", "市盈率(动态)", "pe_dynamic"),
    "pb": ("市净率", "pb"),
}


# ---------------------------------------------------------------------------
# Financial report constants
# ---------------------------------------------------------------------------

#: Mapping from :class:`FinancialPeriod` field name to the indicator
#: row label exposed by ``ak.stock_financial_abstract``. The abstract
#: endpoint returns a single dataframe with one row per metric and one
#: column per ``YYYYMMDD`` reporting date, which lets us project the
#: 8 fields below through a single call without joining 3 separate
#: statement endpoints.
#:
#: A note on duplicates: the abstract endpoint sometimes lists the
#: same metric under two ``选项`` groupings (e.g. ``归母净利润`` is
#: surfaced as both 常用指标 and 成长能力) but the underlying value is
#: identical, so picking the first row is safe.
_AK_FIN_METRIC_ROWS: Final[dict[str, str]] = {
    "revenue": "营业总收入",
    "net_profit": "归母净利润",
    "net_profit_excl_nrgl": "扣非净利润",
    "operating_cash_flow": "经营现金流量净额",
    "equity": "股东权益合计(净资产)",
}

#: ``营业成本`` row label used to derive 毛利 = 营业总收入 - 营业成本.
#: The abstract endpoint does not expose 毛利 directly; we compute it
#: from the two underlying figures so the DTO field is always
#: populated.
_AK_FIN_COST_ROW: Final[str] = "营业成本"

#: ``资产负债率`` row label used to derive 总资产 / 总负债 from equity.
#: 资产负债率 is denominated in *percent* on this endpoint
#: (e.g. ``19.04`` ⇒ 19.04%), so we recover:
#:     total_assets = equity / (1 - debt_ratio/100)
#:     total_liabilities = total_assets - equity
#: This avoids a second call to ``stock_financial_report_sina``
#: (which is slower and uses a different reporting universe).
_AK_FIN_DEBT_RATIO_ROW: Final[str] = "资产负债率"

#: Accepted ``report_type`` values at the adapter boundary
#: (Requirement 4.4).
_AK_FIN_REPORT_TYPES: Final[frozenset[str]] = frozenset({"annual", "quarterly"})

#: ``periods`` lower / upper bounds (Requirement 4.4).
_AK_FIN_MIN_PERIODS: Final[int] = 1
_AK_FIN_MAX_PERIODS: Final[int] = 12


# ---------------------------------------------------------------------------
# Money-flow constants
# ---------------------------------------------------------------------------

#: Accepted ``flow_type`` values at the adapter boundary
#: (Requirement 5.5).
_AK_FLOW_TYPES: Final[frozenset[str]] = frozenset(
    {"north", "main", "dragon_tiger"}
)

#: ``top_n`` lower / upper bounds (Requirement 5.4).
_AK_FLOW_MIN_TOP_N: Final[int] = 1
_AK_FLOW_MAX_TOP_N: Final[int] = 100

#: Lookback window (calendar days) for the dragon-tiger detail
#: endpoint. The list endpoint takes a date range and returns rows for
#: every trading day inside it; 14 calendar days is enough to span
#: long holidays (e.g. 春节) without overshooting the upstream's
#: per-call row budget.
_AK_LHB_LOOKBACK_DAYS: Final[int] = 14


# ---------------------------------------------------------------------------
# Industry-peers constants
# ---------------------------------------------------------------------------

#: Accepted ``metrics`` set at the adapter boundary (Requirement 6.2).
#: The adapter validates defensively so a caller bypassing the service
#: layer still surfaces a unified error type.
_AK_PEER_METRICS: Final[frozenset[str]] = frozenset(
    {"pe", "pb", "roe", "revenue_growth"}
)

#: ``top_n`` lower / upper bounds (Requirement 6.4).
_AK_PEER_MIN_TOP_N: Final[int] = 1
_AK_PEER_MAX_TOP_N: Final[int] = 50

#: Mapping from public peer-metric name to the column produced by
#: :func:`ak.stock_board_industry_cons_em`. ``roe`` and
#: ``revenue_growth`` are *not* on that endpoint -- per-symbol
#: financial-indicator calls would explode the request budget for a
#: 50-stock industry. v1 leaves those metrics as ``None`` (rendered as
#: "-" by the formatter) and documents the limitation; future
#: iterations may join ``ak.stock_zh_a_spot_em`` or batch indicator
#: pulls to populate them.
_AK_PEER_VALUATION_COLS: Final[dict[str, str]] = {
    "pe": "市盈率-动态",
    "pb": "市净率",
}

#: Column on the industry-constituents endpoint used to rank rows.
#: 成交额 is the only liquidity proxy reliably available across
#: different industry boards (总市值 is *not* on this endpoint).
_AK_PEER_RANK_COL: Final[str] = "成交额"

#: ``item`` row label inside ``ak.stock_individual_info_em`` whose
#: value carries the symbol's industry name.
_AK_INDUSTRY_INFO_ROW: Final[str] = "行业"


# ---------------------------------------------------------------------------
# Market overview constants
# ---------------------------------------------------------------------------

#: 6-digit codes for the three core A 股 indices surfaced by the
#: market overview tool: 上证综指 / 深证成指 / 创业板指. Matched in
#: order against the ``代码`` column produced by
#: :func:`ak.stock_zh_index_spot_em`. Only the 6-digit numeric body
#: is compared so we do not depend on whether akshare ships the raw
#: code (e.g. ``000001``) or a prefixed code (e.g. ``sh000001``).
_AK_OVERVIEW_INDICES: Final[tuple[tuple[str, str], ...]] = (
    ("000001", "上证指数"),
    ("399001", "深证成指"),
    ("399006", "创业板指"),
)

#: ``涨跌幅`` thresholds (percent) used to approximate 涨停 / 跌停
#: counts from the spot dataframe. A 股 daily limit is ±10% (with
#: ±20% for 创业板/科创板 and ±30% for 北交所), so the 9.9 / -9.9
#: cutoff captures the conservative "close to limit" definition
#: design §Data Models documents (limit_stats: ``{limit_up,
#: limit_down}``). A symbol that closes exactly at +10.0% may round
#: to 9.99 in the spot frame; using 9.9 keeps the count from
#: under-reporting at the cost of including a thin band of near-limit
#: rows. v1 documents this approximation in the tool docstring.
_AK_OVERVIEW_LIMIT_UP_PCT: Final[float] = 9.9
_AK_OVERVIEW_LIMIT_DOWN_PCT: Final[float] = -9.9

#: Maximum number of 行业热度 rows surfaced by the overview tool.
#: Five fits comfortably within the per-tool 3 000-token Markdown
#: budget (Property 13).
_AK_OVERVIEW_TOP_INDUSTRIES: Final[int] = 5

#: Heat-score adjustments. ``HEAT_BREADTH_WEIGHT`` is the slope of
#: the linear breadth → score mapping (0 ⇒ all 跌, 100 ⇒ all 涨).
#: ``HEAT_NORTH_BUMP`` and ``HEAT_LIMIT_BUMP`` add small positive /
#: negative offsets when the 北向 net inflow or 涨跌停 distribution
#: indicates a one-sided sentiment, then the result is clamped to
#: ``[0, 100]`` so DTO validation (``Annotated[float, Field(ge=0,
#: le=100)]``) holds. The exact constants are documented in the
#: ``MarketOverview.heat_score`` docstring so future iterations can
#: tweak the formula without surprising callers.
_AK_HEAT_BREADTH_WEIGHT: Final[float] = 100.0
_AK_HEAT_NORTH_BUMP: Final[float] = 5.0
_AK_HEAT_LIMIT_BUMP: Final[float] = 5.0
_AK_HEAT_LIMIT_THRESHOLD: Final[int] = 50


# ---------------------------------------------------------------------------
# Fund info constants
# ---------------------------------------------------------------------------

#: Fund code regex: 6 digit string, no exchange suffix.
_FUND_CODE_LEN: Final[int] = 6

#: Number of months back used to derive each return horizon. Trading
#: calendar density on the open-end NAV history is roughly 252
#: trading days per 365 calendar days, so dividing by 12 and
#: multiplying by trading-day-per-month is sufficient when navigating
#: from the tail of the dataframe back N months.
_AK_FUND_RETURN_HORIZONS: Final[dict[str, int]] = {
    "return_1m": 21,    # ~21 trading days per month
    "return_3m": 63,
    "return_6m": 126,
    "return_12m": 252,
}

#: Maximum top holdings to include in :class:`FundInfo.top_holdings`
#: (Requirement 7.1 -- 前十大持仓).
_AK_FUND_MAX_HOLDINGS: Final[int] = 10

#: 元 → 亿 / 万 thresholds reused by :meth:`_coerce_amount` when
#: parsing Chinese-formatted AUM strings like ``"12.34亿"``.
_YI: Final[float] = 1e8
_WAN: Final[float] = 1e4


# ---------------------------------------------------------------------------
# Exception translation helpers
# ---------------------------------------------------------------------------


def _is_network_exception(exc: BaseException) -> bool:
    """Return ``True`` if *exc* looks like a transient network failure."""

    return any(base.__name__ in _NETWORK_EXC_NAMES for base in type(exc).__mro__)


def _is_rate_limit_exception(exc: BaseException) -> bool:
    """Return ``True`` if *exc* looks like an upstream rate-limit response."""

    msg = str(exc).lower()
    return any(pattern.lower() in msg for pattern in _RATE_LIMIT_PATTERNS)


def _call_akshare(
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Call ``func`` and translate its exceptions to :class:`ChinaStockMCPError`.

    Order of checks mirrors design "Error Handling" §Scenarios 2-3:

    1. Pass through exceptions that already belong to our hierarchy.
    2. Map rate-limit indicators to :class:`RateLimitError`.
    3. Map transport-level failures to :class:`NetworkError`.
    4. Wrap every other exception as :class:`DataSourceError` so
       Requirement 13.1 holds: no raw third-party exception escapes.
    """

    try:
        return func(*args, **kwargs)
    except ChinaStockMCPError:
        # Already in the unified hierarchy -- propagate verbatim so
        # ``fetch_with_fallback`` can apply Properties 5/6 correctly.
        raise
    except Exception as exc:
        if _is_rate_limit_exception(exc):
            raise RateLimitError(f"akshare 调用频率受限: {exc}") from exc
        if _is_network_exception(exc):
            raise NetworkError(f"akshare 网络错误: {exc}") from exc
        raise DataSourceError(f"akshare error: {exc}") from exc


# ---------------------------------------------------------------------------
# Scalar coercion helpers (pandas → DTO)
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert ``value`` to ``float``; return ``default`` on NaN / failure."""

    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:  # NaN check without importing math
        return default
    return result


def _safe_optional_float(value: Any) -> float | None:
    """Like :func:`_safe_float` but returns ``None`` when missing / invalid."""

    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert ``value`` to ``int`` via ``float``; return ``default`` on failure."""

    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:
        return default
    return int(result)


def _clamp_turnover_rate(value: Any) -> float:
    """Coerce a turnover rate into ``[0, 100]`` (Quote validation rule)."""

    raw = _safe_float(value, default=0.0)
    if raw < 0.0:
        return 0.0
    if raw > 100.0:
        return 100.0
    return raw


def _a_stock_suffix(code: str) -> str | None:
    """Return ``SH`` / ``SZ`` / ``BJ`` for a bare 6-digit A 股 code, else ``None``."""

    if len(code) != 6 or not code.isdigit():
        return None
    prefix = code[:2]
    if prefix in _A_STOCK_SH_PREFIXES:
        return "SH"
    if prefix in _A_STOCK_SZ_PREFIXES:
        return "SZ"
    if code[0] == _A_STOCK_BJ_FIRST_CHAR:
        return "BJ"
    return None


# ---------------------------------------------------------------------------
# AkshareAdapter
# ---------------------------------------------------------------------------


class AkshareAdapter(BaseAdapter):
    """Primary adapter backed by the ``akshare`` library.

    Only ``search`` and ``quote`` are implemented in task 8.1; the other
    abstract methods raise :class:`NotImplementedError` so the class is
    instantiable for the integration tests in task 8.7. Subsequent
    tasks (10.1 / 11.1 / 12.1 / 13.1 / 14.1 / 15.1 / 17.1) replace
    those placeholders one-by-one.
    """

    name: str = "akshare"

    def __init__(self) -> None:
        # Lazy-import ``akshare`` so that importing this module has no
        # side effects (akshare itself does heavy work at import time).
        # The import is performed once and cached on the instance.
        import akshare as ak

        self._ak = ak

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(self, query: str, market: str) -> list[SymbolHit]:
        """Search standardized symbols by code or Chinese name.

        Implementation strategy (Requirement 1.1):

        1. Pull the spot dataframe(s) for the requested market(s).
        2. Filter rows where ``代码`` or ``名称`` contains *query*
           (case-insensitive substring match).
        3. Map each surviving row into a :class:`SymbolHit` and cap
           the result list at :data:`_MAX_SEARCH_HITS` entries
           (Requirement 1.7 narrows the search space; this method
           additionally caps the response size).

        Parameters
        ----------
        query:
            User-supplied query; trimmed but otherwise passed through
            verbatim. An empty string yields an empty result rather
            than dumping the entire universe.
        market:
            One of ``"a_stock"`` / ``"hk_stock"`` / ``"fund"`` /
            ``"all"``. Any other value raises
            :class:`DataSourceError` so the violation surfaces with
            the unified error type.
        """

        if market not in _VALID_SEARCH_MARKETS:
            raise DataSourceError(
                f"akshare error: 不支持的 market: {market!r}. "
                f"必须是 {sorted(_VALID_SEARCH_MARKETS)} 之一"
            )

        q = (query or "").strip()
        if not q:
            return []

        hits: list[SymbolHit] = []
        if market in {"a_stock", "all"}:
            hits.extend(self._search_a_stock(q))
        if market in {"hk_stock", "all"} and len(hits) < _MAX_SEARCH_HITS:
            hits.extend(self._search_hk_stock(q))
        if market in {"fund", "all"} and len(hits) < _MAX_SEARCH_HITS:
            hits.extend(self._search_fund(q))

        return hits[:_MAX_SEARCH_HITS]

    def _search_a_stock(self, query: str) -> list[SymbolHit]:
        df = _call_akshare(self._ak.stock_zh_a_spot_em)
        df = cast("pd.DataFrame", df)
        if df is None or df.empty:
            return []

        code_col = self._pick_column(df, ("代码", "code", "symbol"))
        name_col = self._pick_column(df, ("名称", "name"))
        if code_col is None or name_col is None:
            return []

        mask = self._build_query_mask(df, code_col, name_col, query)
        matched = df.loc[mask, [code_col, name_col]]
        if matched.empty:
            return []

        hits: list[SymbolHit] = []
        for code_val, name_val in matched.itertuples(index=False, name=None):
            code = str(code_val).strip()
            name = str(name_val).strip()
            if not code or not name:
                continue
            suffix = _a_stock_suffix(code)
            if suffix is None:
                # Skip non-A-share rows that may sneak into the spot
                # frame (B 股, exotic 9xxxxx codes that do not match
                # the design's prefix table).
                continue
            hits.append(
                SymbolHit(
                    code=f"{code}.{suffix}",
                    name=name,
                    market="a_stock",
                )
            )
            if len(hits) >= _MAX_SEARCH_HITS:
                break
        return hits

    def _search_hk_stock(self, query: str) -> list[SymbolHit]:
        df = _call_akshare(self._ak.stock_hk_spot_em)
        df = cast("pd.DataFrame", df)
        if df is None or df.empty:
            return []

        code_col = self._pick_column(df, ("代码", "symbol", "code"))
        name_col = self._pick_column(df, ("名称", "name"))
        if code_col is None or name_col is None:
            return []

        mask = self._build_query_mask(df, code_col, name_col, query)
        matched = df.loc[mask, [code_col, name_col]]
        if matched.empty:
            return []

        hits: list[SymbolHit] = []
        for code_val, name_val in matched.itertuples(index=False, name=None):
            code = str(code_val).strip()
            name = str(name_val).strip()
            if not code or not name:
                continue
            # akshare returns HK codes already padded to 5 digits in
            # most builds, but be defensive: zero-pad shorter codes
            # and drop anything non-numeric.
            if not code.isdigit():
                continue
            code = code.zfill(5)
            if len(code) != 5:
                continue
            hits.append(
                SymbolHit(
                    code=f"{code}.HK",
                    name=name,
                    market="hk_stock",
                )
            )
            if len(hits) >= _MAX_SEARCH_HITS:
                break
        return hits

    def _search_fund(self, query: str) -> list[SymbolHit]:
        df = _call_akshare(self._ak.fund_name_em)
        df = cast("pd.DataFrame", df)
        if df is None or df.empty:
            return []

        code_col = self._pick_column(df, ("基金代码", "代码", "code"))
        name_col = self._pick_column(df, ("基金简称", "基金名称", "名称", "name"))
        if code_col is None or name_col is None:
            return []

        mask = self._build_query_mask(df, code_col, name_col, query)
        matched = df.loc[mask, [code_col, name_col]]
        if matched.empty:
            return []

        hits: list[SymbolHit] = []
        for code_val, name_val in matched.itertuples(index=False, name=None):
            code = str(code_val).strip()
            name = str(name_val).strip()
            if not code or not name or not code.isdigit() or len(code) != 6:
                continue
            hits.append(
                SymbolHit(
                    code=code,
                    name=name,
                    market="fund",
                )
            )
            if len(hits) >= _MAX_SEARCH_HITS:
                break
        return hits

    @staticmethod
    def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
        """Return the first column name from *candidates* present in ``df``."""

        for name in candidates:
            if name in df.columns:
                return name
        return None

    @staticmethod
    def _build_query_mask(
        df: pd.DataFrame,
        code_col: str,
        name_col: str,
        query: str,
    ) -> pd.Series[bool]:
        """Build a boolean mask matching *query* in either column.

        Substring match is case-insensitive; Chinese characters survive
        ``str.casefold`` unchanged so the rule applies uniformly.
        """

        needle = query.casefold()
        codes = df[code_col].astype(str).str.casefold()
        names = df[name_col].astype(str).str.casefold()
        return cast(
            "pd.Series[bool]",
            codes.str.contains(needle, regex=False, na=False)
            | names.str.contains(needle, regex=False, na=False),
        )

    # ------------------------------------------------------------------
    # quote
    # ------------------------------------------------------------------

    def quote(self, symbols: list[str]) -> list[Quote]:
        """Fetch real-time (delayed ~15 min) quote snapshots.

        Pipeline:

        1. Normalize each input via :func:`normalize_symbol` so callers
           may pass bare codes or already-suffixed ones interchangeably.
        2. Group the standardized symbols by market and call the
           corresponding spot endpoint **once per group** to minimize
           upstream load.
        3. Map each requested symbol to its row and build a
           :class:`Quote`. Missing symbols raise
           :class:`DataNotFoundError` listing the absent codes.

        The function preserves the caller-provided order of *symbols*
        in the returned list.
        """

        if not symbols:
            return []

        # Standardize while remembering the caller's order. Multiple
        # raw inputs may map to the same standardized symbol (e.g.
        # "300750" and "300750.SZ"); we keep both in the output so the
        # adapter is order-preserving and idempotent w.r.t. duplicates.
        std_symbols: list[str] = [normalize_symbol(s) for s in symbols]

        unique_by_market: dict[str, list[str]] = {
            "a_stock": [],
            "hk_stock": [],
            "fund": [],
        }
        seen: set[str] = set()
        for std in std_symbols:
            if std in seen:
                continue
            seen.add(std)
            market = detect_market(std)
            unique_by_market[market].append(std)

        # Build a lookup table std_symbol -> Quote for O(1) replay.
        quote_by_symbol: dict[str, Quote] = {}
        if unique_by_market["a_stock"]:
            quote_by_symbol.update(
                self._quote_a_stock(unique_by_market["a_stock"])
            )
        if unique_by_market["hk_stock"]:
            quote_by_symbol.update(
                self._quote_hk_stock(unique_by_market["hk_stock"])
            )
        if unique_by_market["fund"]:
            # Funds are out of scope for task 8.1 (no real-time quote
            # endpoint with the same shape as a-share / hk-share spot).
            # Surface as DataNotFoundError so callers can switch to
            # ``get_fund_info`` (Requirement 7).
            missing = ", ".join(unique_by_market["fund"])
            raise DataNotFoundError(
                f"akshare 适配器暂不支持基金实时行情: {missing}. "
                "请改用 get_fund_info 工具"
            )

        # Replay caller order, surfacing every absent symbol at once.
        missing_symbols: list[str] = []
        results: list[Quote] = []
        for std in std_symbols:
            quote_dto = quote_by_symbol.get(std)
            if quote_dto is None:
                missing_symbols.append(std)
                continue
            results.append(quote_dto)

        if missing_symbols:
            unique_missing = sorted(set(missing_symbols))
            raise DataNotFoundError(
                "未找到行情数据: " + ", ".join(unique_missing)
            )

        return results

    def _quote_a_stock(self, std_symbols: list[str]) -> dict[str, Quote]:
        df = _call_akshare(self._ak.stock_zh_a_spot_em)
        df = cast("pd.DataFrame", df)
        if df is None or df.empty:
            return {}

        code_col = self._pick_column(df, ("代码", "code", "symbol"))
        if code_col is None:
            return {}

        # Map bare 6-digit code -> standardized symbol so we can index
        # into the dataframe using akshare's native code column.
        bare_to_std: dict[str, str] = {std.split(".")[0]: std for std in std_symbols}
        mask = df[code_col].astype(str).isin(list(bare_to_std.keys()))
        matched = df.loc[mask]
        if matched.empty:
            return {}

        timestamp = self._pick_timestamp(df)
        out: dict[str, Quote] = {}
        for row in matched.itertuples(index=False):
            row_dict = row._asdict() if hasattr(row, "_asdict") else dict(
                zip(matched.columns, row, strict=False)
            )
            bare = str(row_dict.get(code_col, "")).strip()
            std = bare_to_std.get(bare)
            if std is None:
                continue
            out[std] = self._row_to_quote(row_dict, std, timestamp)
        return out

    def _quote_hk_stock(self, std_symbols: list[str]) -> dict[str, Quote]:
        df = _call_akshare(self._ak.stock_hk_spot_em)
        df = cast("pd.DataFrame", df)
        if df is None or df.empty:
            return {}

        code_col = self._pick_column(df, ("代码", "symbol", "code"))
        if code_col is None:
            return {}

        bare_to_std: dict[str, str] = {std.split(".")[0]: std for std in std_symbols}
        codes_series = df[code_col].astype(str).str.zfill(5)
        mask = codes_series.isin(list(bare_to_std.keys()))
        matched = df.loc[mask]
        if matched.empty:
            return {}

        timestamp = self._pick_timestamp(df)
        out: dict[str, Quote] = {}
        for row in matched.itertuples(index=False):
            row_dict = row._asdict() if hasattr(row, "_asdict") else dict(
                zip(matched.columns, row, strict=False)
            )
            bare = str(row_dict.get(code_col, "")).strip().zfill(5)
            std = bare_to_std.get(bare)
            if std is None:
                continue
            out[std] = self._row_to_quote(row_dict, std, timestamp)
        return out

    @staticmethod
    def _pick_timestamp(df: pd.DataFrame) -> datetime:
        """Extract a ``datetime`` from the dataframe's 时间 column or now()."""

        for col in ("时间", "更新时间", "timestamp"):
            if col not in df.columns:
                continue
            try:
                # Use the first row's value as the snapshot time;
                # akshare's spot endpoints return a single timestamp
                # shared across the whole frame.
                value = df[col].iloc[0]
            except (IndexError, KeyError):  # pragma: no cover - defensive
                continue
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.isna(parsed):
                continue
            ts = cast("pd.Timestamp", parsed).to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            return cast(datetime, ts)
        return datetime.now(UTC)

    @staticmethod
    def _row_to_quote(
        row: dict[str, Any],
        std_symbol: str,
        timestamp: datetime,
    ) -> Quote:
        """Map an akshare spot dataframe row into a :class:`Quote` DTO.

        Field aliases account for minor schema drift across akshare
        builds; missing optional metrics fall back to ``None`` (PE / PB)
        or ``0`` (numeric required fields) so DTO validation still
        succeeds when an upstream column is absent.
        """

        def pick(*aliases: str) -> Any:
            for alias in aliases:
                if alias in row:
                    return row[alias]
            return None

        name = str(pick("名称", "name") or "").strip() or std_symbol
        return Quote(
            symbol=std_symbol,
            name=name,
            price=_safe_float(pick("最新价", "price")),
            change=_safe_float(pick("涨跌额", "change")),
            change_pct=_safe_float(pick("涨跌幅", "change_pct")),
            volume=_safe_int(pick("成交量", "volume")),
            amount=_safe_float(pick("成交额", "amount")),
            turnover_rate=_clamp_turnover_rate(pick("换手率", "turnover_rate")),
            pe_ttm=_safe_optional_float(pick("市盈率-TTM", "pe_ttm")),
            pe_dynamic=_safe_optional_float(
                pick("市盈率-动态", "市盈率(动态)", "pe_dynamic")
            ),
            pb=_safe_optional_float(pick("市净率", "pb")),
            market_cap=_safe_float(pick("总市值", "market_cap")),
            float_market_cap=_safe_float(
                pick("流通市值", "float_market_cap")
            ),
            timestamp=timestamp,
            delay_seconds=DEFAULT_QUOTE_DELAY_SECONDS,
        )

    # ------------------------------------------------------------------
    # Placeholders for tasks 10.1 / 11.1 / 12.1 / 13.1 / 14.1 / 15.1 / 17.1
    # ------------------------------------------------------------------

    def kline(
        self,
        symbol: str,
        period: str,
        count: int,
        adjust: str,
    ) -> KLineSeries:
        """Fetch K-line bars for ``symbol``.

        Pipeline:

        1. Normalize ``symbol`` and reject HK / fund codes (akshare's
           daily / minute history endpoints used here only cover A-share
           sufficiently for v1).
        2. Translate ``period`` and ``adjust`` to akshare's convention.
        3. For daily / weekly / monthly call
           :func:`ak.stock_zh_a_hist` with a conservative date window
           computed from ``count``; for 60min / 30min call
           :func:`ak.stock_zh_a_hist_min_em` and let the service layer
           keep the most recent ``count`` rows.
        4. Convert each row into :class:`KLineBar` and truncate to the
           most recent ``min(count, 250)`` rows so DTO validation
           (``len(bars) <= 250``) and caller intent both hold.

        ``indicators`` and ``pattern_note`` are intentionally left
        empty here; the service layer (``KLineService``) computes them
        on top of the bars.
        """

        std_symbol = normalize_symbol(symbol)
        market = detect_market(std_symbol)
        if market != "a_stock":
            raise DataNotFoundError(
                f"akshare 适配器暂不支持港股/基金 K 线: {std_symbol}"
            )

        if period not in _AK_DAILY_PERIODS and period not in _AK_MIN_PERIODS:
            raise DataSourceError(
                f"akshare error: 不支持的 period: {period!r}"
            )
        if adjust not in _AK_ADJUST_MAP:
            raise DataSourceError(
                f"akshare error: 不支持的 adjust: {adjust!r}"
            )

        bare = std_symbol.split(".")[0]
        ak_adjust = _AK_ADJUST_MAP[adjust]

        if period in _AK_DAILY_PERIODS:
            df = self._fetch_kline_daily(bare, period, ak_adjust, count)
        else:
            ak_period = _AK_MIN_PERIODS[period]
            df = self._fetch_kline_min(bare, ak_period, ak_adjust)

        bars = self._dataframe_to_bars(df, period)
        # Truncate to the most recent ``min(count, 250)`` rows; the
        # data source returns rows in ascending date order so we slice
        # from the tail.
        keep = max(0, min(count, MAX_KLINE_BARS))
        if len(bars) > keep:
            bars = bars[-keep:]

        return KLineSeries(
            symbol=std_symbol,
            period=cast(Any, period),
            adjust=cast(Any, adjust),
            bars=bars,
            indicators={},
            pattern_note=None,
        )

    def _fetch_kline_daily(
        self,
        bare_symbol: str,
        period: str,
        ak_adjust: str,
        count: int,
    ) -> pd.DataFrame:
        """Call ``ak.stock_zh_a_hist`` with a conservative date window."""

        end_dt = datetime.now(UTC).date()
        days_per_unit = _AK_DAY_LOOKBACK.get(period, 2)
        lookback_days = max(_AK_DAY_LOOKBACK_MIN, days_per_unit * max(count, 1))
        start_dt = end_dt - timedelta(days=lookback_days)
        df = _call_akshare(
            self._ak.stock_zh_a_hist,
            symbol=bare_symbol,
            period=period,
            start_date=start_dt.strftime("%Y%m%d"),
            end_date=end_dt.strftime("%Y%m%d"),
            adjust=ak_adjust,
        )
        return cast("pd.DataFrame", df) if df is not None else pd.DataFrame()

    def _fetch_kline_min(
        self,
        bare_symbol: str,
        ak_period: str,
        ak_adjust: str,
    ) -> pd.DataFrame:
        """Call ``ak.stock_zh_a_hist_min_em`` (defaults cover full range)."""

        df = _call_akshare(
            self._ak.stock_zh_a_hist_min_em,
            symbol=bare_symbol,
            period=ak_period,
            adjust=ak_adjust,
        )
        return cast("pd.DataFrame", df) if df is not None else pd.DataFrame()

    @staticmethod
    def _dataframe_to_bars(df: pd.DataFrame, period: str) -> list[KLineBar]:
        """Convert akshare K-line dataframe rows into :class:`KLineBar`.

        Both daily and minute endpoints share the OHLCV column names
        (``开盘`` / ``收盘`` / ``最高`` / ``最低`` / ``成交量`` / ``成交额``);
        the date column differs (``日期`` for daily, ``时间`` for minute).
        Rows missing OHLC or violating the OHLC inequality are skipped
        rather than failing the whole fetch.
        """

        if df is None or df.empty:
            return []

        date_col: str | None = None
        for candidate in ("日期", "时间", "date"):
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            return []

        required_cols = ("开盘", "收盘", "最高", "最低")
        for col in required_cols:
            if col not in df.columns:
                return []

        bars: list[KLineBar] = []
        for row in df.itertuples(index=False):
            row_dict = (
                row._asdict()
                if hasattr(row, "_asdict")
                else dict(zip(df.columns, row, strict=False))
            )
            bar_date = AkshareAdapter._coerce_bar_date(row_dict.get(date_col))
            if bar_date is None:
                continue
            open_v = _safe_float(row_dict.get("开盘"))
            close_v = _safe_float(row_dict.get("收盘"))
            high_v = _safe_float(row_dict.get("最高"))
            low_v = _safe_float(row_dict.get("最低"))
            volume_v = _safe_int(row_dict.get("成交量"))
            amount_v = _safe_float(row_dict.get("成交额"))

            # Defensive clamp: some upstream rows have rounding noise
            # that violates the OHLC inequality by a hair. Re-derive
            # high / low from the OHLC envelope so downstream DTO
            # validation does not reject the whole series.
            envelope_high = max(open_v, close_v, high_v, low_v)
            envelope_low = min(open_v, close_v, high_v, low_v)
            high_v = max(high_v, envelope_high)
            low_v = min(low_v, envelope_low)

            try:
                bars.append(
                    KLineBar(
                        date=bar_date,
                        open=open_v,
                        high=high_v,
                        low=low_v,
                        close=close_v,
                        volume=max(0, volume_v),
                        amount=max(0.0, amount_v),
                    )
                )
            except ValueError:
                # Skip rows that still fail OHLC validation after the
                # envelope clamp -- preserves partial data instead of
                # throwing the whole fetch away.
                continue

        # Reference period to make the parameter meaningful for callers
        # that grep for it; akshare already returns rows in ascending
        # date order so no further sorting is required.
        _ = period
        return bars

    @staticmethod
    def _coerce_bar_date(value: Any) -> date | None:
        """Coerce a dataframe cell into :class:`date`."""

        if value is None:
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        try:
            parsed = pd.to_datetime(value, errors="coerce")
        except (TypeError, ValueError):
            return None
        if pd.isna(parsed):
            return None
        result: date = cast("pd.Timestamp", parsed).date()
        return result

    def fundamentals(self, symbol: str) -> FundamentalSnapshot:
        """Fetch valuation / profitability / growth / health buckets.

        Pipeline:

        1. Normalize ``symbol``; reject HK / fund codes (A 股 only in
           v1, per task 11.1 spec).
        2. Pull valuation metrics (pe_ttm / pe_dynamic / pb) from the
           spot endpoint :func:`ak.stock_zh_a_spot_em` shared with
           :meth:`AkshareAdapter.quote`.
        3. Pull profitability / growth / health metrics from the most
           recent row returned by
           :func:`ak.stock_financial_analysis_indicator_em`. Values
           are denominated in *percent* (e.g. ``ROEJQ = 12.34`` ⇒
           12.34%) which matches the unit convention used elsewhere
           in the pipeline.
        4. Surface only fields that the upstream actually populated:
           a metric whose column is missing from the upstream payload
           is omitted from the bucket entirely; a metric whose column
           exists but is NaN / null is recorded as ``None`` so the
           formatter renders ``"-"``.

        ``industry_percentile`` is left empty here -- cross-stock peer
        comparison is the responsibility of task 14.1 (industry
        peers). The service layer documents the v1 limitation.
        """

        std_symbol = normalize_symbol(symbol)
        market = detect_market(std_symbol)
        if market != "a_stock":
            raise DataNotFoundError(
                f"akshare 适配器暂不支持港股/基金基本面: {std_symbol}"
            )

        valuation = self._fetch_valuation_bucket(std_symbol)
        prof, growth, health = self._fetch_indicator_buckets(std_symbol)

        return FundamentalSnapshot(
            symbol=std_symbol,
            valuation=valuation,
            profitability=prof,
            growth=growth,
            health=health,
            industry_percentile={},
        )

    def _fetch_valuation_bucket(self, std_symbol: str) -> dict[str, float | None]:
        """Pull pe_ttm / pe_dynamic / pb from the spot endpoint.

        Only metrics whose source column is present in the upstream
        dataframe end up in the bucket; absent columns are omitted
        (per task spec). NaN / null values map to ``None`` so the
        renderer can show a "-" placeholder.
        """

        df = _call_akshare(self._ak.stock_zh_a_spot_em)
        df = cast("pd.DataFrame", df)
        if df is None or df.empty:
            return {}

        code_col = self._pick_column(df, ("代码", "code", "symbol"))
        if code_col is None:
            return {}

        bare = std_symbol.split(".")[0]
        match = df.loc[df[code_col].astype(str) == bare]
        if match.empty:
            raise DataNotFoundError(
                f"未找到基本面行情数据: {std_symbol}"
            )
        row = match.iloc[0].to_dict()

        bucket: dict[str, float | None] = {}
        for metric, aliases in _AK_FUND_VALUATION_ALIASES.items():
            present = False
            value: Any = None
            for alias in aliases:
                if alias in row:
                    present = True
                    value = row[alias]
                    break
            if not present:
                # Upstream column absent -- omit the metric entirely
                # rather than recording None.
                continue
            bucket[metric] = _safe_optional_float(value)
        return bucket

    def _fetch_indicator_buckets(
        self,
        std_symbol: str,
    ) -> tuple[dict[str, float | None], dict[str, float | None], dict[str, float | None]]:
        """Pull profitability / growth / health from the latest report.

        Returns three dicts in the bucket order
        ``(profitability, growth, health)``. The most recent row by
        ``REPORT_DATE`` is used so the snapshot reflects the latest
        published interim or annual report.
        """

        try:
            df = _call_akshare(
                self._ak.stock_financial_analysis_indicator_em,
                symbol=std_symbol,
                indicator="按报告期",
            )
        except DataSourceError:
            # Some symbols (e.g. recently listed shares) raise from
            # the indicator endpoint even though they have a valid
            # spot quote. Surface as DataNotFoundError so
            # ``fetch_with_fallback`` does not silently switch sources.
            raise DataNotFoundError(
                f"未找到财务指标数据: {std_symbol}"
            ) from None

        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty:
            return {}, {}, {}

        # Pick the most recent report row.
        if "REPORT_DATE" in df.columns:
            df_sorted = df.sort_values("REPORT_DATE", ascending=False)
            row = df_sorted.iloc[0].to_dict()
        else:
            row = df.iloc[0].to_dict()

        profitability = self._project_bucket(row, _AK_FUND_PROFITABILITY)
        growth = self._project_bucket(row, _AK_FUND_GROWTH)
        health = self._project_bucket(row, _AK_FUND_HEALTH)
        return profitability, growth, health

    @staticmethod
    def _project_bucket(
        row: dict[str, Any],
        mapping: dict[str, str],
    ) -> dict[str, float | None]:
        """Project ``row`` through ``mapping`` (target_name → source_col).

        Per task spec: only metrics whose source column is present in
        ``row`` are surfaced. Present-but-null columns map to ``None``
        so the formatter can render a ``-`` placeholder.
        """

        bucket: dict[str, float | None] = {}
        for target, source in mapping.items():
            if source not in row:
                continue
            bucket[target] = _safe_optional_float(row[source])
        return bucket

    def financial_report(
        self,
        symbol: str,
        report_type: str,
        periods: int,
    ) -> FinancialReport:
        """Fetch ``periods`` of 财务报告 (annual or quarterly).

        Pipeline:

        1. Normalize ``symbol`` and reject HK / fund codes (A 股 only
           in v1; HK financial endpoints have a different schema).
        2. Validate ``report_type`` and ``periods`` defensively at the
           adapter boundary -- the service layer is the primary
           validator but the adapter must still raise a unified error
           if a caller bypasses it.
        3. Pull a single call against
           :func:`ak.stock_financial_abstract`, which returns one row
           per metric and one column per ``YYYYMMDD`` reporting date.
           Filter the date columns by ``report_type``:

           * ``annual``    → keep only ``MMDD == "1231"`` columns.
           * ``quarterly`` → keep every ``0331/0630/0930/1231`` column.

           Drop columns whose value for the canonical revenue row is
           NaN, then take the most recent ``periods`` columns.
        4. Project each surviving column into a :class:`FinancialPeriod`
           by reading the metric rows and computing 毛利 from
           ``营业总收入 - 营业成本`` plus 总资产 / 总负债 from
           ``equity / (1 - 资产负债率)``.
        5. Sort the periods ascending by ``period_end`` so the most
           recent period is at the end (Requirement 4.6).
        6. When the upstream produces fewer rows than ``periods``,
           raise :class:`DataNotFoundError` with the actual count so
           callers can adjust ``periods`` / ``report_type``
           (Requirement 4.5).
        """

        std_symbol = normalize_symbol(symbol)
        market = detect_market(std_symbol)
        if market != "a_stock":
            raise DataNotFoundError(
                f"akshare 适配器暂不支持港股/基金财务报告: {std_symbol}"
            )

        # Defensive boundary validation -- the service layer is the
        # primary validator so these branches mostly catch internal
        # callers that bypass it.
        if report_type not in _AK_FIN_REPORT_TYPES:
            raise DataSourceError(
                f"akshare error: 不支持的 report_type: {report_type!r}. "
                f"必须是 {sorted(_AK_FIN_REPORT_TYPES)} 之一"
            )
        if (
            not isinstance(periods, int)
            or isinstance(periods, bool)
            or periods < _AK_FIN_MIN_PERIODS
            or periods > _AK_FIN_MAX_PERIODS
        ):
            raise DataSourceError(
                f"akshare error: periods 必须是 [{_AK_FIN_MIN_PERIODS}, "
                f"{_AK_FIN_MAX_PERIODS}] 之间的整数, 实际收到 {periods!r}"
            )

        bare = std_symbol.split(".")[0]
        df = _call_akshare(self._ak.stock_financial_abstract, symbol=bare)
        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty or "指标" not in df.columns:
            raise DataNotFoundError(
                f"未找到 {periods} 期 {report_type} 财务报告: {std_symbol} "
                f"上游返回空数据, 请稍后重试或更换 report_type"
            )

        # Build a metric -> dict[date_str -> raw_value] lookup for O(1)
        # access while iterating over the date columns. ``选项`` and
        # ``指标`` are non-data columns; everything else is a
        # ``YYYYMMDD`` reporting date.
        date_columns: list[str] = [
            c
            for c in df.columns.tolist()
            if c not in ("选项", "指标") and self._is_report_date_column(c)
        ]
        if not date_columns:
            raise DataNotFoundError(
                f"未找到 {periods} 期 {report_type} 财务报告: {std_symbol} "
                f"上游未返回报告期列, 请稍后重试或更换 report_type"
            )

        # Filter by report type. Annual reports end on 1231 only;
        # quarterly reports include 0331 / 0630 / 0930 / 1231.
        if report_type == "annual":
            date_columns = [c for c in date_columns if c.endswith("1231")]
        # else: keep every quarterly column

        if not date_columns:
            raise DataNotFoundError(
                f"未找到 {periods} 期 {report_type} 财务报告: {std_symbol} "
                f"上游缺少匹配的报告期, 请减小 periods 或更换 report_type"
            )

        metric_lookup = self._build_metric_lookup(df)

        # Drop columns whose canonical revenue row is NaN -- those
        # represent reporting dates the upstream has not populated yet
        # (e.g. future-dated quarters that appear as headers).
        revenue_row = metric_lookup.get(_AK_FIN_METRIC_ROWS["revenue"], {})
        valid_columns = [
            c for c in date_columns if _safe_optional_float(revenue_row.get(c)) is not None
        ]
        if not valid_columns:
            raise DataNotFoundError(
                f"未找到 {periods} 期 {report_type} 财务报告: {std_symbol} "
                f"上游未返回有效数据, 请减小 periods 或更换 report_type"
            )

        # Sort newest-first; akshare appears to return descending date
        # order already but explicit sort is cheaper than trusting that.
        valid_columns.sort(reverse=True)

        if len(valid_columns) < periods:
            raise DataNotFoundError(
                f"未找到 {periods} 期 {report_type} 财务报告: {std_symbol} "
                f"上游仅有 {len(valid_columns)} 期, "
                f"请减小 periods 或更换 report_type"
            )

        selected = valid_columns[:periods]

        period_dtos: list[FinancialPeriod] = []
        for date_col in selected:
            period_dto = self._project_financial_period(date_col, metric_lookup)
            if period_dto is not None:
                period_dtos.append(period_dto)

        if len(period_dtos) < periods:
            raise DataNotFoundError(
                f"未找到 {periods} 期 {report_type} 财务报告: {std_symbol} "
                f"上游仅有 {len(period_dtos)} 期可用, "
                f"请减小 periods 或更换 report_type"
            )

        # Sort ascending by period_end so the most recent period is at
        # the tail (Requirement 4.6 -- 升序或降序之一稳定排列, design
        # Markdown convention reads chronologically).
        period_dtos.sort(key=lambda p: p.period_end)

        return FinancialReport(
            symbol=std_symbol,
            report_type=cast(Any, report_type),
            periods=period_dtos,
        )

    @staticmethod
    def _is_report_date_column(column: str) -> bool:
        """Return ``True`` if ``column`` looks like ``YYYYMMDD``."""

        return (
            len(column) == 8
            and column.isdigit()
            and column[4:6] in {"03", "06", "09", "12"}
            and column[6:8] in {"31", "30"}
        )

    @staticmethod
    def _build_metric_lookup(
        df: pd.DataFrame,
    ) -> dict[str, dict[str, Any]]:
        """Build a metric_label → {date_col: raw_value} lookup.

        When the abstract endpoint duplicates a metric across multiple
        ``选项`` groupings, the first occurrence wins -- the values
        are identical (verified empirically) so the choice is purely
        about determinism.

        ``DataFrame.itertuples`` is intentionally avoided here because
        it mangles numeric column names like ``20251231`` into
        positional placeholders (``_2``, ``_3``, ...). Instead we
        iterate by index label and project each row through
        :py:meth:`pandas.Series.to_dict`, which preserves the original
        column names verbatim.
        """

        out: dict[str, dict[str, Any]] = {}
        if "指标" not in df.columns:
            return out
        for idx in df.index:
            row_series = df.loc[idx]
            if isinstance(row_series, pd.DataFrame):
                # Defensive guard: a duplicate index would make
                # ``df.loc[idx]`` return a DataFrame; pick the first
                # row in that case.
                row_series = row_series.iloc[0]
            label = str(row_series.get("指标", "")).strip()
            if not label or label in out:
                continue
            out[label] = cast("dict[str, Any]", row_series.to_dict())
        return out

    @staticmethod
    def _project_financial_period(
        date_col: str,
        metric_lookup: dict[str, dict[str, Any]],
    ) -> FinancialPeriod | None:
        """Project one date column into a :class:`FinancialPeriod`.

        Returns ``None`` when essential fields are missing so the
        caller can drop the period instead of failing the whole fetch.
        """

        def get_metric(label: str) -> float | None:
            row = metric_lookup.get(label)
            if row is None:
                return None
            return _safe_optional_float(row.get(date_col))

        revenue = get_metric(_AK_FIN_METRIC_ROWS["revenue"])
        net_profit = get_metric(_AK_FIN_METRIC_ROWS["net_profit"])
        net_profit_excl = get_metric(_AK_FIN_METRIC_ROWS["net_profit_excl_nrgl"])
        operating_cash_flow = get_metric(_AK_FIN_METRIC_ROWS["operating_cash_flow"])
        equity = get_metric(_AK_FIN_METRIC_ROWS["equity"])
        cost = get_metric(_AK_FIN_COST_ROW)
        debt_ratio_pct = get_metric(_AK_FIN_DEBT_RATIO_ROW)

        # Revenue is the canonical "is this period populated" probe;
        # if it is missing we cannot say the column represents a real
        # period and must drop it.
        if revenue is None:
            return None

        # Parse the YYYYMMDD column header into a date. Failure here
        # indicates a column we should not have selected; bail out.
        try:
            period_end = date(
                int(date_col[0:4]), int(date_col[4:6]), int(date_col[6:8])
            )
        except (TypeError, ValueError):
            return None

        # 毛利 = 营业总收入 - 营业成本; default cost to 0 when missing
        # so the field is still numeric (DTO requires float).
        gross_profit = revenue - (cost if cost is not None else 0.0)

        # Recover total_assets / total_liabilities from equity and
        # 资产负债率 (percent). Guard against the edge case where the
        # debt ratio is 100% (would zero-divide) or missing.
        total_assets: float
        total_liabilities: float
        if equity is None or debt_ratio_pct is None:
            total_assets = 0.0
            total_liabilities = 0.0
            equity_v = equity if equity is not None else 0.0
        else:
            equity_v = equity
            ratio = debt_ratio_pct / 100.0
            if ratio >= 1.0 or ratio < 0.0:
                # Degenerate ratio -- fall back to zeros rather than
                # producing a misleading negative or infinite value.
                total_assets = 0.0
                total_liabilities = 0.0
            else:
                total_assets = equity_v / (1.0 - ratio)
                total_liabilities = total_assets - equity_v

        try:
            return FinancialPeriod(
                period_end=period_end,
                revenue=float(revenue),
                net_profit=float(net_profit if net_profit is not None else 0.0),
                net_profit_excl_nrgl=float(
                    net_profit_excl if net_profit_excl is not None else 0.0
                ),
                gross_profit=float(gross_profit),
                operating_cash_flow=float(
                    operating_cash_flow if operating_cash_flow is not None else 0.0
                ),
                total_assets=float(total_assets),
                total_liabilities=float(total_liabilities),
                equity=float(equity_v),
            )
        except ValueError:
            return None

    def money_flow(
        self,
        symbol: str | None,
        flow_type: str,
        top_n: int,
    ) -> MoneyFlow:
        """Fetch 资金流向 data (north / main / dragon_tiger).

        Pipeline:

        1. Validate ``flow_type`` and ``top_n`` defensively at the
           adapter boundary (Requirements 5.4 / 5.5). The service
           layer is the primary validator but the adapter must still
           raise a unified error if a caller bypasses it.
        2. Dispatch on ``flow_type``:

           * ``north`` -- :func:`ak.stock_hsgt_hist_em` returns the
             daily north-bound aggregate. ``symbol`` is ignored;
             rows ship the most recent ``top_n`` trading days with
             ``当日成交净买额`` (净流入) as the headline number.
           * ``main`` -- requires ``symbol``; calls
             :func:`ak.stock_individual_fund_flow` for the latest
             ``top_n`` trading days of 主力 / 超大单 / 大单 / 中单 /
             小单 净流入.
           * ``dragon_tiger`` -- :func:`ak.stock_lhb_detail_em` over
             the past 14 calendar days; rows are filtered to
             ``symbol`` when provided, otherwise the full board is
             returned. Each row reports 净买额 / 买入额 / 卖出额 /
             换手率 / 上榜原因.

        3. Truncate rows to the most recent ``top_n`` (north / main)
           or to the first ``top_n`` rows from the upstream order
           (dragon_tiger -- the upstream sorts by 上榜日 desc).
        4. Wrap the result in :class:`MoneyFlow` with
           ``snapshot_at = datetime.now(UTC)`` (Requirement 5.6).
        """

        if flow_type not in _AK_FLOW_TYPES:
            raise DataSourceError(
                f"akshare error: 不支持的 flow_type: {flow_type!r}. "
                f"必须是 {sorted(_AK_FLOW_TYPES)} 之一"
            )
        if (
            not isinstance(top_n, int)
            or isinstance(top_n, bool)
            or top_n < _AK_FLOW_MIN_TOP_N
            or top_n > _AK_FLOW_MAX_TOP_N
        ):
            raise DataSourceError(
                f"akshare error: top_n 必须是 [{_AK_FLOW_MIN_TOP_N}, "
                f"{_AK_FLOW_MAX_TOP_N}] 之间的整数, 实际收到 {top_n!r}"
            )

        if flow_type == "north":
            rows = self._fetch_money_flow_north(top_n)
        elif flow_type == "main":
            rows = self._fetch_money_flow_main(symbol, top_n)
        else:
            rows = self._fetch_money_flow_dragon_tiger(symbol, top_n)

        return MoneyFlow(
            flow_type=cast(Any, flow_type),
            rows=rows,
            snapshot_at=datetime.now(UTC),
        )

    def _fetch_money_flow_north(self, top_n: int) -> list[dict[str, object]]:
        """Pull the most recent ``top_n`` north-bound trading days."""

        df = _call_akshare(self._ak.stock_hsgt_hist_em, symbol="北向资金")
        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty or "日期" not in df.columns:
            raise DataNotFoundError("未找到北向资金流向数据")

        net_col = self._pick_column(
            df, ("当日成交净买额", "当日资金流入", "成交净买额")
        )
        buy_col = self._pick_column(df, ("买入成交额",))
        sell_col = self._pick_column(df, ("卖出成交额",))
        hold_col = self._pick_column(df, ("持股市值",))
        cumulative_col = self._pick_column(df, ("历史累计净买额",))

        # Drop rows whose net inflow is NaN (future-dated placeholders
        # that the upstream sometimes ships ahead of the trading day).
        if net_col is not None:
            df = df.loc[df[net_col].notna()].copy()

        # Sort newest-first and take the most recent ``top_n`` rows.
        df = df.sort_values("日期", ascending=False).head(top_n)
        if df.empty:
            raise DataNotFoundError("未找到北向资金流向数据")

        rows: list[dict[str, object]] = []
        for record in df.to_dict(orient="records"):
            row: dict[str, object] = {
                "date": self._coerce_iso_date(record.get("日期")),
            }
            if net_col is not None:
                row["净流入"] = _safe_optional_float(record.get(net_col))
            if buy_col is not None:
                row["买入金额"] = _safe_optional_float(record.get(buy_col))
            if sell_col is not None:
                row["卖出金额"] = _safe_optional_float(record.get(sell_col))
            if hold_col is not None:
                row["持股市值"] = _safe_optional_float(record.get(hold_col))
            if cumulative_col is not None:
                row["累计净流入"] = _safe_optional_float(
                    record.get(cumulative_col)
                )
            rows.append(row)
        return rows

    def _fetch_money_flow_main(
        self,
        symbol: str | None,
        top_n: int,
    ) -> list[dict[str, object]]:
        """Pull the most recent ``top_n`` 主力资金 days for ``symbol``."""

        if symbol is None or not str(symbol).strip():
            raise DataNotFoundError("main 资金流需要 symbol 参数")

        std_symbol = normalize_symbol(symbol)
        market = detect_market(std_symbol)
        if market != "a_stock":
            raise DataNotFoundError(
                f"akshare 适配器暂不支持港股/基金主力资金流向: {std_symbol}"
            )

        bare = std_symbol.split(".")[0]
        suffix = std_symbol.split(".")[1] if "." in std_symbol else ""
        ak_market = "sh" if suffix == "SH" else "sz" if suffix == "SZ" else "bj"

        df = _call_akshare(
            self._ak.stock_individual_fund_flow,
            stock=bare,
            market=ak_market,
        )
        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty or "日期" not in df.columns:
            raise DataNotFoundError(
                f"未找到主力资金流向数据: {std_symbol}"
            )

        # Sort newest-first; akshare returns ascending order.
        df = df.sort_values("日期", ascending=False).head(top_n)

        rows: list[dict[str, object]] = []
        for record in df.to_dict(orient="records"):
            rows.append(
                {
                    "date": self._coerce_iso_date(record.get("日期")),
                    "收盘价": _safe_optional_float(record.get("收盘价")),
                    "涨跌幅": _safe_optional_float(record.get("涨跌幅")),
                    "主力净流入": _safe_optional_float(
                        record.get("主力净流入-净额")
                    ),
                    "超大单净流入": _safe_optional_float(
                        record.get("超大单净流入-净额")
                    ),
                    "大单净流入": _safe_optional_float(
                        record.get("大单净流入-净额")
                    ),
                    "中单净流入": _safe_optional_float(
                        record.get("中单净流入-净额")
                    ),
                    "小单净流入": _safe_optional_float(
                        record.get("小单净流入-净额")
                    ),
                }
            )
        return rows

    def _fetch_money_flow_dragon_tiger(
        self,
        symbol: str | None,
        top_n: int,
    ) -> list[dict[str, object]]:
        """Pull recent 龙虎榜 detail rows; filter by ``symbol`` if provided."""

        end_dt = datetime.now(UTC).date()
        start_dt = end_dt - timedelta(days=_AK_LHB_LOOKBACK_DAYS)
        df = _call_akshare(
            self._ak.stock_lhb_detail_em,
            start_date=start_dt.strftime("%Y%m%d"),
            end_date=end_dt.strftime("%Y%m%d"),
        )
        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty or "代码" not in df.columns:
            raise DataNotFoundError("未找到龙虎榜数据")

        if symbol is not None and str(symbol).strip():
            std_symbol = normalize_symbol(symbol)
            bare = std_symbol.split(".")[0]
            df = df.loc[df["代码"].astype(str) == bare].copy()
            if df.empty:
                raise DataNotFoundError(
                    f"未找到龙虎榜数据: {std_symbol}"
                )

        if "上榜日" in df.columns:
            df = df.sort_values("上榜日", ascending=False)
        df = df.head(top_n)

        rows: list[dict[str, object]] = []
        for record in df.to_dict(orient="records"):
            rows.append(
                {
                    "代码": str(record.get("代码", "")).strip(),
                    "名称": str(record.get("名称", "")).strip(),
                    "上榜日": self._coerce_iso_date(record.get("上榜日")),
                    "收盘价": _safe_optional_float(record.get("收盘价")),
                    "涨跌幅": _safe_optional_float(record.get("涨跌幅")),
                    "净买额": _safe_optional_float(record.get("龙虎榜净买额")),
                    "买入金额": _safe_optional_float(
                        record.get("龙虎榜买入额")
                    ),
                    "卖出金额": _safe_optional_float(
                        record.get("龙虎榜卖出额")
                    ),
                    "换手率": _safe_optional_float(record.get("换手率")),
                    "上榜原因": str(record.get("上榜原因", "")).strip(),
                }
            )
        return rows

    @staticmethod
    def _coerce_iso_date(value: Any) -> str:
        """Coerce a dataframe cell into an ``YYYY-MM-DD`` ISO string.

        Falls back to the value's :func:`str` representation when the
        cell is not parseable as a date so the downstream renderer
        always sees a non-empty label.
        """

        if value is None:
            return ""
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        try:
            parsed = pd.to_datetime(value, errors="coerce")
        except (TypeError, ValueError):
            return str(value)
        if pd.isna(parsed):
            return str(value)
        result: str = cast("pd.Timestamp", parsed).date().isoformat()
        return result

    def industry_peers(
        self,
        symbol: str,
        metrics: list[str],
        top_n: int,
    ) -> PeerTable:
        """Fetch 同行业可比公司 + per-row metrics for ``symbol``.

        Pipeline:

        1. Normalize ``symbol`` and reject HK / fund codes (A 股 only
           in v1; the upstream industry-board endpoints are not
           applicable to those markets).
        2. Validate ``metrics`` (subset of ``{pe, pb, roe,
           revenue_growth}``) and ``top_n`` (``[1, 50]``) defensively
           at the adapter boundary -- the service layer is the
           primary validator but the adapter must still surface a
           unified error if a caller bypasses it.
        3. Look up the symbol's industry via
           :func:`ak.stock_individual_info_em`, picking the row whose
           ``item == "行业"``. Empty / missing → :class:`DataNotFoundError`.
        4. Pull the industry's constituents via
           :func:`ak.stock_board_industry_cons_em`, project each row
           into the requested metric set, sort descending by
           ``成交额`` (the only liquidity proxy on that endpoint) and
           truncate to ``top_n`` rows.

        v1 limitation: ``roe`` / ``revenue_growth`` are *not* available
        from the industry-constituents endpoint, and per-symbol
        financial-indicator calls for every constituent would balloon
        the request budget. The cells are surfaced as ``None`` so the
        formatter renders ``"-"``; the service layer reflects the same
        behaviour in its row-level percentile annotation.
        """

        std_symbol = normalize_symbol(symbol)
        market = detect_market(std_symbol)
        if market != "a_stock":
            raise DataNotFoundError(
                f"akshare 适配器暂不支持港股/基金行业对比: {std_symbol}"
            )

        # Defensive boundary validation -- service is the primary
        # validator, but the adapter must still raise a unified error
        # if a caller bypasses it (Requirement 13.1).
        if not isinstance(metrics, list) or not metrics:
            raise DataSourceError(
                "akshare error: metrics 必须是非空列表, "
                f"实际收到 {metrics!r}"
            )
        invalid_metrics = sorted(set(metrics) - _AK_PEER_METRICS)
        if invalid_metrics:
            raise DataSourceError(
                f"akshare error: 不支持的 metrics: {invalid_metrics}. "
                f"必须是 {sorted(_AK_PEER_METRICS)} 的子集"
            )
        if (
            not isinstance(top_n, int)
            or isinstance(top_n, bool)
            or top_n < _AK_PEER_MIN_TOP_N
            or top_n > _AK_PEER_MAX_TOP_N
        ):
            raise DataSourceError(
                f"akshare error: top_n 必须是 [{_AK_PEER_MIN_TOP_N}, "
                f"{_AK_PEER_MAX_TOP_N}] 之间的整数, 实际收到 {top_n!r}"
            )

        industry_name = self._fetch_industry_name(std_symbol)
        constituents = self._fetch_industry_constituents(industry_name)

        # Preserve the caller-supplied ``metrics`` order so the
        # service / tool layers can rely on it for column rendering.
        # De-dup defensively -- a caller that passed ``["pe", "pe"]``
        # should still get a single ``pe`` column.
        seen_metrics: set[str] = set()
        ordered_metrics: list[str] = []
        for m in metrics:
            if m in seen_metrics:
                continue
            seen_metrics.add(m)
            ordered_metrics.append(m)

        rows = self._project_peer_rows(constituents, ordered_metrics, top_n)
        if not rows:
            raise DataNotFoundError(
                f"未找到行业 '{industry_name}' 的可比公司"
            )

        return PeerTable(
            base_symbol=std_symbol,
            industry=industry_name,
            metrics=ordered_metrics,
            rows=rows,
        )

    def _fetch_industry_name(self, std_symbol: str) -> str:
        """Resolve the symbol's industry via ``stock_individual_info_em``.

        The endpoint returns a 2-column dataframe (``item`` / ``value``)
        with one row per metadata field; the ``行业`` row holds the
        industry classification. Empty / missing ⇒ DataNotFoundError.
        """

        bare = std_symbol.split(".")[0]
        df = _call_akshare(self._ak.stock_individual_info_em, symbol=bare)
        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty or "item" not in df.columns or "value" not in df.columns:
            raise DataNotFoundError(
                f"未找到 {std_symbol} 的行业分类信息"
            )

        match = df.loc[df["item"].astype(str) == _AK_INDUSTRY_INFO_ROW]
        if match.empty:
            raise DataNotFoundError(
                f"未找到 {std_symbol} 的行业分类信息"
            )

        industry = str(match.iloc[0]["value"]).strip()
        if not industry or industry == "-":
            raise DataNotFoundError(
                f"未找到 {std_symbol} 的行业分类信息"
            )
        return industry

    def _fetch_industry_constituents(self, industry_name: str) -> pd.DataFrame:
        """Pull the constituents dataframe for ``industry_name``."""

        df = _call_akshare(
            self._ak.stock_board_industry_cons_em, symbol=industry_name
        )
        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty or "代码" not in df.columns or "名称" not in df.columns:
            raise DataNotFoundError(
                f"未找到行业 '{industry_name}' 的成份股数据"
            )
        return df

    def _project_peer_rows(
        self,
        constituents: pd.DataFrame,
        metrics: list[str],
        top_n: int,
    ) -> list[dict[str, object]]:
        """Project the constituents frame into ``PeerTable.rows`` dicts.

        Sort order: descending by ``成交额`` (the only liquidity proxy
        on this endpoint). Ties keep the upstream order, which is
        already the eastmoney default ranking.
        """

        df = constituents.copy()
        if _AK_PEER_RANK_COL in df.columns:
            df[_AK_PEER_RANK_COL] = pd.to_numeric(
                df[_AK_PEER_RANK_COL], errors="coerce"
            )
            df = df.sort_values(
                _AK_PEER_RANK_COL, ascending=False, kind="stable"
            )
        df = df.head(top_n)

        rows: list[dict[str, object]] = []
        for record in df.to_dict(orient="records"):
            row: dict[str, object] = {
                "代码": str(record.get("代码", "")).strip(),
                "名称": str(record.get("名称", "")).strip(),
            }
            for metric in metrics:
                col = _AK_PEER_VALUATION_COLS.get(metric)
                if col is None or col not in record:
                    # roe / revenue_growth are not on this endpoint
                    # in v1; surface as None so the formatter renders
                    # "-" rather than dropping the column.
                    row[metric] = None
                    continue
                row[metric] = _safe_optional_float(record.get(col))
            rows.append(row)
        return rows

    def fund_info(self, fund_code: str) -> FundInfo:
        """Fetch metadata, returns and holdings for a public fund.

        Pipeline:

        1. Validate ``fund_code`` is a 6-digit string -- non-conforming
           inputs raise :class:`SymbolError` (Requirement 7.3). Funds
           use bare 6-digit codes with no exchange suffix; we do
           **not** route through :func:`normalize_symbol` because the
           normalizer rejects unknown 6-digit codes (it cannot tell a
           valid fund code from a typo without a fund index).
        2. Pull metadata via :func:`ak.fund_individual_basic_info_xq`
           (one-row DataFrame keyed by ``item``/``value``) for code,
           name, manager, inception date and AUM. Missing rows surface
           as :class:`DataNotFoundError`.
        3. Pull NAV history via :func:`ak.fund_open_fund_info_em`
           (``indicator="单位净值走势"``) and derive
           ``return_1m`` / ``return_3m`` / ``return_6m`` /
           ``return_12m`` plus ``max_drawdown`` from the trailing
           series. NAV-history endpoints occasionally fail for very
           young funds; in that case the tracked windows fall back to
           ``0.0``.
        4. Pull top holdings via :func:`ak.fund_portfolio_hold_em`
           (most recent quarter, top 10 by weight). Missing endpoint
           or empty frame leaves the list empty.
        5. Industry distribution and ``sharpe`` / ``rank_in_category``
           are not exposed by a stable akshare endpoint at the moment;
           v1 leaves them at ``None`` / empty / "-" placeholders so
           downstream renderers display a "-" cell (Requirement 7.5).

        ``DataSourceError`` from :func:`_call_akshare` is unwrapped
        into :class:`DataNotFoundError` when the fund code exists in
        akshare's universe but the metadata payload is empty.
        """

        # 1) Validate (Requirement 7.3).
        if not isinstance(fund_code, str):
            raise SymbolError(
                f"非法基金代码: {fund_code!r}, 必须是 6 位数字字符串"
            )
        code = fund_code.strip()
        if len(code) != _FUND_CODE_LEN or not code.isdigit():
            raise SymbolError(
                f"非法基金代码: {fund_code!r}, 必须是 6 位数字"
            )

        # 2) Metadata.
        meta = self._fetch_fund_meta(code)

        # 3) NAV-derived returns + drawdown.
        returns = self._fetch_fund_returns(code)

        # 4) Top holdings.
        holdings = self._fetch_fund_top_holdings(code)

        return FundInfo(
            code=code,
            name=meta["name"],
            manager=meta["manager"],
            inception_date=meta["inception_date"],
            aum=meta["aum"],
            return_1m=returns["return_1m"],
            return_3m=returns["return_3m"],
            return_6m=returns["return_6m"],
            return_12m=returns["return_12m"],
            max_drawdown=returns["max_drawdown"],
            sharpe=None,
            rank_in_category="-",
            top_holdings=holdings,
            industry_distribution=[],
        )

    def _fetch_fund_meta(self, code: str) -> dict[str, Any]:
        """Pull basic-info DataFrame and project to the metadata dict.

        :func:`ak.fund_individual_basic_info_xq` returns a 2-column
        ``item`` / ``value`` frame. Missing endpoint or empty frame
        surfaces as :class:`DataNotFoundError`.
        """

        df = _call_akshare(
            self._ak.fund_individual_basic_info_xq, symbol=code
        )
        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty or "item" not in df.columns or "value" not in df.columns:
            raise DataNotFoundError(
                f"未找到基金基础信息: {code}"
            )

        # Build item -> raw value lookup.
        meta_lookup: dict[str, Any] = {}
        for record in df.to_dict(orient="records"):
            key = str(record.get("item", "")).strip()
            if key:
                meta_lookup[key] = record.get("value")

        # Name candidates: 基金名称 / 简称 / 全称.
        name = ""
        for key in ("基金简称", "基金名称", "基金全称"):
            value = meta_lookup.get(key)
            if value is not None and str(value).strip():
                name = str(value).strip()
                break
        if not name:
            raise DataNotFoundError(
                f"未找到基金基础信息: {code}"
            )

        manager_raw = meta_lookup.get("基金经理") or meta_lookup.get("现任基金经理")
        manager = str(manager_raw).strip() if manager_raw is not None else ""

        # Inception date -- multiple possible labels across akshare
        # versions. Fall back to the unix epoch when missing so the
        # DTO still validates; the renderer surfaces the date verbatim.
        inception_dt: date | None = None
        for key in ("成立日期", "成立时间"):
            value = meta_lookup.get(key)
            parsed = self._coerce_bar_date(value)
            if parsed is not None:
                inception_dt = parsed
                break
        if inception_dt is None:
            inception_dt = date(1970, 1, 1)

        # AUM -- "资产规模" or "基金规模" rows. Often denominated as a
        # string like "12.34亿" -- attempt a best-effort numeric parse,
        # otherwise default to 0.0 (DTO requires NonNegativeFloat).
        aum_value = 0.0
        for key in ("资产规模", "基金规模", "资产净值"):
            raw = meta_lookup.get(key)
            parsed_aum = self._coerce_amount(raw)
            if parsed_aum is not None:
                aum_value = parsed_aum
                break

        return {
            "name": name,
            "manager": manager,
            "inception_date": inception_dt,
            "aum": aum_value,
        }

    @staticmethod
    def _coerce_amount(value: Any) -> float | None:
        """Best-effort parse of a CN-formatted amount string into 元.

        Accepts forms like ``"12.34亿"`` / ``"5,678万"`` / a plain
        number / ``None``. Returns ``None`` when the input cannot be
        coerced so the caller can decide on a default.
        """

        if value is None:
            return None
        if isinstance(value, (int, float)):
            result = float(value)
            if result != result or result < 0:
                return None
            return result
        if not isinstance(value, str):
            return None

        text = value.strip().replace(",", "").replace(" ", "")
        if not text or text in {"-", "--"}:
            return None

        multiplier = 1.0
        if text.endswith("亿"):
            multiplier = _YI
            text = text[:-1]
        elif text.endswith("万"):
            multiplier = _WAN
            text = text[:-1]
        elif text.endswith("元"):
            text = text[:-1]

        try:
            number = float(text)
        except (TypeError, ValueError):
            return None
        if number != number or number < 0:
            return None
        return number * multiplier

    def _fetch_fund_returns(self, code: str) -> dict[str, float]:
        """Pull NAV history and derive trailing-window returns.

        Failures (missing endpoint, empty frame, parse errors) leave
        every metric at ``0.0`` so the DTO still validates -- the
        renderer surfaces them via :func:`format_percent` as
        ``"0.00%"``. Per task spec: missing fields render as "-",
        but the DTO's ``return_*`` fields are required ``float``s so
        we honor that contract; only the optional ``sharpe`` field
        receives the ``None`` → "-" treatment.
        """

        defaults: dict[str, float] = {
            "return_1m": 0.0,
            "return_3m": 0.0,
            "return_6m": 0.0,
            "return_12m": 0.0,
            "max_drawdown": 0.0,
        }

        try:
            df = _call_akshare(
                self._ak.fund_open_fund_info_em,
                symbol=code,
                indicator="单位净值走势",
            )
        except DataSourceError:
            return defaults

        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty:
            return defaults

        nav_col = self._pick_column(df, ("单位净值", "nav"))
        date_col = self._pick_column(df, ("净值日期", "日期", "date"))
        if nav_col is None or date_col is None:
            return defaults

        # Sort ascending so the tail is the most recent value.
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col, ascending=True)

        navs: list[float] = []
        for raw in df[nav_col].tolist():
            parsed = _safe_optional_float(raw)
            if parsed is None or parsed <= 0:
                continue
            navs.append(parsed)

        if len(navs) < 2:
            return defaults

        latest = navs[-1]

        out = dict(defaults)
        for metric, lookback in _AK_FUND_RETURN_HORIZONS.items():
            if len(navs) <= lookback:
                continue
            base = navs[-lookback - 1]
            if base <= 0:
                continue
            out[metric] = (latest / base - 1.0) * 100.0

        # Max drawdown over the full available window, expressed as a
        # negative percent (e.g. -12.34 means a 12.34% drawdown).
        peak = navs[0]
        worst_drawdown = 0.0
        for nav in navs:
            if nav > peak:
                peak = nav
            if peak > 0:
                drawdown = (nav - peak) / peak * 100.0
                if drawdown < worst_drawdown:
                    worst_drawdown = drawdown
        out["max_drawdown"] = worst_drawdown

        return out

    def _fetch_fund_top_holdings(
        self,
        code: str,
    ) -> list[dict[str, object]]:
        """Pull top-10 holdings from the most recent reporting period.

        Returns an empty list on any failure -- holdings are decorative
        in v1 and a missing payload should not fail the whole call.
        """

        try:
            year = datetime.now(UTC).year
            df = _call_akshare(
                self._ak.fund_portfolio_hold_em,
                symbol=code,
                date=str(year),
            )
        except (DataSourceError, NetworkError):
            return []
        except DataNotFoundError:
            return []

        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty:
            return []

        code_col = self._pick_column(df, ("股票代码", "代码"))
        name_col = self._pick_column(df, ("股票名称", "名称"))
        weight_col = self._pick_column(
            df, ("占净值比例", "持仓占比", "占净值比例(%)")
        )
        if code_col is None or name_col is None or weight_col is None:
            return []

        # Sort by weight desc, taking the latest reporting quarter.
        df = df.copy()
        df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
        df = df.dropna(subset=[weight_col])

        # Some payloads ship multiple quarters concatenated; keep the
        # most recent quarter only (largest 季度 / 报告期 column when
        # present).
        period_col = self._pick_column(df, ("季度", "报告期", "截止日期"))
        if period_col is not None:
            try:
                latest_period = df[period_col].astype(str).max()
                df = df.loc[df[period_col].astype(str) == latest_period]
            except (TypeError, ValueError):
                pass

        df = df.sort_values(weight_col, ascending=False, kind="stable")
        df = df.head(_AK_FUND_MAX_HOLDINGS)

        rows: list[dict[str, object]] = []
        for record in df.to_dict(orient="records"):
            symbol_raw = str(record.get(code_col, "")).strip()
            name = str(record.get(name_col, "")).strip()
            weight = _safe_optional_float(record.get(weight_col))
            if not symbol_raw or not name:
                continue
            rows.append(
                {
                    "symbol": symbol_raw,
                    "name": name,
                    "weight": weight,
                }
            )
        return rows

    def market_overview(self) -> MarketOverview:
        """Fetch a snapshot of the overall A 股 market.

        Pipeline:

        1. Pull the index spot frame
           (:func:`ak.stock_zh_index_spot_em` with ``symbol="沪深重要指数"``)
           and project the three core indices (上证 / 深证成指 / 创业板)
           into ``[{name, code, last, change_pct}, ...]`` rows.
        2. Pull the A 股 spot frame
           (:func:`ak.stock_zh_a_spot_em`) and derive 涨跌家数
           (``advance`` / ``decline`` / ``flat``) plus 涨跌停 counts
           (``limit_up`` / ``limit_down``) from the ``涨跌幅`` column.
           The limit definition uses a conservative ±9.9% cutoff
           which captures both 主板 (±10%) and the rounding noise
           in akshare's ``涨跌幅`` cell.
        3. Pull 北向资金 net inflow from the most recent row of
           :func:`ak.stock_hsgt_hist_em` (``symbol="北向资金"``); use
           the ``当日成交净买额`` column with a ``成交净买额`` /
           ``当日资金流入`` fallback for upstream column drift.
        4. Pull 行业资金流向排行 from
           :func:`ak.stock_sector_fund_flow_rank` (``indicator="今日"``)
           and take the top :data:`_AK_OVERVIEW_TOP_INDUSTRIES` rows
           by 主力净流入 (the column name varies between akshare
           builds, so the helper picks the first available
           candidate).
        5. Compose ``heat_score`` from the breadth ratio plus
           directional bumps for north flow and 涨跌停 imbalance,
           then clamp to ``[0, 100]``.
        6. Use the index frame's 时间 column (or ``datetime.now(UTC)``
           when absent) as ``snapshot_at``.

        Any optional sub-pull (北向 / 行业排行) that fails with a
        :class:`DataSourceError` produces a sensible default rather
        than aborting the whole overview -- the indices + breadth
        backbone is enough to answer "今天市场怎么样". Transient
        failures (:class:`NetworkError` / :class:`RateLimitError`)
        propagate so the service-level fallback can take over.
        """

        spot_df = _call_akshare(self._ak.stock_zh_a_spot_em)
        spot_df = (
            cast("pd.DataFrame", spot_df) if spot_df is not None else pd.DataFrame()
        )

        indices, snapshot_at = self._fetch_overview_indices()
        advance_decline = self._compute_advance_decline(spot_df)
        limit_stats = self._compute_limit_stats(spot_df)

        try:
            north_net_inflow = self._fetch_overview_north_flow()
        except DataSourceError:
            north_net_inflow = 0.0

        try:
            top_industries = self._fetch_overview_top_industries()
        except DataSourceError:
            top_industries = []

        heat_score = self._compute_heat_score(
            advance_decline=advance_decline,
            limit_stats=limit_stats,
            north_net_inflow=north_net_inflow,
        )

        return MarketOverview(
            indices=indices,
            advance_decline=advance_decline,
            limit_stats=limit_stats,
            north_net_inflow=north_net_inflow,
            top_inflow_industries=top_industries,
            heat_score=heat_score,
            snapshot_at=snapshot_at,
        )

    def _fetch_overview_indices(
        self,
    ) -> tuple[list[dict[str, object]], datetime]:
        """Pull 上证 / 深证 / 创业板 spot rows + ``snapshot_at``.

        Returns a tuple of ``(rows, snapshot_at)``. When the upstream
        endpoint is unavailable we fall back to empty rows + the
        current UTC time so the rest of the pipeline can still run.
        """

        try:
            df = _call_akshare(
                self._ak.stock_zh_index_spot_em, symbol="沪深重要指数"
            )
        except DataSourceError:
            return [], datetime.now(UTC)

        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty:
            return [], datetime.now(UTC)

        code_col = self._pick_column(df, ("代码", "code", "symbol"))
        name_col = self._pick_column(df, ("名称", "name"))
        last_col = self._pick_column(df, ("最新价", "现价", "last", "price"))
        change_col = self._pick_column(df, ("涨跌幅", "change_pct"))

        rows: list[dict[str, object]] = []
        if code_col is not None:
            for target_code, fallback_name in _AK_OVERVIEW_INDICES:
                code_series = df[code_col].astype(str)
                # akshare may ship the bare code (``000001``) or a
                # prefixed variant (``sh000001``); match by suffix so
                # both cases hit.
                mask = code_series.str.endswith(target_code)
                match = df.loc[mask]
                if match.empty:
                    continue
                record = match.iloc[0].to_dict()
                last = (
                    _safe_optional_float(record.get(last_col))
                    if last_col is not None
                    else None
                )
                change_pct = (
                    _safe_optional_float(record.get(change_col))
                    if change_col is not None
                    else None
                )
                name_value: str = fallback_name
                if name_col is not None:
                    raw = str(record.get(name_col, "")).strip()
                    if raw:
                        name_value = raw
                rows.append(
                    {
                        "name": name_value,
                        "code": target_code,
                        "last": last if last is not None else 0.0,
                        "change_pct": change_pct
                        if change_pct is not None
                        else 0.0,
                    }
                )

        snapshot_at = self._pick_timestamp(df)
        return rows, snapshot_at

    @staticmethod
    def _compute_advance_decline(spot_df: pd.DataFrame) -> dict[str, int]:
        """Derive ``{advance, decline, flat}`` from the A 股 spot frame."""

        if spot_df is None or spot_df.empty or "涨跌幅" not in spot_df.columns:
            return {"advance": 0, "decline": 0, "flat": 0}

        change = pd.to_numeric(spot_df["涨跌幅"], errors="coerce")
        # Drop NaN rows so they do not skew the counts; suspended
        # symbols ship NaN for ``涨跌幅`` until the next session.
        change = change.dropna()
        advance = int((change > 0).sum())
        decline = int((change < 0).sum())
        flat = int((change == 0).sum())
        return {"advance": advance, "decline": decline, "flat": flat}

    @staticmethod
    def _compute_limit_stats(spot_df: pd.DataFrame) -> dict[str, int]:
        """Approximate 涨停 / 跌停 counts from ``涨跌幅 ≥ 9.9 / ≤ -9.9``."""

        if spot_df is None or spot_df.empty or "涨跌幅" not in spot_df.columns:
            return {"limit_up": 0, "limit_down": 0}

        change = pd.to_numeric(spot_df["涨跌幅"], errors="coerce").dropna()
        limit_up = int((change >= _AK_OVERVIEW_LIMIT_UP_PCT).sum())
        limit_down = int((change <= _AK_OVERVIEW_LIMIT_DOWN_PCT).sum())
        return {"limit_up": limit_up, "limit_down": limit_down}

    def _fetch_overview_north_flow(self) -> float:
        """Pull the most recent 北向资金 净流入 (元)."""

        df = _call_akshare(self._ak.stock_hsgt_hist_em, symbol="北向资金")
        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty or "日期" not in df.columns:
            return 0.0

        net_col = self._pick_column(
            df, ("当日成交净买额", "当日资金流入", "成交净买额")
        )
        if net_col is None:
            return 0.0

        # The endpoint sometimes ships future-dated placeholder rows
        # whose net inflow is NaN; drop them before picking the last
        # populated row.
        df = df.loc[df[net_col].notna()].copy()
        if df.empty:
            return 0.0
        df = df.sort_values("日期", ascending=False)
        latest = df.iloc[0].to_dict()
        return _safe_float(latest.get(net_col), default=0.0)

    def _fetch_overview_top_industries(self) -> list[dict[str, object]]:
        """Pull the top-N 行业 by 主力净流入 (today)."""

        try:
            df = _call_akshare(
                self._ak.stock_sector_fund_flow_rank, indicator="今日"
            )
        except DataSourceError:
            # ``sector_type`` defaults vary across akshare builds;
            # surface as an empty list rather than failing the whole
            # overview.
            return []

        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty:
            return []

        name_col = self._pick_column(df, ("名称", "行业", "板块名称"))
        flow_col = self._pick_column(
            df,
            (
                "今日主力净流入-净额",
                "主力净流入-净额",
                "今日主力净流入",
                "主力净流入",
                "净额",
            ),
        )
        if name_col is None or flow_col is None:
            return []

        df = df.copy()
        df[flow_col] = pd.to_numeric(df[flow_col], errors="coerce")
        df = df.dropna(subset=[flow_col])
        df = df.sort_values(flow_col, ascending=False, kind="stable")
        df = df.head(_AK_OVERVIEW_TOP_INDUSTRIES)

        rows: list[dict[str, object]] = []
        for record in df.to_dict(orient="records"):
            name = str(record.get(name_col, "")).strip()
            if not name:
                continue
            net_inflow = _safe_optional_float(record.get(flow_col))
            rows.append(
                {
                    "name": name,
                    "net_inflow": net_inflow if net_inflow is not None else 0.0,
                }
            )
        return rows

    @staticmethod
    def _compute_heat_score(
        advance_decline: dict[str, int],
        limit_stats: dict[str, int],
        north_net_inflow: float,
    ) -> float:
        """Compose a 0..100 市场热度 score from breadth + flow signals.

        Formula (documented for callers; see module constants):

            breadth = advance / max(advance + decline, 1)
            score   = HEAT_BREADTH_WEIGHT * breadth                  # 0..100
                    + sign(north_net_inflow) * HEAT_NORTH_BUMP       # ±5
                    + sign(limit_up - limit_down) * HEAT_LIMIT_BUMP  # ±5
                      when |limit_up - limit_down| >= HEAT_LIMIT_THRESHOLD
            heat_score = clamp(score, 0, 100)
        """

        advance = advance_decline.get("advance", 0)
        decline = advance_decline.get("decline", 0)
        denom = advance + decline
        breadth = (advance / denom) if denom > 0 else 0.5
        score = breadth * _AK_HEAT_BREADTH_WEIGHT

        if north_net_inflow > 0:
            score += _AK_HEAT_NORTH_BUMP
        elif north_net_inflow < 0:
            score -= _AK_HEAT_NORTH_BUMP

        limit_diff = limit_stats.get("limit_up", 0) - limit_stats.get(
            "limit_down", 0
        )
        if abs(limit_diff) >= _AK_HEAT_LIMIT_THRESHOLD:
            score += _AK_HEAT_LIMIT_BUMP if limit_diff > 0 else -_AK_HEAT_LIMIT_BUMP

        if score < 0.0:
            return 0.0
        if score > 100.0:
            return 100.0
        return float(score)


# Silence "unused import" warnings for the ``logger`` symbol; it is
# imported so future enhancements can log without re-importing, and so
# the module signals via its imports that it participates in the shared
# logging configuration.
_ = logger


__all__ = ["AkshareAdapter"]
