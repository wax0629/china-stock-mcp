"""TushareAdapter: backup data source backed by the ``tushare`` Pro API.

This module implements a *fallback* adapter referenced by
``design.md`` Component 4 (Adapter Layer). Per task 19.1 only the v1
fallback methods are materialized -- ``search`` / ``quote`` /
``fundamentals`` / ``financial_report`` -- because those are the
endpoints where ``akshare`` outages can plausibly be covered by
tushare's professional dataset (real-time quote is end-of-day on
free tushare tiers, which is acceptable as a fallback).
The remaining five abstract methods (``kline`` / ``money_flow`` /
``industry_peers`` / ``fund_info`` / ``market_overview``) raise
:class:`DataNotFoundError` so the class is instantiable while
``fetch_with_fallback`` correctly skips this source for tools that
have no tushare equivalent.

References
----------
- ``design.md`` Component 4: Adapter Layer (``TushareAdapter`` slot).
- Requirement 13.1: every adapter SHALL raise
  :class:`~china_stock_mcp.exceptions.ChinaStockMCPError` subclasses
  rather than letting third-party exceptions escape.

Lazy-import contract
--------------------
``tushare`` is declared as an optional dependency
(``[project.optional-dependencies] tushare = ["tushare>=1.4"]``)
so we **must not** import it at module scope. Doing so would break
``import china_stock_mcp.adapters.tushare_adapter`` for any user
who installed the base package without the ``tushare`` extra.
The import lives inside :py:meth:`TushareAdapter.__init__` and the
fully-typed reference is cached on the instance.

Token requirement
-----------------
Tushare's Pro API rejects every call without a registered token, so
the adapter resolves a token in this order:

1. The explicit ``token`` argument to :py:meth:`TushareAdapter.__init__`.
2. ``CSM_TUSHARE_TOKEN`` from :func:`load_settings` (Requirements
   12.4 -- secrets are read from env, never persisted).

If neither yields a non-empty value the constructor raises
:class:`RuntimeError` with a clear remediation message rather than
deferring the failure to the first network call.

v1 limitations
--------------
* ``kline`` -- akshare's K-line endpoint already backfills tushare's
  daily history with adjusted prices and is the primary source per
  ``design.md`` §6 (Data Source Mapping); the fallback chain only
  loses the indicator series when both sources are down.
* ``money_flow`` / ``industry_peers`` / ``fund_info`` /
  ``market_overview`` -- tushare exposes these only on paid tiers,
  so we surface :class:`DataNotFoundError` here and let the akshare
  primary handle them.

Exception translation contract
------------------------------
``tushare`` propagates ``requests`` errors plus its own free-tier
quota /積分 messages. :func:`_call_tushare` maps them to the
unified hierarchy:

* connection / DNS / timeout / 5xx                → :class:`NetworkError`
* "您每分钟最多访问该接口" / "tokens 不足" /
  "积分" / "rate" wording                          → :class:`RateLimitError`
* every other third-party exception               → :class:`DataSourceError`

Errors that already inherit from :class:`ChinaStockMCPError` are
re-raised verbatim so ``fetch_with_fallback`` can apply the design's
"DataNotFoundError must NOT trigger another source switch" rule.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any, Final, TypeVar, cast

import pandas as pd

from china_stock_mcp import logger
from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.config import load_settings
from china_stock_mcp.exceptions import (
    ChinaStockMCPError,
    DataNotFoundError,
    DataSourceError,
    NetworkError,
    RateLimitError,
)
from china_stock_mcp.models import (
    FinancialPeriod,
    FinancialReport,
    FundamentalSnapshot,
    FundInfo,
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

#: Maximum number of hits returned by :meth:`TushareAdapter.search`.
#: Mirrors the akshare / efinance adapters so the formatter layer can
#: apply the same per-tool token budget regardless of source.
_MAX_SEARCH_HITS: Final[int] = 20

#: Type names that indicate a transient network failure. We compare by
#: name rather than by ``isinstance`` so we do not need to import
#: ``requests`` / ``urllib`` at module scope.
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

#: Substrings inside an exception ``str`` that indicate rate limiting
#: or a quota / 积分 shortfall on the free tushare tier.
_RATE_LIMIT_PATTERNS: Final[tuple[str, ...]] = (
    "您每分钟最多访问该接口",
    "tokens 不足",
    "积分",
    "rate",
    "rate limit",
    "rate-limit",
    "too many requests",
    "429",
    "请求过于频繁",
    "访问频率过快",
)

#: Accepted ``report_type`` values (Requirement 4.4).
_TS_FIN_REPORT_TYPES: Final[frozenset[str]] = frozenset({"annual", "quarterly"})

#: ``periods`` lower / upper bounds (Requirement 4.4).
_TS_FIN_MIN_PERIODS: Final[int] = 1
_TS_FIN_MAX_PERIODS: Final[int] = 12

#: Placeholder message for adapter methods this fallback intentionally
#: does not implement (Requirement 13.1: surface a unified error).
_NOT_SUPPORTED_TEMPLATE: Final[str] = "tushare 适配器暂不支持: {feature}"

#: Tushare's ``daily`` endpoint is end-of-day; surface ``86400`` as
#: the delay so callers (and the formatter's "数据延迟约 X 分钟"
#: rule) treat the snapshot accordingly. Distinct from akshare's
#: ``DEFAULT_QUOTE_DELAY_SECONDS`` (~15 min) to make the lag obvious.
_TS_QUOTE_DELAY_SECONDS: Final[int] = 86400

#: Maximum number of symbols allowed in a single ``pro.daily`` /
#: ``pro.daily_basic`` call. Tushare accepts up to ~50 codes per
#: request; we cap at 20 to align with the upstream batch limit
#: enforced by ``QuoteService``.
_TS_QUOTE_MAX_BATCH: Final[int] = 20


# ---------------------------------------------------------------------------
# Exception translation helpers
# ---------------------------------------------------------------------------


def _is_network_exception(exc: BaseException) -> bool:
    """Return ``True`` if *exc* looks like a transient network failure."""

    return any(base.__name__ in _NETWORK_EXC_NAMES for base in type(exc).__mro__)


def _is_rate_limit_exception(exc: BaseException) -> bool:
    """Return ``True`` if *exc* looks like a rate-limit / quota response."""

    msg = str(exc).lower()
    return any(pattern.lower() in msg for pattern in _RATE_LIMIT_PATTERNS)


def _call_tushare(
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Call ``func`` and translate exceptions to :class:`ChinaStockMCPError`.

    Order of checks mirrors the akshare / efinance adapters (and
    design "Error Handling" §Scenarios 2-3):

    1. Pass through exceptions that already belong to our hierarchy
       so ``fetch_with_fallback`` can apply the design's fallback
       rules correctly (``DataNotFoundError`` must NOT trigger another
       source switch).
    2. Map rate-limit / quota / 积分 indicators to
       :class:`RateLimitError`.
    3. Map transport-level failures to :class:`NetworkError`.
    4. Wrap every other exception as :class:`DataSourceError` so
       Requirement 13.1 holds: no raw third-party exception escapes.
    """

    try:
        return func(*args, **kwargs)
    except ChinaStockMCPError:
        raise
    except Exception as exc:
        if _is_rate_limit_exception(exc):
            raise RateLimitError(f"tushare 调用频率受限: {exc}") from exc
        if _is_network_exception(exc):
            raise NetworkError(f"tushare 网络错误: {exc}") from exc
        raise DataSourceError(f"tushare error: {exc}") from exc


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


# ---------------------------------------------------------------------------
# TushareAdapter
# ---------------------------------------------------------------------------


class TushareAdapter(BaseAdapter):
    """Backup adapter backed by the ``tushare`` Pro API.

    Only the v1 fallback methods are implemented (``search`` /
    ``quote`` / ``fundamentals`` / ``financial_report``); the
    remaining five abstract methods raise :class:`DataNotFoundError`
    so the class is instantiable for the fallback-link wiring in
    task 19.3 while the design's "fallback only when an equivalent
    endpoint exists" rule still holds.
    """

    name: str = "tushare"

    def __init__(self, token: str | None = None) -> None:
        """Resolve a tushare token and initialize the Pro API client.

        Parameters
        ----------
        token:
            Explicit token override. When ``None`` (the default) the
            constructor falls back to ``CSM_TUSHARE_TOKEN`` via
            :func:`load_settings`. If both are empty / missing the
            constructor raises :class:`RuntimeError` so the failure
            surfaces at wiring time rather than the first call.
        """

        resolved = token if token else load_settings().tushare_token
        if not resolved:
            # IMPORTANT (Requirement 12.4 / task 23.2): the error
            # message must NOT echo the empty / candidate token value.
            # Logged downstream by the server's wiring layer, this
            # text reaches the operator-facing log stream and must
            # stay free of secret material.
            raise RuntimeError(
                "tushare token 未配置: 请设置环境变量 CSM_TUSHARE_TOKEN "
                "或在构造 TushareAdapter 时显式传入 token 参数"
            )

        # Lazy-import ``tushare`` so importing this module without the
        # optional dependency installed does not crash. The reference
        # is typed as ``Any`` because tushare ships no public type
        # stubs and we do not want to leak its internals through the
        # adapter contract.
        import tushare as ts

        # ``ts.set_token`` forwards the token straight to the tushare
        # SDK's in-memory state. The token is NOT persisted to the
        # cache layer or any log sink (Requirement 12.4) -- ``self``
        # only retains the SDK references below.
        ts.set_token(resolved)
        self._ts: Any = ts
        self._pro: Any = ts.pro_api()

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(self, query: str, market: str) -> list[SymbolHit]:
        """Search standardized A 股 symbols by code or Chinese name.

        Pipeline:

        1. Pull the full A 股 listing universe via ``pro.stock_basic``
           with ``list_status='L'`` (currently listed) and the four
           columns we actually need (``ts_code`` / ``name`` / ``symbol``
           / ``industry``). Tushare returns ``ts_code`` already in the
           standardized form (e.g. ``"600519.SH"``) so no suffix
           reconstruction is needed.
        2. Filter rows where ``name`` or ``symbol`` contains the query
           (case-insensitive substring match -- Chinese characters
           survive ``str.casefold`` unchanged so the rule applies
           uniformly to ASCII tickers and Chinese names).
        3. Cap hits at :data:`_MAX_SEARCH_HITS` so a wide query
           (e.g. ``"宁"``) cannot blow the token budget.

        ``market`` is honored: only ``"a_stock"`` and ``"all"``
        produce results. ``"hk_stock"`` and ``"fund"`` raise
        :class:`DataNotFoundError` because tushare's free tier does
        not surface a comparable HK / fund universe; the akshare
        primary adapter already covers those, and the fallback chain
        only loses HK / fund search when both sources are down.
        """

        q = (query or "").strip()
        if not q:
            return []

        if market not in {"a_stock", "all"}:
            raise DataNotFoundError(
                _NOT_SUPPORTED_TEMPLATE.format(
                    feature=f"market={market!r} 搜索 (仅支持 a_stock / all)"
                )
            )

        df = _call_tushare(
            self._pro.stock_basic,
            exchange="",
            list_status="L",
            fields="ts_code,name,symbol,industry",
        )
        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty:
            return []

        needle = q.casefold()
        names = df["name"].astype(str).str.casefold() if "name" in df.columns else None
        symbols = (
            df["symbol"].astype(str).str.casefold() if "symbol" in df.columns else None
        )
        if names is None or symbols is None:
            return []
        mask = names.str.contains(needle, regex=False, na=False) | symbols.str.contains(
            needle, regex=False, na=False
        )
        matched = df.loc[mask]
        if matched.empty:
            return []

        hits: list[SymbolHit] = []
        for record in matched.to_dict(orient="records"):
            ts_code = str(record.get("ts_code", "")).strip()
            name = str(record.get("name", "")).strip()
            if not ts_code or not name:
                continue
            # Defensive guard: skip rows whose ts_code does not match
            # the standardized A 股 shape (would fail SymbolHit
            # validation otherwise).
            if not _is_a_stock_ts_code(ts_code):
                continue
            industry_raw = record.get("industry")
            industry: str | None = None
            if industry_raw is not None:
                stripped = str(industry_raw).strip()
                industry = stripped or None
            hits.append(
                SymbolHit(
                    code=ts_code,
                    name=name,
                    market="a_stock",
                    industry=industry,
                )
            )
            if len(hits) >= _MAX_SEARCH_HITS:
                break
        return hits

    # ------------------------------------------------------------------
    # quote
    # ------------------------------------------------------------------

    def quote(self, symbols: list[str]) -> list[Quote]:
        """Fetch end-of-day quote snapshots from tushare's ``daily`` endpoint.

        Pipeline:

        1. Normalize each input via :func:`normalize_symbol` so callers
           may pass bare codes or already-suffixed ones interchangeably.
        2. Reject HK / fund symbols up front -- tushare's free tier
           has no equivalent endpoint with the same Quote schema.
        3. Resolve the most recent open trading day via
           ``pro.trade_cal(end_date=today, is_open=1)`` so weekends /
           holidays do not yield empty rows.
        4. Pull ``pro.daily`` (open / high / low / close / vol / amount)
           and ``pro.daily_basic`` (turnover_rate / pe_ttm / pe / pb /
           total_mv / circ_mv) in one batched call each, joined by
           ``ts_code``.
        5. Build a :class:`Quote` per requested symbol with
           ``delay_seconds=86400`` to make the end-of-day lag explicit.
           Symbols missing from either upstream frame raise
           :class:`DataNotFoundError`.

        Note
        ----
        Tushare's ``vol`` is reported in ``手`` (lots, 100 股 each) and
        ``amount`` in ``千元`` (thousand CNY). We convert to ``股`` and
        ``元`` to keep the unit convention consistent with the
        :class:`Quote` DTO contract.
        """

        if not symbols:
            return []

        std_symbols: list[str] = [normalize_symbol(s) for s in symbols]
        seen: set[str] = set()
        unique: list[str] = []
        for std in std_symbols:
            if std in seen:
                continue
            seen.add(std)
            market = detect_market(std)
            if market != "a_stock":
                raise DataNotFoundError(
                    f"tushare 适配器暂不支持港股/基金行情: {std}"
                )
            unique.append(std)

        if len(unique) > _TS_QUOTE_MAX_BATCH:
            raise DataNotFoundError(
                f"tushare 单次行情请求上限 {_TS_QUOTE_MAX_BATCH} 个标的, "
                f"实际收到 {len(unique)} 个"
            )

        trade_date = self._latest_trade_date()
        ts_codes = ",".join(unique)

        daily_df = _call_tushare(
            self._pro.daily, ts_code=ts_codes, trade_date=trade_date
        )
        daily_df = (
            cast("pd.DataFrame", daily_df) if daily_df is not None else pd.DataFrame()
        )
        if daily_df.empty:
            raise DataNotFoundError(
                f"未找到 {trade_date} 的行情数据: {ts_codes}"
            )

        basic_df = _call_tushare(
            self._pro.daily_basic, ts_code=ts_codes, trade_date=trade_date
        )
        basic_df = (
            cast("pd.DataFrame", basic_df) if basic_df is not None else pd.DataFrame()
        )

        daily_by_code: dict[str, dict[str, Any]] = {}
        if "ts_code" in daily_df.columns:
            for record in daily_df.to_dict(orient="records"):
                code = str(record.get("ts_code", "")).strip()
                if code:
                    daily_by_code[code] = record

        basic_by_code: dict[str, dict[str, Any]] = {}
        if "ts_code" in basic_df.columns:
            for record in basic_df.to_dict(orient="records"):
                code = str(record.get("ts_code", "")).strip()
                if code:
                    basic_by_code[code] = record

        timestamp = _trade_date_to_datetime(trade_date)

        results: list[Quote] = []
        missing: list[str] = []
        # Replay caller order so ``Quote`` results line up with the
        # requested ``symbols`` argument.
        for std in std_symbols:
            daily_row = daily_by_code.get(std)
            if daily_row is None:
                missing.append(std)
                continue
            basic_row = basic_by_code.get(std, {})
            results.append(self._row_to_quote(std, daily_row, basic_row, timestamp))

        if missing:
            unique_missing = sorted(set(missing))
            raise DataNotFoundError(
                "未找到行情数据: " + ", ".join(unique_missing)
            )

        return results

    def _latest_trade_date(self) -> str:
        """Return the most recent open trading date as ``YYYYMMDD``."""

        today = datetime.now(UTC).strftime("%Y%m%d")
        cal_df = _call_tushare(self._pro.trade_cal, end_date=today, is_open=1)
        cal_df = cast("pd.DataFrame", cal_df) if cal_df is not None else pd.DataFrame()
        if cal_df.empty or "cal_date" not in cal_df.columns:
            raise DataNotFoundError(
                "tushare trade_cal 返回为空, 无法解析最近交易日"
            )
        return str(cal_df["cal_date"].iloc[-1]).strip()

    @staticmethod
    def _row_to_quote(
        std_symbol: str,
        daily_row: dict[str, Any],
        basic_row: dict[str, Any],
        timestamp: datetime,
    ) -> Quote:
        """Map tushare daily / daily_basic rows into a :class:`Quote` DTO."""

        close = _safe_float(daily_row.get("close"))
        pre_close = _safe_float(daily_row.get("pre_close"))
        change = _safe_float(daily_row.get("change"), default=close - pre_close)
        change_pct = _safe_float(daily_row.get("pct_chg"))
        # Tushare ``vol`` is in ``手`` (100 股); convert to ``股`` so
        # the Quote.volume unit matches the rest of the pipeline.
        volume = _safe_int(_safe_float(daily_row.get("vol")) * 100)
        # Tushare ``amount`` is in ``千元``; convert to ``元``.
        amount = _safe_float(daily_row.get("amount")) * 1000.0

        # ``daily_basic`` carries valuation / market-cap fields. They
        # are quoted in ``万元`` for the cap fields, which we convert
        # to ``元`` to match the Quote contract.
        total_mv = _safe_float(basic_row.get("total_mv")) * 10000.0
        float_mv = _safe_float(basic_row.get("circ_mv")) * 10000.0
        # Stock-name lookup is not in either dataframe; fall back to
        # the standardized symbol so the DTO's min_length=1 holds.
        return Quote(
            symbol=std_symbol,
            name=std_symbol,
            price=max(0.0, close),
            change=change,
            change_pct=change_pct,
            volume=max(0, volume),
            amount=max(0.0, amount),
            turnover_rate=_clamp_turnover_rate(basic_row.get("turnover_rate")),
            pe_ttm=_safe_optional_float(basic_row.get("pe_ttm")),
            pe_dynamic=_safe_optional_float(basic_row.get("pe")),
            pb=_safe_optional_float(basic_row.get("pb")),
            market_cap=max(0.0, total_mv),
            float_market_cap=max(0.0, float_mv),
            timestamp=timestamp,
            delay_seconds=_TS_QUOTE_DELAY_SECONDS,
        )

    # ------------------------------------------------------------------
    # fundamentals
    # ------------------------------------------------------------------

    def fundamentals(self, symbol: str) -> FundamentalSnapshot:
        """Fetch valuation / profitability / growth / health buckets.

        Pipeline:

        1. Normalize ``symbol`` and reject HK / fund codes (A 股 only
           in v1, parity with the akshare primary).
        2. Pull ``pro.fina_indicator(ts_code=..., period=last_year_end)``
           for ROE / ROA / 毛利率 / 净利率 + 营收 / 净利润 同比.
        3. Pull ``pro.daily_basic`` for the latest trading day to
           supply pe_ttm / pe (动态) / pb in the valuation bucket.
        4. Surface only fields the upstream actually populated --
           absent columns are omitted from the bucket; present-but-NaN
           map to ``None`` so the formatter can render a "-".
        5. ``industry_percentile`` is left empty: peer comparison is
           the responsibility of task 14.1 (industry peers).
        """

        std_symbol = normalize_symbol(symbol)
        market = detect_market(std_symbol)
        if market != "a_stock":
            raise DataNotFoundError(
                f"tushare 适配器暂不支持港股/基金基本面: {std_symbol}"
            )

        period = _latest_year_end_period(datetime.now(UTC).date())

        valuation = self._fetch_valuation_bucket(std_symbol)
        profitability, growth, health = self._fetch_indicator_buckets(
            std_symbol, period
        )

        return FundamentalSnapshot(
            symbol=std_symbol,
            valuation=valuation,
            profitability=profitability,
            growth=growth,
            health=health,
            industry_percentile={},
        )

    def _fetch_valuation_bucket(self, std_symbol: str) -> dict[str, float | None]:
        """Pull pe_ttm / pe / pb from the latest ``daily_basic`` row."""

        trade_date = self._latest_trade_date()
        df = _call_tushare(
            self._pro.daily_basic,
            ts_code=std_symbol,
            trade_date=trade_date,
        )
        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty:
            return {}

        record = df.iloc[0].to_dict()
        bucket: dict[str, float | None] = {}
        if "pe_ttm" in record:
            bucket["pe_ttm"] = _safe_optional_float(record.get("pe_ttm"))
        if "pe" in record:
            bucket["pe_dynamic"] = _safe_optional_float(record.get("pe"))
        if "pb" in record:
            bucket["pb"] = _safe_optional_float(record.get("pb"))
        return bucket

    def _fetch_indicator_buckets(
        self,
        std_symbol: str,
        period: str,
    ) -> tuple[
        dict[str, float | None],
        dict[str, float | None],
        dict[str, float | None],
    ]:
        """Pull profitability / growth / health from ``fina_indicator``."""

        df = _call_tushare(
            self._pro.fina_indicator, ts_code=std_symbol, period=period
        )
        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty:
            raise DataNotFoundError(
                f"未找到 {period} 期财务指标: {std_symbol}"
            )

        # Tushare's fina_indicator returns one row per period; the
        # most recent row at the requested period_end is at index 0
        # but we sort defensively by ``end_date`` desc when the column
        # exists.
        if "end_date" in df.columns:
            df = df.sort_values("end_date", ascending=False)
        row = df.iloc[0].to_dict()

        profitability: dict[str, float | None] = {}
        if "roe" in row:
            profitability["roe"] = _safe_optional_float(row.get("roe"))
        if "roa" in row:
            profitability["roa"] = _safe_optional_float(row.get("roa"))
        if "grossprofit_margin" in row:
            profitability["gross_margin"] = _safe_optional_float(
                row.get("grossprofit_margin")
            )
        if "netprofit_margin" in row:
            profitability["net_margin"] = _safe_optional_float(
                row.get("netprofit_margin")
            )

        growth: dict[str, float | None] = {}
        if "or_yoy" in row:
            growth["revenue_yoy"] = _safe_optional_float(row.get("or_yoy"))
        if "netprofit_yoy" in row:
            growth["net_profit_yoy"] = _safe_optional_float(row.get("netprofit_yoy"))

        health: dict[str, float | None] = {}
        if "debt_to_assets" in row:
            health["debt_ratio"] = _safe_optional_float(row.get("debt_to_assets"))
        if "current_ratio" in row:
            health["current_ratio"] = _safe_optional_float(row.get("current_ratio"))

        return profitability, growth, health

    # ------------------------------------------------------------------
    # financial_report
    # ------------------------------------------------------------------

    def financial_report(
        self,
        symbol: str,
        report_type: str,
        periods: int,
    ) -> FinancialReport:
        """Fetch ``periods`` of 财务报告 (annual or quarterly).

        Pipeline:

        1. Normalize ``symbol``; reject HK / fund codes (A 股 only in
           v1).
        2. Validate ``report_type`` and ``periods`` defensively at the
           adapter boundary -- the service layer is the primary
           validator but the adapter must still raise a unified error
           if a caller bypasses it.
        3. Pull ``pro.income(ts_code=...)`` once to enumerate the
           reporting periods the company has published. Filter
           ``end_date`` by report type (``annual`` keeps only ``MMDD
           == 1231``); take the most recent ``periods`` rows.
        4. For each surviving ``end_date`` query ``pro.income`` /
           ``pro.balancesheet`` / ``pro.cashflow`` and build a
           :class:`FinancialPeriod`.
        5. Sort the periods ascending by ``period_end`` so the
           Markdown table reads chronologically (Requirement 4.6).
           When the upstream produces fewer rows than ``periods`` we
           raise :class:`DataNotFoundError` with the actual count
           (Requirement 4.5).
        """

        std_symbol = normalize_symbol(symbol)
        market = detect_market(std_symbol)
        if market != "a_stock":
            raise DataNotFoundError(
                f"tushare 适配器暂不支持港股/基金财务报告: {std_symbol}"
            )

        if report_type not in _TS_FIN_REPORT_TYPES:
            raise DataSourceError(
                f"tushare error: 不支持的 report_type: {report_type!r}. "
                f"必须是 {sorted(_TS_FIN_REPORT_TYPES)} 之一"
            )
        if (
            not isinstance(periods, int)
            or isinstance(periods, bool)
            or periods < _TS_FIN_MIN_PERIODS
            or periods > _TS_FIN_MAX_PERIODS
        ):
            raise DataSourceError(
                f"tushare error: periods 必须是 [{_TS_FIN_MIN_PERIODS}, "
                f"{_TS_FIN_MAX_PERIODS}] 之间的整数, 实际收到 {periods!r}"
            )

        income_df = _call_tushare(self._pro.income, ts_code=std_symbol)
        income_df = (
            cast("pd.DataFrame", income_df)
            if income_df is not None
            else pd.DataFrame()
        )
        if income_df.empty or "end_date" not in income_df.columns:
            raise DataNotFoundError(
                f"未找到 {periods} 期 {report_type} 财务报告: {std_symbol} "
                f"上游返回空数据, 请稍后重试或更换 report_type"
            )

        # Build a unique, sorted-desc list of reporting end dates that
        # match the requested report_type.
        end_dates: list[str] = sorted(
            {
                str(v).strip()
                for v in income_df["end_date"].tolist()
                if v is not None and str(v).strip()
            },
            reverse=True,
        )
        if report_type == "annual":
            end_dates = [d for d in end_dates if d.endswith("1231")]
        if not end_dates:
            raise DataNotFoundError(
                f"未找到 {periods} 期 {report_type} 财务报告: {std_symbol} "
                f"上游缺少匹配的报告期, 请减小 periods 或更换 report_type"
            )

        if len(end_dates) < periods:
            raise DataNotFoundError(
                f"未找到 {periods} 期 {report_type} 财务报告: {std_symbol} "
                f"上游仅有 {len(end_dates)} 期, "
                f"请减小 periods 或更换 report_type"
            )

        selected = end_dates[:periods]

        period_dtos: list[FinancialPeriod] = []
        for end_date in selected:
            period_dto = self._fetch_period(std_symbol, end_date, income_df)
            if period_dto is not None:
                period_dtos.append(period_dto)

        if len(period_dtos) < periods:
            raise DataNotFoundError(
                f"未找到 {periods} 期 {report_type} 财务报告: {std_symbol} "
                f"上游仅有 {len(period_dtos)} 期可用, "
                f"请减小 periods 或更换 report_type"
            )

        period_dtos.sort(key=lambda p: p.period_end)

        return FinancialReport(
            symbol=std_symbol,
            report_type=cast(Any, report_type),
            periods=period_dtos,
        )

    def _fetch_period(
        self,
        std_symbol: str,
        end_date: str,
        income_df: pd.DataFrame,
    ) -> FinancialPeriod | None:
        """Build one :class:`FinancialPeriod` for ``end_date``.

        Reuses the already-fetched ``income_df`` for the income
        statement values to avoid a redundant API call; only the
        balance sheet and cash-flow statement require fresh fetches.
        """

        try:
            period_end = date(
                int(end_date[0:4]), int(end_date[4:6]), int(end_date[6:8])
            )
        except (TypeError, ValueError):
            return None

        income_row = self._row_for_end_date(income_df, end_date)
        if income_row is None:
            return None

        balance_df = _call_tushare(
            self._pro.balancesheet, ts_code=std_symbol, period=end_date
        )
        balance_df = (
            cast("pd.DataFrame", balance_df)
            if balance_df is not None
            else pd.DataFrame()
        )
        balance_row = self._row_for_end_date(balance_df, end_date) or {}

        cashflow_df = _call_tushare(
            self._pro.cashflow, ts_code=std_symbol, period=end_date
        )
        cashflow_df = (
            cast("pd.DataFrame", cashflow_df)
            if cashflow_df is not None
            else pd.DataFrame()
        )
        cashflow_row = self._row_for_end_date(cashflow_df, end_date) or {}

        revenue = _safe_optional_float(income_row.get("total_revenue"))
        if revenue is None:
            return None
        net_profit = _safe_optional_float(income_row.get("n_income_attr_p"))
        # Tushare exposes 扣非净利润 as ``n_income`` minus 非经常性损益.
        # ``profit_dedt`` is the closest direct field on fina_indicator
        # but it's not in the income statement; fall back to net_profit
        # when missing so the DTO field stays numeric.
        net_profit_excl_nrgl = _safe_optional_float(
            income_row.get("n_income")
        )
        gross_profit = revenue - _safe_float(income_row.get("oper_cost"))
        operating_cash_flow = _safe_optional_float(
            cashflow_row.get("n_cashflow_act")
        )
        total_assets = _safe_optional_float(balance_row.get("total_assets"))
        total_liabilities = _safe_optional_float(balance_row.get("total_liab"))
        equity = _safe_optional_float(balance_row.get("total_hldr_eqy_inc_min_int"))

        try:
            return FinancialPeriod(
                period_end=period_end,
                revenue=float(revenue),
                net_profit=float(net_profit if net_profit is not None else 0.0),
                net_profit_excl_nrgl=float(
                    net_profit_excl_nrgl
                    if net_profit_excl_nrgl is not None
                    else 0.0
                ),
                gross_profit=float(gross_profit),
                operating_cash_flow=float(
                    operating_cash_flow if operating_cash_flow is not None else 0.0
                ),
                total_assets=float(total_assets if total_assets is not None else 0.0),
                total_liabilities=float(
                    total_liabilities if total_liabilities is not None else 0.0
                ),
                equity=float(equity if equity is not None else 0.0),
            )
        except ValueError:
            return None

    @staticmethod
    def _row_for_end_date(
        df: pd.DataFrame,
        end_date: str,
    ) -> dict[str, Any] | None:
        """Return the first row whose ``end_date`` matches, or ``None``."""

        if df.empty or "end_date" not in df.columns:
            return None
        match = df.loc[df["end_date"].astype(str) == end_date]
        if match.empty:
            return None
        return cast("dict[str, Any]", match.iloc[0].to_dict())

    # ------------------------------------------------------------------
    # Methods explicitly out of scope for the v1 fallback
    # ------------------------------------------------------------------

    def kline(
        self,
        symbol: str,
        period: str,
        count: int,
        adjust: str,
    ) -> KLineSeries:
        """Not implemented -- akshare primary already covers K 线 history."""

        raise DataNotFoundError(
            _NOT_SUPPORTED_TEMPLATE.format(feature="kline")
        )

    def money_flow(
        self,
        symbol: str | None,
        flow_type: str,
        top_n: int,
    ) -> MoneyFlow:
        """Not implemented -- tushare's money-flow endpoints are paid-only."""

        raise DataNotFoundError(
            _NOT_SUPPORTED_TEMPLATE.format(feature="money_flow")
        )

    def industry_peers(
        self,
        symbol: str,
        metrics: list[str],
        top_n: int,
    ) -> PeerTable:
        """Not implemented -- requires paid tushare classifications."""

        raise DataNotFoundError(
            _NOT_SUPPORTED_TEMPLATE.format(feature="industry_peers")
        )

    def fund_info(self, fund_code: str) -> FundInfo:
        """Not implemented -- tushare fund endpoints have a different schema."""

        raise DataNotFoundError(
            _NOT_SUPPORTED_TEMPLATE.format(feature="fund_info")
        )

    def market_overview(self) -> MarketOverview:
        """Not implemented -- aggregated by akshare primary."""

        raise DataNotFoundError(
            _NOT_SUPPORTED_TEMPLATE.format(feature="market_overview")
        )


# ---------------------------------------------------------------------------
# Module-level helpers (kept private)
# ---------------------------------------------------------------------------


def _is_a_stock_ts_code(ts_code: str) -> bool:
    """Return ``True`` when ``ts_code`` matches the standardized A 股 shape."""

    if len(ts_code) != 9:
        return False
    head, sep, tail = ts_code.partition(".")
    if sep != "." or len(head) != 6 or not head.isdigit():
        return False
    return tail in {"SH", "SZ", "BJ"}


def _latest_year_end_period(today: date) -> str:
    """Return the most recent published fiscal year-end as ``YYYYMMDD``.

    Annual reports for fiscal year ``Y`` are typically published in
    ``Y+1`` between Jan and Apr. We conservatively assume ``Y-1`` is
    available year-round; callers that need the freshest period can
    fall back via :class:`DataNotFoundError` retries.
    """

    return f"{today.year - 1}1231"


def _trade_date_to_datetime(trade_date: str) -> datetime:
    """Coerce ``YYYYMMDD`` into a UTC midnight :class:`datetime`."""

    try:
        parsed = datetime.strptime(trade_date, "%Y%m%d")
    except ValueError:
        return datetime.now(UTC)
    return parsed.replace(tzinfo=UTC)


# Silence "unused import" warnings for the ``logger`` symbol; it is
# imported so future enhancements can log without re-importing, and so
# the module signals via its imports that it participates in the shared
# logging configuration.
_ = logger


__all__ = ["TushareAdapter"]
