"""EfinanceAdapter: backup data source backed by the ``efinance`` library.

This module implements the *fallback* adapter referenced by
``design.md`` Component 4 (Adapter Layer). Per task 19.2 only the v1
fallback methods are materialized -- ``search`` / ``quote`` /
``money_flow`` -- because those three tools are the ones for which
``akshare`` outages can plausibly be covered by a second free upstream
without contradicting the design's TTL and primary-source mapping.
The remaining six abstract methods (``kline`` / ``fundamentals`` /
``financial_report`` / ``industry_peers`` / ``fund_info`` /
``market_overview``) raise :class:`DataNotFoundError` so the class is
instantiable while ``fetch_with_fallback`` correctly skips this source
for tools that have no efinance equivalent.

References
----------
- ``design.md`` Component 4: Adapter Layer (``EfinanceAdapter`` slot).
- Requirement 13.1: every adapter SHALL raise
  :class:`~china_stock_mcp.exceptions.ChinaStockMCPError` subclasses
  rather than letting third-party exceptions escape.

Exception translation
---------------------
``efinance`` re-uses ``requests`` / ``urllib3`` under the hood and
exposes a few of its own helpers (e.g. ``retry``). Any of the
following surfaces are mapped to the unified hierarchy by
:func:`_call_efinance`:

* connection / DNS / timeout / 5xx           → :class:`NetworkError`
* HTTP 429 or "请求过于频繁" wording          → :class:`RateLimitError`
* every other third-party exception          → :class:`DataSourceError`

Lazy-import contract
--------------------
``efinance`` is declared as an optional dependency
(``[project.optional-dependencies] efinance = ["efinance>=0.4"]``)
so we **must not** import it at module scope. Doing so would break
``import china_stock_mcp.adapters.efinance_adapter`` for any user
who installed the base package without the ``efinance`` extra.
The import lives inside :py:meth:`EfinanceAdapter.__init__` and the
fully-typed reference is cached on the instance.
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
)
from china_stock_mcp.models import (
    DEFAULT_QUOTE_DELAY_SECONDS,
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

#: Maximum number of hits returned by :meth:`EfinanceAdapter.search`.
#: Mirrors the akshare adapter so the formatter layer can apply the
#: same per-tool token budget regardless of the source that answered.
_MAX_SEARCH_HITS: Final[int] = 20

#: Markets accepted by :meth:`EfinanceAdapter.search`.
_VALID_SEARCH_MARKETS: Final[frozenset[str]] = frozenset(
    {"a_stock", "hk_stock", "fund", "all"}
)

#: A 股 6-digit code prefixes → exchange suffix.
_A_STOCK_SH_PREFIXES: Final[frozenset[str]] = frozenset({"60", "68", "90"})
_A_STOCK_SZ_PREFIXES: Final[frozenset[str]] = frozenset({"00", "30", "20"})
_A_STOCK_BJ_FIRST_CHAR: Final[str] = "8"

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

#: Substrings inside an exception ``str`` that indicate rate limiting.
_RATE_LIMIT_PATTERNS: Final[tuple[str, ...]] = (
    "too many requests",
    "rate limit",
    "rate-limit",
    "429",
    "请求过于频繁",
    "访问频率过快",
)

#: Accepted ``flow_type`` values at the adapter boundary.
_EF_FLOW_TYPES: Final[frozenset[str]] = frozenset(
    {"north", "main", "dragon_tiger"}
)

#: ``top_n`` lower / upper bounds for money-flow queries.
_EF_FLOW_MIN_TOP_N: Final[int] = 1
_EF_FLOW_MAX_TOP_N: Final[int] = 100

#: Lookback window (calendar days) for the dragon-tiger detail
#: endpoint -- 14 days spans long holidays without overshooting the
#: per-call row budget. Mirrors the akshare adapter's choice so the
#: two sources' outputs stay comparable when ``fetch_with_fallback``
#: switches between them.
_EF_LHB_LOOKBACK_DAYS: Final[int] = 14

#: Placeholder message for adapter methods this fallback intentionally
#: does not implement (Requirement 13.1: surface a unified error).
_NOT_SUPPORTED_TEMPLATE: Final[str] = "efinance 适配器暂不支持: {feature}"


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


def _call_efinance(
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Call ``func`` and translate exceptions to :class:`ChinaStockMCPError`.

    Order of checks mirrors the akshare adapter (and design "Error
    Handling" §Scenarios 2-3):

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
        # ``fetch_with_fallback`` can apply the design's fallback
        # rules correctly (DataNotFoundError must NOT trigger another
        # source switch).
        raise
    except Exception as exc:
        if _is_rate_limit_exception(exc):
            raise RateLimitError(f"efinance 调用频率受限: {exc}") from exc
        if _is_network_exception(exc):
            raise NetworkError(f"efinance 网络错误: {exc}") from exc
        raise DataSourceError(f"efinance error: {exc}") from exc


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


def _coerce_iso_date(value: Any) -> str:
    """Coerce a dataframe cell into a ``YYYY-MM-DD`` ISO string."""

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


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first column from ``candidates`` that exists in ``df``."""

    for name in candidates:
        if name in df.columns:
            return name
    return None


# ---------------------------------------------------------------------------
# EfinanceAdapter
# ---------------------------------------------------------------------------


class EfinanceAdapter(BaseAdapter):
    """Backup adapter backed by the ``efinance`` library.

    Only the v1 fallback methods are implemented (``search`` / ``quote``
    / ``money_flow``); the remaining six abstract methods raise
    :class:`DataNotFoundError` so the class is instantiable for the
    fallback-link wiring in task 19.3 while the design's "fallback
    only when an equivalent endpoint exists" rule still holds.
    """

    name: str = "efinance"

    def __init__(self) -> None:
        # Lazy-import ``efinance`` so importing this module without the
        # optional dependency installed does not crash. The reference is
        # cached on the instance (and typed as ``Any`` because efinance
        # ships no public type stubs) so subsequent method calls do
        # not pay the import cost again.
        import efinance as ef

        self._ef: Any = ef

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(self, query: str, market: str) -> list[SymbolHit]:
        """Search standardized symbols by code or Chinese name.

        efinance does not ship a dedicated multi-market search
        endpoint, but :func:`ef.stock.get_realtime_quotes` returns the
        full universe with both ``股票代码`` and ``股票名称`` columns
        for every requested ``fs`` group (沪深A股 / 港股). We reuse it
        as the search index by pulling the relevant universe(s) once
        and filtering rows where ``代码`` or ``名称`` contains the
        query (case-insensitive substring match).

        Result hits are capped at :data:`_MAX_SEARCH_HITS` so a wide
        query (e.g. ``"宁"``) cannot blow the token budget.

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
                f"efinance error: 不支持的 market: {market!r}. "
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
        # efinance has no stable public-fund universe endpoint; the
        # fund market is intentionally not surfaced here. The akshare
        # primary adapter already covers it, so the fallback chain
        # only loses fund search when *both* sources are down -- a
        # narrower failure mode than v1 is willing to optimize for.
        return hits[:_MAX_SEARCH_HITS]

    def _search_a_stock(self, query: str) -> list[SymbolHit]:
        df = _call_efinance(self._ef.stock.get_realtime_quotes)
        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty:
            return []

        code_col = _pick_column(df, ("股票代码", "代码", "code", "symbol"))
        name_col = _pick_column(df, ("股票名称", "名称", "name"))
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
                # Skip non-A-share rows that can sneak into the spot
                # universe (B 股, exotic 9xxxxx codes that do not match
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
        df = _call_efinance(self._ef.stock.get_realtime_quotes, fs="港股")
        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty:
            return []

        code_col = _pick_column(df, ("股票代码", "代码", "symbol", "code"))
        name_col = _pick_column(df, ("股票名称", "名称", "name"))
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
        2. Group the standardized symbols by market and call
           :func:`ef.stock.get_realtime_quotes` once per group with the
           appropriate ``fs`` filter (``None`` ≡ 沪深A股 / ``"港股"``).
           This minimizes upstream load -- a single endpoint call
           covers all symbols in the same market.
        3. Map each requested symbol to its row and build a
           :class:`Quote`. Missing symbols raise
           :class:`DataNotFoundError` listing the absent codes.

        Funds are out of scope for the v1 fallback (no efinance
        endpoint with the same shape as the spot frame), so a fund
        symbol surfaces :class:`DataNotFoundError` and the upstream
        ``fetch_with_fallback`` reports the failure to the caller.
        """

        if not symbols:
            return []

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
            missing = ", ".join(unique_by_market["fund"])
            raise DataNotFoundError(
                f"efinance 适配器暂不支持基金实时行情: {missing}. "
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
        df = _call_efinance(self._ef.stock.get_realtime_quotes)
        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty:
            return {}

        code_col = _pick_column(df, ("股票代码", "代码", "code", "symbol"))
        if code_col is None:
            return {}

        bare_to_std: dict[str, str] = {std.split(".")[0]: std for std in std_symbols}
        mask = df[code_col].astype(str).isin(list(bare_to_std.keys()))
        matched = df.loc[mask]
        if matched.empty:
            return {}

        timestamp = self._pick_timestamp(df)
        out: dict[str, Quote] = {}
        for row in matched.itertuples(index=False):
            row_dict = (
                row._asdict()
                if hasattr(row, "_asdict")
                else dict(zip(matched.columns, row, strict=False))
            )
            bare = str(row_dict.get(code_col, "")).strip()
            std = bare_to_std.get(bare)
            if std is None:
                continue
            out[std] = self._row_to_quote(row_dict, std, timestamp)
        return out

    def _quote_hk_stock(self, std_symbols: list[str]) -> dict[str, Quote]:
        df = _call_efinance(self._ef.stock.get_realtime_quotes, fs="港股")
        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty:
            return {}

        code_col = _pick_column(df, ("股票代码", "代码", "symbol", "code"))
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
            row_dict = (
                row._asdict()
                if hasattr(row, "_asdict")
                else dict(zip(matched.columns, row, strict=False))
            )
            bare = str(row_dict.get(code_col, "")).strip().zfill(5)
            std = bare_to_std.get(bare)
            if std is None:
                continue
            out[std] = self._row_to_quote(row_dict, std, timestamp)
        return out

    @staticmethod
    def _pick_timestamp(df: pd.DataFrame) -> datetime:
        """Extract a ``datetime`` from the dataframe or fall back to now()."""

        for col in ("时间", "更新时间", "timestamp"):
            if col not in df.columns:
                continue
            try:
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
        """Map an efinance realtime-quotes row into a :class:`Quote` DTO.

        Field aliases account for minor schema drift across efinance
        versions (e.g. ``市盈率(动)`` vs. ``动态市盈率`` vs.
        ``市盈率-动态``). Missing optional metrics fall back to
        ``None`` (PE / PB) or ``0`` (numeric required fields) so DTO
        validation still succeeds when an upstream column is absent.
        """

        def pick(*aliases: str) -> Any:
            for alias in aliases:
                if alias in row:
                    return row[alias]
            return None

        name = (
            str(pick("股票名称", "名称", "name") or "").strip() or std_symbol
        )
        return Quote(
            symbol=std_symbol,
            name=name,
            price=_safe_float(pick("最新价", "现价", "price")),
            change=_safe_float(pick("涨跌额", "change")),
            change_pct=_safe_float(pick("涨跌幅", "change_pct")),
            volume=_safe_int(pick("成交量", "volume")),
            amount=_safe_float(pick("成交额", "amount")),
            turnover_rate=_clamp_turnover_rate(pick("换手率", "turnover_rate")),
            pe_ttm=_safe_optional_float(pick("市盈率-TTM", "动态市盈率(TTM)", "pe_ttm")),
            pe_dynamic=_safe_optional_float(
                pick("动态市盈率", "市盈率(动)", "市盈率-动态", "市盈率(动态)", "pe_dynamic")
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
    # money_flow
    # ------------------------------------------------------------------

    def money_flow(
        self,
        symbol: str | None,
        flow_type: str,
        top_n: int,
    ) -> MoneyFlow:
        """Fetch 资金流向 data (north / main / dragon_tiger).

        Pipeline:

        1. Validate ``flow_type`` and ``top_n`` defensively at the
           adapter boundary -- the service layer is the primary
           validator but the adapter must still raise a unified error
           if a caller bypasses it.
        2. Dispatch on ``flow_type``:

           * ``main`` -- requires ``symbol``; calls
             :func:`ef.stock.get_history_bill` for the latest
             ``top_n`` trading days of 主力 / 超大单 / 大单 / 中单 /
             小单 净流入. This is the strongest equivalent we have
             for akshare's ``stock_individual_fund_flow``.
           * ``dragon_tiger`` -- :func:`ef.stock.get_daily_billboard`
             over the past 14 calendar days; rows are filtered to
             ``symbol`` when provided, otherwise the full board is
             returned.
           * ``north`` -- efinance does not expose a stable
             north-bound aggregate equivalent across versions, so
             this branch raises :class:`DataNotFoundError` rather
             than fabricate data. The akshare primary adapter
             already covers ``north``; the fallback chain only loses
             north flow when both sources are down.

        3. Truncate rows to the most recent ``top_n``.
        4. Wrap the result in :class:`MoneyFlow` with
           ``snapshot_at = datetime.now(UTC)``.
        """

        if flow_type not in _EF_FLOW_TYPES:
            raise DataSourceError(
                f"efinance error: 不支持的 flow_type: {flow_type!r}. "
                f"必须是 {sorted(_EF_FLOW_TYPES)} 之一"
            )
        if (
            not isinstance(top_n, int)
            or isinstance(top_n, bool)
            or top_n < _EF_FLOW_MIN_TOP_N
            or top_n > _EF_FLOW_MAX_TOP_N
        ):
            raise DataSourceError(
                f"efinance error: top_n 必须是 [{_EF_FLOW_MIN_TOP_N}, "
                f"{_EF_FLOW_MAX_TOP_N}] 之间的整数, 实际收到 {top_n!r}"
            )

        if flow_type == "north":
            raise DataNotFoundError(
                _NOT_SUPPORTED_TEMPLATE.format(feature="north 资金流向")
            )
        if flow_type == "main":
            rows = self._fetch_money_flow_main(symbol, top_n)
        else:
            rows = self._fetch_money_flow_dragon_tiger(symbol, top_n)

        return MoneyFlow(
            flow_type=cast(Any, flow_type),
            rows=rows,
            snapshot_at=datetime.now(UTC),
        )

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
                f"efinance 适配器暂不支持港股/基金主力资金流向: {std_symbol}"
            )

        bare = std_symbol.split(".")[0]
        df = _call_efinance(self._ef.stock.get_history_bill, bare)
        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty or "日期" not in df.columns:
            raise DataNotFoundError(
                f"未找到主力资金流向数据: {std_symbol}"
            )

        # efinance returns rows in ascending date order; sort newest-first
        # and take the most recent ``top_n`` rows.
        df = df.sort_values("日期", ascending=False).head(top_n)

        rows: list[dict[str, object]] = []
        for record in df.to_dict(orient="records"):
            rows.append(
                {
                    "date": _coerce_iso_date(record.get("日期")),
                    "收盘价": _safe_optional_float(record.get("收盘价")),
                    "涨跌幅": _safe_optional_float(record.get("涨跌幅")),
                    "主力净流入": _safe_optional_float(record.get("主力净流入")),
                    "超大单净流入": _safe_optional_float(
                        record.get("超大单净流入")
                    ),
                    "大单净流入": _safe_optional_float(record.get("大单净流入")),
                    "中单净流入": _safe_optional_float(record.get("中单净流入")),
                    "小单净流入": _safe_optional_float(record.get("小单净流入")),
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
        start_dt = end_dt - timedelta(days=_EF_LHB_LOOKBACK_DAYS)
        df = _call_efinance(
            self._ef.stock.get_daily_billboard,
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=end_dt.strftime("%Y-%m-%d"),
        )
        df = cast("pd.DataFrame", df) if df is not None else pd.DataFrame()
        if df.empty or "股票代码" not in df.columns:
            raise DataNotFoundError("未找到龙虎榜数据")

        if symbol is not None and str(symbol).strip():
            std_symbol = normalize_symbol(symbol)
            bare = std_symbol.split(".")[0]
            df = df.loc[df["股票代码"].astype(str) == bare].copy()
            if df.empty:
                raise DataNotFoundError(
                    f"未找到龙虎榜数据: {std_symbol}"
                )

        if "上榜日期" in df.columns:
            df = df.sort_values("上榜日期", ascending=False)
        df = df.head(top_n)

        rows: list[dict[str, object]] = []
        for record in df.to_dict(orient="records"):
            rows.append(
                {
                    "代码": str(record.get("股票代码", "")).strip(),
                    "名称": str(record.get("股票名称", "")).strip(),
                    "上榜日": _coerce_iso_date(record.get("上榜日期")),
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
        """Not implemented -- v1 fallback covers search / quote / money_flow only."""

        raise DataNotFoundError(
            _NOT_SUPPORTED_TEMPLATE.format(feature="kline")
        )

    def fundamentals(self, symbol: str) -> FundamentalSnapshot:
        """Not implemented -- v1 fallback covers search / quote / money_flow only."""

        raise DataNotFoundError(
            _NOT_SUPPORTED_TEMPLATE.format(feature="fundamentals")
        )

    def financial_report(
        self,
        symbol: str,
        report_type: str,
        periods: int,
    ) -> FinancialReport:
        """Not implemented -- v1 fallback covers search / quote / money_flow only."""

        raise DataNotFoundError(
            _NOT_SUPPORTED_TEMPLATE.format(feature="financial_report")
        )

    def industry_peers(
        self,
        symbol: str,
        metrics: list[str],
        top_n: int,
    ) -> PeerTable:
        """Not implemented -- v1 fallback covers search / quote / money_flow only."""

        raise DataNotFoundError(
            _NOT_SUPPORTED_TEMPLATE.format(feature="industry_peers")
        )

    def fund_info(self, fund_code: str) -> FundInfo:
        """Not implemented -- v1 fallback covers search / quote / money_flow only."""

        raise DataNotFoundError(
            _NOT_SUPPORTED_TEMPLATE.format(feature="fund_info")
        )

    def market_overview(self) -> MarketOverview:
        """Not implemented -- v1 fallback covers search / quote / money_flow only."""

        raise DataNotFoundError(
            _NOT_SUPPORTED_TEMPLATE.format(feature="market_overview")
        )


# Silence "unused import" warnings for the ``logger`` symbol; it is
# imported so future enhancements can log without re-importing, and so
# the module signals via its imports that it participates in the shared
# logging configuration.
_ = logger


__all__ = ["EfinanceAdapter"]
