"""``valuation_compare`` prompt -- 多标的估值横向对比 Markdown 编排.

Implements *task 20.2* from ``.kiro/specs/china-stock-mcp/tasks.md``.
The prompt orchestrates three service-layer calls per symbol and
stitches the results into a single Markdown deliverable that lets an
AI client compare valuations across 2..10 标的 at a glance:

1. **行情对比** -- ``QuoteService.get_snapshot`` (one batch call
   covering every symbol).
2. **估值横向对比** -- ``FundamentalService.snapshot`` (per symbol).
3. **行业横切** -- ``IndustryService.peers`` (per symbol; defaults to
   ``[pe, pb, roe]`` and ``top_n=5`` so the section stays compact).

Behaviour
---------

- Per-symbol graceful degradation (Requirement 10.5):
  every per-symbol call is wrapped in ``try / except
  ChinaStockMCPError`` and on failure that symbol's row / section is
  replaced by a ``> ⚠️ {symbol} 数据不可用: {error}`` blockquote. The
  remaining symbols still render so the consumer always gets
  something usable.

- The 行情 batch call is also wrapped: if it fails outright the
  行情对比表 falls back to "数据暂不可用" rows but the rest of the
  document keeps rendering.

- The disclaimer is appended exactly once at the end via
  :func:`append_disclaimer` (Requirement 10.6, Property 14).

The prompt is **pure** with respect to the services it receives:
callers (e.g. :mod:`china_stock_mcp.server`) construct the service
instances and pass them in via the ``services`` keyword, which keeps
the function trivially testable without any FastMCP / akshare side
effects.
"""

from __future__ import annotations

import math
from typing import Final, TypedDict

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from china_stock_mcp.error_mapping import bridge_pydantic_error
from china_stock_mcp.exceptions import ChinaStockMCPError
from china_stock_mcp.formatters import (
    NONE_PLACEHOLDER,
    append_disclaimer,
    format_amount,
    format_percent,
    render_change,
    render_table,
)
from china_stock_mcp.models import FundamentalSnapshot, PeerTable, Quote
from china_stock_mcp.services.fundamental_service import FundamentalService
from china_stock_mcp.services.industry_service import IndustryService
from china_stock_mcp.services.quote_service import QuoteService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Lower / upper bounds on the symbol list. Two is the minimum that
#: makes a *comparison* meaningful; ten keeps the rendered Markdown
#: well within the per-tool 3 000-token budget (Property 13) even
#: when every symbol fans out into a per-symbol sub-table.
_MIN_SYMBOLS: Final[int] = 2
_MAX_SYMBOLS: Final[int] = 10

#: Defaults for the 行业横切 peer query. Pinned here (not exposed to
#: the prompt caller) because the prompt's job is to give brief
#: context, not to expose a fully tunable industry tool -- callers
#: who need that should invoke ``get_industry_peers`` directly.
_PEER_METRICS: Final[tuple[str, ...]] = ("pe", "pb", "roe")
_PEER_TOP_N: Final[int] = 5

#: Headers for the 行情对比 multi-row table.
_QUOTE_TABLE_HEADERS: Final[list[str]] = [
    "代码",
    "名称",
    "现价",
    "涨跌幅",
    "PE(TTM)",
    "PB",
    "总市值",
]

#: Headers for the per-symbol 估值横向对比 sub-table.
_VALUATION_TABLE_HEADERS: Final[list[str]] = ["指标", "数值"]

#: Localized labels for the four 估值横向对比 metrics surfaced from
#: :class:`FundamentalSnapshot`. Order is preserved when rendering the
#: per-symbol sub-table.
_VALUATION_METRICS: Final[tuple[tuple[str, str, str], ...]] = (
    # (snapshot_bucket, bucket_key, label)
    ("valuation", "pe_ttm", "PE(TTM)"),
    ("valuation", "pb", "PB"),
    ("profitability", "roe", "ROE"),
    ("growth", "revenue_yoy", "营收增速"),
)

#: Metric keys whose value is denominated in *percent* and should
#: render with :func:`format_percent` (which appends ``%``).
_PERCENT_KEYS: Final[frozenset[str]] = frozenset({"roe", "revenue_yoy"})


class _Services(TypedDict):
    """Typed bundle of pre-wired service instances."""

    quote: QuoteService
    fundamental: FundamentalService
    industry: IndustryService


# ---------------------------------------------------------------------------
# Pydantic input model
# ---------------------------------------------------------------------------


class ValuationCompareInput(BaseModel):
    """Pydantic v2 input schema for :func:`valuation_compare`.

    Constraints:

    - ``symbols`` is a list of ``[2, 10]`` non-empty strings; the
      prompt's job is *comparison* so a single-symbol call is rejected.
    - ``extra="forbid"`` rejects unknown keys.
    - ``str_strip_whitespace=True`` trims surrounding whitespace from
      each entry; downstream services take care of the rest of the
      normalization.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    symbols: list[str] = Field(
        ...,
        min_length=_MIN_SYMBOLS,
        max_length=_MAX_SYMBOLS,
    )


# ---------------------------------------------------------------------------
# Public prompt entry
# ---------------------------------------------------------------------------


def valuation_compare(
    symbols: list[str],
    *,
    services: _Services,
) -> str:
    """Return a 估值对比 Markdown document for ``symbols``.

    Parameters
    ----------
    symbols:
        List of 2..10 standardized or bare symbols (Chinese names /
        pinyin also accepted; the underlying services normalize them).
    services:
        Bundle of pre-wired :class:`QuoteService`,
        :class:`FundamentalService` and :class:`IndustryService`
        instances. The prompt is pure with respect to its services so
        unit tests can supply lightweight stubs.

    Returns
    -------
    str
        Markdown document ending with the standard disclaimer.

    Raises
    ------
    ValidationError
        If ``symbols`` fails Pydantic validation.
    """

    # 1) Input validation -- pydantic failures bridge to the unified
    #    error tree (Requirement 13.7).
    try:
        validated = ValuationCompareInput(symbols=symbols)
    except PydanticValidationError as exc:
        raise bridge_pydantic_error(exc) from exc

    syms = list(validated.symbols)

    # 2) Run a single batched 行情 call covering every symbol. The
    #    QuoteService already preserves caller order and dedups
    #    internally, so we can map the result by symbol for the row
    #    lookup below.
    quotes_by_symbol, quote_error = _fetch_quotes(services["quote"], syms)

    # 3) Render the three sections.
    title = f"# 估值对比 ({len(syms)} 只标的)"

    quote_section = _render_quote_section(syms, quotes_by_symbol, quote_error)
    valuation_section = _render_valuation_section(services["fundamental"], syms)
    industry_section = _render_industry_section(services["industry"], syms)

    body = "\n\n".join(
        [title, quote_section, valuation_section, industry_section]
    )

    # 4) Disclaimer (Requirement 10.6, Property 14).
    return append_disclaimer(body)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _fetch_quotes(
    service: QuoteService,
    symbols: list[str],
) -> tuple[dict[str, Quote], ChinaStockMCPError | None]:
    """Run one batched ``get_snapshot`` call, degrading on errors.

    The map is keyed by *both* the standardized symbol and the raw
    caller-supplied symbol so the row renderer can find a quote
    regardless of which form the caller passed in. When the upstream
    fails outright (e.g. a transient network outage), the helper
    returns an empty map plus the original error so the row renderer
    can degrade per-row.
    """

    try:
        quotes = service.get_snapshot(symbols)
    except ChinaStockMCPError as exc:
        return {}, exc

    by_symbol: dict[str, Quote] = {}
    # Index every returned :class:`Quote` by its *standardized* symbol
    # (``Quote.symbol`` is always the standardized form), and also by
    # the raw caller-supplied symbol at the same offset so we can find
    # a match without re-running :func:`normalize_symbol`. Iterating in
    # parallel is safe: ``QuoteService.get_snapshot`` preserves caller
    # order including duplicates.
    for raw, q in zip(symbols, quotes, strict=False):
        by_symbol[q.symbol] = q
        by_symbol.setdefault(raw, q)
    return by_symbol, None


def _render_quote_section(
    symbols: list[str],
    quotes_by_symbol: dict[str, Quote],
    quote_error: ChinaStockMCPError | None,
) -> str:
    """Render the **行情对比** section.

    When the batched quote call failed outright the table is replaced
    by a single block-quote line citing the upstream error. When only
    individual symbols are missing (e.g. an HK code that the upstream
    quietly dropped), those rows fall back to a "数据暂不可用" row so
    the surrounding rows still render.
    """

    heading = "## 行情对比"

    if quote_error is not None:
        return f"{heading}\n\n{_format_unavailable_inline(quote_error)}"

    rows: list[dict[str, str]] = []
    for sym in symbols:
        quote_dto = quotes_by_symbol.get(sym)
        if quote_dto is None:
            rows.append(
                {
                    "代码": sym,
                    "名称": NONE_PLACEHOLDER,
                    "现价": NONE_PLACEHOLDER,
                    "涨跌幅": NONE_PLACEHOLDER,
                    "PE(TTM)": NONE_PLACEHOLDER,
                    "PB": NONE_PLACEHOLDER,
                    "总市值": NONE_PLACEHOLDER,
                }
            )
            continue
        rows.append(_quote_to_row(quote_dto))

    table = render_table(rows, headers=_QUOTE_TABLE_HEADERS)
    return f"{heading}\n\n{table}"


def _render_valuation_section(
    service: FundamentalService,
    symbols: list[str],
) -> str:
    """Render the **估值横向对比** section.

    Each symbol gets its own labelled sub-table with four rows:
    ``PE(TTM)`` / ``PB`` / ``ROE`` / ``营收增速``. Per-symbol failures
    degrade gracefully per Requirement 10.5.
    """

    heading = "## 估值横向对比"
    blocks: list[str] = [heading]
    for sym in symbols:
        blocks.append(_render_valuation_block(service, sym))
    return "\n\n".join(blocks)


def _render_valuation_block(
    service: FundamentalService,
    symbol: str,
) -> str:
    """Render one 估值横向对比 sub-table for ``symbol``."""

    sub_heading = f"### {symbol}"
    try:
        snapshot = service.snapshot(symbol)
    except ChinaStockMCPError as exc:
        return f"{sub_heading}\n\n{_format_unavailable_inline(exc, prefix=symbol)}"

    rows: list[dict[str, str]] = []
    for bucket_attr, key, label in _VALUATION_METRICS:
        value = _get_metric(snapshot, bucket_attr, key)
        rows.append(
            {
                "指标": label,
                "数值": _format_metric_value(key, value),
            }
        )

    table = render_table(rows, headers=_VALUATION_TABLE_HEADERS)
    return f"{sub_heading}\n\n{table}"


def _render_industry_section(
    service: IndustryService,
    symbols: list[str],
) -> str:
    """Render the **行业横切** section.

    For each symbol, surface the industry name and the count of peers
    found (capped by ``_PEER_TOP_N``). Per-symbol failures degrade
    gracefully per Requirement 10.5.
    """

    heading = "## 行业横切"
    bullets: list[str] = []
    for sym in symbols:
        bullets.append(_render_industry_bullet(service, sym))

    return f"{heading}\n\n" + "\n".join(bullets)


def _render_industry_bullet(
    service: IndustryService,
    symbol: str,
) -> str:
    """Render one 行业横切 bullet line for ``symbol``."""

    try:
        peers: PeerTable = service.peers(
            symbol=symbol,
            metrics=list(_PEER_METRICS),
            top_n=_PEER_TOP_N,
        )
    except ChinaStockMCPError as exc:
        return _format_unavailable_inline(exc, prefix=symbol)

    return (
        f"- **{symbol}** 所属行业: {peers.industry}; "
        f"采样可比公司 {len(peers.rows)} 家 (top {_PEER_TOP_N})"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _quote_to_row(q: Quote) -> dict[str, str]:
    """Project a :class:`Quote` into the 行情对比 table columns."""

    return {
        "代码": q.symbol,
        "名称": q.name,
        "现价": f"{q.price:.2f}",
        "涨跌幅": render_change(q.change_pct),
        "PE(TTM)": _format_optional_ratio(q.pe_ttm),
        "PB": _format_optional_ratio(q.pb),
        "总市值": format_amount(q.market_cap),
    }


def _format_optional_ratio(value: float | None) -> str:
    """Render a 2-decimal ratio (PE / PB), or ``"-"`` when missing."""

    if value is None:
        return NONE_PLACEHOLDER
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return NONE_PLACEHOLDER
    fvalue = float(value)
    if math.isnan(fvalue):
        return NONE_PLACEHOLDER
    return f"{fvalue:.2f}"


def _get_metric(
    snapshot: FundamentalSnapshot,
    bucket_attr: str,
    key: str,
) -> float | None:
    """Look up ``snapshot.<bucket_attr>[key]`` defensively."""

    bucket = getattr(snapshot, bucket_attr, None)
    if not isinstance(bucket, dict):
        return None
    value = bucket.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return float(value)


def _format_metric_value(key: str, value: float | None) -> str:
    """Render a single 估值横向对比 cell with the appropriate unit."""

    if value is None:
        return NONE_PLACEHOLDER
    if math.isnan(value):
        return NONE_PLACEHOLDER
    if key in _PERCENT_KEYS:
        return format_percent(value)
    return f"{value:.2f}"


def _format_unavailable_inline(
    error: ChinaStockMCPError | None,
    *,
    prefix: str | None = None,
) -> str:
    """Render the Requirement 10.5 graceful-degradation block-quote.

    When ``prefix`` is supplied (the per-symbol case) the line reads
    ``> ⚠️ {prefix} 数据不可用: {message}``; otherwise the generic
    ``> ⚠️ 数据不可用: {message}`` form is used. Any newlines in the
    upstream message are collapsed to spaces so the rendered block
    stays a single Markdown block-quote line.
    """

    message = (
        error.to_user_message() if error is not None else "未知错误"
    )
    flat = " ".join(line.strip() for line in message.splitlines() if line.strip())
    if prefix is not None:
        return f"> ⚠️ {prefix} 数据不可用: {flat}"
    return f"> ⚠️ 数据不可用: {flat}"


__all__ = ["ValuationCompareInput", "valuation_compare"]
