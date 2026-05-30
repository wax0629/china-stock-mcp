"""``symbol://{code}/profile`` resource -- 标的概览 Markdown.

Implements the ``symbol_profile`` MCP resource referenced by
*design.md Component 1* and task 21.1.

Unlike the other two resources (which alias an existing tool 1:1),
``symbol_profile`` composes two service calls into a compact
"single-page" snapshot:

1. :meth:`SymbolService.normalize` -- canonicalize the URI parameter
   so a caller can hit ``symbol://300750/profile`` /
   ``symbol://300750.SZ/profile`` / ``symbol://宁德时代/profile`` and
   land on the same resource (Requirement 1.4 / Property 1).
2. :meth:`SymbolService.search` -- fetch the search-hit metadata
   (中文名 / market / 行业) so the resource reads as a self-contained
   "what is this symbol" document. The first hit whose code matches
   the normalized symbol is preferred; otherwise the very first hit
   is used (a defensive fallback for upstreams that occasionally
   return an unordered list).
3. :meth:`FundamentalService.snapshot` -- pull the 估值 / 盈利
   buckets so the resource carries the most-frequently-asked metrics
   inline. ``DataNotFoundError`` from the snapshot call (e.g. HK or
   fund codes that the v1 fundamentals adapter does not cover) is
   handled gracefully: the basic-info section still renders and the
   metrics section is replaced with a "数据不可用" notice so the
   resource never returns a hard error just because the symbol lacks
   fundamentals data.

The function is pure with respect to its services bundle: it accepts
a ``services`` dict produced by the caller (typically
:mod:`china_stock_mcp.server`) and never reaches for module-level
state.
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
    format_percent,
    render_table,
)
from china_stock_mcp.models import FundamentalSnapshot, SymbolHit
from china_stock_mcp.services.fundamental_service import FundamentalService
from china_stock_mcp.services.symbol_service import SymbolService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Localized labels for the 估值 / 盈利 metrics surfaced by this
#: resource. The mapping mirrors the labels used by ``tools/fundamental``
#: so callers see consistent terminology across surfaces. Unknown keys
#: fall back to ``str(key)`` so future bucket additions degrade
#: gracefully.
_METRIC_LABELS: Final[dict[str, str]] = {
    "pe_ttm": "市盈率(TTM)",
    "pe_dynamic": "市盈率(动态)",
    "pb": "市净率",
    "ps": "市销率",
    "peg": "PEG",
    "roe": "净资产收益率",
    "roa": "总资产收益率",
    "gross_margin": "毛利率",
    "net_margin": "净利率",
}

#: Metrics whose value is denominated in *percent* and should be
#: rendered with :func:`format_percent` (which appends ``%``). Every
#: other metric renders as a 2-decimal float.
_PERCENT_METRICS: Final[frozenset[str]] = frozenset(
    {"roe", "roa", "gross_margin", "net_margin"}
)

#: Localized market labels surfaced in the basic-info section.
_MARKET_LABELS: Final[dict[str, str]] = {
    "a_stock": "A 股",
    "hk_stock": "港股",
    "fund": "公募基金",
}


class _Services(TypedDict):
    """Typed bundle of the service instances this resource needs."""

    symbol: SymbolService
    fundamental: FundamentalService


# ---------------------------------------------------------------------------
# Pydantic input model
# ---------------------------------------------------------------------------


class _SymbolProfileInput(BaseModel):
    """Pydantic v2 input schema for :func:`symbol_profile_resource`.

    Constraints:

    - ``symbol`` is required and bounded to ``[1, 64]`` characters.
    - ``extra="forbid"`` rejects unknown keys.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    symbol: str = Field(..., min_length=1, max_length=64)


# ---------------------------------------------------------------------------
# Public resource entry
# ---------------------------------------------------------------------------


def symbol_profile_resource(symbol: str, *, services: _Services) -> str:
    """Return a compact 标的概览 Markdown document for ``symbol``.

    Parameters
    ----------
    symbol:
        Standardized or bare symbol; whitespace is stripped before
        validation. Chinese names / pinyin are also accepted; the
        underlying :class:`SymbolService` normalizes them.
    services:
        Bundle that must contain pre-wired :class:`SymbolService` and
        :class:`FundamentalService` instances under the ``"symbol"``
        and ``"fundamental"`` keys.

    Returns
    -------
    str
        Markdown document with two sub-sections (基本信息 / 估值与盈利)
        and the standard disclaimer (Requirement 12.1, Property 14).

    Raises
    ------
    ValidationError
        If ``symbol`` fails Pydantic validation (e.g. empty string).
    ChinaStockMCPError
        Any subclass raised by :class:`SymbolService.normalize` is
        propagated verbatim (Requirement 13.1). Failures from the
        fundamentals snapshot are absorbed and surfaced as a
        "数据不可用" notice so the resource still produces a usable
        document.
    """

    # 1) Tool-boundary input validation -- bridges Pydantic failures to
    #    the unified error tree (Requirement 13.7).
    try:
        validated = _SymbolProfileInput(symbol=symbol)
    except PydanticValidationError as exc:
        raise bridge_pydantic_error(exc) from exc

    raw_symbol = validated.symbol
    symbol_service = services["symbol"]
    fundamental_service = services["fundamental"]

    # 2) Canonicalize the symbol. ``SymbolError`` from this call
    #    propagates verbatim -- if we cannot resolve the symbol we have
    #    nothing useful to render.
    std_symbol = symbol_service.normalize(raw_symbol)

    # 3) Look up the search-hit metadata. Resolution failures here are
    #    non-fatal: a usable resource is still possible from just the
    #    standardized code + fundamentals.
    hit = _resolve_search_hit(symbol_service, raw_symbol, std_symbol)

    # 4) Pull the fundamentals snapshot, degrading gracefully on
    #    upstream errors so HK / fund codes still produce a document.
    snapshot, snapshot_error = _resolve_snapshot(fundamental_service, std_symbol)

    # 5) Compose the Markdown document.
    body = _render_profile(std_symbol, hit, snapshot, snapshot_error)

    # 6) Disclaimer (Requirement 12.1, Property 14).
    return append_disclaimer(body)


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _resolve_search_hit(
    service: SymbolService,
    raw_symbol: str,
    std_symbol: str,
) -> SymbolHit | None:
    """Best-effort search lookup; returns ``None`` on any failure.

    The search is bounded to ``market="all"`` so a caller-provided
    Chinese name can match across markets. The first hit whose
    standardized code matches ``std_symbol`` wins; otherwise the
    very first hit is used (defensive fallback for adapters that
    occasionally return an unordered list).
    """

    try:
        hits = service.search(raw_symbol, market="all")
    except ChinaStockMCPError:
        return None

    if not hits:
        return None

    for candidate in hits:
        if candidate.code == std_symbol:
            return candidate
    return hits[0]


def _resolve_snapshot(
    service: FundamentalService,
    std_symbol: str,
) -> tuple[FundamentalSnapshot | None, str | None]:
    """Best-effort snapshot lookup; returns ``(snapshot, error_msg)``.

    On success returns ``(snapshot, None)``. On any
    :class:`ChinaStockMCPError` the snapshot is ``None`` and the
    formatted ``to_user_message()`` text is surfaced so the rendering
    layer can show a "数据不可用" notice instead of failing the
    resource outright.
    """

    try:
        return service.snapshot(std_symbol), None
    except ChinaStockMCPError as exc:
        return None, exc.to_user_message()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_profile(
    std_symbol: str,
    hit: SymbolHit | None,
    snapshot: FundamentalSnapshot | None,
    snapshot_error: str | None,
) -> str:
    """Build the Markdown body for a standardized symbol."""

    title = f"## 标的概览 ({std_symbol})"
    sections: list[str] = [title, _render_basic_info(std_symbol, hit)]
    sections.append(_render_metrics(snapshot, snapshot_error))
    return "\n\n".join(sections)


def _render_basic_info(std_symbol: str, hit: SymbolHit | None) -> str:
    """Render the 基本信息 sub-table.

    When the search hit is unavailable we still render a minimal table
    carrying the standardized code so the caller has at least one row
    to display.
    """

    sub_heading = "### 基本信息"

    if hit is None:
        rows = [
            {
                "代码": std_symbol,
                "名称": NONE_PLACEHOLDER,
                "市场": NONE_PLACEHOLDER,
                "行业": NONE_PLACEHOLDER,
            }
        ]
    else:
        rows = [
            {
                "代码": hit.code,
                "名称": hit.name,
                "市场": _MARKET_LABELS.get(hit.market, hit.market),
                "行业": hit.industry if hit.industry else NONE_PLACEHOLDER,
            }
        ]

    table = render_table(rows, headers=["代码", "名称", "市场", "行业"])
    return f"{sub_heading}\n\n{table}"


def _render_metrics(
    snapshot: FundamentalSnapshot | None,
    snapshot_error: str | None,
) -> str:
    """Render the 估值与盈利 sub-table.

    When the snapshot is unavailable a "数据不可用" line is emitted in
    place of the table so the resource degrades gracefully on HK / fund
    symbols (Requirement 13.4-style soft fallback at the resource
    boundary).
    """

    sub_heading = "### 估值与盈利"

    if snapshot is None:
        # Collapse newlines so the rendered notice stays a single
        # block-quote line; some Markdown renderers split ``> `` blocks
        # on embedded newlines.
        if snapshot_error:
            flat = " ".join(
                line.strip()
                for line in snapshot_error.splitlines()
                if line.strip()
            )
            notice = f"> ⚠️ 估值与盈利数据不可用：{flat}"  # noqa: RUF001
        else:
            notice = "> ⚠️ 估值与盈利数据不可用"
        return f"{sub_heading}\n\n{notice}"

    rows: list[dict[str, str]] = []
    for metric in ("pe_ttm", "pe_dynamic", "pb", "ps", "peg"):
        value = snapshot.valuation.get(metric)
        if value is None:
            continue
        rows.append(
            {
                "指标": _METRIC_LABELS.get(metric, metric),
                "数值": _format_metric_value(metric, value),
            }
        )
    for metric in ("roe", "roa", "gross_margin", "net_margin"):
        value = snapshot.profitability.get(metric)
        if value is None:
            continue
        rows.append(
            {
                "指标": _METRIC_LABELS.get(metric, metric),
                "数值": _format_metric_value(metric, value),
            }
        )

    if not rows:
        return f"{sub_heading}\n\n_数据暂无_"

    table = render_table(rows, headers=["指标", "数值"])
    return f"{sub_heading}\n\n{table}"


def _format_metric_value(metric: str, value: float | None) -> str:
    """Render a single metric value with the appropriate unit.

    - Percent-denominated metrics (ROE / margins / ROA) render via
      :func:`format_percent` (appends ``%``).
    - Every other metric renders as a 2-decimal float.
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


__all__ = ["symbol_profile_resource"]
