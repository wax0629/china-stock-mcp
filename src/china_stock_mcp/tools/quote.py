"""``get_quote`` tool -- single or batched 行情快照 Markdown rendering.

Implements *Component 2 (Tools Layer)* from ``design.md`` for the
``get_quote`` entry. The tool layer is intentionally thin:

1. Validate the caller's input via :class:`GetQuoteInput` (pydantic v2).
2. Delegate to :meth:`QuoteService.get_snapshot` for normalization,
   caching, rate-limiting and adapter fallback.
3. Render either a single Markdown card (one symbol) or a multi-column
   pipe table (multiple symbols) using helpers from
   :mod:`china_stock_mcp.formatters`.
4. Optionally prepend "数据延迟约 15 分钟" when ``CSM_DATA_DELAY_NOTICE``
   is enabled, then append the standard disclaimer.

Acceptance criteria covered
---------------------------

- Requirements 2.1 -- single-symbol Markdown card with full quote fields.
- Requirements 2.2 -- list input renders one row per symbol; ordering and
  duplicates are preserved by :class:`QuoteService`.
- Requirements 2.6 -- ``CSM_DATA_DELAY_NOTICE`` toggles the
  "数据延迟约 15 分钟" header.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel as _PydanticBaseModel
from pydantic import ConfigDict, field_validator
from pydantic import ValidationError as PydanticValidationError

from china_stock_mcp.config import Settings
from china_stock_mcp.error_mapping import bridge_pydantic_error
from china_stock_mcp.formatters import (
    DELAY_NOTICE_LINE,
    NONE_PLACEHOLDER,
    finalize_tool_output,
    format_amount,
    render_change,
    render_quote,
    render_table,
)
from china_stock_mcp.models import Quote
from china_stock_mcp.services.quote_service import QuoteService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum number of symbols accepted in a single ``get_quote`` call.
#: Mirrors :attr:`QuoteService.MAX_BATCH_SIZE` so the input model can
#: reject obviously over-sized batches before any service work happens.
_MAX_SYMBOLS: Final[int] = QuoteService.MAX_BATCH_SIZE

#: Header row for the multi-symbol table view (Requirement 2.2).
_TABLE_HEADERS: Final[list[str]] = [
    "代码",
    "名称",
    "现价",
    "涨跌幅",
    "成交额",
    "总市值",
    "PE(TTM)",
    "PB",
]

#: Re-export of :data:`china_stock_mcp.formatters.DELAY_NOTICE_LINE`.
#: Existing tests import this constant from ``tools.quote`` so the
#: alias is kept for backwards compatibility while the canonical
#: definition now lives in :mod:`formatters` (task 23.1).
_DELAY_NOTICE_LINE: Final[str] = DELAY_NOTICE_LINE


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------


class GetQuoteInput(_PydanticBaseModel):
    """Pydantic v2 input model for :func:`get_quote`.

    Constraints:

    - ``symbol`` accepts a single string or a list of strings.
    - When a list is provided it must contain ``1..20`` entries
      (Requirement 2.3).
    - ``extra="forbid"`` rejects unknown keys so typos are caught early.
    - ``str_strip_whitespace=True`` trims surrounding whitespace from
      each entry; ``normalize_symbol`` performs the rest of the
      canonicalization downstream.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    symbol: str | list[str]

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str | list[str]) -> str | list[str]:
        """Reject empty strings, empty lists, and over-sized batches."""

        if isinstance(value, str):
            if value.strip() == "":
                raise ValueError("symbol 不能为空字符串")
            return value

        if not isinstance(value, list):  # pragma: no cover - pydantic guards
            raise ValueError("symbol 必须是字符串或字符串列表")

        if len(value) == 0:
            raise ValueError("symbol 列表不能为空")
        if len(value) > _MAX_SYMBOLS:
            raise ValueError(
                f"单次请求最多 {_MAX_SYMBOLS} 个标的, 实际收到 {len(value)} 个"
            )
        for idx, item in enumerate(value):
            if not isinstance(item, str) or item.strip() == "":
                raise ValueError(
                    f"symbol[{idx}] 必须是非空字符串, 实际为 {item!r}"
                )
        return value


# ---------------------------------------------------------------------------
# Public tool entry
# ---------------------------------------------------------------------------


def get_quote(
    service: QuoteService,
    symbol: str | list[str],
    *,
    settings: Settings | None = None,
) -> str:
    """Return a Markdown snapshot for ``symbol``.

    Pipeline:

    1. Validate ``symbol`` via :class:`GetQuoteInput`. Pydantic failures
       are re-raised as :class:`china_stock_mcp.exceptions.ValidationError`
       so the unified error tree from Requirement 13.7 is preserved.
    2. Call :meth:`QuoteService.get_snapshot`, which performs symbol
       normalization, per-symbol caching, rate-limit admission and
       adapter fallback.
    3. Render a single :class:`Quote` as a card via
       :func:`render_quote`, or a multi-symbol pipe table via
       :func:`render_table`.
    4. When ``settings.data_delay_notice`` is ``True`` (default
       behaviour per :data:`DEFAULT_DATA_DELAY_NOTICE`), prepend the
       blockquote notice line.
    5. Append the canonical disclaimer via :func:`append_disclaimer`.

    Parameters
    ----------
    service:
        The :class:`QuoteService` instance composed at server startup.
    symbol:
        A single symbol string or a list of symbol strings (1..20).
        Whitespace is stripped and the strings are passed verbatim to
        :func:`normalize_symbol` inside the service.
    settings:
        Optional :class:`Settings` injection for tests. Production
        callers omit this argument and the function reads the current
        environment via :func:`load_settings`.

    Raises
    ------
    ValidationError
        If ``symbol`` is empty, contains empty entries, or has more
        than 20 entries.
    ChinaStockMCPError
        Any other domain failure raised by the service layer is
        propagated unchanged (e.g. :class:`SymbolError`,
        :class:`DataNotFoundError`, :class:`RateLimitError`).
    """

    # 1) Input validation (Pydantic -> domain exception bridge).
    try:
        validated = GetQuoteInput(symbol=symbol)
    except PydanticValidationError as exc:
        raise bridge_pydantic_error(exc) from exc

    # 2) Service call -- ordering & duplicates are preserved by the
    #    service so the caller-visible Markdown reflects the input.
    quotes: list[Quote] = service.get_snapshot(validated.symbol)

    # 3) Render body.
    body = (
        render_quote(quotes[0])
        if len(quotes) == 1
        else _render_quote_table(quotes)
    )

    # 4) Unified tool-exit pipeline (task 23.1):
    #    - Optional 数据延迟 banner (Requirement 2.6)
    #    - Token budget enforcement (Property 13 / Requirement 12.2)
    #    - Disclaimer (Property 14 / Requirement 12.1)
    return finalize_tool_output(
        body,
        settings=settings,
        delay_notice=True,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_quote_table(quotes: list[Quote]) -> str:
    """Render multiple quotes as a Markdown pipe table.

    The table is preceded by a level-3 heading announcing the row count
    so AI clients can tell the multi-symbol path was taken even when
    only two symbols were requested.
    """

    heading = f"### 行情快照 ({len(quotes)} 只)"
    rows: list[dict[str, str]] = [_quote_to_row(q) for q in quotes]
    table = render_table(rows, headers=_TABLE_HEADERS)
    return f"{heading}\n\n{table}"


def _quote_to_row(q: Quote) -> dict[str, str]:
    """Project a :class:`Quote` into the multi-symbol table columns."""

    return {
        "代码": q.symbol,
        "名称": q.name,
        "现价": f"{q.price:.2f}",
        "涨跌幅": render_change(q.change_pct),
        "成交额": format_amount(q.amount),
        "总市值": format_amount(q.market_cap),
        "PE(TTM)": _format_optional_ratio(q.pe_ttm),
        "PB": _format_optional_ratio(q.pb),
    }


def _format_optional_ratio(value: float | None) -> str:
    """Render a 2-decimal ratio (PE / PB), or ``"-"`` when missing.

    PE / PB are unitless multiples, so :func:`format_percent` (which
    appends ``%``) is *not* used here; the value is formatted directly
    with two decimal places.
    """

    if value is None:
        return NONE_PLACEHOLDER
    return f"{float(value):.2f}"


__all__ = ["GetQuoteInput", "get_quote"]
