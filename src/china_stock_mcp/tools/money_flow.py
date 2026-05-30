"""``get_money_flow`` tool -- 资金流向 Markdown 渲染.

Implements *Component 2 (Tools Layer)* from ``design.md`` for the
``get_money_flow`` entry. The tool layer is intentionally thin:

1. Validate caller input via :class:`GetMoneyFlowInput`
   (pydantic v2), bridging any pydantic ``ValidationError`` to the
   unified :class:`china_stock_mcp.exceptions.ValidationError`
   (Requirements 13.7).
2. Delegate to :meth:`MoneyFlowService.get`, which performs symbol
   normalization (where applicable), caching, rate-limit admission
   and adapter fallback.
3. Render a Markdown header + ``snapshot_at`` line + dynamic table.
   Each ``flow_type`` carries a different row schema, so the column
   set is derived from the union of keys present in
   ``MoneyFlow.rows``; monetary cells render via
   :func:`format_amount` (亿 / 万 unit selection), percentage cells
   via :func:`format_percent`, dates as-is, and free-text fields
   verbatim.
4. Append the canonical disclaimer via :func:`append_disclaimer`
   (Requirements 12.1, Property 14).

Acceptance criteria covered
---------------------------

- Requirement 5.1 -- 北向资金净流入 + 排行 Markdown.
- Requirement 5.2 -- 主力 / 超大单 / 大单 / 中单 / 小单 (with symbol).
- Requirement 5.3 -- 龙虎榜买卖席位与金额.
- Requirement 5.4 -- ``top_n`` ∈ ``[1, 100]``; 大于 ``top_n`` 的行被截断.
- Requirement 5.5 -- 非法 ``flow_type`` 抛 :class:`ValidationError`.
- Requirement 5.6 -- ``MoneyFlow.snapshot_at`` 渲染至 Markdown 顶部.
- Requirement 12.1 / Property 14 -- 末尾追加免责声明.
"""

from __future__ import annotations

import math
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from china_stock_mcp.error_mapping import bridge_pydantic_error
from china_stock_mcp.formatters import (
    NONE_PLACEHOLDER,
    finalize_tool_output,
    format_amount,
    format_percent,
    render_table,
)
from china_stock_mcp.models import MoneyFlow
from china_stock_mcp.services.money_flow_service import MoneyFlowService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: ``top_n`` lower / upper bounds. Mirrors the service layer's
#: validation but exposed at the schema layer too so pydantic surfaces
#: a clearer per-field error message.
_MIN_TOP_N: Final[int] = 1
_MAX_TOP_N: Final[int] = 100

#: Default ``top_n`` value -- 20 fits comfortably within the per-tool
#: 3 000 token budget while still showing a useful slice.
_DEFAULT_TOP_N: Final[int] = 20

#: Localized labels for ``flow_type`` so the heading reads naturally.
_FLOW_TYPE_LABEL: Final[dict[str, str]] = {
    "north": "北向资金",
    "main": "主力资金",
    "dragon_tiger": "龙虎榜",
}

#: Substrings that mark a column as monetary (元-denominated). Cells
#: matching one of these tokens are rendered through
#: :func:`format_amount`. The order is irrelevant; matching is by
#: substring containment.
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


# ---------------------------------------------------------------------------
# Pydantic input model
# ---------------------------------------------------------------------------


class GetMoneyFlowInput(BaseModel):
    """Pydantic v2 input schema for :func:`get_money_flow`.

    Constraints:

    - ``symbol`` is optional (north flow ignores it; dragon_tiger
      treats ``None`` as the aggregate board); when provided it is
      bounded to ``[1, 64]`` characters.
    - ``flow_type`` is restricted to ``"north"`` / ``"main"`` /
      ``"dragon_tiger"`` (Requirement 5.5).
    - ``top_n`` is bounded to ``[1, 100]`` (Requirement 5.4) and
      defaults to ``20``.
    - ``extra="forbid"`` rejects unknown keys.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    symbol: str | None = Field(default=None, max_length=64)
    flow_type: Literal["north", "main", "dragon_tiger"] = "north"
    top_n: int = Field(_DEFAULT_TOP_N, ge=_MIN_TOP_N, le=_MAX_TOP_N)


# ---------------------------------------------------------------------------
# Public tool entry
# ---------------------------------------------------------------------------


def get_money_flow(
    service: MoneyFlowService,
    symbol: str | None = None,
    flow_type: str = "north",
    top_n: int = _DEFAULT_TOP_N,
) -> str:
    """Return a Markdown 资金流向 table for ``flow_type``.

    Parameters
    ----------
    service:
        Pre-wired :class:`MoneyFlowService` instance.
    symbol:
        Optional standardized or bare symbol. Required for ``main``
        flow; ignored for ``north``; optional for ``dragon_tiger``
        (omit for the aggregate board).
    flow_type:
        ``"north"`` for 北向资金 aggregate flow, ``"main"`` for 主力 /
        分单 flow on a specific symbol, ``"dragon_tiger"`` for 龙虎榜
        rows. Defaults to ``"north"``.
    top_n:
        Maximum number of rows to render, ``[1, 100]``. Defaults to
        ``20``.

    Returns
    -------
    str
        Markdown document ending with the standard disclaimer.

    Raises
    ------
    ValidationError
        If any input fails Pydantic validation, or if the service
        layer rejects an out-of-range value.
    DataNotFoundError
        If the upstream cannot satisfy the request (e.g. ``main``
        called without a symbol, or empty frame).
    ChinaStockMCPError
        Any other subclass raised by the service / adapter layer is
        propagated verbatim (Requirements 13.1).
    """

    # 1) Input validation -- pydantic failures bridge to the unified
    #    error tree.
    try:
        # ``symbol`` may be an empty string from the MCP transport;
        # treat it as omitted so the service applies the per-flow_type
        # rule without bouncing on min_length.
        cleaned_symbol = symbol if symbol and str(symbol).strip() else None
        validated = GetMoneyFlowInput(
            symbol=cleaned_symbol,
            flow_type=flow_type,
            top_n=top_n,
        )
    except PydanticValidationError as exc:
        raise bridge_pydantic_error(exc) from exc

    # 2) Service call -- all heavy lifting (cache + rate-limit +
    #    fallback + flow-type-specific symbol handling) happens here.
    flow: MoneyFlow = service.get(
        symbol=validated.symbol,
        flow_type=validated.flow_type,
        top_n=validated.top_n,
    )

    # 3) Render Markdown.
    body = _render_flow(flow, top_n=validated.top_n)

    # 4) Unified tool-exit pipeline (Property 13 / 14, Requirement 12.1 / 12.2).
    return finalize_tool_output(body)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_flow(flow: MoneyFlow, top_n: int) -> str:
    """Build the Markdown body for a :class:`MoneyFlow`."""

    type_label = _FLOW_TYPE_LABEL.get(flow.flow_type, flow.flow_type)
    actual = len(flow.rows)
    title = f"### 资金流向 ({type_label}, top {top_n})"
    snapshot_line = (
        f"> 数据时间: {flow.snapshot_at.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Reference the requested ``top_n`` so callers see both numbers in
    # the heading -- Requirement 5.4 caps the row count and the
    # service layer already enforced it but we reflect the authority.
    if actual == 0:
        return f"{title}\n\n{snapshot_line}\n\n_暂无可用资金流向数据_"

    # Derive headers from the union of keys, preserving the order in
    # which keys first appear. Each ``flow_type`` ships a stable row
    # schema, so the union typically equals every row's key set, but
    # taking the union is defensive against per-row holes.
    headers: list[str] = []
    seen: set[str] = set()
    for row in flow.rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            headers.append(key)

    rendered_rows: list[dict[str, str]] = [
        {key: _format_cell(key, row.get(key)) for key in headers}
        for row in flow.rows
    ]

    # Map the internal "date" key (north / main flows) to a friendlier
    # label so the rendered column header reads "日期".
    display_headers = [_display_header(h) for h in headers]
    if display_headers != headers:
        rendered_rows = [
            {
                _display_header(key): cell
                for key, cell in row.items()
            }
            for row in rendered_rows
        ]

    table = render_table(rendered_rows, headers=display_headers)
    return f"{title}\n\n{snapshot_line}\n\n{table}"


def _display_header(key: str) -> str:
    """Map an internal row key to its rendered Chinese column label."""

    if key == "date":
        return "日期"
    return key


def _format_cell(key: str, value: Any) -> str:
    """Render a single cell, picking units from the column key.

    - Monetary keys (``金额`` / ``净流入`` / ``买入`` / ...) go through
      :func:`format_amount` (亿 / 万 unit selection).
    - Percentage keys (``涨跌幅`` / ``换手率`` / ``占比``) go through
      :func:`format_percent`.
    - ``None`` and NaN map to :data:`NONE_PLACEHOLDER`.
    - Strings, dates and other free-form values render verbatim
      (with table-cell escaping handled by :func:`render_table`).
    """

    if value is None:
        return NONE_PLACEHOLDER
    if isinstance(value, float) and math.isnan(value):
        return NONE_PLACEHOLDER

    if isinstance(value, (int, float)):
        if any(token in key for token in _AMOUNT_KEY_TOKENS):
            return format_amount(float(value))
        if any(token in key for token in _PERCENT_KEY_TOKENS):
            return format_percent(float(value))
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    return str(value)


__all__ = ["GetMoneyFlowInput", "get_money_flow"]
