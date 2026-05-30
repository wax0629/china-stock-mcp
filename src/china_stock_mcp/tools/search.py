"""``search_symbol`` tool — Markdown 标的搜索结果渲染。

Implements the ``search_symbol`` entrypoint from *design.md* Component 2
(Tools Layer). The tool is a thin shell around
:meth:`SymbolService.search`:

1. Validate inputs with a Pydantic v2 model (extra="forbid", strip
   whitespace) so the protocol layer never propagates invalid kwargs.
2. Convert any pydantic ``ValidationError`` into the unified
   :class:`china_stock_mcp.exceptions.ValidationError` with a message
   that names the offending field, the expected constraint, and the
   actual value (Requirements 13.7).
3. Call :meth:`SymbolService.search`, which already enforces the
   ``market`` whitelist and routes through cache + rate-limit + the
   primary/fallback adapter pair.
4. Render the resulting :class:`SymbolHit` list as a Markdown table
   with localized market labels, mapping ``industry=None`` to the
   shared ``-`` placeholder via :func:`formatters.render_table`.
5. Append the standard disclaimer through
   :func:`formatters.append_disclaimer` (Requirements 12.1, Property
   14) so the tool's output ends with the canonical text.

Acceptance criteria covered
---------------------------

- 1.1  Markdown 表格输出(代码 / 名称 / 市场 / 可选行业)
- 1.7  ``market`` 范围过滤透传至 service 层
- 12.1 末尾自动追加 Disclaimer
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from china_stock_mcp.error_mapping import bridge_pydantic_error
from china_stock_mcp.formatters import finalize_tool_output, render_table
from china_stock_mcp.services.symbol_service import SymbolService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Markdown table headers (design Component 2 §search_symbol).
_HEADERS: Final[list[str]] = ["代码", "名称", "市场", "行业"]

#: Localized market labels surfaced to MCP clients. Keys mirror the
#: ``market`` literal accepted by :class:`SearchSymbolInput` minus the
#: ``"all"`` sentinel — every :class:`SymbolHit` carries one of the
#: three concrete markets (Requirements 1.1).
_MARKET_LABELS: Final[dict[str, str]] = {
    "a_stock": "A股",
    "hk_stock": "港股",
    "fund": "公募基金",
}


# ---------------------------------------------------------------------------
# Pydantic input model
# ---------------------------------------------------------------------------


class SearchSymbolInput(BaseModel):
    """Pydantic v2 input schema for :func:`search_symbol`.

    The schema mirrors the function signature documented in *design.md*
    Component 2 and is what FastMCP advertises to MCP clients. Unknown
    keys are rejected (``extra="forbid"``) so a typo cannot silently
    short-circuit validation, and string fields are stripped to keep
    cache keys stable.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(..., min_length=1, max_length=64)
    market: Literal["a_stock", "hk_stock", "fund", "all"] = "all"


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def search_symbol(service: SymbolService, query: str, market: str = "all") -> str:
    """Search standardized symbols and render a Markdown table.

    Parameters
    ----------
    service:
        Pre-wired :class:`SymbolService` instance. Tests inject a
        service backed by stub adapters; production code wires the
        process-wide singleton in :mod:`server`.
    query:
        Free-text query (code / Chinese name / pinyin). The Service
        layer canonicalizes the value for cache-key purposes so the
        original spelling is forwarded to the adapter unchanged.
    market:
        ``"a_stock"`` / ``"hk_stock"`` / ``"fund"`` / ``"all"``.

    Returns
    -------
    str
        Markdown document ending with the standard disclaimer.

    Raises
    ------
    ValidationError
        If ``query`` / ``market`` fail Pydantic validation. The error
        message lists the offending field, the expected constraint,
        and the actual value (Requirements 13.7).
    ChinaStockMCPError
        Any subclass raised by the Service layer is propagated
        verbatim (Requirements 13.1).
    """

    # 1) Validate inputs and translate any pydantic failure into the
    #    unified ChinaStockMCPError hierarchy.
    try:
        validated = SearchSymbolInput(query=query, market=market)
    except PydanticValidationError as exc:
        raise bridge_pydantic_error(exc) from exc

    # 2) Delegate to the Service layer (cache + rate-limit + fallback
    #    composition lives there).
    hits = service.search(validated.query, validated.market)

    # 3) Build the response Markdown.
    title = f"### 标的搜索: '{validated.query}' (market={validated.market})"

    if not hits:
        body = f"{title}\n\n**未找到匹配的标的**"
        return finalize_tool_output(body)

    rows = [
        {
            "代码": hit.code,
            "名称": hit.name,
            "市场": _MARKET_LABELS.get(hit.market, hit.market),
            # ``render_table`` already maps None → "-", but list it
            # explicitly so the intent is obvious to readers.
            "行业": hit.industry,
        }
        for hit in hits
    ]
    table = render_table(rows, headers=_HEADERS)

    return finalize_tool_output(f"{title}\n\n{table}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["SearchSymbolInput", "search_symbol"]
