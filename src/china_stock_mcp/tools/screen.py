"""``screen_stocks`` tool -- 多因子选股 Markdown 渲染.

Implements *Component 2 (Tools Layer)* from ``design.md`` for the
``screen_stocks`` entry. The tool layer is intentionally thin:

1. Validate caller input via :class:`ScreenStocksInput` (pydantic
   v2), bridging any pydantic ``ValidationError`` to the unified
   :class:`china_stock_mcp.exceptions.ValidationError`
   (Requirement 13.7). The schema enforces:

   - ``criteria`` is an open dict (passed through to
     :class:`ScreenCriteria`; unknown keys become extras so design
     can extend the schema without breaking older callers).
   - ``sort_by`` ∈ ``{"pe_ttm", "pb", "roe", "market_cap",
     "revenue_growth"}`` (Requirement 8.5).
   - ``order`` ∈ ``{"asc", "desc"}`` (Requirement 8.3).
   - ``limit`` ∈ ``[1, 200]`` (Requirement 8.2).

2. Build the :class:`ScreenCriteria` model from the caller's dict;
   invalid range filters surface as :class:`ValidationError` with
   a descriptive message.

3. Delegate to :meth:`ScreenService.filter` which performs the
   universe build → range filter → stable sort → truncate pipeline
   from design Algorithm 5.

4. Render a Markdown header (``len(hits)/limit``) plus a table whose
   columns reflect the criteria the caller actually filtered by
   (Requirement 8.6) -- 代码 / 名称 / 行业 + each criterion field.

5. Append the canonical disclaimer via :func:`append_disclaimer`
   (Requirement 12.1, Property 14).
"""

from __future__ import annotations

import math
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from china_stock_mcp.error_mapping import bridge_pydantic_error, format_pydantic_error
from china_stock_mcp.exceptions import ValidationError
from china_stock_mcp.formatters import (
    NONE_PLACEHOLDER,
    finalize_tool_output,
    format_amount,
    format_percent,
    render_table,
)
from china_stock_mcp.models import ScreenCriteria, ScreenHit
from china_stock_mcp.services.screen_service import (
    SUPPORTED_FIELDS,
    ScreenService,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: ``limit`` lower / upper bounds (Requirement 8.2). Mirrored at the
#: tool boundary so pydantic surfaces a per-field error before the
#: service layer is reached.
_MIN_LIMIT: Final[int] = 1
_MAX_LIMIT: Final[int] = 200

#: Default ``limit`` value -- 30 keeps the rendered table well within
#: the per-tool 3 000-token budget enforced upstream by the formatter
#: layer.
_DEFAULT_LIMIT: Final[int] = 30

#: Localized labels for each criterion field. Surface order in the
#: rendered table follows the order in which the caller specified
#: each criterion so the user sees the columns they asked for.
_FIELD_LABELS: Final[dict[str, str]] = {
    "pe_ttm": "市盈率(TTM)",
    "pb": "市净率",
    "roe": "净资产收益率",
    "market_cap": "总市值",
    "revenue_growth": "营收增速",
}

#: Fields rendered as 元-denominated 亿/万 amounts (via :func:`format_amount`).
_AMOUNT_FIELDS: Final[frozenset[str]] = frozenset({"market_cap"})

#: Fields rendered as percentages with 2 decimals (via :func:`format_percent`).
_PERCENT_FIELDS: Final[frozenset[str]] = frozenset(
    {"roe", "revenue_growth"}
)


# ---------------------------------------------------------------------------
# Pydantic input model
# ---------------------------------------------------------------------------


class ScreenStocksInput(BaseModel):
    """Pydantic v2 input schema for :func:`screen_stocks`.

    Constraints:

    - ``criteria`` is an open mapping; the empty dict is valid (returns
      the unconstrained universe truncated to ``limit``).
    - ``sort_by`` must be one of the supported field names.
    - ``order`` is ``"asc"`` or ``"desc"`` (defaults to ``"desc"``).
    - ``limit`` is bounded to ``[1, 200]`` (defaults to ``30``).
    - ``extra="forbid"`` rejects unknown top-level keys.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    criteria: dict[str, Any] = Field(default_factory=dict)
    sort_by: Literal[
        "pe_ttm", "pb", "roe", "market_cap", "revenue_growth"
    ] = "market_cap"
    order: Literal["asc", "desc"] = "desc"
    limit: int = Field(_DEFAULT_LIMIT, ge=_MIN_LIMIT, le=_MAX_LIMIT)


# ---------------------------------------------------------------------------
# Public tool entry
# ---------------------------------------------------------------------------


def screen_stocks(
    service: ScreenService,
    criteria: dict[str, Any] | None = None,
    sort_by: str = "market_cap",
    order: str = "desc",
    limit: int = _DEFAULT_LIMIT,
) -> str:
    """Return a Markdown 选股结果 table for the given criteria.

    Parameters
    ----------
    service:
        Pre-wired :class:`ScreenService` instance.
    criteria:
        Mapping of criterion name → value. Each criterion lives at
        either ``{min, max}`` (for range filters such as ``pe_ttm`` /
        ``pb`` / ``roe`` / ``market_cap`` / ``revenue_growth``) or a
        list of strings (for ``industry``). Defaults to an empty
        mapping.
    sort_by:
        One of ``"pe_ttm"`` / ``"pb"`` / ``"roe"`` / ``"market_cap"``
        / ``"revenue_growth"``. Defaults to ``"market_cap"``.
    order:
        Either ``"asc"`` or ``"desc"``. Defaults to ``"desc"``.
    limit:
        Maximum number of rows, ``[1, 200]``. Defaults to ``30``.

    Returns
    -------
    str
        Markdown document ending with the standard disclaimer.

    Raises
    ------
    ValidationError
        If any input fails Pydantic validation, or if the service
        layer rejects an out-of-range value.
    ChinaStockMCPError
        Any other subclass raised by the service / adapter layer is
        propagated verbatim (Requirement 13.1).
    """

    # 1) Tool-boundary input validation -- bridge pydantic to the
    #    unified error tree (Requirement 13.7).
    try:
        validated = ScreenStocksInput(
            criteria=dict(criteria) if criteria is not None else {},
            sort_by=sort_by,
            order=order,
            limit=limit,
        )
    except PydanticValidationError as exc:
        raise bridge_pydantic_error(exc) from exc

    # 2) Build the criteria DTO.
    try:
        criteria_dto = ScreenCriteria.model_validate(validated.criteria)
    except PydanticValidationError as exc:
        raise ValidationError(
            f"criteria 校验失败: {format_pydantic_error(exc)}"
        ) from exc

    # 3) Service call -- universe build, filter, sort and truncate
    #    happen inside the service.
    hits = service.filter(
        criteria=criteria_dto,
        sort_by=validated.sort_by,
        order=validated.order,
        limit=validated.limit,
    )

    # 4) Render Markdown.
    body = _render_hits(
        hits,
        criteria=criteria_dto,
        sort_by=validated.sort_by,
        order=validated.order,
        limit=validated.limit,
    )

    # 5) Unified tool-exit pipeline (Property 13 / 14, Requirement 12.1 / 12.2).
    return finalize_tool_output(body)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_hits(
    hits: list[ScreenHit],
    *,
    criteria: ScreenCriteria,
    sort_by: str,
    order: str,
    limit: int,
) -> str:
    """Build the Markdown body for a screen result list."""

    title = (
        f"### 选股结果 ({len(hits)}/{limit} 条, "
        f"按 {sort_by} {order})"
    )

    if not hits:
        return f"{title}\n\n_未找到符合条件的标的_"

    # Build the column list. 代码 / 名称 / 行业 are always present;
    # criterion fields render in the order the caller specified them
    # so the rendered table reflects the user's mental model
    # (Requirement 8.6).
    field_columns = _ordered_criterion_fields(criteria)
    # Ensure ``sort_by`` is visible in the table even when the caller
    # did not filter by it -- otherwise users cannot inspect the
    # value the result was sorted by.
    if sort_by in SUPPORTED_FIELDS and sort_by not in field_columns:
        field_columns = [*field_columns, sort_by]

    headers: list[str] = ["代码", "名称", "行业"]
    for field in field_columns:
        headers.append(_FIELD_LABELS.get(field, field))

    rendered_rows: list[dict[str, str]] = []
    for hit in hits:
        row: dict[str, str] = {
            "代码": hit.code,
            "名称": hit.name,
            "行业": hit.industry or NONE_PLACEHOLDER,
        }
        for field in field_columns:
            label = _FIELD_LABELS.get(field, field)
            row[label] = _format_field_value(field, hit.fields.get(field))
        rendered_rows.append(row)

    table = render_table(rendered_rows, headers=headers)
    return f"{title}\n\n{table}"


def _ordered_criterion_fields(criteria: ScreenCriteria) -> list[str]:
    """Return the criterion field names that are *active* on ``criteria``.

    Order follows the canonical declaration order of
    :class:`ScreenCriteria` so the Markdown table reads predictably
    regardless of how the caller serialized the input dict.
    """

    fields: list[str] = []
    for field in ("pe_ttm", "pb", "roe", "market_cap", "revenue_growth"):
        rng = getattr(criteria, field)
        if rng is not None:
            fields.append(field)
    return fields


def _format_field_value(field: str, value: float | None) -> str:
    """Render a single criterion cell with the appropriate unit.

    - ``market_cap`` is rendered via :func:`format_amount` (亿/万).
    - ``roe`` / ``revenue_growth`` are percent-denominated and go
      through :func:`format_percent`.
    - Everything else (``pe_ttm`` / ``pb``) is rendered as a 2-decimal
      float.
    - ``None`` and NaN map to :data:`NONE_PLACEHOLDER`.
    """

    if value is None:
        return NONE_PLACEHOLDER
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return NONE_PLACEHOLDER
    if isinstance(value, float) and math.isnan(value):
        return NONE_PLACEHOLDER

    if field in _AMOUNT_FIELDS:
        return format_amount(float(value))
    if field in _PERCENT_FIELDS:
        return format_percent(float(value))
    return f"{float(value):.2f}"


__all__ = ["ScreenStocksInput", "screen_stocks"]
