"""``get_industry_peers`` tool -- 行业对比 Markdown 渲染.

Implements *Component 2 (Tools Layer)* from ``design.md`` for the
``get_industry_peers`` entry. The tool layer is intentionally thin:

1. Validate caller input via :class:`GetIndustryPeersInput`
   (pydantic v2), bridging any pydantic ``ValidationError`` to the
   unified :class:`china_stock_mcp.exceptions.ValidationError`
   (Requirement 13.7). The schema enforces ``metrics`` ⊆
   ``{pe, pb, roe, revenue_growth}`` and ``top_n ∈ [1, 50]``
   (Requirements 6.2 / 6.4).
2. Delegate to :meth:`IndustryService.peers`, which performs symbol
   normalization, caching, rate-limit admission, adapter fallback,
   and the per-row 行业分位 annotation.
3. Render a Markdown header (with industry + top_n metadata), a
   metric-keyed table whose monetary / percent / dimensionless cells
   are formatted appropriately, and a per-metric 行业分位 footnote
   describing the percentile column meaning (Requirement 6.3).
4. Append the canonical disclaimer via :func:`append_disclaimer`
   (Requirement 12.1, Property 14).

Acceptance criteria covered
---------------------------

- Requirement 6.1 -- 行业 + top_n 同行业可比公司 Markdown 表.
- Requirement 6.2 -- ``metrics`` ⊆ ``{pe, pb, roe, revenue_growth}``.
- Requirement 6.3 -- 每个数值列附 "行业分位" 说明.
- Requirement 6.4 -- ``top_n`` ∈ ``[1, 50]``.
- Requirement 6.5 -- 不支持的 ``metric`` 抛 :class:`ValidationError`,
  错误消息列出全部支持的 metric 名 (服务层负责).
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
    format_percent,
    render_table,
)
from china_stock_mcp.models import PeerTable
from china_stock_mcp.services.industry_service import IndustryService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: ``top_n`` lower / upper bounds (Requirement 6.4). Mirrors the
#: service-layer bound so pydantic surfaces a per-field error at the
#: tool boundary too.
_MIN_TOP_N: Final[int] = 1
_MAX_TOP_N: Final[int] = 50

#: Default ``top_n`` value -- 10 keeps the rendered table compact and
#: well within the per-tool 3 000-token budget while still showing a
#: useful slice of the industry universe.
_DEFAULT_TOP_N: Final[int] = 10

#: Localized labels for each metric. Surface order in the rendered
#: table follows the caller-supplied ``metrics`` order so the user
#: gets the columns they asked for in the order they asked.
_METRIC_LABELS: Final[dict[str, str]] = {
    "pe": "市盈率(动态)",
    "pb": "市净率",
    "roe": "净资产收益率",
    "revenue_growth": "营收同比",
}

#: Metrics whose value is denominated in *percent* and should be
#: rendered with :func:`format_percent` (which appends ``%``). Every
#: other metric is rendered as a 2-decimal float.
_PERCENT_METRICS: Final[frozenset[str]] = frozenset(
    {"roe", "revenue_growth"}
)

#: Suffix applied by :class:`IndustryService` for percentile columns.
_PERCENTILE_SUFFIX: Final[str] = "_percentile"

#: Default metric set when the caller does not specify one. Typed as
#: a tuple of ``Literal`` values so the pydantic ``default_factory``
#: returns a ``list[Literal[...]]`` matching the field annotation
#: under ``mypy --strict``.
_DefaultMetric = Literal["pe", "pb", "roe", "revenue_growth"]
_DEFAULT_METRICS: Final[tuple[_DefaultMetric, ...]] = (
    "pe",
    "pb",
    "roe",
    "revenue_growth",
)


# ---------------------------------------------------------------------------
# Pydantic input model
# ---------------------------------------------------------------------------


class GetIndustryPeersInput(BaseModel):
    """Pydantic v2 input schema for :func:`get_industry_peers`.

    Constraints:

    - ``symbol`` is required and bounded to ``[1, 64]`` characters.
    - ``metrics`` is a list of one or more values drawn from
      ``{pe, pb, roe, revenue_growth}`` (Requirement 6.2). Defaults to
      all four supported metrics when omitted.
    - ``top_n`` is bounded to ``[1, 50]`` (Requirement 6.4) and
      defaults to ``10``.
    - ``extra="forbid"`` rejects unknown keys.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    symbol: str = Field(..., min_length=1, max_length=64)
    metrics: list[Literal["pe", "pb", "roe", "revenue_growth"]] = Field(
        default_factory=lambda: list(_DEFAULT_METRICS),
        min_length=1,
    )
    top_n: int = Field(_DEFAULT_TOP_N, ge=_MIN_TOP_N, le=_MAX_TOP_N)


# ---------------------------------------------------------------------------
# Public tool entry
# ---------------------------------------------------------------------------


def get_industry_peers(
    service: IndustryService,
    symbol: str,
    metrics: list[str] | None = None,
    top_n: int = _DEFAULT_TOP_N,
) -> str:
    """Return a Markdown 同行业可比 table for ``symbol``.

    Parameters
    ----------
    service:
        Pre-wired :class:`IndustryService` instance.
    symbol:
        Standardized or bare A-share symbol; whitespace is stripped
        before validation. Normalization happens inside the service.
    metrics:
        Optional list of metric names; supported names are
        ``"pe"`` / ``"pb"`` / ``"roe"`` / ``"revenue_growth"``.
        Defaults to all four when ``None``.
    top_n:
        Maximum number of peer rows to render, ``[1, 50]``. Defaults
        to ``10``.

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
        If the symbol's industry cannot be resolved or the industry
        has no constituents.
    ChinaStockMCPError
        Any other subclass raised by the service / adapter layer is
        propagated verbatim (Requirement 13.1).
    """

    # 1) Input validation -- pydantic failures bridge to the unified
    #    error tree.
    try:
        validated = GetIndustryPeersInput(
            symbol=symbol,
            metrics=metrics if metrics is not None else list(_DEFAULT_METRICS),
            top_n=top_n,
        )
    except PydanticValidationError as exc:
        raise bridge_pydantic_error(exc) from exc

    # 2) Service call -- all heavy lifting (cache + rate-limit +
    #    fallback + percentile annotation) happens here.
    peers: PeerTable = service.peers(
        symbol=validated.symbol,
        metrics=list(validated.metrics),
        top_n=validated.top_n,
    )

    # 3) Render Markdown.
    body = _render_peers(peers, requested_top_n=validated.top_n)

    # 4) Unified tool-exit pipeline (Property 13 / 14, Requirement 12.1 / 12.2).
    return finalize_tool_output(body)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_peers(peers: PeerTable, requested_top_n: int) -> str:
    """Build the Markdown body for a :class:`PeerTable`."""

    title = (
        f"### 行业对比 ({peers.base_symbol}, 行业: {peers.industry}, "
        f"top {requested_top_n})"
    )
    actual = len(peers.rows)

    if actual == 0:
        return f"{title}\n\n_未找到行业可比数据_"

    # Build a table whose columns reflect the caller-supplied ``metrics``
    # order, prefixed by 代码 / 名称.
    headers: list[str] = ["代码", "名称"]
    for metric in peers.metrics:
        headers.append(_METRIC_LABELS.get(metric, metric))

    rendered_rows: list[dict[str, str]] = []
    for row in peers.rows:
        rendered: dict[str, str] = {
            "代码": str(row.get("代码", "")),
            "名称": str(row.get("名称", "")),
        }
        for metric in peers.metrics:
            rendered[_METRIC_LABELS.get(metric, metric)] = _format_metric_value(
                metric, row.get(metric)
            )
        rendered_rows.append(rendered)

    table = render_table(rendered_rows, headers=headers)

    # Build per-metric 行业分位 footnote describing the percentile
    # column meaning + the symbol's own rank (Requirement 6.3). When
    # the symbol is one of the rows, surface its percentile directly;
    # when the symbol is *not* in the row set (e.g. it ranks below
    # ``top_n``), describe the column meaning generically.
    base_row = _find_row(peers.rows, peers.base_symbol)
    notes: list[str] = []
    for metric in peers.metrics:
        notes.append(_render_percentile_note(metric, peers, base_row, actual))

    body_parts: list[str] = [title, "", table]
    if notes:
        body_parts.append("")
        body_parts.extend(notes)
    return "\n".join(body_parts)


def _find_row(
    rows: list[dict[str, object]],
    base_symbol: str,
) -> dict[str, object] | None:
    """Return the row whose ``代码`` matches the bare ``base_symbol``."""

    bare = base_symbol.split(".")[0]
    for row in rows:
        code = str(row.get("代码", "")).strip()
        if code == bare:
            return row
    return None


def _render_percentile_note(
    metric: str,
    peers: PeerTable,
    base_row: dict[str, object] | None,
    total_rows: int,
) -> str:
    """Render a single per-metric 行业分位 footnote line."""

    label = _METRIC_LABELS.get(metric, metric)

    # Count rows with a non-null numeric value for this metric so the
    # note matches the percentile denominator.
    numeric_count = sum(
        1
        for row in peers.rows
        if isinstance(row.get(metric), (int, float))
        and not isinstance(row.get(metric), bool)
        and not _is_nan(row.get(metric))
    )

    if numeric_count == 0:
        return (
            f"> {label} 行业分位: 该指标在行业 {total_rows} 家公司中"
            f"暂无可比数值"
        )

    if base_row is None:
        return (
            f"> {label} 行业分位: 数值越大分位越高 "
            f"(基于 {numeric_count}/{total_rows} 家公司, 0~100 分制)"
        )

    raw_value = base_row.get(metric)
    percentile = base_row.get(f"{metric}{_PERCENTILE_SUFFIX}")
    if (
        not isinstance(raw_value, (int, float))
        or isinstance(raw_value, bool)
        or _is_nan(raw_value)
        or not isinstance(percentile, (int, float))
        or isinstance(percentile, bool)
        or _is_nan(percentile)
    ):
        return (
            f"> {label} 行业分位: {peers.base_symbol} 暂无该指标数值, "
            f"基于 {numeric_count}/{total_rows} 家公司"
        )

    return (
        f"> {label} 行业分位: {peers.base_symbol} 在行业 "
        f"{numeric_count} 家公司中位于 {float(percentile):.1f} 分位 "
        f"(数值越大分位越高)"
    )


def _format_metric_value(metric: str, value: object) -> str:
    """Render a single metric cell with the appropriate unit.

    - Percent-denominated metrics (``roe`` / ``revenue_growth``) go
      through :func:`format_percent` (appends ``%``).
    - Every other metric (PE / PB) renders as a 2-decimal float.
    - ``None`` and NaN map to :data:`NONE_PLACEHOLDER`.
    """

    if value is None:
        return NONE_PLACEHOLDER
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return NONE_PLACEHOLDER
    if isinstance(value, float) and math.isnan(value):
        return NONE_PLACEHOLDER

    if metric in _PERCENT_METRICS:
        return format_percent(float(value))
    return f"{float(value):.2f}"


def _is_nan(value: Any) -> bool:
    """Return ``True`` for float NaN; integers / non-floats are not NaN."""

    return isinstance(value, float) and math.isnan(value)


__all__ = ["GetIndustryPeersInput", "get_industry_peers"]
