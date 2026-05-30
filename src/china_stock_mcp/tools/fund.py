"""``get_fund_info`` tool -- 公募基金信息 Markdown 渲染.

Implements *Component 2 (Tools Layer)* from ``design.md`` for the
``get_fund_info`` entry. The tool layer is intentionally thin:

1. Validate caller input via :class:`GetFundInfoInput` (pydantic v2),
   bridging any pydantic ``ValidationError`` to
   :class:`china_stock_mcp.exceptions.SymbolError` -- per
   Requirement 7.3, an invalid 6-digit code raises ``SymbolError``
   (not the generic ``ValidationError``).
2. Delegate to :meth:`FundService.info`, which performs the
   6-digit validation + caching + rate-limit admission + adapter
   fallback. The double-validation (here + service) is intentional:
   the tool surfaces a clean message at the protocol boundary and
   the service stays usable directly from prompts / tests.
3. Render two Markdown sub-tables -- 基本信息 (name, manager,
   inception date, AUM, returns, drawdown, sharpe, rank) and
   前十大持仓 (代码 / 名称 / 权重) -- with optional 行业分布 when
   the upstream populates it. Missing optional fields surface as
   :data:`NONE_PLACEHOLDER` ("-") per Requirement 7.5.
4. Append the canonical disclaimer via :func:`append_disclaimer`
   (Requirement 12.1, Property 14).

Acceptance criteria covered
---------------------------

- Requirement 7.1 -- 基金代码 / 名称 / 经理 / 成立日期 / AUM / 收益 /
  回撤 / 夏普 / 同类排名 / 持仓.
- Requirement 7.2 -- 6 位代码不追加交易所后缀 (服务层负责).
- Requirement 7.3 -- 非法 6 位代码抛 :class:`SymbolError`.
- Requirement 7.4 -- 前十大持仓权重 2 位小数百分比.
- Requirement 7.5 -- 缺失字段以 "-" 占位.
- Requirement 12.1 / Property 14 -- 末尾追加免责声明.
"""

from __future__ import annotations

import math
from typing import Final

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from china_stock_mcp.error_mapping import format_pydantic_error
from china_stock_mcp.exceptions import SymbolError
from china_stock_mcp.formatters import (
    NONE_PLACEHOLDER,
    finalize_tool_output,
    format_amount,
    format_percent,
    render_table,
)
from china_stock_mcp.models import FundInfo
from china_stock_mcp.services.fund_service import FundService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Required length of a 公募基金 6-digit code (mirrored at every
#: layer so the schema-level error surfaces with the same wording).
_FUND_CODE_LEN: Final[int] = 6

#: 6-digit fund code regex used by :class:`GetFundInfoInput`.
_FUND_CODE_PATTERN: Final[str] = r"^\d{6}$"


# ---------------------------------------------------------------------------
# Pydantic input model
# ---------------------------------------------------------------------------


class GetFundInfoInput(BaseModel):
    """Pydantic v2 input schema for :func:`get_fund_info`.

    Constraints:

    - ``fund_code`` is required and must match ``^\\d{6}$``
      (Requirement 7.3).
    - ``extra="forbid"`` rejects unknown keys.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    fund_code: str = Field(..., pattern=_FUND_CODE_PATTERN)


# ---------------------------------------------------------------------------
# Public tool entry
# ---------------------------------------------------------------------------


def get_fund_info(service: FundService, fund_code: str) -> str:
    """Return a Markdown 公募基金 information document for ``fund_code``.

    Parameters
    ----------
    service:
        Pre-wired :class:`FundService` instance.
    fund_code:
        6-digit bare fund code (no exchange suffix). Whitespace is
        stripped before validation.

    Returns
    -------
    str
        Markdown document ending with the standard disclaimer.

    Raises
    ------
    SymbolError
        If ``fund_code`` fails the 6-digit pattern check
        (Requirement 7.3) or the service-layer 6-digit validation.
    DataNotFoundError
        If the adapter cannot resolve the fund code.
    ChinaStockMCPError
        Any other subclass raised by the service / adapter layer is
        propagated verbatim (Requirement 13.1).
    """

    # 1) Input validation -- pydantic failures bridge to SymbolError
    #    (Requirement 7.3 specifically calls for SymbolError on an
    #    invalid 6-digit code, not the generic ValidationError).
    try:
        validated = GetFundInfoInput(fund_code=fund_code)
    except PydanticValidationError as exc:
        raise SymbolError(
            f"非法基金代码: {fund_code!r}, 必须是 6 位数字 ({format_pydantic_error(exc)})"
        ) from exc

    # 2) Service call -- 6-digit revalidation + cache + rate-limit +
    #    fallback happens here.
    info: FundInfo = service.info(validated.fund_code)

    # 3) Render Markdown.
    body = _render_fund(info)

    # 4) Unified tool-exit pipeline (Property 13 / 14, Requirement 12.1 / 12.2).
    return finalize_tool_output(body)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_fund(info: FundInfo) -> str:
    """Build the Markdown body for a :class:`FundInfo`."""

    title = f"### 基金信息 ({info.code})"

    basic_table = _render_basic_table(info)
    holdings_table = _render_holdings_table(info)
    industry_table = _render_industry_table(info)

    parts: list[str] = [title, "", "#### 基本信息", "", basic_table]
    parts.extend(["", "#### 前十大持仓", "", holdings_table])
    if industry_table is not None:
        parts.extend(["", "#### 行业分布", "", industry_table])
    return "\n".join(parts)


def _render_basic_table(info: FundInfo) -> str:
    """Build the 基本信息 table (name / manager / returns / risk)."""

    rows: list[dict[str, str]] = [
        {"指标": "名称", "值": info.name or NONE_PLACEHOLDER},
        {"指标": "基金经理", "值": info.manager or NONE_PLACEHOLDER},
        {"指标": "成立日期", "值": info.inception_date.isoformat()},
        {"指标": "规模", "值": format_amount(info.aum)},
        {"指标": "近1月收益", "值": format_percent(info.return_1m)},
        {"指标": "近3月收益", "值": format_percent(info.return_3m)},
        {"指标": "近6月收益", "值": format_percent(info.return_6m)},
        {"指标": "近12月收益", "值": format_percent(info.return_12m)},
        {"指标": "最大回撤", "值": format_percent(info.max_drawdown)},
        {"指标": "夏普比率", "值": _format_optional_sharpe(info.sharpe)},
        {"指标": "同类排名", "值": info.rank_in_category or NONE_PLACEHOLDER},
    ]
    return render_table(rows, headers=["指标", "值"])


def _render_holdings_table(info: FundInfo) -> str:
    """Build the 前十大持仓 table.

    Per Requirement 7.4, weights render as 2-decimal percentages via
    :func:`format_percent`. Missing weight cells fall back to
    :data:`NONE_PLACEHOLDER` ("-").
    """

    if not info.top_holdings:
        return "_暂无持仓数据_"

    rendered: list[dict[str, str]] = []
    for holding in info.top_holdings:
        symbol_raw = holding.get("symbol", "")
        name_raw = holding.get("name", "")
        weight_raw = holding.get("weight")
        rendered.append(
            {
                "代码": str(symbol_raw) if symbol_raw is not None else "",
                "名称": str(name_raw) if name_raw is not None else "",
                "权重": _format_weight(weight_raw),
            }
        )
    return render_table(rendered, headers=["代码", "名称", "权重"])


def _render_industry_table(info: FundInfo) -> str | None:
    """Build the optional 行业分布 table; return ``None`` when empty."""

    if not info.industry_distribution:
        return None

    rendered: list[dict[str, str]] = []
    for slice_ in info.industry_distribution:
        industry_raw = slice_.get("industry", "")
        weight_raw = slice_.get("weight")
        rendered.append(
            {
                "行业": str(industry_raw) if industry_raw is not None else "",
                "权重": _format_weight(weight_raw),
            }
        )
    return render_table(rendered, headers=["行业", "权重"])


def _format_optional_sharpe(value: float | None) -> str:
    """Render ``sharpe`` (2-decimal) or ``-`` when missing.

    Per Requirement 7.5 the field renders as ``"-"`` rather than being
    omitted from the table when the upstream did not provide it.
    """

    if value is None:
        return NONE_PLACEHOLDER
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return NONE_PLACEHOLDER
    if isinstance(value, float) and math.isnan(value):
        return NONE_PLACEHOLDER
    return f"{float(value):.2f}"


def _format_weight(value: object) -> str:
    """Render a holding / industry weight as a 2-decimal percent."""

    if value is None:
        return NONE_PLACEHOLDER
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return NONE_PLACEHOLDER
    if isinstance(value, float) and math.isnan(value):
        return NONE_PLACEHOLDER
    return format_percent(float(value))


__all__ = ["GetFundInfoInput", "get_fund_info"]
