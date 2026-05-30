"""Markdown rendering helpers (pure functions).

This module is the *Formatter* component referenced in ``design.md``
(Component 6) and is the single place responsible for turning DTOs
into the Markdown strings that MCP tools return to the client.

Public API
----------

- :func:`render_table`       -- Markdown pipe-style table.
- :func:`render_quote`       -- Quote card for :class:`~models.Quote`.
- :func:`format_amount`      -- Convert a 元 amount into 亿 / 万 units.
- :func:`format_percent`     -- Format a percent value with 2 decimals.
- :func:`render_change`      -- 🔴 / 🟢 涨跌色 (design Property 16).
- :func:`append_disclaimer`  -- Append the fixed disclaimer (Property 14).

Acceptance criteria covered
---------------------------

- 2.4  涨跌色规则 (Property 16) -- :func:`render_change`.
- 2.6  数据延迟提示语义        -- :func:`render_quote` 标注 ``delay_seconds``.
- 7.4  权重 2 位小数            -- :func:`format_percent` / :func:`render_table`.
- 12.1 末尾免责声明 (Property 14) -- :func:`append_disclaimer`.

The module is intentionally free of I/O, configuration and adapter
imports so it can be exercised by property-based tests without any
fixtures.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Final

from china_stock_mcp.config import Settings, load_settings
from china_stock_mcp.models import Quote

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Fixed disclaimer text from ``design.md`` §Security Considerations
#: (Requirements 12.1 / Property 14). Do **not** localize or rewrite —
#: the full-width punctuation is part of the canonical Chinese text.
DISCLAIMER: Final[str] = (
    "⚠️ 数据来源于公开第三方，可能存在延迟或误差。"  # noqa: RUF001
    "本服务仅供研究学习使用，不构成任何投资建议。"  # noqa: RUF001
)

#: Placeholder used when a cell value is ``None`` (Requirements 7.5).
NONE_PLACEHOLDER: Final[str] = "-"

#: Canonical 数据延迟 blockquote prepended to 行情-style outputs when
#: ``Settings.data_delay_notice`` is enabled (Requirement 2.6). The
#: full blockquote prefix (``> ℹ️``) is part of the canonical line so  # noqa: RUF003
#: the idempotency check in :func:`finalize_tool_output` matches on
#: the exact string and never collides with substring "数据延迟约"
#: that appears inside a Quote card row.
DELAY_NOTICE_LINE: Final[str] = "> ℹ️ 数据延迟约 15 分钟"  # noqa: RUF001

#: Approximate per-tool token budget expressed in characters. Property 13 /
#: Requirement 12.2 caps ``token_count(markdown)`` at 3000; the project-
#: wide proxy used in tests is ``(len + 3) // 4`` (4 chars ≈ 1 token),
#: so 12000 chars ≈ 3000 tokens. We reserve ~200 chars of headroom for
#: the disclaimer + truncation marker + paragraph separators.
_MAX_BODY_CHARS: Final[int] = 11_800

#: Marker appended when the rendered Markdown was truncated to satisfy
#: the token budget. Surfaced in plain text so the AI client can detect
#: the cutoff and re-issue a tighter request when needed.
_TRUNCATION_NOTE: Final[str] = "_(输出已截断至 ~3000 tokens)_"

# Unit thresholds for :func:`format_amount`. Values are denominated in
# 元 throughout the project (see :class:`~models.Quote`).
_YI: Final[float] = 1e8  # 一亿
_WAN: Final[float] = 1e4  # 一万

# Emoji used by :func:`render_change`. Matches A 股 convention: 红涨 / 绿跌.
_UP_EMOJI: Final[str] = "🔴"
_DOWN_EMOJI: Final[str] = "🟢"


# ---------------------------------------------------------------------------
# Primitive formatters
# ---------------------------------------------------------------------------


def format_amount(value: float | int | None) -> str:
    """Render a 元 amount with automatic 亿 / 万 unit selection.

    Behavior::

        format_amount(1_234_567_890)  -> "12.35 亿"
        format_amount(1_500_000)      -> "150.00 万"
        format_amount(987.6)          -> "987.60"
        format_amount(0)              -> "0.00"
        format_amount(-2.5e8)         -> "-2.50 亿"
        format_amount(None)           -> "-"

    Parameters
    ----------
    value:
        Amount in 元. ``None`` is rendered as :data:`NONE_PLACEHOLDER`
        so the function can be safely called on optional fields (e.g.
        a missing market cap).
    """

    if value is None:
        return NONE_PLACEHOLDER
    if not isinstance(value, (int, float)) or _is_nan(value):
        return NONE_PLACEHOLDER

    abs_value = abs(float(value))
    if abs_value >= _YI:
        return f"{value / _YI:.2f} 亿"
    if abs_value >= _WAN:
        return f"{value / _WAN:.2f} 万"
    return f"{float(value):.2f}"


def format_percent(value: float | int | None) -> str:
    """Format a percent value with 2 decimal places.

    The input is expected to already be expressed as a percentage
    number (e.g. ``5.23`` for 5.23%); no scaling is applied. ``None``
    is rendered as :data:`NONE_PLACEHOLDER`.

    Examples::

        format_percent(5.234)   -> "5.23%"
        format_percent(-1)      -> "-1.00%"
        format_percent(0)       -> "0.00%"
        format_percent(None)    -> "-"
    """

    if value is None:
        return NONE_PLACEHOLDER
    if not isinstance(value, (int, float)) or _is_nan(value):
        return NONE_PLACEHOLDER
    return f"{float(value):.2f}%"


def render_change(change: float | int | None) -> str:
    """Render a percent change with A 股 涨跌色 emoji.

    Implements design **Property 16**:

    - ``change > 0``  → output contains ``🔴`` and a leading ``+`` sign.
    - ``change < 0``  → output contains ``🟢`` and a leading ``-`` sign.
    - ``change == 0`` → output contains *no* color emoji.

    The numeric portion is always formatted with 2 decimal places and
    suffixed with ``%``; ``change`` is treated as a percentage value
    (``1.23`` → ``"+1.23%"``).

    Parameters
    ----------
    change:
        Percent change. ``None`` returns :data:`NONE_PLACEHOLDER`.
    """

    if change is None:
        return NONE_PLACEHOLDER
    if not isinstance(change, (int, float)) or _is_nan(change):
        return NONE_PLACEHOLDER

    value = float(change)
    if value > 0:
        return f"{_UP_EMOJI} +{value:.2f}%"
    if value < 0:
        # The numeric formatter already supplies the leading minus sign.
        return f"{_DOWN_EMOJI} {value:.2f}%"
    return f"{value:.2f}%"


# ---------------------------------------------------------------------------
# Table renderer
# ---------------------------------------------------------------------------


def render_table(
    rows: Iterable[Mapping[str, Any]],
    headers: list[str],
) -> str:
    """Render a Markdown pipe-style table.

    The output uses GitHub-flavored Markdown::

        | h1 | h2 |
        |---|---|
        | a  | b  |

    Behavior:

    - Cell values are looked up by ``headers`` keys; missing keys and
      ``None`` values render as :data:`NONE_PLACEHOLDER`.
    - ``|`` and newline characters inside cell values are escaped so a
      single value cannot break the table layout.
    - ``float`` cells are rendered with :func:`str` to avoid surprising
      precision changes; callers that need fixed precision should
      pre-format with :func:`format_amount` / :func:`format_percent`.
    - When ``rows`` is empty, the function still emits the header and
      separator line so downstream consumers can detect "no rows" by
      checking for the absence of data lines rather than parsing an
      empty string.

    Parameters
    ----------
    rows:
        Iterable of mappings; each mapping represents one row.
    headers:
        Ordered list of column keys. Must be non-empty.

    Raises
    ------
    ValueError
        If ``headers`` is empty.
    """

    if not headers:
        raise ValueError("render_table requires at least one header")

    header_line = "| " + " | ".join(_escape_cell(h) for h in headers) + " |"
    separator = "|" + "|".join(["---"] * len(headers)) + "|"

    body_lines: list[str] = []
    for row in rows:
        cells = [_format_cell(row.get(h)) for h in headers]
        body_lines.append("| " + " | ".join(cells) + " |")

    if body_lines:
        return "\n".join([header_line, separator, *body_lines])
    return "\n".join([header_line, separator])


# ---------------------------------------------------------------------------
# Quote card
# ---------------------------------------------------------------------------


def render_quote(q: Quote) -> str:
    """Render a single :class:`~models.Quote` as a Markdown card.

    Layout::

        ### {name} ({symbol})

        | 指标 | 值 |
        |---|---|
        | 现价 | 12.34 |
        | 涨跌 | 🔴 +0.20 |
        | 涨跌幅 | 🔴 +1.65% |
        | 成交量 | 12.34 万 股 |
        | 成交额 | 1.50 亿 |
        | 换手率 | 0.50% |
        | PE(TTM) | 10.50 |
        | PE(动) | 9.80 |
        | PB | 1.20 |
        | 总市值 | 1.00 亿 |
        | 流通市值 | 80.00 万 |
        | 数据时间 | 2024-01-01 15:00:00 |
        | 时延 | 数据延迟约 15 分钟 |

    The disclaimer is **not** appended here; tool callers concatenate
    one disclaimer per response via :func:`append_disclaimer`.
    """

    title = f"### {q.name} ({q.symbol})"

    # Render the change column as a single cell that combines the
    # absolute amount with the colored percent change so callers see
    # both at a glance.
    change_cell = _render_change_cell(q.change, q.change_pct)
    change_pct_cell = render_change(q.change_pct)

    # Volume is given in shares; convert into 万股 / 亿股 only when the
    # number is large enough to warrant a unit, otherwise show as-is.
    volume_cell = _format_volume(q.volume)

    delay_cell = _format_delay(q.delay_seconds)

    rows: list[dict[str, str]] = [
        {"指标": "现价", "值": f"{q.price:.2f}"},
        {"指标": "涨跌", "值": change_cell},
        {"指标": "涨跌幅", "值": change_pct_cell},
        {"指标": "成交量", "值": volume_cell},
        {"指标": "成交额", "值": format_amount(q.amount)},
        {"指标": "换手率", "值": format_percent(q.turnover_rate)},
        {"指标": "PE(TTM)", "值": _format_optional_float(q.pe_ttm)},
        {"指标": "PE(动)", "值": _format_optional_float(q.pe_dynamic)},
        {"指标": "PB", "值": _format_optional_float(q.pb)},
        {"指标": "总市值", "值": format_amount(q.market_cap)},
        {"指标": "流通市值", "值": format_amount(q.float_market_cap)},
        {"指标": "数据时间", "值": q.timestamp.strftime("%Y-%m-%d %H:%M:%S")},
        {"指标": "时延", "值": delay_cell},
    ]

    table = render_table(rows, headers=["指标", "值"])
    return f"{title}\n\n{table}"


# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------


def append_disclaimer(text: str) -> str:
    """Append the fixed disclaimer to ``text`` (idempotent).

    Implements design **Property 14** (``markdown.endswith(DISCLAIMER)``)
    and the exact string from ``design.md`` §Security Considerations.

    The function is idempotent: if ``text`` already ends with the
    disclaimer (after stripping trailing whitespace) it is returned
    unchanged so callers can compose pipelines without worrying about
    double-appending.
    """

    stripped = text.rstrip()
    if stripped.endswith(DISCLAIMER):
        return text

    if stripped == "":
        return DISCLAIMER

    return f"{stripped}\n\n{DISCLAIMER}"


# ---------------------------------------------------------------------------
# Unified tool-exit pipeline (task 23.1)
# ---------------------------------------------------------------------------


def finalize_tool_output(
    markdown: str,
    *,
    settings: Settings | None = None,
    delay_notice: bool = False,
) -> str:
    """Apply the cross-cutting tool-exit pipeline to ``markdown``.

    This is the single entry point every ``tools/*.py`` calls before
    returning a Markdown payload. It enforces three orthogonal rules
    drawn from the design's *Security Considerations* section:

    1. **Token budget (Property 13 / Requirement 12.2).** When the
       rendered body exceeds ~3000 tokens (≈ 12000 chars), the body is
       cut at the last paragraph boundary that still fits and a small
       ``_(输出已截断至 ~3000 tokens)_`` marker is appended so the AI
       client can tell the response was clipped.
    2. **Data-delay notice (Requirement 2.6).** When ``delay_notice``
       is ``True`` *and* ``settings.data_delay_notice`` is also ``True``,
       the canonical :data:`DELAY_NOTICE_LINE` blockquote is prepended.
       The check is idempotent: a body that already starts with the
       same blockquote (e.g. tools/quote.py composing it manually for
       backwards compatibility) is left untouched.
    3. **Disclaimer (Property 14 / Requirement 12.1).** :func:`append_disclaimer`
       is invoked last so every response ends with the canonical
       disclaimer. ``append_disclaimer`` itself is idempotent so this
       step is safe even when a caller already appended the disclaimer
       manually.

    Parameters
    ----------
    markdown:
        The Markdown body produced by a tool's renderer.
    settings:
        Optional :class:`Settings` injection for tests. When ``None``
        the function reads the current environment via
        :func:`load_settings` so production callers do not need to
        thread settings through every layer.
    delay_notice:
        When ``True`` the function consults ``settings.data_delay_notice``
        to decide whether to prepend the 数据延迟 banner. Defaults to
        ``False`` because only quote-style tools advertise a data delay
        in v1; the rest of the tool surface is end-of-day or static
        data.

    Returns
    -------
    str
        The transformed Markdown ending with :data:`DISCLAIMER`.
    """

    body = markdown

    # 1) Optional 数据延迟 banner -- prepend before the truncation
    #    pass so the banner is never the part that gets clipped.
    if delay_notice:
        resolved: Settings = (
            settings if settings is not None else load_settings()
        )
        if resolved.data_delay_notice and not body.lstrip().startswith(
            DELAY_NOTICE_LINE
        ):
            # Use a blank line between the notice and the body so the
            # blockquote renders cleanly in any Markdown viewer.
            body = f"{DELAY_NOTICE_LINE}\n\n{body}"

    # 2) Token-budget enforcement. The disclaimer + truncation marker
    #    add a known overhead (~200 chars including the two paragraph
    #    separators), reserved upfront in :data:`_MAX_BODY_CHARS`.
    if len(body) > _MAX_BODY_CHARS:
        body = _truncate_to_budget(body)

    # 3) Disclaimer (idempotent).
    return append_disclaimer(body)


def _truncate_to_budget(body: str) -> str:
    """Cut ``body`` at a paragraph boundary within the budget.

    Picks the last paragraph break (``"\\n\\n"``) that keeps the body
    within :data:`_MAX_BODY_CHARS`; if no paragraph break is reachable,
    falls back to the last newline; if neither is available, hard-cuts
    at the budget. The truncation marker is appended on its own
    paragraph so AI clients can detect the cutoff.
    """

    cutoff = _MAX_BODY_CHARS
    head = body[:cutoff]
    boundary = head.rfind("\n\n")
    if boundary == -1:
        boundary = head.rfind("\n")
    truncated = head if boundary == -1 else body[:boundary]

    truncated = truncated.rstrip()
    if truncated:
        return f"{truncated}\n\n{_TRUNCATION_NOTE}"
    return _TRUNCATION_NOTE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_nan(value: float | int) -> bool:
    """Return ``True`` for float NaN; integers are never NaN."""

    return isinstance(value, float) and value != value


def _escape_cell(value: str) -> str:
    """Escape a value so it cannot break the surrounding pipe table."""

    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _format_cell(value: Any) -> str:
    """Render a single table cell, applying the standard placeholder."""

    if value is None:
        return NONE_PLACEHOLDER
    if isinstance(value, float) and _is_nan(value):
        return NONE_PLACEHOLDER
    return _escape_cell(str(value))


def _format_optional_float(value: float | None) -> str:
    """Render an optional float with 2 decimals, or "-" when missing."""

    if value is None:
        return NONE_PLACEHOLDER
    if not isinstance(value, (int, float)) or _is_nan(value):
        return NONE_PLACEHOLDER
    return f"{float(value):.2f}"


def _format_volume(volume: int) -> str:
    """Render a share-volume integer with 万股 / 亿股 unit selection."""

    if volume >= int(_YI):
        return f"{volume / _YI:.2f} 亿股"
    if volume >= int(_WAN):
        return f"{volume / _WAN:.2f} 万股"
    return f"{volume} 股"


def _format_delay(delay_seconds: int) -> str:
    """Render the quote delay in a human-readable form.

    - ``0`` seconds → "实时".
    - Otherwise render minutes (rounded) so the message reads
      "数据延迟约 15 分钟" for the canonical 900-second case
      (Requirements 2.6 / Property 15).
    """

    if delay_seconds <= 0:
        return "实时"
    minutes = max(1, round(delay_seconds / 60))
    return f"数据延迟约 {minutes} 分钟"


def _render_change_cell(change: float, change_pct: float) -> str:
    """Render the *change* (绝对额) cell paired with涨跌色 emoji.

    The percent column uses :func:`render_change` directly; this helper
    is dedicated to the absolute-amount column so the two stay
    visually consistent. ``change`` here is in the same unit as
    :class:`~models.Quote.price` (元).
    """

    if change > 0:
        return f"{_UP_EMOJI} +{change:.2f}"
    if change < 0:
        return f"{_DOWN_EMOJI} {change:.2f}"
    # Reference change_pct only to keep the column visually aligned
    # with the percent column when no movement occurred.
    _ = change_pct
    return f"{change:.2f}"


__all__ = [
    "DELAY_NOTICE_LINE",
    "DISCLAIMER",
    "NONE_PLACEHOLDER",
    "append_disclaimer",
    "finalize_tool_output",
    "format_amount",
    "format_percent",
    "render_change",
    "render_quote",
    "render_table",
]
