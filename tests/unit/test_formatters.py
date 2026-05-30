"""Tests for :mod:`china_stock_mcp.formatters`.

Covers:

- Property 16 -- 涨跌色规则 (Requirements 2.4) — :func:`render_change`.
- Property 14 -- 免责声明 (Requirements 12.1) — :func:`append_disclaimer`.
- Property 13 -- Token 预算 (Requirements 12.2) — ``render_quote`` + disclaimer.
- Unit coverage for :func:`format_amount`, :func:`format_percent`,
  :func:`render_table`.
"""

from __future__ import annotations

from datetime import UTC, datetime

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from china_stock_mcp.formatters import (
    DISCLAIMER,
    append_disclaimer,
    format_amount,
    format_percent,
    render_change,
    render_quote,
    render_table,
)
from china_stock_mcp.models import Quote

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

#: Maximum tokens a single tool response may emit (design Property 13 /
#: Requirement 12.2). Mirror the value from ``design.md`` so the test
#: fails loudly if the budget is ever changed without updating callers.
TOKEN_BUDGET: int = 3000

#: Approximate tokens-per-character ratio used by Claude's tokenizer
#: guideline (~4 chars per token). The proxy intentionally rounds up
#: so we never under-estimate the cost of CJK-heavy output.
def _token_count(markdown: str) -> int:
    """Conservative token-count proxy for Markdown text."""

    return (len(markdown) + 3) // 4


# Float strategy that excludes NaN / inf so we can reason about sign.
_finite_floats = st.floats(allow_nan=False, allow_infinity=False)

# Constrain magnitudes to avoid astronomical numbers that explode the
# rendered string length and slow down hypothesis without adding signal.
_change_floats = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)


def _quote_strategy() -> st.SearchStrategy[Quote]:
    """Hypothesis strategy producing valid :class:`Quote` instances.

    Bounds are chosen to stay within plausible market values while
    still exercising the formatter for a wide range of magnitudes.
    """

    bounded_float = st.floats(
        min_value=0.0,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
    )
    optional_ratio = st.one_of(
        st.none(),
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    )

    return st.builds(
        Quote,
        symbol=st.sampled_from(["300750.SZ", "600519.SH", "00700.HK", "510300"]),
        name=st.text(
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Lo", "Nd"),
                blacklist_characters="|\n\r",
            ),
            min_size=1,
            max_size=20,
        ),
        price=bounded_float,
        change=st.floats(
            min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False
        ),
        change_pct=st.floats(
            min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        volume=st.integers(min_value=0, max_value=10**12),
        amount=st.floats(
            min_value=0.0, max_value=1e13, allow_nan=False, allow_infinity=False
        ),
        turnover_rate=st.floats(
            min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        pe_ttm=optional_ratio,
        pe_dynamic=optional_ratio,
        pb=optional_ratio,
        market_cap=st.floats(
            min_value=0.0, max_value=1e13, allow_nan=False, allow_infinity=False
        ),
        float_market_cap=st.floats(
            min_value=0.0, max_value=1e13, allow_nan=False, allow_infinity=False
        ),
        timestamp=st.datetimes(
            min_value=datetime(2000, 1, 1),
            max_value=datetime(2100, 1, 1),
            timezones=st.just(UTC),
        ),
        delay_seconds=st.integers(min_value=0, max_value=86400),
    )


# ---------------------------------------------------------------------------
# Property 16 — render_change colour rules (Requirements 2.4)
# ---------------------------------------------------------------------------


class TestRenderChangeColor:
    """**Validates: Requirements 2.4** (design Property 16).

    For any finite ``change``:

    - ``change > 0``  ⇒ output contains ``🔴`` and never ``🟢``.
    - ``change < 0``  ⇒ output contains ``🟢`` and never ``🔴``.
    - ``change == 0`` ⇒ output contains neither emoji.
    """

    @given(change=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_positive_change_renders_red(self, change: float) -> None:
        rendered = render_change(change)
        assert "🔴" in rendered
        assert "🟢" not in rendered

    @given(change=st.floats(min_value=-1e6, max_value=-1e-6, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_negative_change_renders_green(self, change: float) -> None:
        rendered = render_change(change)
        assert "🟢" in rendered
        assert "🔴" not in rendered

    def test_zero_change_renders_no_color(self) -> None:
        rendered = render_change(0.0)
        assert "🔴" not in rendered
        assert "🟢" not in rendered

    def test_zero_int_change_renders_no_color(self) -> None:
        rendered = render_change(0)
        assert "🔴" not in rendered
        assert "🟢" not in rendered


# ---------------------------------------------------------------------------
# Property 14 — append_disclaimer (Requirements 12.1)
# ---------------------------------------------------------------------------


class TestAppendDisclaimer:
    """**Validates: Requirements 12.1** (design Property 14).

    For any input ``text``:

    - ``append_disclaimer(text).rstrip().endswith(DISCLAIMER)`` is true.
    - The disclaimer appears exactly once in the result.
    - The function is idempotent:
      ``append_disclaimer(append_disclaimer(t)) == append_disclaimer(t)``.
    """

    @given(text=st.text(max_size=2000))
    @settings(max_examples=200)
    def test_output_ends_with_disclaimer(self, text: str) -> None:
        result = append_disclaimer(text)
        assert result.rstrip().endswith(DISCLAIMER)

    @given(text=st.text(max_size=2000))
    @settings(max_examples=200)
    def test_disclaimer_appears_exactly_once(self, text: str) -> None:
        # Filter out the (vanishingly unlikely) case where Hypothesis
        # produces the disclaimer string itself; that scenario is
        # exercised by ``test_idempotent`` instead.
        assume(DISCLAIMER not in text)
        result = append_disclaimer(text)
        assert result.count(DISCLAIMER) == 1

    @given(text=st.text(max_size=2000))
    @settings(max_examples=200)
    def test_idempotent(self, text: str) -> None:
        once = append_disclaimer(text)
        twice = append_disclaimer(once)
        assert once == twice

    def test_empty_input_returns_disclaimer_only(self) -> None:
        assert append_disclaimer("") == DISCLAIMER

    def test_whitespace_only_input_returns_disclaimer(self) -> None:
        assert append_disclaimer("   \n\t").rstrip().endswith(DISCLAIMER)


# ---------------------------------------------------------------------------
# Property 13 — token budget (Requirements 12.2)
# ---------------------------------------------------------------------------


class TestTokenBudget:
    """**Validates: Requirements 12.2** (design Property 13).

    For any well-formed :class:`Quote`, ``render_quote(q)`` followed by
    ``append_disclaimer`` produces Markdown whose approximate token
    count stays within the per-tool budget of 3000 tokens.
    """

    @given(quote=_quote_strategy())
    @settings(max_examples=200, deadline=None)
    def test_render_quote_within_token_budget(self, quote: Quote) -> None:
        markdown = append_disclaimer(render_quote(quote))
        assert _token_count(markdown) <= TOKEN_BUDGET

    def test_token_count_proxy_matches_claude_guideline(self) -> None:
        # Sanity check the proxy: 4 ASCII chars ≈ 1 token.
        assert _token_count("abcd") == 1
        assert _token_count("a" * 12_001) > TOKEN_BUDGET


# ---------------------------------------------------------------------------
# Unit tests — format_amount / format_percent / render_table
# ---------------------------------------------------------------------------


class TestFormatAmount:
    def test_yi_unit(self) -> None:
        assert format_amount(1.5e8) == "1.50 亿"

    def test_wan_unit(self) -> None:
        assert format_amount(1.5e4) == "1.50 万"

    def test_negative_yi(self) -> None:
        assert format_amount(-2.5e8) == "-2.50 亿"

    def test_below_wan(self) -> None:
        assert format_amount(987.6) == "987.60"

    def test_zero(self) -> None:
        assert format_amount(0) == "0.00"

    def test_none_placeholder(self) -> None:
        assert format_amount(None) == "-"

    def test_nan_placeholder(self) -> None:
        assert format_amount(float("nan")) == "-"


class TestFormatPercent:
    def test_two_decimal_truncation(self) -> None:
        assert format_percent(5.234) == "5.23%"

    def test_negative_value(self) -> None:
        assert format_percent(-1.0) == "-1.00%"

    def test_zero(self) -> None:
        assert format_percent(0) == "0.00%"

    def test_none_placeholder(self) -> None:
        assert format_percent(None) == "-"

    def test_nan_placeholder(self) -> None:
        assert format_percent(float("nan")) == "-"


class TestRenderTable:
    def test_basic_layout(self) -> None:
        markdown = render_table(
            [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            headers=["a", "b"],
        )
        assert markdown == "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"

    def test_missing_keys_use_placeholder(self) -> None:
        markdown = render_table(
            [{"a": 1}],
            headers=["a", "b"],
        )
        assert markdown.endswith("| 1 | - |")

    def test_pipe_in_cell_is_escaped(self) -> None:
        markdown = render_table(
            [{"col": "a|b"}],
            headers=["col"],
        )
        # The literal pipe must be escaped so it cannot be parsed as a
        # column boundary by downstream Markdown renderers.
        assert "a\\|b" in markdown
        # And the unescaped pipe count equals the table-frame pipes:
        # 2 frame pipes per data line + header (2) + separator (2).
        # One escaped pipe in the body should not be counted.
        body_line = markdown.splitlines()[-1]
        assert body_line.count("|") - body_line.count("\\|") == 2

    def test_newline_in_cell_is_replaced(self) -> None:
        markdown = render_table(
            [{"col": "line1\nline2"}],
            headers=["col"],
        )
        # The cell must collapse to a single line so the table layout
        # is preserved.
        assert "line1\nline2" not in markdown
        assert "line1 line2" in markdown

    def test_none_cell_uses_placeholder(self) -> None:
        markdown = render_table([{"a": None}], headers=["a"])
        assert markdown.endswith("| - |")

    def test_empty_rows_emits_header_and_separator_only(self) -> None:
        markdown = render_table([], headers=["a", "b"])
        assert markdown == "| a | b |\n|---|---|"
