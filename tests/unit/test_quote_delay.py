# ruff: noqa: RUF002
"""Quote 校验与延迟相关测试 (任务 8.5).

Covers:

- **Property 15** (Requirements 2.5 / 2.7) — 行情时延标注:
  对任意 ``delay_seconds ∈ [0, 86400]``, ``Quote.delay_seconds >= 0``.
- :class:`Quote` 校验:
  - ``delay_seconds = -1`` 触发 pydantic ``ValidationError``.
  - ``delay_seconds = 0`` 接受, 渲染为 "实时".
  - ``delay_seconds = 900`` (默认) 渲染为 "数据延迟约 15 分钟".
  - ``delay_seconds = 300`` 渲染为 "数据延迟约 5 分钟".
  - 缺省字段 → ``delay_seconds == 900`` (Requirements 2.5).
- :func:`get_quote` 工具层延迟提示开关 (Requirements 2.5 / 2.6):
  - ``data_delay_notice=True`` ⇒ 输出包含
    ``"> ℹ️ 数据延迟约 15 分钟"`` 行.
  - ``data_delay_notice=False`` ⇒ 不包含该行.

This file is intentionally separate from ``test_models.py`` /
``test_formatters.py`` so 延迟相关性质集中在一处, 便于后续追溯
Property 15 的验证范围。其他 Quote 校验 (price / volume / amount /
turnover_rate) 已由 ``test_models.py::TestQuote`` 覆盖, 这里不重复。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st
from pydantic import ValidationError as PydanticValidationError

from china_stock_mcp.config import Settings
from china_stock_mcp.formatters import DISCLAIMER, render_quote
from china_stock_mcp.models import DEFAULT_QUOTE_DELAY_SECONDS, Quote
from china_stock_mcp.tools.quote import _DELAY_NOTICE_LINE, get_quote

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def make_quote(**overrides: Any) -> Quote:
    """Build a valid :class:`Quote` with overridable fields.

    Defaults match a plausible A-share snapshot so tests only have to
    name the field they care about (e.g. ``make_quote(delay_seconds=0)``).
    """

    base: dict[str, Any] = dict(
        symbol="600519.SH",
        name="贵州茅台",
        price=1700.0,
        change=12.5,
        change_pct=0.74,
        volume=1_000_000,
        amount=1_700_000_000.0,
        turnover_rate=0.85,
        pe_ttm=30.0,
        pe_dynamic=28.0,
        pb=10.0,
        market_cap=2_100_000_000_000.0,
        float_market_cap=2_100_000_000_000.0,
        timestamp=datetime(2024, 1, 2, 15, 0, tzinfo=UTC),
    )
    base.update(overrides)
    return Quote(**base)


class _StubQuoteService:
    """Minimal :class:`QuoteService` stand-in for tool-level tests.

    Returns a fixed ``list[Quote]`` regardless of input so tests don't
    have to wire up the cache, rate limiter, or adapter layer. Duck-
    typing is sufficient because :func:`get_quote` only calls
    ``service.get_snapshot``.
    """

    def __init__(self, quotes: list[Quote]) -> None:
        self._quotes = quotes
        self.last_input: str | list[str] | None = None

    def get_snapshot(
        self, symbols: str | list[str]
    ) -> list[Quote]:
        self.last_input = symbols
        return self._quotes


# ---------------------------------------------------------------------------
# Property 15 — 行情时延标注 (Validates: Requirements 2.5, 2.7)
# ---------------------------------------------------------------------------


class TestQuoteDelayProperty:
    """**Validates: Requirements 2.5, 2.7** (design Property 15).

    For any ``delay_seconds`` drawn from ``[0, 86400]``, constructing a
    valid :class:`Quote` succeeds and the resulting field satisfies
    the non-negativity invariant.
    """

    @given(delay_seconds=st.integers(min_value=0, max_value=86400))
    @hyp_settings(max_examples=200)
    def test_non_negative_delay_seconds_accepted(
        self, delay_seconds: int
    ) -> None:
        q = make_quote(delay_seconds=delay_seconds)
        assert q.delay_seconds >= 0
        assert q.delay_seconds == delay_seconds


# ---------------------------------------------------------------------------
# Quote model validation — delay-specific
# ---------------------------------------------------------------------------


class TestQuoteDelayValidation:
    def test_default_delay_seconds_is_900(self) -> None:
        # Requirement 2.5 — 默认延迟应为 15 分钟 (=900 秒).
        q = make_quote()
        assert q.delay_seconds == 900
        assert q.delay_seconds == DEFAULT_QUOTE_DELAY_SECONDS

    def test_negative_delay_seconds_rejected(self) -> None:
        # Requirement 2.7 — pydantic ``Field(ge=0)`` 拒绝负值.
        with pytest.raises(PydanticValidationError, match="delay_seconds"):
            make_quote(delay_seconds=-1)

    def test_zero_delay_seconds_accepted(self) -> None:
        q = make_quote(delay_seconds=0)
        assert q.delay_seconds == 0


# ---------------------------------------------------------------------------
# render_quote — delay rendering branches
# ---------------------------------------------------------------------------


class TestRenderQuoteDelay:
    def test_zero_delay_renders_real_time_label(self) -> None:
        q = make_quote(delay_seconds=0)
        markdown = render_quote(q)
        assert "实时" in markdown
        # 数据时延行不会同时出现 "数据延迟" 文案.
        assert "数据延迟" not in markdown

    def test_default_delay_renders_15_minutes(self) -> None:
        q = make_quote()  # 默认 900 秒.
        assert "数据延迟约 15 分钟" in render_quote(q)

    def test_explicit_900_seconds_renders_15_minutes(self) -> None:
        q = make_quote(delay_seconds=900)
        assert "数据延迟约 15 分钟" in render_quote(q)

    def test_300_seconds_renders_5_minutes(self) -> None:
        q = make_quote(delay_seconds=300)
        assert "数据延迟约 5 分钟" in render_quote(q)


# ---------------------------------------------------------------------------
# get_quote — data_delay_notice toggle (Requirements 2.5 + 2.6)
# ---------------------------------------------------------------------------


def _settings(*, data_delay_notice: bool) -> Settings:
    """Construct a :class:`Settings` with only the toggle that matters.

    Other fields keep their dataclass defaults so the test stays
    insensitive to unrelated config changes.
    """

    return Settings(data_delay_notice=data_delay_notice)


class TestGetQuoteDelayNotice:
    def test_notice_present_when_enabled(self) -> None:
        service = _StubQuoteService([make_quote()])
        markdown = get_quote(
            service,  # type: ignore[arg-type]
            "600519.SH",
            settings=_settings(data_delay_notice=True),
        )

        assert _DELAY_NOTICE_LINE in markdown
        # Notice should sit above the body, before the disclaimer.
        notice_idx = markdown.index(_DELAY_NOTICE_LINE)
        disclaimer_idx = markdown.index(DISCLAIMER)
        assert notice_idx < disclaimer_idx

    def test_notice_absent_when_disabled(self) -> None:
        service = _StubQuoteService([make_quote()])
        markdown = get_quote(
            service,  # type: ignore[arg-type]
            "600519.SH",
            settings=_settings(data_delay_notice=False),
        )

        assert _DELAY_NOTICE_LINE not in markdown
        # 关闭通知开关并不会影响 Quote 卡片内的"时延"行本身.
        assert "数据延迟约 15 分钟" in markdown
        # Disclaimer must still be appended (Requirement 12.1).
        assert markdown.rstrip().endswith(DISCLAIMER)

    def test_notice_present_for_multi_symbol_table(self) -> None:
        # 多标的也应在表格上方加入延迟提示行.
        service = _StubQuoteService(
            [
                make_quote(symbol="600519.SH", name="贵州茅台"),
                make_quote(symbol="300750.SZ", name="宁德时代"),
            ]
        )
        markdown = get_quote(
            service,  # type: ignore[arg-type]
            ["600519.SH", "300750.SZ"],
            settings=_settings(data_delay_notice=True),
        )

        assert _DELAY_NOTICE_LINE in markdown
        # 多标的视图使用 "行情快照 (N 只)" 标题.
        assert "行情快照 (2 只)" in markdown
