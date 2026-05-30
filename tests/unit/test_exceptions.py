"""Unit tests for :mod:`china_stock_mcp.exceptions`.

Covers Requirement 13.1 (every domain failure inherits
:class:`ChinaStockMCPError`), Requirement 13.2 (``to_user_message``
returns text without any Python traceback or stack frames -- the
"AI-friendly" guarantee that backs Property 17), and Requirement
13.6 (``SymbolError`` surfaces up to three candidate suggestions plus
an optional ``market`` hint).

Task 22.1 specifies that *each* subclass must produce a traceback-free
human-readable message. The tests below exercise every concrete
subclass in the tree.
"""

from __future__ import annotations

import traceback

import pytest

from china_stock_mcp.exceptions import (
    ChinaStockMCPError,
    DataNotFoundError,
    DataSourceError,
    NetworkError,
    RateLimitError,
    SymbolError,
    ValidationError,
)

# ---------------------------------------------------------------------------
# Forbidden tokens
# ---------------------------------------------------------------------------

#: Substrings whose presence would indicate a Python traceback or
#: stack-frame leak. ``to_user_message`` MUST NOT include any of them
#: per Requirement 13.2 / Property 17.
_FORBIDDEN_TRACEBACK_TOKENS: tuple[str, ...] = (
    "Traceback (most recent call last)",
    'File "',
    "  File ",
    ", line ",
    "  at 0x",
    "<frame at",
    "self.__traceback__",
)


def _assert_no_traceback(message: str) -> None:
    """Assert ``message`` carries no Python traceback or stack frames."""

    for token in _FORBIDDEN_TRACEBACK_TOKENS:
        assert token not in message, (
            f"to_user_message leaked traceback marker {token!r}: {message!r}"
        )


# ---------------------------------------------------------------------------
# Hierarchy invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subclass",
    [
        SymbolError,
        DataSourceError,
        NetworkError,
        RateLimitError,
        DataNotFoundError,
        ValidationError,
    ],
)
def test_every_subclass_inherits_base(subclass: type[ChinaStockMCPError]) -> None:
    """Requirement 13.1: every domain error is a ``ChinaStockMCPError``."""

    assert issubclass(subclass, ChinaStockMCPError)


def test_network_and_rate_limit_inherit_data_source() -> None:
    """``NetworkError`` / ``RateLimitError`` extend :class:`DataSourceError`.

    The fallback policy in ``fetch_with_fallback`` (Requirement 13.3)
    catches these two specifically; keeping them under a shared parent
    matches the design.
    """

    assert issubclass(NetworkError, DataSourceError)
    assert issubclass(RateLimitError, DataSourceError)
    # ``DataNotFoundError`` must NOT be a ``DataSourceError`` so the
    # fallback path does not accidentally swallow it.
    assert not issubclass(DataNotFoundError, DataSourceError)


# ---------------------------------------------------------------------------
# to_user_message: no traceback, prefix present
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("subclass", "expected_prefix"),
    [
        (ChinaStockMCPError, "错误"),
        (SymbolError, "标的代码错误"),
        (DataSourceError, "数据源错误"),
        (NetworkError, "网络错误"),
        (RateLimitError, "调用频率受限"),
        (DataNotFoundError, "数据不存在"),
        (ValidationError, "参数校验失败"),
    ],
)
def test_to_user_message_prefix_and_no_traceback(
    subclass: type[ChinaStockMCPError],
    expected_prefix: str,
) -> None:
    """Each subclass renders the localized prefix and no traceback."""

    err = subclass("内部错误描述")
    msg = err.to_user_message()

    assert msg.startswith(f"{expected_prefix}:")
    assert "内部错误描述" in msg
    _assert_no_traceback(msg)


def test_to_user_message_with_hint_appends_line() -> None:
    """``hint`` is rendered on a new line under the main message."""

    err = NetworkError("连接超时", hint="请稍后重试")
    msg = err.to_user_message()

    assert "网络错误: 连接超时" in msg
    assert "请稍后重试" in msg
    _assert_no_traceback(msg)


# ---------------------------------------------------------------------------
# SymbolError specifics (Requirement 13.6)
# ---------------------------------------------------------------------------


def test_symbol_error_renders_candidates() -> None:
    """Up to three candidates surface in :meth:`to_user_message`."""

    err = SymbolError(
        "无法识别的代码: '苹果'",
        candidates=["AAPL", "00700.HK", "300750.SZ"],
        market="all",
    )
    msg = err.to_user_message()

    assert "AAPL" in msg
    assert "00700.HK" in msg
    assert "300750.SZ" in msg
    _assert_no_traceback(msg)


def test_symbol_error_truncates_extra_candidates() -> None:
    """Requirement 13.6 caps candidate suggestions at three entries."""

    err = SymbolError(
        "ambiguous",
        candidates=["a", "b", "c", "d", "e"],
    )

    assert err.candidates == ("a", "b", "c")
    msg = err.to_user_message()
    assert "d" not in msg
    assert "e" not in msg


def test_symbol_error_with_no_candidates_is_concise() -> None:
    """Empty / ``None`` candidates omit the candidate line entirely."""

    err = SymbolError("无法识别", candidates=None)
    msg = err.to_user_message()

    assert "候选项" not in msg
    _assert_no_traceback(msg)


def test_symbol_error_includes_market_hint_when_specific() -> None:
    """A non-``all`` market value is surfaced so callers can widen scope."""

    err = SymbolError(
        "未找到",
        candidates=["600519.SH"],
        market="hk_stock",
    )
    msg = err.to_user_message()

    assert "hk_stock" in msg


def test_symbol_error_omits_market_when_all() -> None:
    """``market='all'`` adds no extra noise; the line is suppressed."""

    err = SymbolError("未找到", candidates=("600519.SH",), market="all")
    msg = err.to_user_message()

    assert "market" not in msg.lower()


# ---------------------------------------------------------------------------
# Chained exception handling -- still no traceback in user message
# ---------------------------------------------------------------------------


def test_chained_exception_message_remains_clean() -> None:
    """Chaining a domain error from another exception must not leak frames.

    This guards Requirement 13.2 / 13.8: even when a service catches
    a low-level error and re-raises a domain exception with ``raise
    ... from exc``, ``to_user_message`` must not pick up the original
    traceback. The test simulates that scenario explicitly so the
    invariant holds even when the exception object carries a populated
    ``__cause__`` and ``__traceback__``.
    """

    try:
        try:
            raise RuntimeError("low-level boom")
        except RuntimeError as low_level:
            raise ValidationError("symbol: 必须是字符串") from low_level
    except ValidationError as caught:
        # Sanity-check the traceback is actually populated; otherwise
        # the assertion below would be a no-op.
        assert caught.__cause__ is not None
        assert caught.__traceback__ is not None
        assert traceback.format_exc()  # the *current* traceback exists.

        msg = caught.to_user_message()
        _assert_no_traceback(msg)
        assert "symbol" in msg
        assert "low-level boom" not in msg
