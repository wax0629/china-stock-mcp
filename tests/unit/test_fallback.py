"""Property and unit tests for :mod:`china_stock_mcp.adapters.fallback`.

Covers tasks 5.3 and 5.4 of the china-stock-mcp spec:

- 5.3 Property 5 -- fallback 不吞 NotFound (Validates: Requirements 13.4)
- 5.4 Property 6 -- fallback 透明性    (Validates: Requirements 13.5)

Property 5 says ``DataNotFoundError`` raised by the primary must
propagate unchanged and the fallback must never be invoked. Property
6 says when the primary succeeds, the fallback must never be invoked.

We also include a small set of unit tests for the supporting branches
of :func:`fetch_with_fallback` (NetworkError / RateLimitError trigger
fallback, ``fallback=None`` re-raises, non-fallback-eligible domain
errors propagate, and exceptions raised by the fallback propagate
unchanged) so the property tests above are anchored in clear, named
regression cases.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from china_stock_mcp.adapters.fallback import fetch_with_fallback
from china_stock_mcp.exceptions import (
    ChinaStockMCPError,
    DataNotFoundError,
    DataSourceError,
    NetworkError,
    RateLimitError,
    SymbolError,
    ValidationError,
)


class _Spy:
    """Tiny callable used as a fallback to record whether it ran.

    Using a class instead of ``unittest.mock.Mock`` keeps the property
    tests dependency-free and the assertion text clear.
    """

    def __init__(self, return_value: Any = None) -> None:
        self.return_value = return_value
        self.calls: int = 0

    def __call__(self) -> Any:
        self.calls += 1
        return self.return_value

    @property
    def called(self) -> bool:
        return self.calls > 0


# ---------------------------------------------------------------------------
# Task 5.3 -- Property 5: fallback 不吞 NotFound
# ---------------------------------------------------------------------------


@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(message=st.text(min_size=0, max_size=200))
def test_fallback_does_not_swallow_data_not_found(message: str) -> None:
    """**Validates: Requirements 13.4**

    Property 5: ``primary`` raising :class:`DataNotFoundError` must
    propagate unchanged, and ``fallback`` must never be invoked --
    even though a fallback is configured. The error message is
    forwarded verbatim so callers can show the original explanation.
    """

    def primary() -> Any:
        raise DataNotFoundError(message)

    fallback = _Spy(return_value="should-never-be-returned")

    with pytest.raises(DataNotFoundError) as excinfo:
        fetch_with_fallback(primary, fallback)

    assert excinfo.value.message == message
    assert fallback.called is False, (
        "fallback was invoked despite DataNotFoundError; this would "
        "violate Property 5 / Requirement 13.4."
    )


# ---------------------------------------------------------------------------
# Task 5.4 -- Property 6: fallback 透明性
# ---------------------------------------------------------------------------


# Strategy covering the kinds of payloads adapters return in practice
# (DTO-shaped values would obscure the property; the property only
# cares that *some* value passes through unchanged).
_payloads = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False),
    st.text(max_size=80),
    st.lists(st.integers(), max_size=8),
    st.dictionaries(st.text(max_size=8), st.integers(), max_size=4),
)


@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(value=_payloads)
def test_fallback_not_invoked_on_primary_success(value: Any) -> None:
    """**Validates: Requirements 13.5**

    Property 6: when ``primary`` returns a value, ``fetch_with_fallback``
    forwards it unchanged and the fallback is never invoked.
    """

    def primary() -> Any:
        return value

    fallback = _Spy(return_value="should-never-be-returned")

    result = fetch_with_fallback(primary, fallback)

    assert result == value
    assert fallback.called is False, (
        "fallback was invoked even though primary succeeded; this "
        "would violate Property 6 / Requirement 13.5."
    )


# ---------------------------------------------------------------------------
# Supporting unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        NetworkError("connection reset"),
        RateLimitError("upstream 429"),
    ],
)
def test_network_and_rate_limit_trigger_fallback(
    exc: ChinaStockMCPError,
) -> None:
    """Requirements 13.3 -- only ``NetworkError`` / ``RateLimitError``
    trigger the fallback path."""

    def primary() -> str:
        raise exc

    fallback = _Spy(return_value="from-fallback")

    result = fetch_with_fallback(primary, fallback)

    assert result == "from-fallback"
    assert fallback.calls == 1


def test_fallback_none_reraises_original_network_error() -> None:
    """``fallback=None`` preserves the original exception type."""

    def primary() -> str:
        raise NetworkError("upstream timeout")

    with pytest.raises(NetworkError) as excinfo:
        fetch_with_fallback(primary, None)

    assert excinfo.value.message == "upstream timeout"


def test_fallback_none_reraises_original_rate_limit_error() -> None:
    """``fallback=None`` re-raises :class:`RateLimitError` unchanged."""

    def primary() -> str:
        raise RateLimitError("upstream 429")

    with pytest.raises(RateLimitError) as excinfo:
        fetch_with_fallback(primary, None)

    assert excinfo.value.message == "upstream 429"


@pytest.mark.parametrize(
    "exc",
    [
        SymbolError("无法识别的代码"),
        ValidationError("字段错误"),
        DataSourceError("malformed payload"),
    ],
)
def test_non_fallback_errors_do_not_trigger_fallback(
    exc: ChinaStockMCPError,
) -> None:
    """Only ``NetworkError`` / ``RateLimitError`` switch sources.

    ``SymbolError`` / ``ValidationError`` / ``DataSourceError`` (the
    bare parent class) propagate unchanged so callers see the precise
    failure type. Requirements 13.3 / 13.4.
    """

    def primary() -> str:
        raise exc

    fallback = _Spy(return_value="should-never-be-returned")

    with pytest.raises(type(exc)):
        fetch_with_fallback(primary, fallback)

    assert fallback.called is False


def test_fallback_exception_propagates_unchanged() -> None:
    """A ``RateLimitError`` from the fallback bubbles up verbatim.

    The fallback's failure is not silently masked or wrapped; callers
    rely on the original type to drive their retry policy.
    """

    def primary() -> str:
        raise NetworkError("primary timeout")

    def fallback() -> str:
        raise RateLimitError("fallback 429")

    with pytest.raises(RateLimitError) as excinfo:
        fetch_with_fallback(primary, fallback)

    assert excinfo.value.message == "fallback 429"


def test_primary_success_returns_value_with_no_fallback_configured() -> None:
    """``fallback=None`` is allowed on the success path (sanity check)."""

    def primary() -> str:
        return "ok"

    assert fetch_with_fallback(primary, None) == "ok"
