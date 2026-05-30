"""Property test 17 — ``ChinaStockMCPError.to_user_message()`` is traceback-free.

Covers task 22.3 of the china-stock-mcp spec:

- 22.3 Property 17 — 错误对 AI 友好 (**Validates: Requirements 13.2**)

Property 17 says ``to_user_message()`` must never leak Python traceback
or stack-frame information into the text the client receives,
regardless of:

* the message / hint string the exception carries
* the ``candidates`` / ``market`` fields specific to :class:`SymbolError`
* whether the exception was raised through ``raise ... from cause``
  (which populates ``__cause__`` and ``__traceback__``)

The test fuzzes arbitrary strings into every concrete subclass and
asserts the rendered message stays clean. Inputs that themselves
contain a forbidden marker are filtered out so the test only flags
output where ``to_user_message`` *adds* traceback markers — the
substantive guarantee.
"""

from __future__ import annotations

from typing import Final

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

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
# Forbidden traceback / stack-frame markers
# ---------------------------------------------------------------------------

#: Substrings whose presence indicates a Python traceback or stack
#: frame leaked into the user-facing message. ``to_user_message`` MUST
#: NOT include any of them per Requirements 13.2 / Property 17.
_FORBIDDEN_TOKENS: Final[tuple[str, ...]] = (
    "Traceback",
    'File "',
    ", line ",
    "<frame at",
    "self.__traceback__",
)


def _contains_forbidden(value: str) -> bool:
    """Return ``True`` if ``value`` literally contains any forbidden marker."""

    return any(token in value for token in _FORBIDDEN_TOKENS)


def _assert_no_traceback(message: str) -> None:
    """Fail the test if ``message`` contains any traceback marker."""

    for token in _FORBIDDEN_TOKENS:
        assert token not in message, (
            f"to_user_message leaked traceback marker {token!r}: "
            f"{message!r}"
        )


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Free-form text drawn from the full unicode space, filtered so that
# user-supplied inputs do not themselves contain a forbidden marker.
# The property under test is whether ``to_user_message`` *itself*
# injects traceback information at render time — it should never
# fabricate a marker that wasn't in the input. Letting a marker leak
# in via the input string would test a different (and uninteresting)
# property.
_message_text = st.text(max_size=500).filter(lambda s: not _contains_forbidden(s))

# Optional remediation hint — exercises both ``hint=None`` and
# ``hint=str`` branches of ``to_user_message``.
_hint = st.one_of(
    st.none(),
    st.text(max_size=200).filter(lambda s: not _contains_forbidden(s)),
)

# Up to 5 candidate strings of length up to 30 each for
# :class:`SymbolError` (the class itself caps the displayed list at 3,
# so we test both the under- and over-cap paths).
_candidate_text = st.text(max_size=30).filter(lambda s: not _contains_forbidden(s))
_candidates = st.one_of(
    st.none(),
    st.lists(_candidate_text, min_size=0, max_size=5),
)

# Market hint values: the four canonical literals plus arbitrary text
# (so even a malformed value cannot smuggle traceback markers in).
_market = st.one_of(
    st.none(),
    st.sampled_from(["a_stock", "hk_stock", "fund", "all"]),
    st.text(max_size=20).filter(lambda s: not _contains_forbidden(s)),
)

#: Subclasses with the simple ``__init__(message, *, hint=None)`` shape.
#: :class:`SymbolError` is exercised separately because it carries
#: extra structural fields.
_SIMPLE_SUBCLASSES: Final[tuple[type[ChinaStockMCPError], ...]] = (
    ChinaStockMCPError,
    DataSourceError,
    NetworkError,
    RateLimitError,
    DataNotFoundError,
    ValidationError,
)


# ---------------------------------------------------------------------------
# Property 17 — random inputs, every subclass
# ---------------------------------------------------------------------------


@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(message=_message_text, hint=_hint)
@pytest.mark.parametrize("subclass", _SIMPLE_SUBCLASSES)
def test_simple_subclass_to_user_message_has_no_traceback(
    subclass: type[ChinaStockMCPError],
    message: str,
    hint: str | None,
) -> None:
    """**Validates: Requirements 13.2**

    Every simple subclass renders a traceback-free user message for
    arbitrary ``message`` / ``hint`` values.
    """

    err = subclass(message, hint=hint)
    _assert_no_traceback(err.to_user_message())


@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(
    message=_message_text,
    hint=_hint,
    candidates=_candidates,
    market=_market,
)
def test_symbol_error_to_user_message_has_no_traceback(
    message: str,
    hint: str | None,
    candidates: list[str] | None,
    market: str | None,
) -> None:
    """**Validates: Requirements 13.2**

    :class:`SymbolError` carries extra structural fields (``candidates``
    + ``market``); the rendered output must still stay traceback-free.
    """

    err = SymbolError(
        message,
        candidates=candidates,
        market=market,
        hint=hint,
    )
    _assert_no_traceback(err.to_user_message())


# ---------------------------------------------------------------------------
# Property 17 — chained exceptions with populated __cause__ / __traceback__
# ---------------------------------------------------------------------------


@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(message=_message_text, hint=_hint)
@pytest.mark.parametrize("subclass", _SIMPLE_SUBCLASSES)
def test_chained_simple_subclass_message_remains_clean(
    subclass: type[ChinaStockMCPError],
    message: str,
    hint: str | None,
) -> None:
    """**Validates: Requirements 13.2**

    A ``raise X from low_level_runtime_error`` chain must not leak
    frames into the rendered message even though ``__cause__`` and
    ``__traceback__`` are populated.
    """

    try:
        try:
            raise RuntimeError("low-level boom")
        except RuntimeError as low_level:
            raise subclass(message, hint=hint) from low_level
    except subclass as caught:
        # Sanity-check: the framing only matters if the traceback /
        # cause are actually populated. If a future Python release
        # changes this behaviour the test will still be sound, but
        # the assumption is worth recording.
        assert caught.__cause__ is not None
        assert caught.__traceback__ is not None
        user_msg = caught.to_user_message()
        _assert_no_traceback(user_msg)
        # The original cause's message must not bleed into the user
        # message via any accidental ``str(__cause__)`` rendering.
        assert "low-level boom" not in user_msg


@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(
    message=_message_text,
    hint=_hint,
    candidates=_candidates,
    market=_market,
)
def test_chained_symbol_error_message_remains_clean(
    message: str,
    hint: str | None,
    candidates: list[str] | None,
    market: str | None,
) -> None:
    """**Validates: Requirements 13.2**

    :class:`SymbolError` raised through a cause stays traceback-free
    even with the extra ``candidates`` / ``market`` fields rendered
    into the output.
    """

    try:
        try:
            raise RuntimeError("low-level boom")
        except RuntimeError as low_level:
            raise SymbolError(
                message,
                candidates=candidates,
                market=market,
                hint=hint,
            ) from low_level
    except SymbolError as caught:
        assert caught.__cause__ is not None
        assert caught.__traceback__ is not None
        user_msg = caught.to_user_message()
        _assert_no_traceback(user_msg)
        assert "low-level boom" not in user_msg
