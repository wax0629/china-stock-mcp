"""Property and unit tests for :mod:`china_stock_mcp.normalizer`.

Covers tasks 2.2, 2.3 and 2.4 of the china-stock-mcp spec:

- 2.2 Property 1 — 归一化幂等 (Validates: Requirements 1.4)
- 2.3 Property 2 — 归一化形态 (Validates: Requirements 1.5)
- 2.4 ``SymbolError`` 候选项与 ``market`` 过滤单元测试
        (Validates: Requirements 1.6, 1.7)
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Final

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from china_stock_mcp.exceptions import SymbolError
from china_stock_mcp.models import SymbolHit
from china_stock_mcp.normalizer import (
    detect_market,
    get_symbol_index,
    normalize_symbol,
    set_symbol_index,
)

# ---------------------------------------------------------------------------
# Standardized-symbol shape (Property 2 / Requirements 1.5).
# ---------------------------------------------------------------------------

_STANDARDIZED_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:\d{6}\.(?:SH|SZ|BJ)|\d{5}\.HK|\d{6})$"
)


# ---------------------------------------------------------------------------
# Stub SymbolIndex used to drive Chinese-name / pinyin / candidate paths
# without touching real data sources.
# ---------------------------------------------------------------------------


class StubSymbolIndex:
    """In-memory :class:`SymbolIndex` for tests.

    ``hits`` maps any case-folded query string to a :class:`SymbolHit`.
    ``candidates`` is the suggestion list returned to :class:`SymbolError`.
    """

    def __init__(
        self,
        hits: dict[str, SymbolHit] | None = None,
        candidates: list[str] | None = None,
    ) -> None:
        self._hits = {k.upper(): v for k, v in (hits or {}).items()}
        self._candidates = list(candidates or [])

    def lookup(self, query: str) -> SymbolHit | None:
        return self._hits.get(query.upper())

    def suggest(self, query: str, limit: int = 3) -> list[str]:
        return self._candidates[:limit]


@pytest.fixture(autouse=True)
def _reset_symbol_index() -> Iterator[None]:
    """Restore the global :data:`_default_index` between tests."""
    original = get_symbol_index()
    try:
        # Start every test from a clean (empty) index so nothing leaks.
        set_symbol_index(StubSymbolIndex())
        yield
    finally:
        set_symbol_index(original)


# ---------------------------------------------------------------------------
# Hypothesis strategies — produce inputs that should normalize successfully.
# ---------------------------------------------------------------------------


def _digits(n: int) -> st.SearchStrategy[str]:
    return st.text(alphabet="0123456789", min_size=n, max_size=n)


# 6-digit A-share codes whose prefix maps to a known exchange. Building
# digit by digit lets Hypothesis shrink toward small numbers.
_a_share_sh = st.tuples(
    st.sampled_from(["60", "68", "90"]), _digits(4)
).map(lambda parts: "".join(parts))

_a_share_sz = st.tuples(
    st.sampled_from(["00", "30", "20"]), _digits(4)
).map(lambda parts: "".join(parts))

_a_share_bj = st.tuples(
    st.just("8"), _digits(5)
).map(lambda parts: "".join(parts))

_bare_a_share = st.one_of(_a_share_sh, _a_share_sz, _a_share_bj)

# 5-digit HK codes.
_bare_hk = _digits(5)

# Already-standardized inputs (mix of upper / mixed casing — the
# normalizer upper-cases internally, so casing should not matter).
_standardized_a = st.tuples(
    _digits(6),
    st.sampled_from([".SH", ".SZ", ".BJ", ".sh", ".sz", ".bj"]),
).map(lambda parts: "".join(parts))

_standardized_hk = st.tuples(
    _digits(5),
    st.sampled_from([".HK", ".hk"]),
).map(lambda parts: "".join(parts))

# Surround inputs with whitespace occasionally to exercise trim().
_padded = lambda inner: st.tuples(  # noqa: E731
    st.sampled_from(["", " ", "\t", "  "]),
    inner,
    st.sampled_from(["", " ", "\t", "  "]),
).map(lambda parts: "".join(parts))

# Final strategy: a mix of every shape that normalize_symbol accepts
# without consulting the symbol index. We deliberately exclude bare
# 6-digit codes whose prefix does NOT resolve (e.g. "500000") because
# those rely on the index, which is empty in these property tests.
normalizable_input = st.one_of(
    _padded(_bare_a_share),
    _padded(_bare_hk),
    _padded(_standardized_a),
    _padded(_standardized_hk),
)


# ---------------------------------------------------------------------------
# Task 2.2 — Property 1: idempotence (Validates: Requirements 1.4)
# ---------------------------------------------------------------------------


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(raw=normalizable_input)
def test_normalize_symbol_is_idempotent(raw: str) -> None:
    """**Validates: Requirements 1.4**

    ``normalize_symbol(normalize_symbol(s)) == normalize_symbol(s)``
    whenever the first call succeeds.
    """
    try:
        first = normalize_symbol(raw)
    except SymbolError:
        assume(False)
        return
    second = normalize_symbol(first)
    assert second == first


# ---------------------------------------------------------------------------
# Task 2.3 — Property 2: shape (Validates: Requirements 1.5)
# ---------------------------------------------------------------------------


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(raw=normalizable_input)
def test_normalize_symbol_shape(raw: str) -> None:
    """**Validates: Requirements 1.5**

    Every successful return must match
    ``^\\d{6}\\.(SH|SZ|BJ)$``, ``^\\d{5}\\.HK$`` or ``^\\d{6}$``.
    """
    try:
        out = normalize_symbol(raw)
    except SymbolError:
        assume(False)
        return
    assert _STANDARDIZED_RE.fullmatch(out), f"unexpected shape: {out!r}"


# ---------------------------------------------------------------------------
# Task 2.4 — Unit tests for SymbolError candidates and market filter
#            (Validates: Requirements 1.6, 1.7)
# ---------------------------------------------------------------------------


def test_unrecognized_input_raises_symbol_error() -> None:
    """Requirements 1.6 — totally unknown input becomes ``SymbolError``."""
    set_symbol_index(StubSymbolIndex())  # no hits, no candidates
    with pytest.raises(SymbolError) as excinfo:
        normalize_symbol("不存在的股票名")
    err = excinfo.value
    assert err.candidates == ()
    assert "不存在的股票名" in err.message


def test_symbol_error_carries_up_to_three_candidates() -> None:
    """Requirements 1.6 — ``SymbolError`` exposes ≤3 candidate suggestions."""
    suggestions = [
        "苹果园(831175.BJ)",
        "苹果手机概念-A(000001.SZ)",
        "苹果种植(600001.SH)",
        "ignored-fourth-suggestion",
    ]
    set_symbol_index(StubSymbolIndex(candidates=suggestions))

    with pytest.raises(SymbolError) as excinfo:
        normalize_symbol("苹果")

    err = excinfo.value
    assert len(err.candidates) == 3
    assert err.candidates == tuple(suggestions[:3])


def test_symbol_error_to_user_message_includes_candidates_and_market() -> None:
    """Requirements 1.6, 1.7, 13.2 — user message lists candidates and market."""
    set_symbol_index(
        StubSymbolIndex(candidates=["苹果园(831175.BJ)", "苹果种植(600001.SH)"])
    )

    with pytest.raises(SymbolError) as excinfo:
        normalize_symbol("苹果", market="a_stock")

    msg = excinfo.value.to_user_message()
    assert "苹果园(831175.BJ)" in msg
    assert "苹果种植(600001.SH)" in msg
    assert "a_stock" in msg
    # Property 17 — no traceback / stack frames.
    assert "Traceback" not in msg
    assert "File \"" not in msg


def test_market_hint_omitted_when_all() -> None:
    """Requirements 1.7 — ``market='all'`` is treated as no restriction."""
    set_symbol_index(StubSymbolIndex())

    with pytest.raises(SymbolError) as excinfo:
        normalize_symbol("纯属乱码", market="all")

    msg = excinfo.value.to_user_message()
    assert "market" not in msg.lower()


def test_market_hint_propagated_when_provided() -> None:
    """Requirements 1.7 — explicit market scope surfaces in the error."""
    set_symbol_index(StubSymbolIndex())

    with pytest.raises(SymbolError) as excinfo:
        normalize_symbol("纯属乱码", market="hk_stock")

    assert excinfo.value.market == "hk_stock"
    assert "hk_stock" in excinfo.value.to_user_message()


def test_chinese_name_resolves_via_symbol_index() -> None:
    """Requirements 1.6 — index hits short-circuit the failure path."""
    hit = SymbolHit(
        code="300750.SZ",
        name="宁德时代",
        market="a_stock",
        industry="电池",
    )
    set_symbol_index(StubSymbolIndex(hits={"宁德时代": hit}))

    assert normalize_symbol("宁德时代") == "300750.SZ"


def test_pinyin_alias_resolves_via_symbol_index() -> None:
    """Index lookup is case-insensitive for ASCII pinyin aliases."""
    hit = SymbolHit(
        code="300750.SZ",
        name="宁德时代",
        market="a_stock",
        industry="电池",
    )
    set_symbol_index(StubSymbolIndex(hits={"NINGDE": hit}))

    assert normalize_symbol("ningde") == "300750.SZ"
    assert normalize_symbol("Ningde") == "300750.SZ"


def test_index_override_param_takes_precedence() -> None:
    """Caller-supplied ``index`` overrides the process-wide default."""
    global_hit = SymbolHit(
        code="600000.SH", name="A", market="a_stock"
    )
    local_hit = SymbolHit(
        code="000001.SZ", name="B", market="a_stock"
    )
    set_symbol_index(StubSymbolIndex(hits={"FOO": global_hit}))
    local_index = StubSymbolIndex(hits={"FOO": local_hit})

    assert normalize_symbol("foo", index=local_index) == "000001.SZ"


def test_already_standardized_inputs_are_returned_as_is() -> None:
    """Requirements 1.4 — standardized inputs round-trip unchanged (upper-case)."""
    assert normalize_symbol("300750.SZ") == "300750.SZ"
    assert normalize_symbol("600519.SH") == "600519.SH"
    assert normalize_symbol("831175.BJ") == "831175.BJ"
    assert normalize_symbol("00700.HK") == "00700.HK"
    # Lower-case suffix still normalizes (whitespace + case are folded).
    assert normalize_symbol(" 300750.sz ") == "300750.SZ"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("600519", "600519.SH"),
        ("688981", "688981.SH"),
        ("900001", "900001.SH"),
        ("000001", "000001.SZ"),
        ("300750", "300750.SZ"),
        ("200001", "200001.SZ"),
        ("831175", "831175.BJ"),
        ("00700", "00700.HK"),
    ],
)
def test_six_and_five_digit_codes_get_correct_suffix(
    raw: str, expected: str
) -> None:
    """Requirements 1.2, 1.3 — prefix rules and HK 5-digit handling."""
    assert normalize_symbol(raw) == expected


def test_empty_input_raises_symbol_error() -> None:
    for value in ("", "   ", "\t"):
        with pytest.raises(SymbolError):
            normalize_symbol(value)


def test_detect_market_resolves_known_shapes() -> None:
    """``detect_market`` maps each standardized shape onto its market."""
    assert detect_market("300750.SZ") == "a_stock"
    assert detect_market("600519.SH") == "a_stock"
    assert detect_market("831175.BJ") == "a_stock"
    assert detect_market("00700.HK") == "hk_stock"
    # 6-digit code without exchange suffix is treated as a fund.
    assert detect_market("510300") == "fund"


def test_detect_market_rejects_non_standardized_input() -> None:
    """``detect_market`` rejects anything outside the standardized shapes."""
    with pytest.raises(SymbolError):
        detect_market("乱码")
    with pytest.raises(SymbolError):
        detect_market("")
    with pytest.raises(SymbolError):
        detect_market("12345678")
