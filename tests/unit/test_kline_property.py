"""K 线属性测试 (任务 10.2 / 10.3).

This module covers the two K 线 properties enumerated in
``design.md`` §"Correctness Properties":

- **Property 8: K 线一致性** (Validates: Requirements 3.4) -- for any
  ``KLineBar``::

      low <= min(open, close) <= max(open, close) <= high

  The rule is enforced by the ``_check_ohlc_inequality`` model
  validator on :class:`china_stock_mcp.models.KLineBar`. The property
  test exercises both the accepting branch (random valid bars
  construct successfully) and the rejecting branch (random invalid
  bars trigger pydantic's ``ValidationError``).

- **Property 9: 指标对齐** (Validates: Requirements 3.5) -- after
  :meth:`KLineService.get_series` returns, every indicator series in
  ``KLineSeries.indicators`` MUST have the same length as ``bars``,
  regardless of bar count or which subset of supported indicators was
  requested. NaN-padding is allowed; the constraint is purely on
  length.

Both properties run against the real :class:`KLineService` wired up to
an in-memory stub adapter / cache / rate limiter so the tests stay
hermetic (no disk I/O, no network).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError as PydanticValidationError

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.cache import make_key
from china_stock_mcp.models import (
    FinancialReport,
    FundamentalSnapshot,
    FundInfo,
    KLineBar,
    KLineSeries,
    MarketOverview,
    MoneyFlow,
    PeerTable,
    Quote,
    SymbolHit,
)
from china_stock_mcp.rate_limiter import RateLimiter
from china_stock_mcp.services.kline_service import KLineService

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubCache:
    """In-memory cache matching the :class:`Cache` protocol.

    Mirrors the stub used in ``tests/unit/test_cache.py`` so each test
    runs in isolation without touching ``diskcache``.
    """

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self.store.get(key)

    def set(self, key: str, value: Any, ttl: int) -> None:
        if ttl <= 0:
            raise ValueError(f"ttl must be > 0, got {ttl}")
        self.store[key] = value

    def make_key(
        self,
        tool: str,
        symbol: str,
        params: Any,
        schema_version: int,
    ) -> str:
        return make_key(tool, symbol, params, schema_version)

    def close(self) -> None:  # pragma: no cover - nothing to release
        return None


class _StubAdapter(BaseAdapter):
    """Adapter returning a pre-baked :class:`KLineSeries` payload.

    Only :meth:`kline` is exercised by :class:`KLineService`; every
    other abstract method raises :class:`NotImplementedError` so a
    misuse (for example a service accidentally calling ``quote`` from
    K 线 code) fails loudly during tests.
    """

    name = "stub"

    def __init__(self, payload: KLineSeries) -> None:
        self._payload = payload
        self.kline_calls = 0
        self.last_args: tuple[str, str, int, str] | None = None

    def kline(
        self,
        symbol: str,
        period: str,
        count: int,
        adjust: str,
    ) -> KLineSeries:
        self.kline_calls += 1
        self.last_args = (symbol, period, count, adjust)
        return self._payload

    # ---- unused abstract methods ----------------------------------------

    def search(self, query: str, market: str) -> list[SymbolHit]:
        raise NotImplementedError

    def quote(self, symbols: list[str]) -> list[Quote]:
        raise NotImplementedError

    def fundamentals(self, symbol: str) -> FundamentalSnapshot:
        raise NotImplementedError

    def financial_report(
        self,
        symbol: str,
        report_type: str,
        periods: int,
    ) -> FinancialReport:
        raise NotImplementedError

    def money_flow(
        self,
        symbol: str | None,
        flow_type: str,
        top_n: int,
    ) -> MoneyFlow:
        raise NotImplementedError

    def industry_peers(
        self,
        symbol: str,
        metrics: list[str],
        top_n: int,
    ) -> PeerTable:
        raise NotImplementedError

    def fund_info(self, fund_code: str) -> FundInfo:
        raise NotImplementedError

    def market_overview(self) -> MarketOverview:
        raise NotImplementedError


def _make_service(payload: KLineSeries) -> tuple[KLineService, _StubAdapter]:
    """Build a :class:`KLineService` wired to an isolated stub stack."""

    adapter = _StubAdapter(payload)
    cache = _StubCache()
    # A generous limiter so property runs never trip on rate limiting;
    # Property 7 is exercised separately in ``test_rate_limiter.py``.
    limiter = RateLimiter(rate_per_minute=1_000_000, capacity=1_000_000)
    service = KLineService(adapter, cache=cache, rate_limiter=limiter)
    return service, adapter


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Bound prices to a sane range so floats stay well-conditioned and
# hypothesis shrinks quickly. Real A 股 prices live well within
# (0, 5000); 10000 leaves ample headroom without causing rounding
# pitfalls in the OHLC arithmetic.
_PRICE_MIN: float = 0.01
_PRICE_MAX: float = 10_000.0


def _price_strategy() -> st.SearchStrategy[float]:
    return st.floats(
        min_value=_PRICE_MIN,
        max_value=_PRICE_MAX,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    )


@st.composite
def _valid_ohlc(draw: st.DrawFn) -> dict[str, float]:
    """Draw an ``(open, high, low, close)`` tuple guaranteed valid.

    Strategy: pick the four monotonic levels
    ``low <= body_low <= body_high <= high`` independently, then
    randomly assign ``body_low`` / ``body_high`` to ``open`` / ``close``.
    This covers both bullish (``open < close``) and bearish
    (``open > close``) bars, plus the doji edge case (``body_low ==
    body_high``).
    """

    levels = sorted(
        [
            draw(_price_strategy()),
            draw(_price_strategy()),
            draw(_price_strategy()),
            draw(_price_strategy()),
        ]
    )
    low, body_low, body_high, high = levels
    bullish = draw(st.booleans())
    open_ = body_low if bullish else body_high
    close_ = body_high if bullish else body_low
    return {"open": open_, "high": high, "low": low, "close": close_}


def _make_bars(
    ohlc_seq: list[dict[str, float]],
    *,
    start: date = date(2024, 1, 1),
) -> list[KLineBar]:
    """Wrap a sequence of OHLC dicts into ``KLineBar`` instances.

    Dates are deterministic (``start + i days``) so each bar carries a
    unique value -- avoiding accidental collisions if a future
    KLineSeries-level uniqueness check is added.
    """

    return [
        KLineBar(
            date=start + timedelta(days=i),
            open=ohlc["open"],
            high=ohlc["high"],
            low=ohlc["low"],
            close=ohlc["close"],
            volume=1_000,
            amount=1_000.0 * ohlc["close"],
        )
        for i, ohlc in enumerate(ohlc_seq)
    ]


# ---------------------------------------------------------------------------
# Property 8 -- OHLC inequality (Validates: Requirements 3.4)
# ---------------------------------------------------------------------------


class TestOHLCInequalityProperty:
    """**Validates: Requirements 3.4** (design Property 8).

    For any bar where
    ``low <= min(open, close) <= max(open, close) <= high`` holds,
    :class:`KLineBar` construction succeeds; otherwise it raises
    pydantic's ``ValidationError`` -- enforced by the
    ``_check_ohlc_inequality`` model validator.
    """

    @given(ohlc=_valid_ohlc())
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_valid_ohlc_accepted(self, ohlc: dict[str, float]) -> None:
        bar = KLineBar(
            date=date(2024, 1, 1),
            open=ohlc["open"],
            high=ohlc["high"],
            low=ohlc["low"],
            close=ohlc["close"],
            volume=100,
            amount=1_000.0,
        )
        # Postcondition: round-tripped fields preserve the inequality.
        assert bar.low <= min(bar.open, bar.close)
        assert max(bar.open, bar.close) <= bar.high

    @given(
        low=_price_strategy(),
        open_=_price_strategy(),
        close_=_price_strategy(),
        high=_price_strategy(),
    )
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_invalid_ohlc_rejected(
        self,
        low: float,
        open_: float,
        close_: float,
        high: float,
    ) -> None:
        body_low = min(open_, close_)
        body_high = max(open_, close_)
        # Only consider quadruples that genuinely violate the OHLC rule.
        assume(not (low <= body_low <= body_high <= high))

        with pytest.raises(PydanticValidationError, match="OHLC"):
            KLineBar(
                date=date(2024, 1, 1),
                open=open_,
                high=high,
                low=low,
                close=close_,
                volume=100,
                amount=1_000.0,
            )


# ---------------------------------------------------------------------------
# Property 9 -- indicator alignment (Validates: Requirements 3.5)
# ---------------------------------------------------------------------------

# All names that ``KLineService.get_series`` is willing to accept.
_SUPPORTED_INDICATORS: tuple[str, ...] = (
    "MA5",
    "MA10",
    "MA20",
    "MA60",
    "MACD",
    "RSI14",
    "BOLL",
)


@st.composite
def _indicator_subset(draw: st.DrawFn) -> list[str]:
    """Draw a (possibly empty) subset of supported indicator names.

    Order matters because :meth:`KLineService.get_series` preserves
    insertion order in its ``indicators`` dict; randomizing the order
    helps catch any iteration-order assumption.
    """

    flags = draw(
        st.lists(
            st.booleans(),
            min_size=len(_SUPPORTED_INDICATORS),
            max_size=len(_SUPPORTED_INDICATORS),
        )
    )
    selected = [
        name for name, keep in zip(_SUPPORTED_INDICATORS, flags, strict=True) if keep
    ]
    if not selected:
        return []
    perm = draw(st.permutations(selected))
    return list(perm)


class TestIndicatorAlignmentProperty:
    """**Validates: Requirements 3.5** (design Property 9).

    For any bar count ``n`` and any subset of supported indicators,
    every series returned by :meth:`KLineService.get_series` has
    ``len(series) == n``. NaN-padding is allowed; only the length is
    checked.
    """

    @given(
        ohlc_seq=st.lists(_valid_ohlc(), min_size=1, max_size=80),
        indicators=_indicator_subset(),
    )
    @settings(
        max_examples=60,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_indicator_lengths_match_bar_count(
        self,
        ohlc_seq: list[dict[str, float]],
        indicators: list[str],
    ) -> None:
        bars = _make_bars(ohlc_seq)
        n = len(bars)

        # Stub adapter returns these bars verbatim; the service computes
        # indicators on top of them.
        payload = KLineSeries(
            symbol="300750.SZ",
            period="daily",
            adjust="qfq",
            bars=bars,
            indicators={},
        )
        service, _ = _make_service(payload)

        result = service.get_series(
            symbol="300750.SZ",
            period="daily",
            count=n,
            adjust="qfq",
            indicators=indicators,
        )

        assert len(result.bars) == n
        # Property 9: every emitted indicator series matches len(bars).
        for name, values in result.indicators.items():
            assert len(values) == n, (
                f"indicator {name!r} length {len(values)} != bar count {n}"
            )
