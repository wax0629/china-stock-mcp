"""K 线 service / model tests for tasks 10.2, 10.3 and 10.4.

This module consolidates:

- **Property 8 (Task 10.2 / Validates: Requirements 3.4)** -- OHLC inequality
  ``low <= min(open, close) <= max(open, close) <= high`` is enforced by
  the :class:`china_stock_mcp.models.KLineBar` model validator. Random
  valid quadruples in ``[0.01, 10000]`` are accepted; random invalid
  quadruples are rejected with pydantic's ``ValidationError``.

- **Property 9 (Task 10.3 / Validates: Requirements 3.5)** -- indicator
  alignment ``len(indicators[k]) == len(bars)`` is enforced both at the
  :class:`china_stock_mcp.models.KLineSeries` level (DTO validation) and
  by :func:`china_stock_mcp.services.kline_service._compute_indicators`
  (NaN-padded series for any subset of supported indicators).

- **Task 10.4** -- service-level validation of ``period`` / ``adjust`` /
  ``count`` / ``indicators`` and the ``pattern_note`` heuristic
  (Requirements 3.2, 3.6, 3.7).

All tests are hermetic: they wire :class:`KLineService` against an
in-memory :class:`_StubCache` and a :class:`_StubKlineAdapter` so no
network or disk I/O is performed.
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
from china_stock_mcp.exceptions import ValidationError
from china_stock_mcp.models import (
    AdjustMode,
    FinancialReport,
    FundamentalSnapshot,
    FundInfo,
    KLineBar,
    KLinePeriod,
    KLineSeries,
    MarketOverview,
    MoneyFlow,
    PeerTable,
    Quote,
    SymbolHit,
)
from china_stock_mcp.rate_limiter import RateLimiter
from china_stock_mcp.services.kline_service import (
    KLineService,
    _compute_indicators,
    _derive_pattern_note,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubCache:
    """In-memory :class:`Cache` matching the protocol in ``cache.py``."""

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


class _StubKlineAdapter(BaseAdapter):
    """Adapter returning a configurable :class:`KLineSeries` payload.

    Every other abstract method raises :class:`NotImplementedError` so
    accidental misuse fails loudly during tests.
    """

    name = "stub"

    def __init__(self, payload: KLineSeries) -> None:
        self._payload = payload
        self.kline_calls = 0
        self.last_args: tuple[str, str, int, str] | None = None

    def set_payload(self, payload: KLineSeries) -> None:
        """Swap the canned payload between assertions."""

        self._payload = payload

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
        self, symbol: str, report_type: str, periods: int
    ) -> FinancialReport:
        raise NotImplementedError

    def money_flow(
        self, symbol: str | None, flow_type: str, top_n: int
    ) -> MoneyFlow:
        raise NotImplementedError

    def industry_peers(
        self, symbol: str, metrics: list[str], top_n: int
    ) -> PeerTable:
        raise NotImplementedError

    def fund_info(self, fund_code: str) -> FundInfo:
        raise NotImplementedError

    def market_overview(self) -> MarketOverview:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat_bar(d: date, price: float = 10.0) -> KLineBar:
    """Construct an OHLC=price bar (degenerate but valid)."""

    return KLineBar(
        date=d,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1_000,
        amount=1_000.0 * price,
    )


def _make_series(
    bars: list[KLineBar],
    *,
    period: KLinePeriod = "daily",
    adjust: AdjustMode = "qfq",
    symbol: str = "300750.SZ",
) -> KLineSeries:
    return KLineSeries(
        symbol=symbol,
        period=period,
        adjust=adjust,
        bars=bars,
        indicators={},
    )


def _make_service(
    payload: KLineSeries | None = None,
) -> tuple[KLineService, _StubKlineAdapter, _StubCache]:
    """Build a :class:`KLineService` wired to isolated stubs.

    Uses ``RateLimiter(rate_per_minute=10000)`` so the limiter never
    throttles property runs; admission semantics are covered separately
    in ``tests/unit/test_rate_limiter.py``.
    """

    if payload is None:
        payload = _make_series([_flat_bar(date(2024, 1, 1))])

    adapter = _StubKlineAdapter(payload)
    cache = _StubCache()
    limiter = RateLimiter(rate_per_minute=10_000, capacity=10_000)
    service = KLineService(adapter, cache=cache, rate_limiter=limiter)
    return service, adapter, cache


# ---------------------------------------------------------------------------
# Hypothesis strategies (Property 8 / Property 9)
# ---------------------------------------------------------------------------

_PRICE_MIN: float = 0.01
_PRICE_MAX: float = 10_000.0


def _price() -> st.SearchStrategy[float]:
    """Floats in ``[0.01, 10000]`` -- per task 10.2 spec."""

    return st.floats(
        min_value=_PRICE_MIN,
        max_value=_PRICE_MAX,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    )


@st.composite
def _valid_ohlc(draw: st.DrawFn) -> dict[str, float]:
    """Draw a quadruple guaranteed to satisfy the OHLC inequality.

    Strategy: pick four monotonic levels
    ``low <= body_low <= body_high <= high`` independently, then
    randomly assign body extremes to ``open`` / ``close``. This covers
    bullish, bearish and doji bars in one strategy.
    """

    levels = sorted(
        [draw(_price()), draw(_price()), draw(_price()), draw(_price())]
    )
    low, body_low, body_high, high = levels
    bullish = draw(st.booleans())
    open_ = body_low if bullish else body_high
    close_ = body_high if bullish else body_low
    return {"open": open_, "high": high, "low": low, "close": close_}


def _wrap_bars(ohlc_seq: list[dict[str, float]]) -> list[KLineBar]:
    """Wrap OHLC dicts into ``KLineBar``s with deterministic dates."""

    start = date(2024, 1, 1)
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


# All names the service is willing to accept (matching
# ``KLineService._SUPPORTED_INDICATORS``).
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
    """Draw a (possibly empty) ordered subset of supported indicators."""

    flags = draw(
        st.lists(
            st.booleans(),
            min_size=len(_SUPPORTED_INDICATORS),
            max_size=len(_SUPPORTED_INDICATORS),
        )
    )
    selected = [
        name
        for name, keep in zip(_SUPPORTED_INDICATORS, flags, strict=True)
        if keep
    ]
    if not selected:
        return []
    return list(draw(st.permutations(selected)))


# ---------------------------------------------------------------------------
# Property 8 -- OHLC inequality (Task 10.2 / Requirements 3.4)
# ---------------------------------------------------------------------------


class TestOHLCInequalityProperty:
    """**Validates: Requirements 3.4** (design Property 8).

    For every random quadruple in ``[0.01, 10000]``:

    - When ``low <= min(open, close) <= max(open, close) <= high`` holds,
      :class:`KLineBar` construction succeeds.
    - Otherwise pydantic raises ``ValidationError``.
    """

    @given(ohlc=_valid_ohlc())
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_valid_ohlc_constructs(self, ohlc: dict[str, float]) -> None:
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
        low=_price(),
        open_=_price(),
        close_=_price(),
        high=_price(),
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
# Property 9 -- indicator alignment (Task 10.3 / Requirements 3.5)
# ---------------------------------------------------------------------------


class TestIndicatorAlignmentProperty:
    """**Validates: Requirements 3.5** (design Property 9).

    Three aspects of the alignment property are exercised:

    1. :class:`KLineSeries` accepts any ``(bars, indicators)`` pair where
       every indicator list matches ``len(bars)`` (any bar count in
       ``0..50``).
    2. :class:`KLineSeries` raises ``ValidationError`` when any indicator
       list length differs from ``len(bars)``.
    3. The service-level :func:`_compute_indicators` returns NaN-padded
       lists matching ``len(bars)`` for every supported indicator name,
       so DTOs built from its output always satisfy the alignment rule.
    """

    @given(
        ohlc_seq=st.lists(_valid_ohlc(), min_size=0, max_size=50),
        indicators=_indicator_subset(),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_aligned_indicators_construct(
        self,
        ohlc_seq: list[dict[str, float]],
        indicators: list[str],
    ) -> None:
        bars = _wrap_bars(ohlc_seq)
        n = len(bars)
        # Build a deterministic value per index so round-tripping is
        # easy to reason about; the property only cares about lengths.
        ind_dict: dict[str, list[float]] = {
            name: [float(i) for i in range(n)] for name in indicators
        }

        series = KLineSeries(
            symbol="300750.SZ",
            period="daily",
            adjust="qfq",
            bars=bars,
            indicators=ind_dict,
        )
        for name, values in series.indicators.items():
            assert len(values) == n, (
                f"indicator {name!r} length {len(values)} != bar count {n}"
            )

    @given(
        ohlc_seq=st.lists(_valid_ohlc(), min_size=1, max_size=20),
        delta=st.integers(min_value=-5, max_value=5).filter(lambda x: x != 0),
    )
    @settings(
        max_examples=80,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_mismatched_indicator_length_rejected(
        self,
        ohlc_seq: list[dict[str, float]],
        delta: int,
    ) -> None:
        bars = _wrap_bars(ohlc_seq)
        n = len(bars)
        bad_len = max(0, n + delta)
        assume(bad_len != n)

        with pytest.raises(PydanticValidationError, match="MA20"):
            KLineSeries(
                symbol="300750.SZ",
                period="daily",
                adjust="qfq",
                bars=bars,
                indicators={"MA20": [0.0] * bad_len},
            )

    @given(
        ohlc_seq=st.lists(_valid_ohlc(), min_size=0, max_size=50),
        indicators=_indicator_subset(),
    )
    @settings(
        max_examples=60,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_compute_indicators_returns_aligned_lists(
        self,
        ohlc_seq: list[dict[str, float]],
        indicators: list[str],
    ) -> None:
        bars = _wrap_bars(ohlc_seq)
        n = len(bars)

        result = _compute_indicators(bars, indicators)
        for name, values in result.items():
            assert len(values) == n, (
                f"indicator {name!r} length {len(values)} != bar count {n}"
            )

    def test_compute_indicators_full_set_matches_bar_count(self) -> None:
        """Service spec exemplar: the canonical indicator subset
        ``["MA20","MA60","MACD","RSI14","BOLL"]`` returns lists whose
        lengths all equal ``len(bars)``."""

        bars = _wrap_bars([{"open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5}] * 25)
        n = len(bars)
        result = _compute_indicators(bars, ["MA20", "MA60", "MACD", "RSI14", "BOLL"])
        # Each requested indicator emits exactly one series.
        assert set(result.keys()) == {"MA20", "MA60", "MACD", "RSI14", "BOLL_MID"}
        for values in result.values():
            assert len(values) == n


# ---------------------------------------------------------------------------
# Task 10.4 -- period / adjust / count / indicators / pattern_note
# ---------------------------------------------------------------------------


class TestGetSeriesValidation:
    """Service-level input validation (Requirements 3.2, 3.6)."""

    def test_invalid_period_lists_accepted_set(self) -> None:
        service, _, _ = _make_service()

        with pytest.raises(ValidationError) as excinfo:
            service.get_series(
                symbol="300750.SZ",
                period="hourly",
                count=10,
                adjust="qfq",
                indicators=[],
            )
        msg = str(excinfo.value)
        assert "period" in msg
        # Every accepted value surfaces verbatim.
        for accepted in ("daily", "weekly", "monthly", "60min", "30min"):
            assert accepted in msg

    @pytest.mark.parametrize(
        "period",
        ["daily", "weekly", "monthly", "60min", "30min"],
    )
    def test_valid_period_accepted(self, period: str) -> None:
        service, _, _ = _make_service()

        result = service.get_series(
            symbol="300750.SZ",
            period=period,
            count=1,
            adjust="qfq",
            indicators=[],
        )
        assert result.period == period

    def test_invalid_adjust_lists_accepted_set(self) -> None:
        service, _, _ = _make_service()

        with pytest.raises(ValidationError) as excinfo:
            service.get_series(
                symbol="300750.SZ",
                period="daily",
                count=10,
                adjust="forward",
                indicators=[],
            )
        msg = str(excinfo.value)
        assert "adjust" in msg
        for accepted in ("qfq", "hfq", "none"):
            assert accepted in msg

    @pytest.mark.parametrize("adjust", ["qfq", "hfq", "none"])
    def test_valid_adjust_accepted(self, adjust: str) -> None:
        service, _, _ = _make_service()

        result = service.get_series(
            symbol="300750.SZ",
            period="daily",
            count=1,
            adjust=adjust,
            indicators=[],
        )
        assert result.adjust == adjust

    @pytest.mark.parametrize("count", [0, -1, 251, 1000])
    def test_invalid_count_rejected(self, count: int) -> None:
        service, _, _ = _make_service()

        with pytest.raises(ValidationError, match="count"):
            service.get_series(
                symbol="300750.SZ",
                period="daily",
                count=count,
                adjust="qfq",
                indicators=[],
            )

    @pytest.mark.parametrize("count", [1, 60, 250])
    def test_count_at_inclusive_bounds_accepted(self, count: int) -> None:
        service, _, _ = _make_service()

        # Stub returns a single bar regardless of count; this test
        # only verifies that the validator admits the boundary values.
        result = service.get_series(
            symbol="300750.SZ",
            period="daily",
            count=count,
            adjust="qfq",
            indicators=[],
        )
        assert isinstance(result, KLineSeries)

    def test_unsupported_indicator_lists_accepted_set(self) -> None:
        service, _, _ = _make_service()

        with pytest.raises(ValidationError) as excinfo:
            service.get_series(
                symbol="300750.SZ",
                period="daily",
                count=10,
                adjust="qfq",
                indicators=["STOCH"],
            )
        msg = str(excinfo.value)
        assert "STOCH" in msg
        # Every supported name must appear in the error.
        for accepted in _SUPPORTED_INDICATORS:
            assert accepted in msg

    def test_supported_indicators_all_accepted(self) -> None:
        service, _, _ = _make_service()

        result = service.get_series(
            symbol="300750.SZ",
            period="daily",
            count=1,
            adjust="qfq",
            indicators=list(_SUPPORTED_INDICATORS),
        )
        # ``BOLL`` is published as ``BOLL_MID`` (middle band only).
        expected_keys = {"MA5", "MA10", "MA20", "MA60", "MACD", "RSI14", "BOLL_MID"}
        assert set(result.indicators.keys()) == expected_keys


class TestPatternNote:
    """Pattern note heuristic (Requirement 3.7).

    - < 60 bars → ``pattern_note is None``.
    - ≥ 60 bars + sustained rising trend → ``"上升趋势"``.
    - ≥ 60 bars + sustained falling trend → ``"下降趋势"``.
    - ≥ 60 bars + sideways prices → ``"震荡"``.
    """

    def test_insufficient_bars_returns_none(self) -> None:
        # 59 flat bars -- one short of the 60-bar threshold.
        bars = [
            _flat_bar(date(2024, 1, 1) + timedelta(days=i), price=10.0)
            for i in range(59)
        ]
        payload = _make_series(bars)
        service, _, _ = _make_service(payload)

        result = service.get_series(
            symbol="300750.SZ",
            period="daily",
            count=59,
            adjust="qfq",
            indicators=[],
        )
        assert result.pattern_note is None

    def test_pure_helper_insufficient_bars(self) -> None:
        """Verify the underlying helper as well, independent of caching."""

        bars = [
            _flat_bar(date(2024, 1, 1) + timedelta(days=i), price=10.0)
            for i in range(59)
        ]
        assert _derive_pattern_note(bars) is None

    def test_rising_trend_returns_up(self) -> None:
        # 80 strictly rising bars -- final close is well above MA20,
        # which in turn is well above MA60.
        bars = [
            _flat_bar(date(2024, 1, 1) + timedelta(days=i), price=10.0 + i * 0.5)
            for i in range(80)
        ]
        payload = _make_series(bars)
        service, _, _ = _make_service(payload)

        result = service.get_series(
            symbol="300750.SZ",
            period="daily",
            count=80,
            adjust="qfq",
            indicators=[],
        )
        assert result.pattern_note == "上升趋势"

    def test_falling_trend_returns_down(self) -> None:
        # 80 strictly falling bars; start high enough to keep prices > 0.
        bars = [
            _flat_bar(date(2024, 1, 1) + timedelta(days=i), price=100.0 - i * 0.5)
            for i in range(80)
        ]
        payload = _make_series(bars)
        service, _, _ = _make_service(payload)

        result = service.get_series(
            symbol="300750.SZ",
            period="daily",
            count=80,
            adjust="qfq",
            indicators=[],
        )
        assert result.pattern_note == "下降趋势"

    def test_sideways_returns_oscillation(self) -> None:
        # 80 perfectly flat bars -- close, MA20 and MA60 are all equal,
        # well within the 1% trend tolerance.
        bars = [
            _flat_bar(date(2024, 1, 1) + timedelta(days=i), price=10.0)
            for i in range(80)
        ]
        payload = _make_series(bars)
        service, _, _ = _make_service(payload)

        result = service.get_series(
            symbol="300750.SZ",
            period="daily",
            count=80,
            adjust="qfq",
            indicators=[],
        )
        assert result.pattern_note == "震荡"
