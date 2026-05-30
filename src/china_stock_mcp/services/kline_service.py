"""KLineService -- K 线序列 + 技术指标 + pattern_note.

Implements the ``KLineService`` half of *design.md* Component 3
(Service Layer). The service composes:

1. Strict input validation for ``period`` / ``adjust`` / ``count`` /
   ``indicators`` (Requirements 3.2 / 3.3 / 3.6) -- failures surface as
   :class:`ValidationError` with both the offending value and the
   accepted set.
2. :func:`normalize_symbol` so callers may pass either a bare 6-digit
   A-share code or an already-standardized symbol.
3. A read-through :func:`cache_get_or_fetch` keyed by
   ``(symbol, period, adjust, count)`` with :data:`TTL_WARM` (300s,
   the K 线 / 资金流 grade defined in Requirements 11.4).
4. A single rate-limit token per upstream batch
   (Requirements 11.6 / 11.7 / Property 7) and
   :func:`fetch_with_fallback` for transient failure transparency
   (Requirements 13.3 / 13.5, Property 6).
5. Pure technical-indicator helpers (MA / MACD / RSI14 / BOLL_MID) that
   return NaN-padded series matching ``len(bars)`` so DTO validation
   (Property 9) holds for every output series.
6. A ``pattern_note`` heuristic that compares the last close against
   the trailing MA20 / MA60 once at least 60 bars are available
   (Requirement 3.7).

The service deliberately re-raises every
:class:`ChinaStockMCPError` subclass verbatim -- the unified hierarchy
is the contract relied on by the Tool layer.
"""

from __future__ import annotations

import math
from typing import Final, cast

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.adapters.fallback import fetch_with_fallback
from china_stock_mcp.cache import TTL_WARM, Cache, cache_get_or_fetch, get_default_cache
from china_stock_mcp.exceptions import ValidationError
from china_stock_mcp.models import (
    MAX_KLINE_BARS,
    AdjustMode,
    KLineBar,
    KLinePeriod,
    KLineSeries,
)
from china_stock_mcp.normalizer import normalize_symbol
from china_stock_mcp.rate_limiter import RateLimiter, get_default_rate_limiter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Cache schema version for K 线 payloads. Bump whenever the
#: :class:`KLineSeries` shape or indicator semantics change so old
#: entries are invalidated (Requirement 11.3, Property 4).
_KLINE_SCHEMA_VERSION: Final[int] = 1

#: Cache "tool" namespace for K 线 payloads.
_KLINE_TOOL_NAMESPACE: Final[str] = "kline_service.series"

#: Accepted ``period`` values (Requirement 3.2).
_VALID_PERIODS: Final[frozenset[str]] = frozenset(
    {"daily", "weekly", "monthly", "60min", "30min"}
)

#: Accepted ``adjust`` values (Requirement 3.2).
_VALID_ADJUSTS: Final[frozenset[str]] = frozenset({"qfq", "hfq", "none"})

#: Indicator names accepted by :meth:`KLineService.get_series`. Any
#: other name triggers :class:`ValidationError` listing this exact set
#: (Requirement 3.6).
_SUPPORTED_INDICATORS: Final[frozenset[str]] = frozenset(
    {"MA5", "MA10", "MA20", "MA60", "MACD", "RSI14", "BOLL"}
)

#: Mapping of MA indicator name to window length.
_MA_WINDOWS: Final[dict[str, int]] = {
    "MA5": 5,
    "MA10": 10,
    "MA20": 20,
    "MA60": 60,
}

#: MACD parameters (DIF only; the standard fast/slow EMA difference).
_MACD_FAST: Final[int] = 12
_MACD_SLOW: Final[int] = 26

#: RSI period (Wilder's smoothing).
_RSI_PERIOD: Final[int] = 14

#: BOLL middle-band window (= MA20 by convention).
_BOLL_WINDOW: Final[int] = 20

#: Minimum number of bars required before computing ``pattern_note``
#: (Requirement 3.7).
_PATTERN_MIN_BARS: Final[int] = 60

#: Tolerance (relative) used when classifying "震荡" -- when the last
#: close is within this fraction of MA20 *and* MA20 is within this
#: fraction of MA60, the trend is considered range-bound. 1% matches
#: typical A 股 noise without dampening genuine moves.
_TREND_TOLERANCE: Final[float] = 0.01


# ---------------------------------------------------------------------------
# KLineService
# ---------------------------------------------------------------------------


class KLineService:
    """K 线序列 + 技术指标计算服务.

    Parameters
    ----------
    primary:
        Primary :class:`BaseAdapter` used for upstream calls
        (typically :class:`AkshareAdapter`).
    fallback:
        Optional backup adapter; when ``None``, transient failures from
        ``primary`` propagate verbatim.
    cache:
        Optional :class:`Cache` injection. Defaults to the process-wide
        instance returned by :func:`get_default_cache`.
    rate_limiter:
        Optional :class:`RateLimiter` injection. Defaults to the
        process-wide instance from :func:`get_default_rate_limiter`.
    """

    __slots__ = ("_cache", "_fallback", "_primary", "_rate_limiter")

    def __init__(
        self,
        primary: BaseAdapter,
        fallback: BaseAdapter | None = None,
        *,
        cache: Cache | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._primary: BaseAdapter = primary
        self._fallback: BaseAdapter | None = fallback
        self._cache: Cache = cache if cache is not None else get_default_cache()
        self._rate_limiter: RateLimiter = (
            rate_limiter if rate_limiter is not None else get_default_rate_limiter()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_series(
        self,
        symbol: str,
        period: str,
        count: int,
        adjust: str,
        indicators: list[str],
    ) -> KLineSeries:
        """Fetch K 线 bars and attach the requested indicators.

        Pipeline:

        1. Validate ``period`` / ``adjust`` / ``count`` / ``indicators``.
           Each failure raises :class:`ValidationError` whose message
           lists the offending value plus the accepted set.
        2. Normalize ``symbol`` via :func:`normalize_symbol` so an
           A-share input may be bare or already-standardized.
        3. Read-through cache keyed by
           ``(symbol, period, adjust, count)`` with TTL_WARM.
        4. On a cache miss, acquire one rate-limit token and call
           :func:`fetch_with_fallback` against the adapter's ``kline``
           endpoint.
        5. Compute the requested indicators and ``pattern_note`` and
           return a fresh :class:`KLineSeries` instance (the cached
           bar list is **not** mutated, so cached series stay reusable
           across calls with different ``indicators`` selections).

        Raises
        ------
        ValidationError
            If any of ``period`` / ``adjust`` / ``count`` /
            ``indicators`` is invalid.
        ChinaStockMCPError
            Any subclass raised by the adapter is propagated verbatim.
        """

        # 1) Strict input validation.
        if period not in _VALID_PERIODS:
            raise ValidationError(
                f"period 必须是 {sorted(_VALID_PERIODS)} 之一, "
                f"实际收到 {period!r}"
            )
        if adjust not in _VALID_ADJUSTS:
            raise ValidationError(
                f"adjust 必须是 {sorted(_VALID_ADJUSTS)} 之一, "
                f"实际收到 {adjust!r}"
            )
        if not isinstance(count, int) or isinstance(count, bool):
            # ``bool`` subclasses ``int`` in Python; reject it explicitly.
            raise ValidationError(
                f"count 必须是 int 类型, 实际收到 {type(count).__name__}"
            )
        if count < 1 or count > MAX_KLINE_BARS:
            raise ValidationError(
                f"count 必须在 [1, {MAX_KLINE_BARS}] 之间, 实际收到 {count}"
            )

        requested = list(indicators)
        unsupported = [name for name in requested if name not in _SUPPORTED_INDICATORS]
        if unsupported:
            raise ValidationError(
                f"不支持的指标: {unsupported}. "
                f"支持的指标: {sorted(_SUPPORTED_INDICATORS)}"
            )

        # 2) Symbol normalization.
        std_symbol = normalize_symbol(symbol)

        # 3) Read-through cache for the *bar* payload (indicators are
        #    derived per-request so different indicator selections share
        #    the same cached bars).
        params: dict[str, object] = {
            "period": period,
            "adjust": adjust,
            "count": count,
        }
        cached_series: KLineSeries = cache_get_or_fetch(
            tool=_KLINE_TOOL_NAMESPACE,
            symbol=std_symbol,
            params=params,
            ttl=TTL_WARM,
            fetcher=lambda: self._fetch_series(std_symbol, period, count, adjust),
            schema_version=_KLINE_SCHEMA_VERSION,
            cache=self._cache,
        )

        # 4) Compute indicators on top of the cached bars without
        #    mutating the cached object.
        indicator_series = _compute_indicators(cached_series.bars, requested)
        pattern_note = _derive_pattern_note(cached_series.bars)

        return KLineSeries(
            symbol=cached_series.symbol,
            period=cast(KLinePeriod, period),
            adjust=cast(AdjustMode, adjust),
            bars=cached_series.bars,
            indicators=indicator_series,
            pattern_note=pattern_note,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_series(
        self,
        std_symbol: str,
        period: str,
        count: int,
        adjust: str,
    ) -> KLineSeries:
        """Acquire one rate-limit token and call ``adapter.kline`` once."""

        self._rate_limiter.acquire()

        primary = self._primary
        fallback = self._fallback

        def _primary_call() -> KLineSeries:
            return primary.kline(std_symbol, period, count, adjust)

        def _fallback_call() -> KLineSeries:
            assert fallback is not None  # narrow for mypy --strict
            return fallback.kline(std_symbol, period, count, adjust)

        return fetch_with_fallback(
            primary=_primary_call,
            fallback=_fallback_call if fallback is not None else None,
            primary_name=primary.name,
            fallback_name=fallback.name if fallback is not None else "none",
        )


# ---------------------------------------------------------------------------
# Indicator helpers (pure functions)
# ---------------------------------------------------------------------------


def _compute_indicators(
    bars: list[KLineBar],
    requested: list[str],
) -> dict[str, list[float]]:
    """Compute every requested indicator series.

    The output keys preserve insertion order matching ``requested``.
    Every returned list has length ``len(bars)``; positions where the
    rolling window is incomplete are filled with NaN so DTO validation
    (Property 9, ``len(indicator) == len(bars)``) holds without
    distorting the meaningful tail.
    """

    closes = [bar.close for bar in bars]
    out: dict[str, list[float]] = {}

    for name in requested:
        if name in _MA_WINDOWS:
            out[name] = _rolling_mean(closes, _MA_WINDOWS[name])
        elif name == "MACD":
            out["MACD"] = _macd_dif(closes, _MACD_FAST, _MACD_SLOW)
        elif name == "RSI14":
            out["RSI14"] = _rsi_wilder(closes, _RSI_PERIOD)
        elif name == "BOLL":
            # Spec: BOLL middle band only, emitted under the
            # ``BOLL_MID`` key so callers can later add upper / lower
            # bands without changing the existing key.
            out["BOLL_MID"] = _rolling_mean(closes, _BOLL_WINDOW)
        # _SUPPORTED_INDICATORS gating in :meth:`get_series` makes the
        # final ``else`` branch unreachable; assert defensively.

    return out


def _rolling_mean(values: list[float], window: int) -> list[float]:
    """Simple rolling mean, NaN-padded for the leading ``window-1`` rows."""

    n = len(values)
    if window <= 0:
        # Defensive -- never reached because windows are constants > 0.
        return [math.nan] * n  # pragma: no cover - defensive

    out: list[float] = [math.nan] * n
    if n < window:
        return out

    running_sum = sum(values[:window])
    out[window - 1] = running_sum / window
    for i in range(window, n):
        running_sum += values[i] - values[i - window]
        out[i] = running_sum / window
    return out


def _ema(values: list[float], window: int) -> list[float]:
    """Exponential moving average aligned with the input list.

    Seeded with the simple mean of the first ``window`` values so the
    first ``window-1`` outputs are NaN; subsequent outputs use the
    standard ``alpha = 2 / (window + 1)`` smoothing. Matches the
    convention used by most charting libraries (e.g. TA-Lib EMA).
    """

    n = len(values)
    out: list[float] = [math.nan] * n
    if n < window or window <= 0:
        return out

    alpha = 2.0 / (window + 1.0)
    seed = sum(values[:window]) / window
    out[window - 1] = seed
    prev = seed
    for i in range(window, n):
        prev = alpha * values[i] + (1.0 - alpha) * prev
        out[i] = prev
    return out


def _macd_dif(values: list[float], fast: int, slow: int) -> list[float]:
    """MACD DIF line = EMA(fast) - EMA(slow), NaN-padded.

    Only the DIF line is exposed (per task 10.1 spec); callers that
    need DEA / histogram can be added later without breaking this
    output's shape.
    """

    n = len(values)
    if n == 0:
        return []
    fast_ema = _ema(values, fast)
    slow_ema = _ema(values, slow)
    out: list[float] = [math.nan] * n
    for i in range(n):
        f = fast_ema[i]
        s = slow_ema[i]
        if math.isnan(f) or math.isnan(s):
            continue
        out[i] = f - s
    return out


def _rsi_wilder(values: list[float], period: int) -> list[float]:
    """Wilder's RSI(14) implementation, NaN-padded for the warmup window.

    The first ``period`` outputs are NaN (need ``period`` differences,
    which require ``period+1`` closes). Output position ``period`` uses
    the simple average of the first ``period`` gains / losses; later
    positions use Wilder's smoothing
    (``avg = (avg * (period - 1) + current) / period``).
    """

    n = len(values)
    out: list[float] = [math.nan] * n
    if n <= period or period <= 0:
        return out

    gains: list[float] = [0.0] * n
    losses: list[float] = [0.0] * n
    for i in range(1, n):
        delta = values[i] - values[i - 1]
        if delta > 0:
            gains[i] = delta
        elif delta < 0:
            losses[i] = -delta

    # Seed averages over the first ``period`` deltas (indices 1..period).
    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period
    out[period] = _rsi_from_avgs(avg_gain, avg_loss)

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i] = _rsi_from_avgs(avg_gain, avg_loss)
    return out


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    """Convert Wilder rolling averages into an RSI value in ``[0, 100]``."""

    if avg_loss == 0.0:
        # All recent moves were non-negative -> maximum strength.
        return 100.0 if avg_gain > 0.0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _derive_pattern_note(bars: list[KLineBar]) -> str | None:
    """Classify the trailing trend based on close vs MA20 / MA60.

    Returns ``None`` when fewer than :data:`_PATTERN_MIN_BARS` bars are
    available (Requirement 3.7); otherwise compares the last close
    with the trailing MA20 and MA60:

    - close > MA20 > MA60 (with margin)  → ``"上升趋势"``
    - close < MA20 < MA60 (with margin)  → ``"下降趋势"``
    - everything else                    → ``"震荡"``
    """

    if len(bars) < _PATTERN_MIN_BARS:
        return None

    closes = [bar.close for bar in bars]
    ma20 = _rolling_mean(closes, _MA_WINDOWS["MA20"])[-1]
    ma60 = _rolling_mean(closes, _MA_WINDOWS["MA60"])[-1]
    last_close = closes[-1]

    if math.isnan(ma20) or math.isnan(ma60):
        return None

    # Use ratios so the threshold scales with price level. ``ma60``
    # cannot be zero in practice (price is non-negative and the window
    # average of ``MA60`` would only be zero if every close were zero,
    # which is rejected upstream), but guard against the degenerate
    # case anyway to avoid ``ZeroDivisionError``.
    if ma60 == 0.0 or ma20 == 0.0:
        return "震荡"

    close_vs_ma20 = (last_close - ma20) / ma20
    ma20_vs_ma60 = (ma20 - ma60) / ma60

    if close_vs_ma20 > _TREND_TOLERANCE and ma20_vs_ma60 > _TREND_TOLERANCE:
        return "上升趋势"
    if close_vs_ma20 < -_TREND_TOLERANCE and ma20_vs_ma60 < -_TREND_TOLERANCE:
        return "下降趋势"
    return "震荡"


__all__ = ["KLineService"]
