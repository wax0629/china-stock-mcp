"""Global Token Bucket rate limiter for china-stock-mcp.

This module implements design Algorithm 4 (`rate_limit_check`) and the
non-functional requirements 11.6 / 11.7 / 11.8:

- 11.6 / Property 7 -- when ``rate_limit_check`` returns ``True``, the
  bucket had ``tokens >= 1`` before the call and exactly one token was
  consumed.
- 11.7 -- callers must raise :class:`RateLimitError` when the bucket
  rejects a request; this module exposes :meth:`RateLimiter.acquire`
  for that exact purpose so adapter code does not duplicate the
  message.
- 11.8 -- the default budget is 30 requests per minute and is
  overridable via ``CSM_RATE_LIMIT`` (read through
  :class:`Settings.rate_limit`).

The :class:`TokenBucket` dataclass is the pure state container; the
:func:`rate_limit_check` function applies Algorithm 4 in-place and is
intentionally side-effect free apart from mutating the bucket. The
:class:`RateLimiter` class wraps a single bucket with a
``threading.Lock`` so the FastMCP server may call into it from
arbitrary worker threads without losing tokens to a race.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

from china_stock_mcp.config import Settings, load_settings
from china_stock_mcp.exceptions import RateLimitError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of seconds in one minute. Centralized so tests can reference
#: the same constant when reasoning about the refill rate.
_SECONDS_PER_MINUTE: float = 60.0

#: Default user-facing message for :class:`RateLimitError` raised by
#: :meth:`RateLimiter.acquire`. Matches the wording suggested in the
#: design's "Error Handling" section for Scenario 2.
_RATE_LIMIT_MESSAGE: str = "数据源调用频率过高，请稍后重试"  # noqa: RUF001


# ---------------------------------------------------------------------------
# Token bucket state + Algorithm 4
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TokenBucket:
    """Mutable token-bucket state used by :func:`rate_limit_check`.

    Parameters
    ----------
    capacity:
        Maximum tokens the bucket can hold. Must be positive.
    refill_rate:
        Tokens added per second. Must be positive.
    tokens:
        Current token balance. Defaults to ``capacity`` (a fully
        primed bucket) when omitted; callers may pass an explicit
        starting value for tests. ``NaN`` is treated as the sentinel
        for "use the default" because it survives ``slots=True``
        without forcing every caller to recompute the default.
    last_ts:
        Last timestamp the bucket was updated, in seconds. The
        absolute origin does not matter (``time.monotonic`` reference
        works fine) as long as ``now`` passed to
        :func:`rate_limit_check` uses the same clock.

    Raises
    ------
    ValueError
        When ``capacity <= 0``, ``refill_rate <= 0``, or ``tokens`` is
        negative / exceeds ``capacity``.
    """

    capacity: float
    refill_rate: float
    tokens: float = math.nan
    last_ts: float = 0.0

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(
                f"capacity must be > 0, got {self.capacity}"
            )
        if self.refill_rate <= 0:
            raise ValueError(
                f"refill_rate must be > 0, got {self.refill_rate}"
            )
        if math.isnan(self.tokens):
            # Default to a fully primed bucket so the first request
            # never blocks. This matches the design's "首次启动" note
            # and keeps tests deterministic.
            self.tokens = self.capacity
            return
        if self.tokens < 0:
            raise ValueError(
                f"tokens must be >= 0, got {self.tokens}"
            )
        if self.tokens > self.capacity:
            raise ValueError(
                f"tokens must be <= capacity ({self.capacity}), "
                f"got {self.tokens}"
            )


def rate_limit_check(bucket: TokenBucket, now: float) -> bool:
    """Apply design Algorithm 4 in-place and report admission.

    Pseudocode::

        elapsed ← max(0, now - bucket.last_ts)
        bucket.tokens ← min(capacity, tokens + elapsed * refill_rate)
        bucket.last_ts ← now
        IF tokens >= 1: tokens -= 1; return True
        ELSE: return False

    Mutates ``bucket`` regardless of the return value (``last_ts`` is
    always advanced to ``now`` when ``now`` is in the future of the
    previous timestamp). Returning ``True`` guarantees the caller
    consumed exactly one token (Property 7 / Requirements 11.6).

    Parameters
    ----------
    bucket:
        The :class:`TokenBucket` to mutate.
    now:
        Current timestamp in seconds, on the same clock as
        ``bucket.last_ts``. Values strictly less than ``last_ts`` are
        clamped (``elapsed`` becomes ``0``) so a non-monotonic
        timestamp source cannot inflate the bucket past its capacity.

    Returns
    -------
    bool
        ``True`` if a token was consumed, ``False`` otherwise. When
        ``False`` is returned, callers should raise
        :class:`RateLimitError` (Requirements 11.7).
    """

    elapsed = max(0.0, now - bucket.last_ts)
    bucket.tokens = min(
        bucket.capacity,
        bucket.tokens + elapsed * bucket.refill_rate,
    )
    bucket.last_ts = now

    if bucket.tokens >= 1:
        bucket.tokens -= 1
        return True
    return False


# ---------------------------------------------------------------------------
# Thread-safe RateLimiter wrapper
# ---------------------------------------------------------------------------


class RateLimiter:
    """Thread-safe wrapper around a single :class:`TokenBucket`.

    The bucket starts full so the first ``rate_per_minute`` requests
    are admitted instantly, after which admission is paced by
    ``refill_rate = rate_per_minute / 60`` tokens/second. Set
    ``capacity`` explicitly to allow shorter or longer bursts than the
    default per-minute budget.

    Parameters
    ----------
    rate_per_minute:
        Steady-state budget (Requirements 11.8 default = ``30``).
        Must be positive.
    capacity:
        Maximum burst size in tokens. Defaults to ``rate_per_minute``
        so a full minute's worth of calls may run back-to-back.

    Notes
    -----
    All public methods take an optional ``now`` parameter so tests can
    drive the limiter with a deterministic clock; production callers
    should leave it ``None`` to use :func:`time.monotonic`.
    """

    __slots__ = ("_bucket", "_lock", "rate_per_minute")

    def __init__(
        self,
        rate_per_minute: int,
        capacity: int | None = None,
    ) -> None:
        if rate_per_minute <= 0:
            raise ValueError(
                f"rate_per_minute must be > 0, got {rate_per_minute}"
            )
        if capacity is not None and capacity <= 0:
            raise ValueError(
                f"capacity must be > 0, got {capacity}"
            )

        bucket_capacity = float(
            capacity if capacity is not None else rate_per_minute
        )
        refill_rate = rate_per_minute / _SECONDS_PER_MINUTE

        self.rate_per_minute = rate_per_minute
        self._bucket = TokenBucket(
            capacity=bucket_capacity,
            refill_rate=refill_rate,
            tokens=bucket_capacity,
            last_ts=time.monotonic(),
        )
        self._lock = threading.Lock()

    # ----- public API -----------------------------------------------------

    def try_acquire(self, now: float | None = None) -> bool:
        """Attempt to consume one token; return whether it succeeded.

        Used by callers that want to perform their own bookkeeping
        before raising. Most code should prefer :meth:`acquire`.
        """

        timestamp = now if now is not None else time.monotonic()
        with self._lock:
            return rate_limit_check(self._bucket, timestamp)

    def acquire(self, now: float | None = None) -> None:
        """Consume one token or raise :class:`RateLimitError`.

        Implements the requirement 11.7 contract: the bucket is the
        single source of truth and the message is the unified one
        callers should surface to AI clients.
        """

        if not self.try_acquire(now):
            raise RateLimitError(_RATE_LIMIT_MESSAGE)

    # ----- introspection (mostly for tests) -------------------------------

    @property
    def capacity(self) -> float:
        """Maximum bucket size in tokens."""

        return self._bucket.capacity

    @property
    def refill_rate(self) -> float:
        """Refill rate in tokens per second."""

        return self._bucket.refill_rate

    def snapshot(self) -> TokenBucket:
        """Return a copy of the current bucket state.

        The returned dataclass is detached from the limiter; mutating
        it does not affect future :meth:`acquire` calls. Useful for
        tests that need to assert the post-condition of Property 7.
        """

        with self._lock:
            return TokenBucket(
                capacity=self._bucket.capacity,
                refill_rate=self._bucket.refill_rate,
                tokens=self._bucket.tokens,
                last_ts=self._bucket.last_ts,
            )


# ---------------------------------------------------------------------------
# Factory + process-wide singleton
# ---------------------------------------------------------------------------


def build_rate_limiter(settings: Settings | None = None) -> RateLimiter:
    """Construct a :class:`RateLimiter` from ``settings.rate_limit``.

    ``settings`` defaults to :func:`load_settings()` so callers can
    rely on the current environment (Requirements 11.8). The default
    when ``CSM_RATE_LIMIT`` is unset is ``30`` requests per minute
    (see :data:`china_stock_mcp.config.DEFAULT_RATE_LIMIT`).
    """

    cfg = settings if settings is not None else load_settings()
    return RateLimiter(rate_per_minute=cfg.rate_limit)


_default_rate_limiter: RateLimiter | None = None
_default_rate_limiter_lock = threading.Lock()


def get_default_rate_limiter() -> RateLimiter:
    """Return the process-wide limiter, building it on demand.

    Mirrors :func:`china_stock_mcp.cache.get_default_cache`: the first
    call materializes a limiter from the current settings; subsequent
    calls return the same instance. Tests that need isolation should
    use :func:`build_rate_limiter` directly or call
    :func:`reset_default_rate_limiter` between cases.
    """

    global _default_rate_limiter
    if _default_rate_limiter is not None:
        return _default_rate_limiter
    with _default_rate_limiter_lock:
        if _default_rate_limiter is None:
            _default_rate_limiter = build_rate_limiter()
        return _default_rate_limiter


def reset_default_rate_limiter() -> None:
    """Drop the cached default limiter (primarily for tests)."""

    global _default_rate_limiter
    with _default_rate_limiter_lock:
        _default_rate_limiter = None


__all__ = [
    "RateLimiter",
    "TokenBucket",
    "build_rate_limiter",
    "get_default_rate_limiter",
    "rate_limit_check",
    "reset_default_rate_limiter",
]
