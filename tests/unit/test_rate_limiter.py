"""Property and unit tests for :mod:`china_stock_mcp.rate_limiter`.

Covers task 6.2 of the china-stock-mcp spec:

- 6.2 Property 7 -- 限流单调性 (Validates: Requirements 11.6)

Property 7 says: when ``rate_limit_check(bucket, now)`` returns
``True``, the bucket had ``tokens >= 1`` *before* the call and the
balance decreased by exactly one token *after* the call. This file
also covers two important edge cases that protect the property:

- The refill phase never inflates the balance above ``capacity``
  (a side-condition of Algorithm 4 that keeps the property finite).
- A non-monotonic ``now`` (i.e. ``now < bucket.last_ts``) is clamped
  so the bucket cannot gain tokens by going back in time.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterator
from dataclasses import replace

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from china_stock_mcp.config import DEFAULT_RATE_LIMIT, load_settings
from china_stock_mcp.exceptions import RateLimitError
from china_stock_mcp.rate_limiter import (
    RateLimiter,
    TokenBucket,
    build_rate_limiter,
    get_default_rate_limiter,
    rate_limit_check,
    reset_default_rate_limiter,
)

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Capacity ∈ [1, 100] (integers since the API hands us ``int`` for the
# minute budget; the dataclass stores a float internally).
_capacity = st.integers(min_value=1, max_value=100).map(float)

# Refill rate ∈ (0.0, 100.0]. ``min_value`` is excluded because the
# bucket asserts ``refill_rate > 0`` in ``__post_init__``.
_refill_rate = st.floats(
    min_value=1e-6,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
)

# Timestamps in [0, 1_000_000] seconds. Bucket starts at ``last_ts=0``.
_timestamp = st.floats(
    min_value=0.0,
    max_value=1_000_000.0,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def _bucket_and_calls(
    draw: st.DrawFn,
) -> tuple[TokenBucket, list[float]]:
    """Build a fresh bucket and a sequence of call timestamps.

    The bucket starts full (``tokens == capacity``) at ``last_ts == 0``
    so the first calls in the sequence are admitted; later calls may
    or may not be admitted depending on the inter-arrival times.
    """

    capacity = draw(_capacity)
    refill_rate = draw(_refill_rate)
    raw_calls = draw(
        st.lists(_timestamp, min_size=1, max_size=20)
    )
    bucket = TokenBucket(
        capacity=capacity,
        refill_rate=refill_rate,
        tokens=capacity,
        last_ts=0.0,
    )
    return bucket, raw_calls


# ---------------------------------------------------------------------------
# Task 6.2 -- Property 7: 限流单调性 (Validates: Requirements 11.6)
# ---------------------------------------------------------------------------


@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(state=_bucket_and_calls())
def test_rate_limit_check_monotonic_decrement(
    state: tuple[TokenBucket, list[float]],
) -> None:
    """**Validates: Requirements 11.6**

    Property 7 -- whenever ``rate_limit_check`` returns ``True``:

    1. The bucket had ``tokens >= 1`` immediately before the call
       (after the refill step but before the consumption step), and
    2. The bucket's ``tokens`` decreased by exactly ``1.0`` from
       before to after the call.

    The check is repeated for every timestamp in a generated sequence
    so the property holds for sequences of admitted / rejected calls,
    not just isolated ones.
    """

    bucket, calls = state

    for now in calls:
        # Snapshot the would-be post-refill state without mutating the
        # real bucket: this is the value of ``tokens`` the
        # consumption step sees when it decides admission.
        elapsed = max(0.0, now - bucket.last_ts)
        post_refill_tokens = min(
            bucket.capacity,
            bucket.tokens + elapsed * bucket.refill_rate,
        )

        before = replace(bucket)
        admitted = rate_limit_check(bucket, now)

        if admitted:
            # (1) Pre-call (post-refill) balance was at least one token.
            assert post_refill_tokens >= 1.0, (
                f"admitted with insufficient tokens: "
                f"post_refill_tokens={post_refill_tokens!r}, "
                f"bucket_before={before!r}, now={now!r}"
            )
            # (2) Exactly one token was consumed.
            delta = post_refill_tokens - bucket.tokens
            assert math.isclose(delta, 1.0, rel_tol=1e-9, abs_tol=1e-9), (
                f"admitted call did not consume exactly one token: "
                f"delta={delta!r}, post_refill_tokens={post_refill_tokens!r}, "
                f"after={bucket!r}"
            )
        else:
            # When rejected, the post-refill balance must be < 1; the
            # algorithm still updates ``last_ts`` and the refilled
            # ``tokens`` value, but never consumes anything.
            assert post_refill_tokens < 1.0, (
                f"rejected even though tokens were available: "
                f"post_refill_tokens={post_refill_tokens!r}, "
                f"bucket_before={before!r}, now={now!r}"
            )
            assert math.isclose(
                bucket.tokens, post_refill_tokens, rel_tol=1e-9, abs_tol=1e-9
            )

        # The bucket never holds more than its capacity and never
        # drops below zero; these are side-conditions of Algorithm 4.
        assert -1e-9 <= bucket.tokens <= bucket.capacity + 1e-9


# ---------------------------------------------------------------------------
# Edge cases supporting the property
# ---------------------------------------------------------------------------


def test_tokens_never_exceed_capacity_with_long_idle() -> None:
    """A long idle period must not let ``tokens`` exceed ``capacity``.

    Requirements 11.6 implicitly relies on this cap -- without it, the
    bucket would behave like an unlimited reservoir after a quiet
    minute and "spend" more than ``capacity`` tokens in a burst,
    breaking the per-call decrement guarantee.
    """

    bucket = TokenBucket(
        capacity=10.0,
        refill_rate=1.0,
        tokens=10.0,
        last_ts=0.0,
    )

    # Pretend the bucket sat idle for 10 000 seconds.
    admitted = rate_limit_check(bucket, now=10_000.0)

    assert admitted is True
    # After consumption, ``tokens`` must be at most ``capacity``.
    assert bucket.tokens <= bucket.capacity
    # And specifically: capacity (10) - 1 consumed = 9.
    assert math.isclose(bucket.tokens, 9.0, rel_tol=1e-9, abs_tol=1e-9)


def test_non_monotonic_now_clamps_elapsed_to_zero() -> None:
    """``now < bucket.last_ts`` clamps ``elapsed`` to ``0``.

    A non-monotonic timestamp source (clock skew, daylight-saving
    jumps, tests injecting an explicit ``now``) must not retroactively
    refill the bucket, otherwise an attacker controlling timestamps
    could mint extra tokens.
    """

    bucket = TokenBucket(
        capacity=5.0,
        refill_rate=1.0,
        tokens=2.0,
        last_ts=1_000.0,
    )

    # ``now`` is in the past relative to ``last_ts``.
    admitted = rate_limit_check(bucket, now=500.0)

    assert admitted is True
    # No refill happened; we simply consumed one token from 2.0 → 1.0.
    assert math.isclose(bucket.tokens, 1.0, rel_tol=1e-9, abs_tol=1e-9)
    # ``last_ts`` is unconditionally advanced to the supplied ``now``.
    assert bucket.last_ts == 500.0


def test_rate_limit_check_rejects_when_bucket_empty() -> None:
    """When ``tokens < 1`` and no time has elapsed, the call is rejected
    and ``tokens`` is not driven negative."""

    bucket = TokenBucket(
        capacity=5.0,
        refill_rate=0.001,  # negligible refill within the test window
        tokens=0.0,
        last_ts=0.0,
    )

    admitted = rate_limit_check(bucket, now=0.0)

    assert admitted is False
    assert bucket.tokens == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Additional unit tests for RateLimiter wrapper
# ---------------------------------------------------------------------------


@pytest.fixture()
def _reset_singleton() -> Iterator[None]:
    """Reset the process-wide limiter singleton around each test.

    Avoids leaking state from earlier tests / fixtures that may have
    materialised the default limiter under a different env var value.
    """

    reset_default_rate_limiter()
    try:
        yield
    finally:
        reset_default_rate_limiter()


class TestRateLimiterDefaults:
    """Requirements 11.8 -- default 30 req/min when ``CSM_RATE_LIMIT`` unset."""

    def test_default_rate_per_minute_is_thirty(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _reset_singleton: None,
    ) -> None:
        monkeypatch.delenv("CSM_RATE_LIMIT", raising=False)
        # Re-read settings so the env var change takes effect.
        settings = load_settings()
        assert settings.rate_limit == DEFAULT_RATE_LIMIT == 30

        limiter = build_rate_limiter(settings)
        assert limiter.rate_per_minute == 30
        # Refill rate ⇒ 30 / 60 = 0.5 tokens/sec.
        assert math.isclose(limiter.refill_rate, 0.5, rel_tol=1e-9)
        # Capacity defaults to ``rate_per_minute`` so a full minute
        # of calls may run back-to-back at startup.
        assert math.isclose(limiter.capacity, 30.0, rel_tol=1e-9)


class TestRateLimiterEnvOverride:
    """Requirements 11.8 -- ``CSM_RATE_LIMIT`` env var overrides default."""

    @pytest.mark.parametrize("override", [1, 5, 60, 120, 600])
    def test_env_var_override_applies(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _reset_singleton: None,
        override: int,
    ) -> None:
        monkeypatch.setenv("CSM_RATE_LIMIT", str(override))
        settings = load_settings()
        assert settings.rate_limit == override

        limiter = build_rate_limiter(settings)
        assert limiter.rate_per_minute == override
        assert math.isclose(
            limiter.refill_rate, override / 60.0, rel_tol=1e-9
        )

    def test_env_var_override_propagates_through_singleton(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _reset_singleton: None,
    ) -> None:
        monkeypatch.setenv("CSM_RATE_LIMIT", "120")
        # First materialisation reads the env var via load_settings.
        limiter = get_default_rate_limiter()
        assert limiter.rate_per_minute == 120
        # Subsequent calls return the same instance.
        assert get_default_rate_limiter() is limiter

    def test_invalid_env_var_raises_value_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _reset_singleton: None,
    ) -> None:
        monkeypatch.setenv("CSM_RATE_LIMIT", "not-a-number")
        with pytest.raises(ValueError, match="CSM_RATE_LIMIT"):
            load_settings()

    @pytest.mark.parametrize("bad", ["0", "-1", "-30"])
    def test_non_positive_env_var_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _reset_singleton: None,
        bad: str,
    ) -> None:
        monkeypatch.setenv("CSM_RATE_LIMIT", bad)
        with pytest.raises(ValueError):
            load_settings()


class TestTryAcquireVsAcquireSemantics:
    """``try_acquire`` returns bool; ``acquire`` raises :class:`RateLimitError`."""

    def test_try_acquire_returns_true_when_admitted(self) -> None:
        limiter = RateLimiter(rate_per_minute=60, capacity=2)
        # Drive the limiter with deterministic ``now`` so the test does
        # not depend on wall-clock progression.
        assert limiter.try_acquire(now=0.0) is True
        assert limiter.try_acquire(now=0.0) is True

    def test_try_acquire_returns_false_when_empty(self) -> None:
        # capacity=1 with negligible refill ⇒ second call within the
        # same instant has no tokens left and is rejected.
        limiter = RateLimiter(rate_per_minute=1, capacity=1)
        assert limiter.try_acquire(now=0.0) is True
        assert limiter.try_acquire(now=0.0) is False

    def test_acquire_succeeds_silently_when_admitted(self) -> None:
        limiter = RateLimiter(rate_per_minute=60, capacity=5)
        # No exception, no return value.
        result = limiter.acquire(now=0.0)
        assert result is None

    def test_acquire_raises_rate_limit_error_when_empty(self) -> None:
        limiter = RateLimiter(rate_per_minute=1, capacity=1)
        limiter.acquire(now=0.0)  # consume the only token
        with pytest.raises(RateLimitError) as excinfo:
            limiter.acquire(now=0.0)
        # Requirements 11.7 -- the error carries the unified user-facing
        # message; we only assert the substring "频率" so the canonical
        # wording can evolve without breaking the test.
        assert "频率" in excinfo.value.message

    def test_try_acquire_does_not_raise_when_empty(self) -> None:
        """``try_acquire`` is the non-throwing counterpart of ``acquire``."""

        limiter = RateLimiter(rate_per_minute=1, capacity=1)
        limiter.acquire(now=0.0)
        # No exception, returns False instead.
        assert limiter.try_acquire(now=0.0) is False

    def test_both_apis_share_underlying_bucket(self) -> None:
        """``acquire`` and ``try_acquire`` consume from the same bucket."""

        limiter = RateLimiter(rate_per_minute=60, capacity=3)
        limiter.acquire(now=0.0)            # tokens: 3 → 2
        assert limiter.try_acquire(now=0.0) is True   # tokens: 2 → 1
        limiter.acquire(now=0.0)            # tokens: 1 → 0
        assert limiter.try_acquire(now=0.0) is False  # bucket empty


class TestRateLimiterThreadSafety:
    """Concurrent ``acquire`` calls must not hand out more tokens than
    the bucket's capacity (the lock protects against torn reads of
    ``bucket.tokens``)."""

    def test_concurrent_acquire_never_exceeds_capacity(self) -> None:
        # Pin ``now`` by passing an explicit timestamp; each thread
        # passes the same value so refill cannot mask a missing lock.
        capacity = 50
        threads_count = 100
        limiter = RateLimiter(
            rate_per_minute=capacity, capacity=capacity
        )

        admitted: list[bool] = []
        admitted_lock = threading.Lock()
        barrier = threading.Barrier(threads_count)

        def worker() -> None:
            # Synchronise so every thread races for the same tokens.
            barrier.wait()
            ok = limiter.try_acquire(now=0.0)
            with admitted_lock:
                admitted.append(ok)

        threads = [
            threading.Thread(target=worker) for _ in range(threads_count)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive(), "worker thread hung"

        admitted_count = sum(1 for ok in admitted if ok)
        # The contract: at most ``capacity`` admissions when ``now``
        # is fixed and refill therefore contributes zero tokens.
        assert admitted_count == capacity, (
            f"expected exactly {capacity} admissions, got {admitted_count}"
        )

        # Bucket balance is non-negative and matches the algorithm
        # post-condition (``capacity - admitted == 0`` here).
        snapshot = limiter.snapshot()
        assert snapshot.tokens == pytest.approx(0.0, abs=1e-9)

    def test_snapshot_returns_independent_copy(self) -> None:
        """Mutating the snapshot must not affect future admissions."""

        limiter = RateLimiter(rate_per_minute=60, capacity=5)
        snap = limiter.snapshot()
        snap.tokens = 999.0  # would be a free pass without isolation

        # The next admission still consumes from the real bucket.
        assert limiter.try_acquire(now=0.0) is True
        # The snapshot is unaffected by the real ``acquire`` call.
        assert snap.tokens == 999.0
