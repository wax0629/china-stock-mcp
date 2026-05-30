"""Tests for :mod:`china_stock_mcp.cache`.

Covers tasks 3.2, 3.3 and 3.4 of the spec at
``.kiro/specs/china-stock-mcp/tasks.md``:

- Property 3 (Requirements 11.2): cache key stability under dict
  insertion-order permutation.
- Property 4 (Requirements 11.3): schema upgrade invalidates cache
  keys for the same ``(tool, symbol, params)`` triple.
- Property 18 (Requirements 11.4, 11.5): TTL grade constants and
  backend selection via :func:`build_cache`.
"""

from __future__ import annotations

import string
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from china_stock_mcp import cache as cache_mod
from china_stock_mcp.cache import (
    TTL_COLD,
    TTL_FROZEN,
    TTL_HOT,
    TTL_STATIC,
    TTL_WARM,
    DiskCache,
    build_cache,
    cache_get_or_fetch,
    make_key,
    reset_default_cache,
    stable_json,
)
from china_stock_mcp.config import Settings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_default_cache() -> Iterator[None]:
    """Ensure each test starts with a clean process-wide cache."""

    reset_default_cache()
    try:
        yield
    finally:
        reset_default_cache()


# ---------------------------------------------------------------------------
# Hypothesis strategies for nested JSON-compatible params
# ---------------------------------------------------------------------------

# Keep alphabets restricted enough that shrinking is fast yet still
# exercises non-ASCII (Chinese) characters, since `stable_json` keeps
# Chinese intact and we want to catch any encoding regression.
_KEY_ALPHABET = string.ascii_letters + string.digits + "_-中文"

_PRIMITIVE = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-10_000, max_value=10_000),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(alphabet=_KEY_ALPHABET, min_size=0, max_size=12),
)


def _params_strategy() -> st.SearchStrategy[dict[str, Any]]:
    """Nested dict/list/primitive strategy with bounded depth."""

    keys = st.text(alphabet=_KEY_ALPHABET, min_size=1, max_size=8)

    def _extend(children: st.SearchStrategy[Any]) -> st.SearchStrategy[Any]:
        return st.one_of(
            st.lists(children, max_size=4),
            st.dictionaries(keys, children, max_size=4),
        )

    leaf = _PRIMITIVE
    nested = st.recursive(leaf, _extend, max_leaves=8)
    return st.dictionaries(keys, nested, max_size=4)


def _shuffle_dict_keys(d: dict[str, Any], rng_seed: int) -> dict[str, Any]:
    """Return a new dict whose keys are inserted in a different order.

    Recursively rebuilds nested dicts so insertion order differs at
    every level. Lists are NOT reordered (their order is semantic).
    """

    import random

    rng = random.Random(rng_seed)

    def _rebuild(value: Any) -> Any:
        if isinstance(value, dict):
            keys = list(value.keys())
            rng.shuffle(keys)
            return {k: _rebuild(value[k]) for k in keys}
        if isinstance(value, list):
            return [_rebuild(v) for v in value]
        return value

    rebuilt = _rebuild(d)
    assert isinstance(rebuilt, dict)
    return rebuilt


# ---------------------------------------------------------------------------
# Property 3: cache key stability (Requirements 11.2)
# ---------------------------------------------------------------------------


class TestCacheKeyStability:
    """**Validates: Requirements 11.2** (Property 3)."""

    @given(params=_params_strategy(), seed=st.integers(min_value=0, max_value=10_000))
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_make_key_stable_under_key_reordering(
        self, params: dict[str, Any], seed: int
    ) -> None:
        """Equal-by-value params yield identical keys regardless of key order."""

        permuted = _shuffle_dict_keys(params, seed)
        # Sanity check: equal by value despite possibly different ordering
        assert permuted == params

        key_a = make_key("get_quote", "300750.SZ", params, schema_version=1)
        key_b = make_key("get_quote", "300750.SZ", permuted, schema_version=1)
        assert key_a == key_b

    @given(params=_params_strategy(), seed=st.integers(min_value=0, max_value=10_000))
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_stable_json_invariant_under_key_reordering(
        self, params: dict[str, Any], seed: int
    ) -> None:
        """``stable_json`` is the deterministic primitive behind ``make_key``."""

        permuted = _shuffle_dict_keys(params, seed)
        assert stable_json(params) == stable_json(permuted)


# ---------------------------------------------------------------------------
# Property 4: schema upgrade invalidation (Requirements 11.3)
# ---------------------------------------------------------------------------


class TestSchemaUpgradeInvalidation:
    """**Validates: Requirements 11.3** (Property 4)."""

    @given(
        tool=st.sampled_from(
            [
                "get_quote",
                "get_kline",
                "get_fundamentals",
                "get_money_flow",
                "screen_stocks",
            ]
        ),
        symbol=st.sampled_from(
            ["300750.SZ", "600519.SH", "00700.HK", "159915", "_"]
        ),
        params=_params_strategy(),
        v1=st.integers(min_value=1, max_value=100),
        v2=st.integers(min_value=1, max_value=100),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_distinct_schema_versions_yield_distinct_keys(
        self,
        tool: str,
        symbol: str,
        params: dict[str, Any],
        v1: int,
        v2: int,
    ) -> None:
        assume(v1 != v2)
        key_v1 = make_key(tool, symbol, params, schema_version=v1)
        key_v2 = make_key(tool, symbol, params, schema_version=v2)
        assert key_v1 != key_v2
        # Sanity: each key still ends with the corresponding suffix.
        assert key_v1.endswith(f":v{v1}")
        assert key_v2.endswith(f":v{v2}")


# ---------------------------------------------------------------------------
# Property 18 / TTL & backend selection (Requirements 11.4, 11.5)
# ---------------------------------------------------------------------------


class TestTtlConstants:
    """**Validates: Requirements 11.4** (Property 18)."""

    def test_ttl_hot_is_60_seconds(self) -> None:
        assert TTL_HOT == 60

    def test_ttl_warm_is_300_seconds(self) -> None:
        assert TTL_WARM == 300

    def test_ttl_cold_is_3600_seconds(self) -> None:
        assert TTL_COLD == 3600

    def test_ttl_frozen_is_86400_seconds(self) -> None:
        assert TTL_FROZEN == 86_400

    def test_ttl_static_is_604800_seconds(self) -> None:
        assert TTL_STATIC == 604_800

    def test_ttl_grades_are_strictly_increasing(self) -> None:
        """Tier ordering reflects the design's hot→static progression."""

        assert TTL_HOT < TTL_WARM < TTL_COLD < TTL_FROZEN < TTL_STATIC


class TestBackendSelection:
    """**Validates: Requirements 11.5**."""

    def test_build_cache_disk_returns_diskcache_instance(
        self, tmp_path: Path
    ) -> None:
        settings = Settings(cache_backend="disk", cache_dir=tmp_path)
        cache = build_cache(settings)
        try:
            assert isinstance(cache, DiskCache)
        finally:
            cache.close()

    def test_build_cache_disk_creates_cache_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "csm-cache"
        assert not target.exists()
        settings = Settings(cache_backend="disk", cache_dir=target)
        cache = build_cache(settings)
        try:
            assert target.is_dir()
        finally:
            cache.close()

    def test_disk_backend_round_trips_payload_and_expires(
        self, tmp_path: Path
    ) -> None:
        settings = Settings(cache_backend="disk", cache_dir=tmp_path)
        cache = build_cache(settings)
        try:
            key = cache.make_key(
                "get_quote", "300750.SZ", {"foo": 1}, schema_version=1
            )
            payload = {"price": 123.45, "name": "宁德时代"}
            cache.set(key, payload, ttl=60)

            assert cache.get(key) == payload

            # ``set`` rejects non-positive TTLs (Requirements 11.4 mandates a
            # tier-based positive TTL for every cached item).
            with pytest.raises(ValueError):
                cache.set(key, payload, ttl=0)
            with pytest.raises(ValueError):
                cache.set(key, payload, ttl=-5)
        finally:
            cache.close()


# ---------------------------------------------------------------------------
# cache_get_or_fetch behaviour (Requirements 11.1, 11.3, plus guards)
# ---------------------------------------------------------------------------


class _StubCache:
    """In-memory cache used to assert ``cache_get_or_fetch`` semantics.

    The stub deliberately matches the :class:`Cache` protocol without
    introducing a real backend so the tests stay hermetic.
    """

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.set_calls: list[tuple[str, int]] = []

    def get(self, key: str) -> Any | None:
        return self.store.get(key)

    def set(self, key: str, value: Any, ttl: int) -> None:
        if ttl <= 0:
            raise ValueError(f"ttl must be > 0, got {ttl}")
        self.store[key] = value
        self.set_calls.append((key, ttl))

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


class TestCacheGetOrFetch:
    """**Validates: Requirements 11.1, 11.3, 11.4**."""

    def test_miss_invokes_fetcher_and_stores_payload(self) -> None:
        cache = _StubCache()
        calls = {"n": 0}

        def fetcher() -> dict[str, Any]:
            calls["n"] += 1
            return {"price": 1.23}

        result = cache_get_or_fetch(
            "get_quote",
            "300750.SZ",
            {"foo": "bar"},
            ttl=TTL_HOT,
            fetcher=fetcher,
            schema_version=1,
            cache=cache,
        )

        assert result == {"price": 1.23}
        assert calls["n"] == 1
        assert len(cache.store) == 1
        assert cache.set_calls[0][1] == TTL_HOT

    def test_hit_does_not_call_fetcher(self) -> None:
        cache = _StubCache()

        def fetcher() -> dict[str, Any]:
            return {"price": 1.23}

        # Prime cache via a first call.
        cache_get_or_fetch(
            "get_quote",
            "300750.SZ",
            {"foo": "bar"},
            ttl=TTL_HOT,
            fetcher=fetcher,
            schema_version=1,
            cache=cache,
        )

        sentinel = {"called": False}

        def boom() -> dict[str, Any]:
            sentinel["called"] = True
            raise AssertionError("fetcher must not be called on cache hit")

        result = cache_get_or_fetch(
            "get_quote",
            "300750.SZ",
            {"foo": "bar"},
            ttl=TTL_HOT,
            fetcher=boom,
            schema_version=1,
            cache=cache,
        )

        assert result == {"price": 1.23}
        assert sentinel["called"] is False

    def test_schema_version_bump_triggers_refetch(self) -> None:
        """**Validates: Requirements 11.3** at the read-through layer."""

        cache = _StubCache()
        invocations: list[int] = []

        def make_fetcher(schema: int):
            def _fetch() -> dict[str, Any]:
                invocations.append(schema)
                return {"schema": schema}

            return _fetch

        first = cache_get_or_fetch(
            "get_fundamentals",
            "300750.SZ",
            {"period": "annual"},
            ttl=TTL_FROZEN,
            fetcher=make_fetcher(1),
            schema_version=1,
            cache=cache,
        )
        second = cache_get_or_fetch(
            "get_fundamentals",
            "300750.SZ",
            {"period": "annual"},
            ttl=TTL_FROZEN,
            fetcher=make_fetcher(2),
            schema_version=2,
            cache=cache,
        )

        assert first == {"schema": 1}
        assert second == {"schema": 2}
        assert invocations == [1, 2]
        # Two distinct keys end up in the store.
        assert len(cache.store) == 2

    def test_dict_key_order_does_not_cause_miss(self) -> None:
        """**Validates: Requirements 11.2** at the read-through layer."""

        cache = _StubCache()
        calls = {"n": 0}

        def fetcher() -> dict[str, Any]:
            calls["n"] += 1
            return {"value": 42}

        params_a = {"a": 1, "b": 2, "c": 3}
        params_b = {"c": 3, "b": 2, "a": 1}

        cache_get_or_fetch(
            "get_quote",
            "300750.SZ",
            params_a,
            ttl=TTL_HOT,
            fetcher=fetcher,
            schema_version=1,
            cache=cache,
        )
        cache_get_or_fetch(
            "get_quote",
            "300750.SZ",
            params_b,
            ttl=TTL_HOT,
            fetcher=fetcher,
            schema_version=1,
            cache=cache,
        )

        assert calls["n"] == 1
        assert len(cache.store) == 1

    def test_ttl_must_be_positive(self) -> None:
        cache = _StubCache()

        def fetcher() -> dict[str, Any]:  # pragma: no cover - never reached
            return {}

        for bad_ttl in (0, -1, -3600):
            with pytest.raises(ValueError, match="ttl"):
                cache_get_or_fetch(
                    "get_quote",
                    "300750.SZ",
                    {},
                    ttl=bad_ttl,
                    fetcher=fetcher,
                    schema_version=1,
                    cache=cache,
                )

    def test_schema_version_must_be_at_least_one(self) -> None:
        cache = _StubCache()

        def fetcher() -> dict[str, Any]:  # pragma: no cover - never reached
            return {}

        for bad_version in (0, -1, -10):
            with pytest.raises(ValueError, match="schema_version"):
                cache_get_or_fetch(
                    "get_quote",
                    "300750.SZ",
                    {},
                    ttl=TTL_HOT,
                    fetcher=fetcher,
                    schema_version=bad_version,
                    cache=cache,
                )

    def test_uses_default_cache_when_none_supplied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sanity check: omitting ``cache`` falls back to the process default."""

        # Force the default cache to materialize a disk backend rooted at
        # tmp_path so this test does not touch the user's home cache.
        settings = Settings(cache_backend="disk", cache_dir=tmp_path)
        monkeypatch.setattr(cache_mod, "load_settings", lambda: settings)
        reset_default_cache()

        calls = {"n": 0}

        def fetcher() -> dict[str, Any]:
            calls["n"] += 1
            return {"price": 9.99}

        result_one = cache_get_or_fetch(
            "get_quote",
            "300750.SZ",
            {"foo": 1},
            ttl=TTL_HOT,
            fetcher=fetcher,
            schema_version=1,
        )
        result_two = cache_get_or_fetch(
            "get_quote",
            "300750.SZ",
            {"foo": 1},
            ttl=TTL_HOT,
            fetcher=fetcher,
            schema_version=1,
        )

        assert result_one == {"price": 9.99}
        assert result_two == {"price": 9.99}
        assert calls["n"] == 1

    def test_make_key_rejects_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="tool"):
            make_key("", "300750.SZ", {}, schema_version=1)
        with pytest.raises(ValueError, match="symbol"):
            make_key("get_quote", "", {}, schema_version=1)
        with pytest.raises(ValueError, match="schema_version"):
            make_key("get_quote", "300750.SZ", {}, schema_version=0)
