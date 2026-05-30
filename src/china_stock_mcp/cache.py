"""Cache layer for china-stock-mcp.

This module implements design Component 5 (`Cache Layer`) and
Algorithm 3 (`cache_get_or_fetch`). It provides:

- Two pluggable backends (``diskcache`` -- default; ``redis`` -- optional)
  selected at construction time via :class:`Settings.cache_backend`.
- A canonical, recursively sorted JSON serializer (:func:`stable_json`)
  used to build deterministic cache keys regardless of caller-supplied
  dict ordering (Property 3, Requirements 11.2).
- :func:`make_key`, which composes the
  ``{tool}:{symbol}:{sha1(stable_json(params))}:v{schema_version}``
  format mandated by Requirements 11.1; bumping ``schema_version``
  guarantees a different key (Property 4, Requirements 11.3).
- TTL grade constants (:data:`TTL_HOT` ... :data:`TTL_STATIC`) covering
  the five tiers in Requirements 11.4.
- :func:`cache_get_or_fetch`, the read-through helper every Service
  layer entry point calls (design Algorithm 3 / Requirements 11.1).

The module is intentionally side-effect free at import time. The first
call to :func:`get_default_cache` materializes a backend instance based
on the current environment; tests can construct backends directly.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final, Protocol, TypeVar, cast, runtime_checkable

from china_stock_mcp.config import Settings, load_settings

# ---------------------------------------------------------------------------
# TTL grades (seconds) -- design Component 5 / Requirements 11.4
# ---------------------------------------------------------------------------

TTL_HOT: Final[int] = 60          # 实时行情
TTL_WARM: Final[int] = 300        # K 线 / 资金流
TTL_COLD: Final[int] = 3600       # 基金净值 / 市场总览
TTL_FROZEN: Final[int] = 86400    # 财务数据
TTL_STATIC: Final[int] = 604800   # 公司信息 / 行业分类


T = TypeVar("T")


# ---------------------------------------------------------------------------
# Canonical JSON / key derivation
# ---------------------------------------------------------------------------


def _canonicalize(obj: Any) -> Any:
    """Recursively convert ``obj`` into a JSON-friendly canonical form.

    - ``Mapping`` instances become dicts with keys sorted as strings.
    - ``tuple`` / ``set`` / ``frozenset`` become lists; sets are sorted
      so equal sets yield identical output regardless of insertion
      order.
    - ``Path`` becomes its string form.
    - Other primitives (str, int, float, bool, None) pass through.
    - Anything else falls back to its ``repr`` so the helper never
      raises on unfamiliar types -- callers should still prefer simple
      JSON-compatible parameters.
    """

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _canonicalize(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(item) for item in obj]
    if isinstance(obj, (set, frozenset)):
        # Sort by canonical JSON of each element so heterogeneous
        # sets still produce a stable ordering.
        canon_items = [_canonicalize(item) for item in obj]
        return sorted(
            canon_items,
            key=lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False),
        )
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def stable_json(params: Mapping[str, Any] | Any) -> str:
    """Serialize ``params`` to a deterministic JSON string.

    Equal-by-value inputs produce byte-identical output (Property 3,
    Requirements 11.2). Dict keys are sorted recursively, separators
    are compact, and non-ASCII characters are preserved so Chinese
    parameter values (e.g. ``"金融"``) remain human-readable in logs.
    """

    return json.dumps(
        _canonicalize(params),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def make_key(
    tool: str,
    symbol: str,
    params: Mapping[str, Any],
    schema_version: int,
) -> str:
    """Compose the canonical cache key.

    Format::

        {tool}:{symbol}:{sha1(stable_json(params))}:v{schema_version}

    Requirements / properties:
    - 11.1, P3 -- equal ``stable_json(params)`` ⇒ equal key.
    - 11.3, P4 -- different ``schema_version`` ⇒ different key.

    Raises
    ------
    ValueError
        If ``tool`` or ``symbol`` is empty, or ``schema_version < 1``.
    """

    if not tool:
        raise ValueError("tool must be a non-empty string")
    if not symbol:
        raise ValueError("symbol must be a non-empty string")
    if schema_version < 1:
        raise ValueError(
            f"schema_version must be >= 1, got {schema_version}"
        )

    digest = _sha1(stable_json(params))
    return f"{tool}:{symbol}:{digest}:v{schema_version}"


# ---------------------------------------------------------------------------
# Backend protocol & implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class Cache(Protocol):
    """Backend-agnostic cache protocol.

    Implementations are expected to be safe for concurrent use from
    multiple threads (the FastMCP server may dispatch tools in
    parallel). ``set`` always honors ``ttl`` (seconds) and ``get``
    returns ``None`` for a miss.
    """

    def get(self, key: str) -> Any | None: ...

    def set(self, key: str, value: Any, ttl: int) -> None: ...

    def make_key(
        self,
        tool: str,
        symbol: str,
        params: Mapping[str, Any],
        schema_version: int,
    ) -> str: ...

    def close(self) -> None: ...


class _BaseCache:
    """Shared :meth:`make_key` implementation for backends.

    Backends only need to implement :meth:`get`, :meth:`set` and
    :meth:`close`; key derivation is centralized here so all backends
    produce identical keys for identical inputs.
    """

    def make_key(
        self,
        tool: str,
        symbol: str,
        params: Mapping[str, Any],
        schema_version: int,
    ) -> str:
        return make_key(tool, symbol, params, schema_version)

    def close(self) -> None:  # pragma: no cover - default no-op
        return None


class DiskCache(_BaseCache):
    """``diskcache``-backed implementation (default backend).

    Stores entries under ``cache_dir`` (created if missing). Values
    are pickled by ``diskcache`` natively so any picklable DTO is
    accepted; TTL is enforced by the library's ``expire`` argument.
    """

    def __init__(self, cache_dir: Path) -> None:
        # Lazy import keeps the module importable even if the optional
        # backend is somehow missing in a stripped-down install; in
        # practice ``diskcache`` is a hard dependency.
        import diskcache

        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(str(cache_dir))

    def get(self, key: str) -> Any | None:
        return self._cache.get(key, default=None)

    def set(self, key: str, value: Any, ttl: int) -> None:
        if ttl <= 0:
            raise ValueError(f"ttl must be > 0, got {ttl}")
        self._cache.set(key, value, expire=ttl)

    def close(self) -> None:
        self._cache.close()


class RedisCache(_BaseCache):
    """``redis``-backed implementation (optional backend).

    Values are pickled before being written to Redis since Redis only
    stores byte strings; this preserves the same DTO round-tripping
    semantics offered by :class:`DiskCache`. The connection URL is
    sourced from the ``CSM_REDIS_URL`` environment variable, falling
    back to ``redis://localhost:6379/0`` so local development works
    out of the box.
    """

    def __init__(self, url: str | None = None) -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "CSM_CACHE_BACKEND=redis requires the optional 'redis' "
                "extra: install with `pip install china-stock-mcp[redis]`."
            ) from exc

        import os

        resolved_url = url or os.environ.get(
            "CSM_REDIS_URL", "redis://localhost:6379/0"
        )
        self._client = redis.Redis.from_url(resolved_url)

    def get(self, key: str) -> Any | None:
        import pickle

        raw = cast(bytes | None, self._client.get(key))
        if raw is None:
            return None
        return pickle.loads(raw)

    def set(self, key: str, value: Any, ttl: int) -> None:
        import pickle

        if ttl <= 0:
            raise ValueError(f"ttl must be > 0, got {ttl}")
        self._client.set(key, pickle.dumps(value), ex=ttl)

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def build_cache(settings: Settings | None = None) -> Cache:
    """Construct a :class:`Cache` matching ``settings.cache_backend``.

    ``settings`` defaults to :func:`load_settings()` so callers can
    rely on the current environment.
    """

    cfg = settings if settings is not None else load_settings()
    if cfg.cache_backend == "disk":
        return DiskCache(cfg.cache_dir)
    if cfg.cache_backend == "redis":
        return RedisCache()
    # Should be unreachable: ``Settings.__post_init__`` already
    # validated the backend name. Guard anyway so future expansion
    # cannot silently fall through.
    raise ValueError(  # pragma: no cover - defensive
        f"Unsupported cache backend: {cfg.cache_backend!r}"
    )


_default_cache: Cache | None = None
_default_cache_lock = threading.Lock()


def get_default_cache() -> Cache:
    """Return the process-wide cache instance, building it on demand.

    The first call materializes a backend from the current settings;
    subsequent calls return the same instance. Tests that need
    isolation should use :func:`build_cache` directly or call
    :func:`reset_default_cache` between cases.
    """

    global _default_cache
    if _default_cache is not None:
        return _default_cache
    with _default_cache_lock:
        if _default_cache is None:
            _default_cache = build_cache()
        return _default_cache


def reset_default_cache() -> None:
    """Drop the cached default backend (primarily for tests)."""

    global _default_cache
    with _default_cache_lock:
        if _default_cache is not None:
            with contextlib.suppress(Exception):
                _default_cache.close()
        _default_cache = None


# ---------------------------------------------------------------------------
# Algorithm 3: cache_get_or_fetch
# ---------------------------------------------------------------------------


def cache_get_or_fetch(
    tool: str,
    symbol: str,
    params: Mapping[str, Any],
    ttl: int,
    fetcher: Callable[[], T],
    schema_version: int,
    *,
    cache: Cache | None = None,
) -> T:
    """Read-through cache helper -- design Algorithm 3.

    Procedure:

    1. Validate ``ttl > 0`` and ``schema_version >= 1``.
    2. Build the canonical key via :func:`make_key`.
    3. Return the cached payload if present.
    4. Otherwise invoke ``fetcher`` (must be idempotent and side-effect
       free), persist the result with the requested TTL, and return.

    Parameters
    ----------
    cache:
        Optional explicit backend; defaults to :func:`get_default_cache`.
        Inject a stub in tests to keep them hermetic.

    Raises
    ------
    ValueError
        If ``ttl <= 0`` or ``schema_version < 1``.
    """

    if ttl <= 0:
        raise ValueError(f"ttl must be > 0, got {ttl}")
    if schema_version < 1:
        raise ValueError(
            f"schema_version must be >= 1, got {schema_version}"
        )

    backend = cache if cache is not None else get_default_cache()
    key = backend.make_key(tool, symbol, params, schema_version)

    cached = backend.get(key)
    if cached is not None:
        return cached  # type: ignore[no-any-return]

    payload = fetcher()
    backend.set(key, payload, ttl)
    return payload


__all__ = [
    "TTL_COLD",
    "TTL_FROZEN",
    "TTL_HOT",
    "TTL_STATIC",
    "TTL_WARM",
    "Cache",
    "DiskCache",
    "RedisCache",
    "build_cache",
    "cache_get_or_fetch",
    "get_default_cache",
    "make_key",
    "reset_default_cache",
    "stable_json",
]
