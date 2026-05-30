"""Runtime configuration loaded from environment variables.

Environment variables (all prefixed with ``CSM_``):

- ``CSM_CACHE_BACKEND``     -- ``disk`` (default) or ``redis``.
- ``CSM_CACHE_DIR``         -- diskcache root directory; defaults to a
                               per-user cache directory under the home folder.
- ``CSM_LOG_LEVEL``         -- log level name (``DEBUG`` / ``INFO`` /
                               ``WARNING`` / ``ERROR`` / ``CRITICAL``);
                               defaults to ``INFO``.
- ``CSM_TUSHARE_TOKEN``     -- optional tushare API token.
- ``CSM_RATE_LIMIT``        -- requests per minute; defaults to ``30``
                               (see Requirements 11.8).
- ``CSM_DATA_DELAY_NOTICE`` -- ``true`` / ``false``; whether to append
                               "数据延迟约 15 分钟" to quote responses.
                               Defaults to ``true`` (Requirements 2.6).
- ``CSM_TRANSPORT``         -- ``stdio`` (default) or ``streamable-http``
                               (Requirements 12.6).

The :func:`load_settings` function reads the environment once and returns
an immutable :class:`Settings` instance that is safe to reuse across the
server lifetime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Allowed values
# ---------------------------------------------------------------------------

CACHE_BACKENDS: Final[frozenset[str]] = frozenset({"disk", "redis"})
TRANSPORTS: Final[frozenset[str]] = frozenset({"stdio", "streamable-http"})
LOG_LEVELS: Final[frozenset[str]] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CACHE_BACKEND: Final[str] = "disk"
DEFAULT_LOG_LEVEL: Final[str] = "INFO"
DEFAULT_RATE_LIMIT: Final[int] = 30  # requests per minute (Requirements 11.8)
DEFAULT_DATA_DELAY_NOTICE: Final[bool] = True  # Requirements 2.6
DEFAULT_TRANSPORT: Final[str] = "stdio"  # Requirements 12.6

_TRUE_VALUES: Final[frozenset[str]] = frozenset(
    {"1", "true", "yes", "y", "on"}
)
_FALSE_VALUES: Final[frozenset[str]] = frozenset(
    {"0", "false", "no", "n", "off"}
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_cache_dir() -> Path:
    """Default diskcache directory (``~/.cache/china-stock-mcp``)."""

    return Path.home() / ".cache" / "china-stock-mcp"


def _read_str(
    name: str,
    default: str,
    allowed: frozenset[str] | None = None,
    *,
    normalize: str | None = None,
) -> str:
    """Read a string env var.

    ``normalize`` may be ``"upper"`` or ``"lower"`` to normalize the input
    before validation; this allows callers to accept e.g. ``info`` for a
    log level whose canonical form is ``INFO``.
    """

    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip()
    if normalize == "upper":
        value = value.upper()
    elif normalize == "lower":
        value = value.lower()
    if allowed is not None and value not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValueError(
            f"Invalid value for {name}: {raw!r}. "
            f"Expected one of: {allowed_list}."
        )
    return value


def _read_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"Invalid integer for {name}: {raw!r}."
        ) from exc
    if minimum is not None and value < minimum:
        raise ValueError(
            f"{name} must be >= {minimum}, got {value}."
        )
    return value


def _read_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(
        f"Invalid boolean for {name}: {raw!r}. "
        f"Expected one of: true/false/yes/no/1/0."
    )


def _read_optional_str(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def _read_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return Path(raw.strip()).expanduser()


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime settings.

    Use :func:`load_settings` to build an instance from environment
    variables; instantiate directly only in tests.
    """

    cache_backend: str = DEFAULT_CACHE_BACKEND
    cache_dir: Path = field(default_factory=_default_cache_dir)
    log_level: str = DEFAULT_LOG_LEVEL
    tushare_token: str | None = None
    rate_limit: int = DEFAULT_RATE_LIMIT
    data_delay_notice: bool = DEFAULT_DATA_DELAY_NOTICE
    transport: str = DEFAULT_TRANSPORT

    def __post_init__(self) -> None:
        if self.cache_backend not in CACHE_BACKENDS:
            raise ValueError(
                f"cache_backend must be one of {sorted(CACHE_BACKENDS)}, "
                f"got {self.cache_backend!r}."
            )
        if self.transport not in TRANSPORTS:
            raise ValueError(
                f"transport must be one of {sorted(TRANSPORTS)}, "
                f"got {self.transport!r}."
            )
        if self.log_level not in LOG_LEVELS:
            raise ValueError(
                f"log_level must be one of {sorted(LOG_LEVELS)}, "
                f"got {self.log_level!r}."
            )
        if self.rate_limit < 1:
            raise ValueError(
                f"rate_limit must be >= 1, got {self.rate_limit}."
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_settings() -> Settings:
    """Read configuration from environment variables.

    Environment variables are parsed lazily at call time, so tests can
    override values via :func:`os.environ` and re-invoke this function.
    """

    return Settings(
        cache_backend=_read_str(
            "CSM_CACHE_BACKEND",
            DEFAULT_CACHE_BACKEND,
            CACHE_BACKENDS,
            normalize="lower",
        ),
        cache_dir=_read_path("CSM_CACHE_DIR", _default_cache_dir()),
        log_level=_read_str(
            "CSM_LOG_LEVEL",
            DEFAULT_LOG_LEVEL,
            LOG_LEVELS,
            normalize="upper",
        ),
        tushare_token=_read_optional_str("CSM_TUSHARE_TOKEN"),
        rate_limit=_read_int("CSM_RATE_LIMIT", DEFAULT_RATE_LIMIT, minimum=1),
        data_delay_notice=_read_bool(
            "CSM_DATA_DELAY_NOTICE", DEFAULT_DATA_DELAY_NOTICE
        ),
        transport=_read_str(
            "CSM_TRANSPORT",
            DEFAULT_TRANSPORT,
            TRANSPORTS,
            normalize="lower",
        ),
    )


__all__ = [
    "CACHE_BACKENDS",
    "DEFAULT_CACHE_BACKEND",
    "DEFAULT_DATA_DELAY_NOTICE",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_RATE_LIMIT",
    "DEFAULT_TRANSPORT",
    "LOG_LEVELS",
    "TRANSPORTS",
    "Settings",
    "load_settings",
]
