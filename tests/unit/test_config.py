"""Smoke tests for :mod:`china_stock_mcp.config`.

These tests cover the requirements wired up in task 1.1:

- 11.5 cache backend selection
- 11.8 default rate limit and override
- 12.6 transport selection
"""

from __future__ import annotations

from pathlib import Path

import pytest

from china_stock_mcp.config import (
    DEFAULT_DATA_DELAY_NOTICE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_RATE_LIMIT,
    Settings,
    load_settings,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "CSM_CACHE_BACKEND",
        "CSM_CACHE_DIR",
        "CSM_LOG_LEVEL",
        "CSM_TUSHARE_TOKEN",
        "CSM_RATE_LIMIT",
        "CSM_DATA_DELAY_NOTICE",
        "CSM_TRANSPORT",
    ):
        monkeypatch.delenv(var, raising=False)


def test_load_settings_defaults() -> None:
    settings = load_settings()

    assert settings.cache_backend == "disk"
    assert settings.log_level == DEFAULT_LOG_LEVEL
    assert settings.rate_limit == DEFAULT_RATE_LIMIT
    assert settings.data_delay_notice is DEFAULT_DATA_DELAY_NOTICE
    assert settings.transport == "stdio"
    assert settings.tushare_token is None
    assert isinstance(settings.cache_dir, Path)


def test_load_settings_reads_all_env_vars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CSM_CACHE_BACKEND", "redis")
    monkeypatch.setenv("CSM_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CSM_LOG_LEVEL", "debug")
    monkeypatch.setenv("CSM_TUSHARE_TOKEN", "secret-token")
    monkeypatch.setenv("CSM_RATE_LIMIT", "120")
    monkeypatch.setenv("CSM_DATA_DELAY_NOTICE", "false")
    monkeypatch.setenv("CSM_TRANSPORT", "streamable-http")

    settings = load_settings()

    assert settings.cache_backend == "redis"
    assert settings.cache_dir == tmp_path
    assert settings.log_level == "DEBUG"
    assert settings.tushare_token == "secret-token"
    assert settings.rate_limit == 120
    assert settings.data_delay_notice is False
    assert settings.transport == "streamable-http"


def test_invalid_cache_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CSM_CACHE_BACKEND", "memcached")
    with pytest.raises(ValueError, match="CSM_CACHE_BACKEND"):
        load_settings()


def test_invalid_transport_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CSM_TRANSPORT", "websocket")
    with pytest.raises(ValueError, match="CSM_TRANSPORT"):
        load_settings()


def test_invalid_rate_limit_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CSM_RATE_LIMIT", "0")
    with pytest.raises(ValueError, match="CSM_RATE_LIMIT"):
        load_settings()


def test_invalid_log_level_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CSM_LOG_LEVEL", "verbose")
    with pytest.raises(ValueError, match="CSM_LOG_LEVEL"):
        load_settings()


def test_settings_is_immutable() -> None:
    settings = load_settings()
    with pytest.raises((AttributeError, TypeError)):
        settings.rate_limit = 99  # type: ignore[misc]


def test_data_delay_notice_truthy_values(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("CSM_DATA_DELAY_NOTICE", value)
        assert load_settings().data_delay_notice is True


def test_data_delay_notice_falsy_values(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("0", "false", "NO", "off"):
        monkeypatch.setenv("CSM_DATA_DELAY_NOTICE", value)
        assert load_settings().data_delay_notice is False


def test_settings_construct_directly() -> None:
    settings = Settings(
        cache_backend="disk",
        cache_dir=Path("/tmp/csm"),
        log_level="WARNING",
        tushare_token=None,
        rate_limit=10,
        data_delay_notice=False,
        transport="stdio",
    )
    assert settings.rate_limit == 10
    assert settings.transport == "stdio"
