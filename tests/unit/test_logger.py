"""Verify that the package exposes a working loguru logger."""

from __future__ import annotations

import io

import pytest

import china_stock_mcp
from china_stock_mcp import logger, reconfigure_logger
from china_stock_mcp.config import Settings


def test_package_exposes_logger() -> None:
    # ``logger`` is the loguru logger; it should at least support the
    # standard severity methods.
    assert hasattr(logger, "info")
    assert hasattr(logger, "warning")
    assert hasattr(logger, "error")


def test_package_version_is_string() -> None:
    assert isinstance(china_stock_mcp.__version__, str)
    assert china_stock_mcp.__version__


def test_reconfigure_logger_applies_level() -> None:
    sink = io.StringIO()

    settings = Settings(log_level="WARNING")
    reconfigure_logger(settings)

    handler_id = logger.add(sink, level=settings.log_level, format="{level}|{message}")
    try:
        logger.debug("debug-should-not-appear")
        logger.warning("warning-should-appear")
    finally:
        logger.remove(handler_id)

    output = sink.getvalue()
    assert "warning-should-appear" in output
    assert "debug-should-not-appear" not in output


@pytest.fixture(autouse=True)
def _restore_logger() -> None:
    """Make sure other tests see the default-configured logger."""

    yield
    reconfigure_logger()
