"""Unit tests for :mod:`china_stock_mcp.tools.fundamental` (task 11.2).

Covers tool-layer rendering and validation:

- Requirement 4.1 -- Markdown contains all four sub-table headings
  (估值 / 盈利 / 成长 / 健康).
- Requirement 4.2 -- "行业分位" column is appended when
  ``industry_percentile`` is non-empty and omitted when empty.
  Boundary values 0.0 / 100.0 round-trip through the renderer.
- Requirement 4.5 / 13.4 -- a :class:`DataNotFoundError` raised by
  the adapter (e.g. for HK / fund codes the fundamentals upstream
  refuses to serve) propagates verbatim to the caller.
- Requirement 12.1 / Property 14 -- Markdown ends with the canonical
  disclaimer.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.cache import Cache, make_key, reset_default_cache
from china_stock_mcp.exceptions import DataNotFoundError
from china_stock_mcp.formatters import DISCLAIMER
from china_stock_mcp.models import (
    FinancialReport,
    FundamentalSnapshot,
    FundInfo,
    KLineSeries,
    MarketOverview,
    MoneyFlow,
    PeerTable,
    Quote,
    SymbolHit,
)
from china_stock_mcp.rate_limiter import RateLimiter
from china_stock_mcp.services.fundamental_service import FundamentalService
from china_stock_mcp.tools.fundamental import get_fundamentals

# ---------------------------------------------------------------------------
# Hermetic cache (in-memory dict matching the :class:`Cache` protocol)
# ---------------------------------------------------------------------------


class _StubCache:
    """In-memory cache backed by a plain dict."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self._store.get(key)

    def set(self, key: str, value: Any, ttl: int) -> None:
        if ttl <= 0:
            raise ValueError(f"ttl must be > 0, got {ttl}")
        self._store[key] = value

    def make_key(
        self,
        tool: str,
        symbol: str,
        params: Any,
        schema_version: int,
    ) -> str:
        return make_key(tool, symbol, params, schema_version)

    def close(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Stub adapter (StubAdapter pattern from
# tests/integration/test_search_quote_flow.py)
# ---------------------------------------------------------------------------


class _StubAdapter(BaseAdapter):
    """Adapter exposing only ``fundamentals``; everything else raises."""

    name: str = "stub"

    def __init__(
        self,
        *,
        snapshot: FundamentalSnapshot | None = None,
        raises: Exception | None = None,
    ) -> None:
        self._snapshot = snapshot
        self._raises = raises
        self.fundamentals_call_count: int = 0

    def fundamentals(self, symbol: str) -> FundamentalSnapshot:
        self.fundamentals_call_count += 1
        if self._raises is not None:
            raise self._raises
        assert self._snapshot is not None
        return self._snapshot

    # ----- not implemented ------------------------------------------------

    def search(self, query: str, market: str) -> list[SymbolHit]:
        raise NotImplementedError

    def quote(self, symbols: list[str]) -> list[Quote]:
        raise NotImplementedError

    def kline(
        self, symbol: str, period: str, count: int, adjust: str
    ) -> KLineSeries:
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_default_cache() -> Iterator[None]:
    reset_default_cache()
    try:
        yield
    finally:
        reset_default_cache()


@pytest.fixture()
def cache() -> _StubCache:
    return _StubCache()


@pytest.fixture()
def rate_limiter() -> RateLimiter:
    return RateLimiter(rate_per_minute=10_000)


# ---------------------------------------------------------------------------
# Snapshot helper
# ---------------------------------------------------------------------------


def _snapshot(
    *,
    symbol: str = "300750.SZ",
    industry_percentile: dict[str, float] | None = None,
) -> FundamentalSnapshot:
    return FundamentalSnapshot(
        symbol=symbol,
        valuation={"pe_ttm": 18.5, "pe_dynamic": 17.2, "pb": 3.4},
        profitability={"roe": 22.5, "net_margin": 18.7},
        growth={"revenue_yoy": 30.5, "net_profit_yoy": 45.6},
        health={"debt_ratio": 45.0, "current_ratio": 1.8},
        industry_percentile=(
            industry_percentile if industry_percentile is not None else {}
        ),
    )


def _make_service(
    adapter: _StubAdapter,
    cache: Cache,
    rate_limiter: RateLimiter,
) -> FundamentalService:
    return FundamentalService(
        adapter, cache=cache, rate_limiter=rate_limiter
    )


# ---------------------------------------------------------------------------
# Bucket headings (Requirement 4.1)
# ---------------------------------------------------------------------------


class TestFundamentalBucketHeadings:
    """**Validates: Requirements 4.1**."""

    def test_markdown_contains_all_four_sub_table_headings(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        adapter = _StubAdapter(snapshot=_snapshot())
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_fundamentals(service, "300750.SZ")

        # The four bucket sub-headings (rendered as ``#### ...``).
        assert "估值指标" in markdown
        assert "盈利能力" in markdown
        assert "成长性" in markdown
        assert "财务健康" in markdown

        # Disclaimer terminator (Property 14).
        assert markdown.rstrip().endswith(DISCLAIMER)

    def test_markdown_renders_per_metric_labels(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        """Each populated metric surfaces under its localized label."""

        adapter = _StubAdapter(snapshot=_snapshot())
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_fundamentals(service, "300750.SZ")

        # A few representative metric labels from each bucket.
        assert "市盈率(TTM)" in markdown  # valuation.pe_ttm
        assert "净资产收益率" in markdown  # profitability.roe
        assert "营收同比" in markdown  # growth.revenue_yoy
        assert "资产负债率" in markdown  # health.debt_ratio


# ---------------------------------------------------------------------------
# Industry-percentile column (Requirement 4.2)
# ---------------------------------------------------------------------------


class TestIndustryPercentileColumn:
    """**Validates: Requirements 4.2**."""

    def test_column_appended_when_percentile_non_empty(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        adapter = _StubAdapter(
            snapshot=_snapshot(
                industry_percentile={"pe_ttm": 60.0, "roe": 85.0}
            )
        )
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_fundamentals(service, "300750.SZ")

        assert "行业分位" in markdown
        # Numeric percentile values render with one decimal place.
        assert "60.0" in markdown
        assert "85.0" in markdown

    def test_column_omitted_when_percentile_empty(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        adapter = _StubAdapter(snapshot=_snapshot(industry_percentile={}))
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_fundamentals(service, "300750.SZ")

        # When the upstream did not provide percentile data, the entire
        # column is dropped to keep the Markdown clean (design v1).
        assert "行业分位" not in markdown

    @pytest.mark.parametrize("value", [0.0, 50.0, 100.0])
    def test_boundary_percentile_values_round_trip(
        self,
        cache: _StubCache,
        rate_limiter: RateLimiter,
        value: float,
    ) -> None:
        """Boundary values 0 / 50 / 100 are accepted by the DTO and rendered."""

        adapter = _StubAdapter(
            snapshot=_snapshot(industry_percentile={"pe_ttm": value})
        )
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_fundamentals(service, "300750.SZ")

        # Renderer prints percentile to 1 decimal place.
        assert f"{value:.1f}" in markdown


# ---------------------------------------------------------------------------
# DataNotFoundError surfaces for HK / fund codes (Requirement 4.5 / 13.4)
# ---------------------------------------------------------------------------


class TestDataNotFoundSurfaces:
    """**Validates: Requirements 4.5, 13.4** (Property 5)."""

    def test_data_not_found_from_adapter_propagates(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        """Adapter raises -> service propagates -> tool surfaces."""

        not_found = DataNotFoundError(
            "00700.HK 暂无 A 股式基本面数据, 请改用 get_quote"
        )
        adapter = _StubAdapter(raises=not_found)
        service = _make_service(adapter, cache, rate_limiter)

        with pytest.raises(DataNotFoundError) as exc_info:
            # ``00700.HK`` is a valid Standardized_Symbol so it passes
            # ``normalize_symbol`` and reaches the adapter.
            get_fundamentals(service, "00700.HK")

        # Verbatim propagation -- same instance escapes the tool.
        assert exc_info.value is not_found
        assert adapter.fundamentals_call_count == 1
