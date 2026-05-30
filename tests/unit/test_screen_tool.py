"""Unit tests for :mod:`china_stock_mcp.tools.screen` (task 16.5).

Covers tool-layer validation and rendering:

- Requirement 8.5 -- ``sort_by`` ∈ :data:`SUPPORTED_FIELDS`; unknown
  values raise :class:`ValidationError`. The service layer's own
  rejection lists every supported field name.
- Requirement 8.3 -- ``order`` ∈ ``{"asc", "desc"}``; ``"ascending"``
  raises :class:`ValidationError`.
- Requirement 8.2 -- ``limit`` ∈ ``[1, 200]``; ``0`` and ``201`` raise
  :class:`ValidationError`.
- Requirement 8.6 -- the rendered Markdown contains 代码 / 名称 / 行业
  plus a column for every criterion field used by the caller.
- Requirement 12.1 / Property 14 -- Markdown ends with the canonical
  disclaimer.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.cache import Cache, make_key, reset_default_cache
from china_stock_mcp.exceptions import ValidationError
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
    ScreenCriteria,
    ScreenHit,
    SymbolHit,
)
from china_stock_mcp.rate_limiter import RateLimiter
from china_stock_mcp.services.screen_service import (
    SUPPORTED_FIELDS,
    ScreenService,
)
from china_stock_mcp.tools.screen import screen_stocks

# ---------------------------------------------------------------------------
# Hermetic cache + stub adapter
# ---------------------------------------------------------------------------


class _StubCache:
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


class _StubAdapter(BaseAdapter):
    """Adapter whose every method raises :class:`NotImplementedError`.

    The screen service does not actually call into the adapter -- it
    pulls data via ``akshare`` directly. The adapter is required only
    by the :class:`ScreenService` constructor signature.
    """

    name: str = "stub"

    def search(self, query: str, market: str) -> list[SymbolHit]:
        raise NotImplementedError

    def quote(self, symbols: list[str]) -> list[Quote]:
        raise NotImplementedError

    def kline(
        self, symbol: str, period: str, count: int, adjust: str
    ) -> KLineSeries:
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


class _StubScreenService(ScreenService):
    """ScreenService whose universe is a fixed in-memory list.

    We override ``_fetch_universe`` so the test never touches
    ``akshare``; ``_validate_inputs`` and the filter / sort / truncate
    pipeline still execute exactly as in production.
    """

    def __init__(
        self,
        universe: list[ScreenHit],
        *,
        cache: Cache,
        rate_limiter: RateLimiter,
    ) -> None:
        super().__init__(
            primary=_StubAdapter(),
            cache=cache,
            rate_limiter=rate_limiter,
        )
        self._stub_universe: list[ScreenHit] = list(universe)

    def _fetch_universe(self, criteria: ScreenCriteria) -> list[ScreenHit]:
        return list(self._stub_universe)


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


def _universe() -> list[ScreenHit]:
    return [
        ScreenHit(
            code="300750.SZ",
            name="宁德时代",
            industry="电池",
            fields={
                "pe_ttm": 18.5,
                "pb": 4.2,
                "roe": 22.5,
                "market_cap": 1.2e12,
                "revenue_growth": 35.6,
            },
        ),
        ScreenHit(
            code="002594.SZ",
            name="比亚迪",
            industry="新能源车",
            fields={
                "pe_ttm": 22.1,
                "pb": 3.8,
                "roe": 18.2,
                "market_cap": 8.5e11,
                "revenue_growth": 42.0,
            },
        ),
    ]


def _make_service(
    cache: Cache, rate_limiter: RateLimiter
) -> _StubScreenService:
    return _StubScreenService(
        _universe(), cache=cache, rate_limiter=rate_limiter
    )


# ---------------------------------------------------------------------------
# sort_by validation (Requirement 8.5)
# ---------------------------------------------------------------------------


class TestSortByValidation:
    """**Validates: Requirements 8.5**."""

    @pytest.mark.parametrize("sort_by", sorted(SUPPORTED_FIELDS))
    def test_supported_sort_by_accepted(
        self,
        cache: _StubCache,
        rate_limiter: RateLimiter,
        sort_by: str,
    ) -> None:
        service = _make_service(cache, rate_limiter)

        markdown = screen_stocks(service, sort_by=sort_by)

        assert markdown.rstrip().endswith(DISCLAIMER)

    def test_unknown_sort_by_raises_validation_error(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        service = _make_service(cache, rate_limiter)

        with pytest.raises(ValidationError):
            screen_stocks(service, sort_by="xyz")

    def test_service_layer_lists_full_supported_set(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        """Bypass pydantic so the service-layer validator is exercised."""

        service = _make_service(cache, rate_limiter)

        with pytest.raises(ValidationError) as exc_info:
            service.filter(
                criteria=ScreenCriteria(),
                sort_by="xyz",
                order="desc",
                limit=10,
            )

        message = str(exc_info.value)
        for field in SUPPORTED_FIELDS:
            assert field in message


# ---------------------------------------------------------------------------
# order validation (Requirement 8.3)
# ---------------------------------------------------------------------------


class TestOrderValidation:
    """**Validates: Requirements 8.3**."""

    @pytest.mark.parametrize("order", ["asc", "desc"])
    def test_valid_order_accepted(
        self,
        cache: _StubCache,
        rate_limiter: RateLimiter,
        order: str,
    ) -> None:
        service = _make_service(cache, rate_limiter)

        markdown = screen_stocks(service, order=order)
        assert markdown.rstrip().endswith(DISCLAIMER)

    @pytest.mark.parametrize("bad_order", ["ascending", "ASC", "down", ""])
    def test_invalid_order_raises_validation_error(
        self,
        cache: _StubCache,
        rate_limiter: RateLimiter,
        bad_order: str,
    ) -> None:
        service = _make_service(cache, rate_limiter)

        with pytest.raises(ValidationError):
            screen_stocks(service, order=bad_order)


# ---------------------------------------------------------------------------
# limit validation (Requirement 8.2)
# ---------------------------------------------------------------------------


class TestLimitValidation:
    """**Validates: Requirements 8.2**."""

    @pytest.mark.parametrize("limit", [1, 30, 200])
    def test_limit_in_range_accepted(
        self,
        cache: _StubCache,
        rate_limiter: RateLimiter,
        limit: int,
    ) -> None:
        service = _make_service(cache, rate_limiter)

        markdown = screen_stocks(service, limit=limit)
        assert markdown.rstrip().endswith(DISCLAIMER)

    @pytest.mark.parametrize("bad_limit", [0, 201, -1, 1000])
    def test_out_of_range_limit_raises_validation_error(
        self,
        cache: _StubCache,
        rate_limiter: RateLimiter,
        bad_limit: int,
    ) -> None:
        service = _make_service(cache, rate_limiter)

        with pytest.raises(ValidationError):
            screen_stocks(service, limit=bad_limit)


# ---------------------------------------------------------------------------
# Markdown structure (Requirement 8.6)
# ---------------------------------------------------------------------------


class TestMarkdownStructure:
    """**Validates: Requirements 8.6**."""

    def test_markdown_contains_required_baseline_columns(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        service = _make_service(cache, rate_limiter)

        markdown = screen_stocks(service, sort_by="market_cap")

        # Baseline columns are always present.
        assert "代码" in markdown
        assert "名称" in markdown
        assert "行业" in markdown

        # Both seed rows render verbatim.
        assert "300750.SZ" in markdown
        assert "宁德时代" in markdown
        assert "002594.SZ" in markdown
        assert "比亚迪" in markdown

    def test_markdown_includes_every_criterion_field_column(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        """Each criterion the caller filters by surfaces as a column."""

        service = _make_service(cache, rate_limiter)

        markdown = screen_stocks(
            service,
            criteria={
                "pe_ttm": {"min": 0, "max": 30},
                "roe": {"min": 10},
                "market_cap": {"min": 1e9},
            },
            sort_by="market_cap",
            order="desc",
            limit=10,
        )

        # Localized labels for the active criterion fields appear.
        assert "市盈率(TTM)" in markdown
        assert "净资产收益率" in markdown
        assert "总市值" in markdown

        # Non-active criterion fields stay out of the table to keep
        # the output focused on the user's mental model.
        assert "市净率" not in markdown
        assert "营收增速" not in markdown
