"""sort_by / order / limit validation + ScreenHit rendering unit tests.

Covers task 16.5 of the china-stock-mcp spec:

- 8.5 -- ``sort_by`` not in :data:`SUPPORTED_FIELDS` SHALL surface as
  :class:`ValidationError` whose message lists every supported field
  name.
- 8.5 (extended) -- ``order`` outside ``{"asc", "desc"}`` and
  ``limit`` outside ``[1, 200]`` SHALL surface as
  :class:`ValidationError`. ``bool`` inputs for ``limit`` are
  rejected explicitly because ``bool`` is a subclass of ``int`` in
  Python.
- 8.6 -- the rendered Markdown table SHALL include 代码 / 名称 / 行业
  columns plus every criterion field used by the caller.

These tests exercise both the tool boundary
(:class:`ScreenStocksInput`) and the service boundary
(:meth:`ScreenService._validate_inputs`) so we get coverage on both
layers without re-running the whole pipeline.
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
# Hermetic stand-ins (StubAdapter pattern from
# tests/integration/test_search_quote_flow.py)
# ---------------------------------------------------------------------------


class _InMemoryCache:
    """In-process :class:`Cache` backed by a plain dict."""

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
    """Adapter whose every method raises :class:`NotImplementedError`."""

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
    """ScreenService whose universe is a fixed list of :class:`ScreenHit`."""

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
def cache() -> _InMemoryCache:
    return _InMemoryCache()


@pytest.fixture()
def rate_limiter() -> RateLimiter:
    return RateLimiter(rate_per_minute=1_000_000)


def _bare_universe() -> list[ScreenHit]:
    """Reusable universe for the rendering tests."""

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


# ---------------------------------------------------------------------------
# 8.5 -- sort_by validation (lists every supported field on rejection)
# ---------------------------------------------------------------------------


class TestSortByValidation:
    """Requirement 8.5 -- unknown ``sort_by`` SHALL raise ``ValidationError``.

    The error message must enumerate every field in
    :data:`SUPPORTED_FIELDS` so the AI client can pick a valid one
    without re-reading the docs.
    """

    def test_unknown_sort_by_raises_with_supported_list(
        self,
        cache: _InMemoryCache,
        rate_limiter: RateLimiter,
    ) -> None:
        service = _StubScreenService(
            _bare_universe(), cache=cache, rate_limiter=rate_limiter
        )

        with pytest.raises(ValidationError) as excinfo:
            screen_stocks(service, sort_by="not_a_field")

        message = str(excinfo.value)
        # The Pydantic literal validator names the bad field directly.
        assert "sort_by" in message

    def test_service_layer_rejects_unknown_sort_by(
        self,
        cache: _InMemoryCache,
        rate_limiter: RateLimiter,
    ) -> None:
        """The service layer's own validator surfaces the supported set.

        The tool layer's pydantic ``Literal`` rejects unknown values
        first; this test bypasses pydantic and calls the service
        directly so the listing branch in
        :meth:`ScreenService._validate_inputs` is exercised.
        """

        service = _StubScreenService(
            _bare_universe(), cache=cache, rate_limiter=rate_limiter
        )

        with pytest.raises(ValidationError) as excinfo:
            service.filter(
                criteria=ScreenCriteria(),
                sort_by="bogus",
                order="desc",
                limit=10,
            )

        message = str(excinfo.value)
        assert "sort_by" in message
        # Every supported field appears verbatim in the error.
        for field in SUPPORTED_FIELDS:
            assert field in message


# ---------------------------------------------------------------------------
# 8.5 -- order / limit validation
# ---------------------------------------------------------------------------


class TestOrderValidation:
    @pytest.mark.parametrize("bad_order", ["ASC", "ascending", "down", ""])
    def test_invalid_order_raises(
        self,
        cache: _InMemoryCache,
        rate_limiter: RateLimiter,
        bad_order: str,
    ) -> None:
        service = _StubScreenService(
            _bare_universe(), cache=cache, rate_limiter=rate_limiter
        )

        with pytest.raises(ValidationError) as excinfo:
            screen_stocks(service, order=bad_order)

        assert "order" in str(excinfo.value)

    def test_service_layer_rejects_invalid_order(
        self,
        cache: _InMemoryCache,
        rate_limiter: RateLimiter,
    ) -> None:
        service = _StubScreenService(
            _bare_universe(), cache=cache, rate_limiter=rate_limiter
        )

        with pytest.raises(ValidationError, match="order"):
            service.filter(
                criteria=ScreenCriteria(),
                sort_by="market_cap",
                order="ASC",
                limit=10,
            )


class TestLimitValidation:
    @pytest.mark.parametrize("bad_limit", [0, 201, -1, 1000])
    def test_out_of_range_limit_raises(
        self,
        cache: _InMemoryCache,
        rate_limiter: RateLimiter,
        bad_limit: int,
    ) -> None:
        service = _StubScreenService(
            _bare_universe(), cache=cache, rate_limiter=rate_limiter
        )

        with pytest.raises(ValidationError) as excinfo:
            screen_stocks(service, limit=bad_limit)

        assert "limit" in str(excinfo.value)

    @pytest.mark.parametrize("bad_limit", [True, False])
    def test_bool_limit_rejected_by_service(
        self,
        cache: _InMemoryCache,
        rate_limiter: RateLimiter,
        bad_limit: bool,
    ) -> None:
        """``bool`` is an ``int`` subclass; the service must reject it."""

        service = _StubScreenService(
            _bare_universe(), cache=cache, rate_limiter=rate_limiter
        )

        with pytest.raises(ValidationError) as excinfo:
            service.filter(
                criteria=ScreenCriteria(),
                sort_by="market_cap",
                order="desc",
                limit=bad_limit,
            )

        # The message names the offending type explicitly.
        assert "limit" in str(excinfo.value)
        assert "bool" in str(excinfo.value)

    def test_boundary_limits_accepted(
        self,
        cache: _InMemoryCache,
        rate_limiter: RateLimiter,
    ) -> None:
        service = _StubScreenService(
            _bare_universe(), cache=cache, rate_limiter=rate_limiter
        )

        # 1 and 200 are inclusive.
        md_lo = screen_stocks(service, limit=1)
        md_hi = screen_stocks(service, limit=200)
        assert "选股结果" in md_lo
        assert "选股结果" in md_hi


# ---------------------------------------------------------------------------
# 8.6 -- ScreenHit table rendering (代码 / 名称 / 行业 + criterion fields)
# ---------------------------------------------------------------------------


class TestScreenHitRendering:
    """Requirement 8.6 -- the table SHALL include 代码 / 名称 / 行业
    plus every criterion field the caller used."""

    def test_table_contains_default_columns_and_disclaimer(
        self,
        cache: _InMemoryCache,
        rate_limiter: RateLimiter,
    ) -> None:
        service = _StubScreenService(
            _bare_universe(), cache=cache, rate_limiter=rate_limiter
        )

        markdown = screen_stocks(service, sort_by="market_cap", limit=10)

        # The three baseline columns are always present.
        assert "代码" in markdown
        assert "名称" in markdown
        assert "行业" in markdown

        # Both rows from the universe surface.
        assert "300750.SZ" in markdown
        assert "宁德时代" in markdown
        assert "002594.SZ" in markdown
        assert "比亚迪" in markdown

        # Industry strings render verbatim.
        assert "电池" in markdown
        assert "新能源车" in markdown

        # Disclaimer (Property 14 / Requirement 12.1).
        assert markdown.rstrip().endswith(DISCLAIMER)

    def test_table_includes_every_criterion_column(
        self,
        cache: _InMemoryCache,
        rate_limiter: RateLimiter,
    ) -> None:
        """When the caller filters by a field, the rendered table must
        carry that field as an explicit column.

        The label mapping is private to ``tools/screen.py``; we assert
        on the localized labels which mirror ``_FIELD_LABELS`` there.
        """

        service = _StubScreenService(
            _bare_universe(), cache=cache, rate_limiter=rate_limiter
        )

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

        # Localized column labels for each criterion the caller used.
        assert "市盈率(TTM)" in markdown
        assert "净资产收益率" in markdown
        assert "总市值" in markdown

        # PB / revenue_growth were not in the criteria so their
        # columns must NOT appear (Requirement 8.6 only mandates
        # used fields).
        assert "市净率" not in markdown
        assert "营收增速" not in markdown

    def test_sort_by_column_visible_even_without_criterion(
        self,
        cache: _InMemoryCache,
        rate_limiter: RateLimiter,
    ) -> None:
        """When the caller sorts by a field they did not filter by,
        the sort_by field must still surface as a column so the user
        can see what they sorted on.
        """

        service = _StubScreenService(
            _bare_universe(), cache=cache, rate_limiter=rate_limiter
        )

        markdown = screen_stocks(
            service,
            criteria={"pe_ttm": {"min": 0, "max": 30}},
            sort_by="market_cap",
            order="desc",
            limit=10,
        )

        # Both the criterion and the sort_by columns appear.
        assert "市盈率(TTM)" in markdown
        assert "总市值" in markdown

    def test_empty_result_renders_friendly_notice(
        self,
        cache: _InMemoryCache,
        rate_limiter: RateLimiter,
    ) -> None:
        """No matches → headline + "_未找到符合条件的标的_"."""

        service = _StubScreenService(
            _bare_universe(), cache=cache, rate_limiter=rate_limiter
        )

        markdown = screen_stocks(
            service,
            # Force empty -- impossible PE range.
            criteria={"pe_ttm": {"min": 99999, "max": 100000}},
            sort_by="market_cap",
            order="desc",
            limit=10,
        )

        assert "未找到符合条件的标的" in markdown
        assert markdown.rstrip().endswith(DISCLAIMER)
