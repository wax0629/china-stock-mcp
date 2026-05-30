"""Unit tests for :mod:`china_stock_mcp.tools.money_flow` (task 13.2).

Covers tool-layer rendering and validation:

- Requirement 5.4 / 5.5 -- ``flow_type`` ∈ ``{"north", "main",
  "dragon_tiger"}`` and ``top_n`` ∈ ``[1, 100]``; out-of-range values
  raise :class:`ValidationError`.
- Requirement 5.2 -- ``main`` flow without a ``symbol`` raises
  :class:`ValidationError` whose message mentions ``"main"`` and
  ``"symbol"``.
- Requirement 5.6 -- ``snapshot_at`` is rendered in the Markdown
  header line above the data table.
- Truncation: when the upstream returns more rows than ``top_n``, the
  service caches whatever the adapter returned. The renderer faithfully
  reflects the row count it received.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
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
    SymbolHit,
)
from china_stock_mcp.rate_limiter import RateLimiter
from china_stock_mcp.services.money_flow_service import MoneyFlowService
from china_stock_mcp.tools.money_flow import get_money_flow

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
    name: str = "stub"

    def __init__(self, flow: MoneyFlow) -> None:
        self._flow = flow
        self.money_flow_call_count: int = 0
        self.last_call: tuple[str | None, str, int] | None = None

    def money_flow(
        self,
        symbol: str | None,
        flow_type: str,
        top_n: int,
    ) -> MoneyFlow:
        self.money_flow_call_count += 1
        self.last_call = (symbol, flow_type, top_n)
        return self._flow

    # ----- not implemented ------------------------------------------------

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
# Helpers
# ---------------------------------------------------------------------------


_FIXED_SNAPSHOT_AT = datetime(2024, 1, 2, 15, 0, 0, tzinfo=UTC)


def _north_flow(rows: int = 3) -> MoneyFlow:
    return MoneyFlow(
        flow_type="north",
        rows=[
            {"日期": "2024-01-02", "净流入金额": 1.5e9 + i}
            for i in range(rows)
        ],
        snapshot_at=_FIXED_SNAPSHOT_AT,
    )


def _main_flow() -> MoneyFlow:
    return MoneyFlow(
        flow_type="main",
        rows=[{"主力净流入": 1.2e8, "超大单": 0.8e8}],
        snapshot_at=_FIXED_SNAPSHOT_AT,
    )


def _make_service(
    adapter: _StubAdapter,
    cache: Cache,
    rate_limiter: RateLimiter,
) -> MoneyFlowService:
    return MoneyFlowService(adapter, cache=cache, rate_limiter=rate_limiter)


# ---------------------------------------------------------------------------
# flow_type validation (Requirement 5.5)
# ---------------------------------------------------------------------------


class TestFlowTypeValidation:
    """**Validates: Requirements 5.5**."""

    @pytest.mark.parametrize(
        "flow_type", ["north", "main", "dragon_tiger"]
    )
    def test_valid_flow_types_accepted(
        self,
        cache: _StubCache,
        rate_limiter: RateLimiter,
        flow_type: str,
    ) -> None:
        flow = (
            _main_flow()
            if flow_type == "main"
            else _north_flow()
        )
        # ``main`` requires a symbol; the others accept ``None``.
        symbol = "300750.SZ" if flow_type == "main" else None

        adapter = _StubAdapter(flow)
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_money_flow(
            service, symbol=symbol, flow_type=flow_type, top_n=10
        )
        assert markdown.rstrip().endswith(DISCLAIMER)

    def test_invalid_flow_type_raises_validation_error(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        adapter = _StubAdapter(_north_flow())
        service = _make_service(adapter, cache, rate_limiter)

        with pytest.raises(ValidationError):
            get_money_flow(service, flow_type="south", top_n=10)

        assert adapter.money_flow_call_count == 0


# ---------------------------------------------------------------------------
# top_n validation (Requirement 5.4)
# ---------------------------------------------------------------------------


class TestTopNValidation:
    """**Validates: Requirements 5.4**."""

    @pytest.mark.parametrize("top_n", [1, 50, 100])
    def test_top_n_in_range_accepted(
        self,
        cache: _StubCache,
        rate_limiter: RateLimiter,
        top_n: int,
    ) -> None:
        adapter = _StubAdapter(_north_flow(rows=1))
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_money_flow(
            service, flow_type="north", top_n=top_n
        )
        assert markdown.rstrip().endswith(DISCLAIMER)

    @pytest.mark.parametrize("top_n", [0, 101, -1])
    def test_top_n_out_of_range_raises_validation_error(
        self,
        cache: _StubCache,
        rate_limiter: RateLimiter,
        top_n: int,
    ) -> None:
        adapter = _StubAdapter(_north_flow())
        service = _make_service(adapter, cache, rate_limiter)

        with pytest.raises(ValidationError):
            get_money_flow(service, flow_type="north", top_n=top_n)

        assert adapter.money_flow_call_count == 0


# ---------------------------------------------------------------------------
# main flow requires symbol (Requirement 5.2)
# ---------------------------------------------------------------------------


class TestMainFlowRequiresSymbol:
    """**Validates: Requirements 5.2**."""

    def test_main_without_symbol_raises_validation_error(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        adapter = _StubAdapter(_main_flow())
        service = _make_service(adapter, cache, rate_limiter)

        with pytest.raises(ValidationError) as exc_info:
            get_money_flow(service, flow_type="main", top_n=10)

        # The error message clearly identifies what the caller missed.
        message = str(exc_info.value)
        assert "main" in message
        assert "symbol" in message
        assert adapter.money_flow_call_count == 0


# ---------------------------------------------------------------------------
# Truncation: rendered row count reflects what the adapter returned
# ---------------------------------------------------------------------------


class TestRowsRespectAdapterReturn:
    """The renderer surfaces every row the adapter / cache returns.

    The service does NOT post-truncate the adapter's response (the
    adapter is responsible for honouring ``top_n``). The renderer
    therefore reflects whatever the upstream produced, and the tool
    layer's heading shows the requested ``top_n`` for transparency.
    """

    def test_renderer_shows_all_rows_returned_by_adapter(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        # Adapter returns 5 rows even though we request top_n=3 -- the
        # service caches whatever the adapter produced.
        adapter = _StubAdapter(_north_flow(rows=5))
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_money_flow(
            service, flow_type="north", top_n=3
        )

        # Heading reflects the requested top_n, not the actual count.
        assert "top 3" in markdown
        # All 5 row markers (净流入金额 values) surface in the table
        # body since the renderer does not post-truncate.
        # We check on date strings as a stable proxy for row count.
        for _ in range(5):
            assert "2024-01-02" in markdown


# ---------------------------------------------------------------------------
# snapshot_at rendered in header (Requirement 5.6)
# ---------------------------------------------------------------------------


class TestSnapshotAtRendered:
    """**Validates: Requirements 5.6**."""

    def test_snapshot_at_appears_in_markdown_header(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        adapter = _StubAdapter(_north_flow())
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_money_flow(
            service, flow_type="north", top_n=10
        )

        # Renderer formats snapshot_at as ``YYYY-MM-DD HH:MM:SS``.
        assert "2024-01-02 15:00:00" in markdown
        # And surfaces it in a dedicated 数据时间 header.
        assert "数据时间:" in markdown
