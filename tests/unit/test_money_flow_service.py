"""Unit tests for :mod:`china_stock_mcp.services.money_flow_service`.

Covers task 13.2 in ``.kiro/specs/china-stock-mcp/tasks.md``:

- Requirement 5.4 -- ``top_n`` ∈ ``[1, 100]``; out-of-range values
  raise :class:`ValidationError`.
- Requirement 5.5 -- non-whitelisted ``flow_type`` raises
  :class:`ValidationError` whose message lists the three accepted
  values.
- Requirement 5.2 -- ``main`` flow without a ``symbol`` raises
  :class:`ValidationError` ("需要 symbol").
- Requirement 5.1 -- ``north`` flow ignores any caller-provided
  ``symbol``; the upstream call always receives ``None``.
- Requirement 5.6 -- the returned :class:`MoneyFlow` carries the
  upstream ``snapshot_at`` timestamp verbatim.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.cache import Cache, build_cache, reset_default_cache
from china_stock_mcp.config import Settings
from china_stock_mcp.exceptions import ValidationError
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

# ---------------------------------------------------------------------------
# Stub adapter
# ---------------------------------------------------------------------------


class StubAdapter(BaseAdapter):
    """Hermetic :class:`BaseAdapter` exposing only ``money_flow``.

    Records the (symbol, flow_type, top_n) triple it was called with
    plus a call counter, so tests can assert that the service forwards
    exactly the expected arguments to the upstream layer.
    """

    name: str = "stub"

    def __init__(self, flow: MoneyFlow) -> None:
        self._flow = flow
        self.money_flow_call_count: int = 0
        self.last_money_flow_call: tuple[str | None, str, int] | None = None

    def money_flow(
        self,
        symbol: str | None,
        flow_type: str,
        top_n: int,
    ) -> MoneyFlow:
        self.money_flow_call_count += 1
        self.last_money_flow_call = (symbol, flow_type, top_n)
        return self._flow

    # ----- not implemented in 13.2 ---------------------------------------

    def search(self, query: str, market: str) -> list[SymbolHit]:
        raise NotImplementedError("StubAdapter does not implement search")

    def quote(self, symbols: list[str]) -> list[Quote]:
        raise NotImplementedError("StubAdapter does not implement quote")

    def kline(
        self,
        symbol: str,
        period: str,
        count: int,
        adjust: str,
    ) -> KLineSeries:
        raise NotImplementedError("StubAdapter does not implement kline")

    def fundamentals(self, symbol: str) -> FundamentalSnapshot:
        raise NotImplementedError("StubAdapter does not implement fundamentals")

    def financial_report(
        self,
        symbol: str,
        report_type: str,
        periods: int,
    ) -> FinancialReport:
        raise NotImplementedError(
            "StubAdapter does not implement financial_report"
        )

    def industry_peers(
        self,
        symbol: str,
        metrics: list[str],
        top_n: int,
    ) -> PeerTable:
        raise NotImplementedError("StubAdapter does not implement industry_peers")

    def fund_info(self, fund_code: str) -> FundInfo:
        raise NotImplementedError("StubAdapter does not implement fund_info")

    def market_overview(self) -> MarketOverview:
        raise NotImplementedError("StubAdapter does not implement market_overview")


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
def cache(tmp_path: Path) -> Iterator[Cache]:
    backend = build_cache(Settings(cache_backend="disk", cache_dir=tmp_path))
    try:
        yield backend
    finally:
        backend.close()


@pytest.fixture()
def rate_limiter() -> RateLimiter:
    return RateLimiter(rate_per_minute=10_000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_FIXED_SNAPSHOT_AT = datetime(2024, 1, 2, 15, 0, 0, tzinfo=UTC)


def _make_north_flow() -> MoneyFlow:
    """Build a minimal north-flow :class:`MoneyFlow` payload."""

    return MoneyFlow(
        flow_type="north",
        rows=[
            {"行业": "电池", "净流入金额": 1.5e9},
            {"行业": "白酒", "净流入金额": -3.2e8},
        ],
        snapshot_at=_FIXED_SNAPSHOT_AT,
    )


def _make_main_flow() -> MoneyFlow:
    """Build a minimal main-flow :class:`MoneyFlow` payload."""

    return MoneyFlow(
        flow_type="main",
        rows=[{"主力净流入": 1.2e8, "超大单": 0.8e8}],
        snapshot_at=_FIXED_SNAPSHOT_AT,
    )


# ---------------------------------------------------------------------------
# Flow-type validation (Requirement 5.5)
# ---------------------------------------------------------------------------


class TestFlowTypeValidation:
    """**Validates: Requirements 5.5**."""

    def test_invalid_flow_type_lists_three_accepted_values(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        adapter = StubAdapter(_make_north_flow())
        service = MoneyFlowService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        with pytest.raises(ValidationError) as exc_info:
            service.get(symbol=None, flow_type="bogus", top_n=10)

        message = str(exc_info.value)
        # All three accepted values surface in the error message.
        for accepted in ("north", "main", "dragon_tiger"):
            assert accepted in message
        # The offending value is also surfaced for clarity.
        assert "bogus" in message
        # No upstream call was made.
        assert adapter.money_flow_call_count == 0


# ---------------------------------------------------------------------------
# top_n validation (Requirement 5.4)
# ---------------------------------------------------------------------------


class TestTopNValidation:
    """**Validates: Requirements 5.4**."""

    def test_top_n_zero_raises_validation_error(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        adapter = StubAdapter(_make_north_flow())
        service = MoneyFlowService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        with pytest.raises(ValidationError) as exc_info:
            service.get(symbol=None, flow_type="north", top_n=0)

        message = str(exc_info.value)
        assert "1" in message
        assert "100" in message
        assert adapter.money_flow_call_count == 0

    def test_top_n_above_upper_bound_raises_validation_error(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        adapter = StubAdapter(_make_north_flow())
        service = MoneyFlowService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        with pytest.raises(ValidationError) as exc_info:
            service.get(symbol=None, flow_type="north", top_n=101)

        message = str(exc_info.value)
        assert "1" in message
        assert "100" in message
        assert adapter.money_flow_call_count == 0


# ---------------------------------------------------------------------------
# main flow requires symbol (Requirement 5.2)
# ---------------------------------------------------------------------------


class TestMainFlowRequiresSymbol:
    """**Validates: Requirements 5.2**."""

    def test_main_flow_without_symbol_raises_validation_error(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        adapter = StubAdapter(_make_main_flow())
        service = MoneyFlowService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        with pytest.raises(ValidationError) as exc_info:
            service.get(symbol=None, flow_type="main", top_n=20)

        assert "symbol" in str(exc_info.value)
        # No upstream call was made.
        assert adapter.money_flow_call_count == 0

    def test_main_flow_with_blank_symbol_raises_validation_error(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        adapter = StubAdapter(_make_main_flow())
        service = MoneyFlowService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        with pytest.raises(ValidationError):
            service.get(symbol="   ", flow_type="main", top_n=20)


# ---------------------------------------------------------------------------
# north flow ignores symbol (Requirement 5.1)
# ---------------------------------------------------------------------------


class TestNorthFlowIgnoresSymbol:
    """**Validates: Requirements 5.1**."""

    def test_north_flow_passes_none_to_upstream_regardless_of_input(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        adapter = StubAdapter(_make_north_flow())
        service = MoneyFlowService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        # Caller supplies an irrelevant symbol; service must drop it
        # before the adapter call so two callers share a cache slot.
        service.get(symbol="300750.SZ", flow_type="north", top_n=10)

        assert adapter.last_money_flow_call is not None
        upstream_symbol, upstream_flow_type, upstream_top_n = (
            adapter.last_money_flow_call
        )
        assert upstream_symbol is None
        assert upstream_flow_type == "north"
        assert upstream_top_n == 10


# ---------------------------------------------------------------------------
# snapshot_at preservation (Requirement 5.6)
# ---------------------------------------------------------------------------


class TestSnapshotAtPreserved:
    """**Validates: Requirements 5.6**."""

    def test_snapshot_at_round_trips_through_service(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        adapter = StubAdapter(_make_north_flow())
        service = MoneyFlowService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        result = service.get(symbol=None, flow_type="north", top_n=10)

        assert result.snapshot_at == _FIXED_SNAPSHOT_AT
        assert result.flow_type == "north"
        assert len(result.rows) == 2
