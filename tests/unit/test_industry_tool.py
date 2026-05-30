"""Unit tests for :mod:`china_stock_mcp.tools.industry` (task 14.2).

Covers tool-layer rendering and validation:

- Requirement 6.2 / 6.5 -- ``metrics`` ⊆ ``{"pe", "pb", "roe",
  "revenue_growth"}``; an unsupported metric raises
  :class:`ValidationError`.
- Requirement 6.4 -- ``top_n`` ∈ ``[1, 50]``; out-of-range values raise
  :class:`ValidationError`.
- Requirement 6.3 -- per-metric "行业分位" notes are rendered when at
  least one numeric value is present for that metric.
- Requirement 6.1 / 6.6 -- the Markdown table contains 代码 / 名称 +
  every requested metric column in the caller-supplied order.
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
    SymbolHit,
)
from china_stock_mcp.rate_limiter import RateLimiter
from china_stock_mcp.services.industry_service import IndustryService
from china_stock_mcp.tools.industry import get_industry_peers

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

    def __init__(self, peers: PeerTable) -> None:
        self._peers = peers
        self.industry_peers_call_count: int = 0

    def industry_peers(
        self,
        symbol: str,
        metrics: list[str],
        top_n: int,
    ) -> PeerTable:
        self.industry_peers_call_count += 1
        return self._peers

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

    def money_flow(
        self, symbol: str | None, flow_type: str, top_n: int
    ) -> MoneyFlow:
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


def _peers(metrics: list[str]) -> PeerTable:
    """Build a :class:`PeerTable` with three peers and the given metrics.

    The 300750 row is included so the renderer surfaces a 行业分位
    annotation that mentions the base symbol.
    """

    rows: list[dict[str, object]] = [
        {
            "代码": "300750",
            "名称": "宁德时代",
            **{m: 18.5 + i for i, m in enumerate(metrics)},
        },
        {
            "代码": "002594",
            "名称": "比亚迪",
            **{m: 22.1 + i for i, m in enumerate(metrics)},
        },
        {
            "代码": "601012",
            "名称": "隆基绿能",
            **{m: 12.4 + i for i, m in enumerate(metrics)},
        },
    ]
    return PeerTable(
        base_symbol="300750.SZ",
        industry="电池",
        metrics=metrics,
        rows=rows,
    )


def _make_service(
    adapter: _StubAdapter,
    cache: Cache,
    rate_limiter: RateLimiter,
) -> IndustryService:
    return IndustryService(adapter, cache=cache, rate_limiter=rate_limiter)


# ---------------------------------------------------------------------------
# metrics validation (Requirement 6.2 / 6.5)
# ---------------------------------------------------------------------------


class TestMetricsValidation:
    """**Validates: Requirements 6.2, 6.5**."""

    @pytest.mark.parametrize(
        "metrics",
        [
            ["pe"],
            ["pb"],
            ["roe"],
            ["revenue_growth"],
            ["pe", "pb"],
            ["pe", "pb", "roe", "revenue_growth"],
        ],
    )
    def test_supported_metrics_accepted(
        self,
        cache: _StubCache,
        rate_limiter: RateLimiter,
        metrics: list[str],
    ) -> None:
        adapter = _StubAdapter(_peers(metrics))
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_industry_peers(
            service, "300750.SZ", metrics=metrics, top_n=10
        )
        assert markdown.rstrip().endswith(DISCLAIMER)

    def test_unsupported_metric_raises_validation_error(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        adapter = _StubAdapter(_peers(["pe"]))
        service = _make_service(adapter, cache, rate_limiter)

        with pytest.raises(ValidationError):
            get_industry_peers(
                service,
                "300750.SZ",
                metrics=["pe", "ev_ebitda"],
                top_n=10,
            )

        assert adapter.industry_peers_call_count == 0

    def test_unsupported_metric_error_lists_full_supported_set(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        """The pydantic Literal validator reports the offending value;
        the service-layer validator (when the tool boundary is bypassed)
        lists every accepted metric. Either way, the error names the
        offending value so the AI client can recover."""

        adapter = _StubAdapter(_peers(["pe"]))
        service = _make_service(adapter, cache, rate_limiter)

        # Bypass the pydantic input model and call the service layer
        # directly so the 6.5 listing branch is exercised.
        with pytest.raises(ValidationError) as exc_info:
            service.peers(
                symbol="300750.SZ",
                metrics=["ev_ebitda"],
                top_n=10,
            )

        message = str(exc_info.value)
        for accepted in ("pe", "pb", "roe", "revenue_growth"):
            assert accepted in message


# ---------------------------------------------------------------------------
# top_n validation (Requirement 6.4)
# ---------------------------------------------------------------------------


class TestTopNValidation:
    """**Validates: Requirements 6.4**."""

    @pytest.mark.parametrize("top_n", [1, 25, 50])
    def test_top_n_in_range_accepted(
        self,
        cache: _StubCache,
        rate_limiter: RateLimiter,
        top_n: int,
    ) -> None:
        adapter = _StubAdapter(_peers(["pe"]))
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_industry_peers(
            service, "300750.SZ", metrics=["pe"], top_n=top_n
        )
        assert markdown.rstrip().endswith(DISCLAIMER)

    @pytest.mark.parametrize("top_n", [0, 51, -1])
    def test_top_n_out_of_range_raises_validation_error(
        self,
        cache: _StubCache,
        rate_limiter: RateLimiter,
        top_n: int,
    ) -> None:
        adapter = _StubAdapter(_peers(["pe"]))
        service = _make_service(adapter, cache, rate_limiter)

        with pytest.raises(ValidationError):
            get_industry_peers(
                service, "300750.SZ", metrics=["pe"], top_n=top_n
            )

        assert adapter.industry_peers_call_count == 0


# ---------------------------------------------------------------------------
# Per-metric percentile notes (Requirement 6.3)
# ---------------------------------------------------------------------------


class TestPercentileNotes:
    """**Validates: Requirements 6.3**."""

    def test_percentile_note_rendered_for_each_metric_with_numeric_values(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        metrics = ["pe", "roe"]
        adapter = _StubAdapter(_peers(metrics))
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_industry_peers(
            service, "300750.SZ", metrics=metrics, top_n=10
        )

        # 行业分位 footnote lines exist for both metrics.
        assert markdown.count("行业分位") >= 2
        # Localized labels show up in the notes.
        assert "市盈率(动态)" in markdown
        assert "净资产收益率" in markdown


# ---------------------------------------------------------------------------
# Markdown structure (Requirement 6.1)
# ---------------------------------------------------------------------------


class TestMarkdownStructure:
    """**Validates: Requirements 6.1, 6.6**."""

    def test_markdown_contains_required_columns_in_caller_order(
        self, cache: _StubCache, rate_limiter: RateLimiter
    ) -> None:
        # Caller-specified metric order is significant.
        metrics = ["pb", "pe", "roe"]
        adapter = _StubAdapter(_peers(metrics))
        service = _make_service(adapter, cache, rate_limiter)

        markdown = get_industry_peers(
            service, "300750.SZ", metrics=metrics, top_n=10
        )

        assert "代码" in markdown
        assert "名称" in markdown
        # Every selected metric label appears.
        assert "市净率" in markdown
        assert "市盈率(动态)" in markdown
        assert "净资产收益率" in markdown

        # Caller-supplied order is honored: 市净率 comes before
        # 市盈率(动态), which comes before 净资产收益率.
        idx_pb = markdown.index("市净率")
        idx_pe = markdown.index("市盈率(动态)")
        idx_roe = markdown.index("净资产收益率")
        assert idx_pb < idx_pe < idx_roe

        # Industry + base symbol metadata is in the heading.
        assert "300750.SZ" in markdown
        assert "电池" in markdown

        assert markdown.rstrip().endswith(DISCLAIMER)
