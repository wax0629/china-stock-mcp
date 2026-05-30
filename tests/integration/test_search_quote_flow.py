"""Integration tests for the ``search_symbol`` → ``get_quote`` chain.

Exercises the full pipeline ``tool entry → Service layer → Adapter`` with a
hand-rolled :class:`StubAdapter`, so that the integration suite never
hits a real network. The tests pin behavior for tasks 8.3, 8.4 and 8.7
of ``.kiro/specs/china-stock-mcp/tasks.md`` and assert the contracts
below:

- Requirements 1.1 -- ``search_symbol`` returns a Markdown table.
- Requirements 2.1 -- single ``symbol`` renders a 行情卡片.
- Requirements 2.2 -- list ``symbol`` renders a multi-row table.
- Requirements 2.3 -- batch size > 20 raises :class:`ValidationError`.
- Requirements 2.6 -- ``CSM_DATA_DELAY_NOTICE`` toggles the delay line.
- Requirements 11.1 -- per-symbol cache hits short-circuit upstream calls.
- Requirements 12.1 / Property 14 -- every response ends with the
  canonical disclaimer.
- Requirements 13.7 -- invalid ``market`` surfaces as
  :class:`ValidationError`.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.cache import Cache, build_cache, reset_default_cache
from china_stock_mcp.config import Settings
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
from china_stock_mcp.services.quote_service import QuoteService
from china_stock_mcp.services.symbol_service import SymbolService
from china_stock_mcp.tools.quote import get_quote
from china_stock_mcp.tools.search import search_symbol

# ---------------------------------------------------------------------------
# Stub adapter
# ---------------------------------------------------------------------------


class StubAdapter(BaseAdapter):
    """In-memory :class:`BaseAdapter` for hermetic integration tests.

    Records call counts so tests can assert that caching short-circuits
    a second invocation. Methods unrelated to ``search`` / ``quote``
    raise :class:`NotImplementedError` because task 8.7 only exercises
    the search → quote chain.
    """

    name: str = "stub"

    def __init__(
        self,
        *,
        search_hits: list[SymbolHit] | None = None,
        quotes_by_symbol: dict[str, Quote] | None = None,
    ) -> None:
        self._search_hits: list[SymbolHit] = list(search_hits or [])
        self._quotes_by_symbol: dict[str, Quote] = dict(quotes_by_symbol or {})
        self.search_call_count: int = 0
        self.quote_call_count: int = 0
        self.last_search_call: tuple[str, str] | None = None
        self.last_quote_call: list[str] | None = None

    # ----- exercised by tests --------------------------------------------

    def search(self, query: str, market: str) -> list[SymbolHit]:
        self.search_call_count += 1
        self.last_search_call = (query, market)
        return list(self._search_hits)

    def quote(self, symbols: list[str]) -> list[Quote]:
        self.quote_call_count += 1
        self.last_quote_call = list(symbols)
        out: list[Quote] = []
        for sym in symbols:
            quote_dto = self._quotes_by_symbol.get(sym)
            if quote_dto is not None:
                out.append(quote_dto)
        return out

    # ----- not implemented in 8.7 ----------------------------------------

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
        raise NotImplementedError("StubAdapter does not implement financial_report")

    def money_flow(
        self,
        symbol: str | None,
        flow_type: str,
        top_n: int,
    ) -> MoneyFlow:
        raise NotImplementedError("StubAdapter does not implement money_flow")

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
    """Each test starts and ends with a fresh process-wide cache."""

    reset_default_cache()
    try:
        yield
    finally:
        reset_default_cache()


@pytest.fixture()
def cache(tmp_path: Path) -> Iterator[Cache]:
    """Disk-backed :class:`Cache` rooted at ``tmp_path``."""

    backend = build_cache(Settings(cache_backend="disk", cache_dir=tmp_path))
    try:
        yield backend
    finally:
        backend.close()


@pytest.fixture()
def rate_limiter() -> RateLimiter:
    """Generous limiter so tests never trip the budget accidentally."""

    return RateLimiter(rate_per_minute=10_000)


# ---------------------------------------------------------------------------
# Quote helper
# ---------------------------------------------------------------------------


_FIXED_TIMESTAMP = datetime(2024, 1, 2, 14, 30, 0, tzinfo=UTC)


def _make_quote(
    symbol: str,
    name: str,
    *,
    price: float = 100.0,
    change: float = 1.5,
    change_pct: float = 1.5,
    market_cap: float = 5e10,
) -> Quote:
    """Build a fully populated :class:`Quote` for fixture data."""

    return Quote(
        symbol=symbol,
        name=name,
        price=price,
        change=change,
        change_pct=change_pct,
        volume=10_000_000,
        amount=1.5e9,
        turnover_rate=1.25,
        pe_ttm=18.5,
        pe_dynamic=17.2,
        pb=3.4,
        market_cap=market_cap,
        float_market_cap=market_cap * 0.8,
        timestamp=_FIXED_TIMESTAMP,
        delay_seconds=900,
    )


# ---------------------------------------------------------------------------
# search → tool entrypoint tests (Requirements 1.1, 12.1, 13.7)
# ---------------------------------------------------------------------------


class TestSearchSymbolFlow:
    """End-to-end coverage of :func:`search_symbol`."""

    def test_search_returns_markdown_table_with_disclaimer(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirements 1.1 + 12.1 -- table, market labels, disclaimer."""

        hits = [
            SymbolHit(
                code="300750.SZ", name="宁德时代", market="a_stock", industry="电池"
            ),
            SymbolHit(
                code="00700.HK", name="腾讯控股", market="hk_stock", industry="互联网"
            ),
        ]
        adapter = StubAdapter(search_hits=hits)
        service = SymbolService(adapter, cache=cache, rate_limiter=rate_limiter)

        markdown = search_symbol(service, query="宁德", market="all")

        # Heading + both codes + both names.
        assert "### 标的搜索:" in markdown
        assert "300750.SZ" in markdown
        assert "00700.HK" in markdown
        assert "宁德时代" in markdown
        assert "腾讯控股" in markdown

        # Localized market labels (not the raw enum names).
        assert "A股" in markdown
        assert "港股" in markdown
        assert "a_stock" not in markdown
        assert "hk_stock" not in markdown

        # Industry column rendered.
        assert "电池" in markdown
        assert "互联网" in markdown

        # Disclaimer terminator (Property 14).
        assert markdown.rstrip().endswith(DISCLAIMER)

    def test_search_empty_result(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirements 1.1 -- empty list still renders a heading + notice."""

        adapter = StubAdapter(search_hits=[])
        service = SymbolService(adapter, cache=cache, rate_limiter=rate_limiter)

        markdown = search_symbol(service, query="不存在的标的", market="all")

        assert "未找到匹配的标的" in markdown
        assert markdown.rstrip().endswith(DISCLAIMER)

    def test_search_invalid_market_raises_validation_error(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirements 13.7 -- pydantic Literal violation surfaces."""

        adapter = StubAdapter(search_hits=[])
        service = SymbolService(adapter, cache=cache, rate_limiter=rate_limiter)

        with pytest.raises(ValidationError):
            search_symbol(service, query="宁德", market="us_stock")

        # The adapter must NOT be invoked when validation fails.
        assert adapter.search_call_count == 0


# ---------------------------------------------------------------------------
# get_quote tests (Requirements 2.1, 2.2, 2.3, 2.6, 11.1, 12.1)
# ---------------------------------------------------------------------------


class TestGetQuoteFlow:
    """End-to-end coverage of :func:`get_quote`."""

    def test_quote_single_renders_card(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirements 2.1 -- single string input renders a card."""

        quotes = {
            "300750.SZ": _make_quote("300750.SZ", "宁德时代", change_pct=2.5),
        }
        adapter = StubAdapter(quotes_by_symbol=quotes)
        service = QuoteService(adapter, cache=cache, rate_limiter=rate_limiter)
        settings = Settings(
            cache_backend="disk",
            cache_dir=Path("."),  # unused -- tool only reads ``data_delay_notice``
            data_delay_notice=False,
        )

        markdown = get_quote(service, "300750.SZ", settings=settings)

        assert "300750.SZ" in markdown
        assert "宁德时代" in markdown
        assert "现价" in markdown
        # Single-symbol path renders the card heading, not the table heading.
        assert "### 行情快照" not in markdown
        assert markdown.rstrip().endswith(DISCLAIMER)

    def test_quote_batch_renders_table(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirements 2.2 + Property 16 -- table with 涨跌色 emoji."""

        quotes = {
            "300750.SZ": _make_quote("300750.SZ", "宁德时代", change_pct=2.5),
            "002594.SZ": _make_quote("002594.SZ", "比亚迪", change_pct=-1.8),
            "600519.SH": _make_quote("600519.SH", "贵州茅台", change_pct=0.7),
        }
        adapter = StubAdapter(quotes_by_symbol=quotes)
        service = QuoteService(adapter, cache=cache, rate_limiter=rate_limiter)
        settings = Settings(
            cache_backend="disk", cache_dir=Path("."), data_delay_notice=False
        )

        markdown = get_quote(
            service,
            ["300750.SZ", "002594.SZ", "600519.SH"],
            settings=settings,
        )

        assert "### 行情快照 (3 只)" in markdown
        for code in ("300750.SZ", "002594.SZ", "600519.SH"):
            assert code in markdown

        # Mixed-direction batch -- both emojis must appear in the
        # 涨跌幅 column (positive + negative percent moves).
        assert "🔴" in markdown
        assert "🟢" in markdown
        assert markdown.rstrip().endswith(DISCLAIMER)

    def test_quote_batch_size_limit(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirements 2.3 -- 21 symbols → :class:`ValidationError`."""

        adapter = StubAdapter(quotes_by_symbol={})
        service = QuoteService(adapter, cache=cache, rate_limiter=rate_limiter)

        oversized = [f"30075{i:01d}.SZ" for i in range(10)] + [
            f"00060{i:01d}.SZ" for i in range(11)
        ]
        assert len(oversized) == 21

        with pytest.raises(ValidationError):
            get_quote(service, oversized)

        # The adapter must NOT have been touched.
        assert adapter.quote_call_count == 0

    def test_quote_caches_per_symbol(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirements 11.1 -- per-symbol cache; duplicates dedup; second
        call does not re-hit the adapter."""

        quotes = {
            "300750.SZ": _make_quote("300750.SZ", "宁德时代", change_pct=1.0),
            "002594.SZ": _make_quote("002594.SZ", "比亚迪", change_pct=-0.5),
        }
        adapter = StubAdapter(quotes_by_symbol=quotes)
        service = QuoteService(adapter, cache=cache, rate_limiter=rate_limiter)

        # Caller-order with one duplicate; service must dedup before
        # calling the adapter and must replay the duplicate from cache.
        first_call = service.get_snapshot(["300750.SZ", "002594.SZ", "300750.SZ"])
        assert [q.symbol for q in first_call] == [
            "300750.SZ",
            "002594.SZ",
            "300750.SZ",
        ]

        # Adapter saw exactly one batch, with the deduplicated symbols.
        assert adapter.quote_call_count == 1
        assert adapter.last_quote_call == ["300750.SZ", "002594.SZ"]

        # Second call: cache must service everything, so the adapter
        # is never invoked again (Requirements 11.1).
        second_call = service.get_snapshot(["300750.SZ", "002594.SZ"])
        assert [q.symbol for q in second_call] == ["300750.SZ", "002594.SZ"]
        assert adapter.quote_call_count == 1  # unchanged

    def test_quote_data_delay_notice_toggle(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirements 2.6 -- toggle on/off via ``Settings``."""

        quotes = {"300750.SZ": _make_quote("300750.SZ", "宁德时代")}
        adapter = StubAdapter(quotes_by_symbol=quotes)
        service = QuoteService(adapter, cache=cache, rate_limiter=rate_limiter)

        with_notice = get_quote(
            service,
            "300750.SZ",
            settings=Settings(
                cache_backend="disk",
                cache_dir=Path("."),
                data_delay_notice=True,
            ),
        )
        without_notice = get_quote(
            service,
            "300750.SZ",
            settings=Settings(
                cache_backend="disk",
                cache_dir=Path("."),
                data_delay_notice=False,
            ),
        )

        assert "数据延迟约 15 分钟" in with_notice
        # The blockquote prefix is part of the canonical notice line.
        assert "> ℹ️ 数据延迟约 15 分钟" in with_notice  # noqa: RUF001

        # ``without_notice`` must not include the blockquote header
        # line; the ``时延`` row inside the card is allowed and is
        # part of the quote body, not the prefixed notice.
        assert "> ℹ️ 数据延迟约 15 分钟" not in without_notice  # noqa: RUF001

    # ------------------------------------------------------------------
    # search → quote pipeline (Requirements 1.1 + 2.1)
    # ------------------------------------------------------------------

    def test_search_to_quote_pipeline(
        self, cache: Cache, rate_limiter: RateLimiter
    ) -> None:
        """Requirements 1.1 + 2.1 -- chain ``search_symbol`` into
        ``get_quote`` end-to-end."""

        # Adapter knows the same symbol on both endpoints so the chain
        # is feasible without any additional plumbing.
        std_code = "300750.SZ"
        hits = [SymbolHit(code=std_code, name="宁德时代", market="a_stock")]
        quotes = {std_code: _make_quote(std_code, "宁德时代", change_pct=1.2)}
        adapter = StubAdapter(search_hits=hits, quotes_by_symbol=quotes)

        symbol_service = SymbolService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )
        quote_service = QuoteService(
            adapter, cache=cache, rate_limiter=rate_limiter
        )

        # 1) Search -> Markdown table containing the standardized code.
        search_md = search_symbol(symbol_service, query="宁德", market="a_stock")
        match = re.search(r"\d{6}\.(?:SH|SZ|BJ)", search_md)
        assert match is not None, f"could not parse code from: {search_md!r}"
        parsed_code = match.group(0)
        assert parsed_code == std_code

        # 2) Feed parsed code into get_quote.
        quote_md = get_quote(
            quote_service,
            parsed_code,
            settings=Settings(
                cache_backend="disk",
                cache_dir=Path("."),
                data_delay_notice=False,
            ),
        )

        assert std_code in quote_md
        assert "宁德时代" in quote_md
        assert "现价" in quote_md
        assert quote_md.rstrip().endswith(DISCLAIMER)
