"""Integration tests for FastMCP Resource registration + read.

Covers task 21.2 (validates Requirement 9.1, 5.1, 12.1, 1.4) by
asserting the FastMCP server exposes the three documented resources
and that each one renders a Markdown document terminated by the
canonical :data:`DISCLAIMER`.

Resources verified
------------------

- ``market://overview``      -- :func:`market_overview_resource`
- ``flow://north``           -- :func:`north_flow_resource`
- ``symbol://{code}/profile`` -- :func:`symbol_profile_resource` (template)

Network avoidance
-----------------

The :class:`FastMCP` server in :mod:`china_stock_mcp.server` builds
its services lazily via :func:`_build_services` on first invocation.
That helper instantiates :class:`AkshareAdapter`, which talks to the
network when its endpoints are exercised. To keep the integration
suite hermetic we monkey-patch the module-level ``_services_singleton``
*before* the resource handlers run, swapping in a stub services
tuple where every method returns deterministic in-memory DTOs. The
monkey-patch is reset between tests via ``autouse`` fixture so a
later test that *wants* the real services (none currently do, but
the safety net guards future additions) starts from a clean state.

The tests use :class:`fastmcp.FastMCP` API surface directly:

- :meth:`FastMCP.list_resources` -- the two static resources.
- :meth:`FastMCP.list_resource_templates` -- the one URI template.
- :meth:`FastMCP.read_resource` -- the byte-identical Markdown body.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from china_stock_mcp import server as server_module
from china_stock_mcp.formatters import DISCLAIMER
from china_stock_mcp.models import (
    FundamentalSnapshot,
    MarketOverview,
    MoneyFlow,
    SymbolHit,
)
from china_stock_mcp.server import mcp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The three URIs the server is contractually required to register.
#: Two are static resources, one is a URI template (variable ``code``).
_EXPECTED_STATIC_URIS: frozenset[str] = frozenset(
    {"market://overview", "flow://north"}
)
_EXPECTED_TEMPLATE_URI: str = "symbol://{code}/profile"

_FIXED_TIMESTAMP = datetime(2024, 6, 3, 14, 30, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Stub DTO builders
# ---------------------------------------------------------------------------


def _make_market_overview() -> MarketOverview:
    return MarketOverview(
        indices=[
            {
                "name": "上证指数",
                "code": "000001.SH",
                "last": 3050.0,
                "change_pct": 0.85,
            },
        ],
        advance_decline={"advance": 3000, "decline": 1500, "flat": 200},
        limit_stats={"limit_up": 50, "limit_down": 10},
        north_net_inflow=1.5e9,
        top_inflow_industries=[
            {"name": "电池", "net_inflow": 5e8},
        ],
        heat_score=72.5,
        snapshot_at=_FIXED_TIMESTAMP,
    )


def _make_north_flow() -> MoneyFlow:
    return MoneyFlow(
        flow_type="north",
        rows=[
            {
                "date": "2024-06-03",
                "净流入金额": 1.5e9,
                "买入金额": 5.0e9,
                "卖出金额": 3.5e9,
                "持股市值": 2.5e12,
            },
        ],
        snapshot_at=_FIXED_TIMESTAMP,
    )


def _make_snapshot(symbol: str) -> FundamentalSnapshot:
    return FundamentalSnapshot(
        symbol=symbol,
        valuation={"pe_ttm": 18.5, "pb": 3.4, "pe_dynamic": 17.2},
        profitability={"roe": 18.0, "gross_margin": 28.0, "net_margin": 12.0},
        growth={"revenue_yoy": 22.5, "net_profit_yoy": 18.5},
        health={"debt_ratio": 45.0, "current_ratio": 1.8},
        industry_percentile={},
    )


# ---------------------------------------------------------------------------
# Stub services
# ---------------------------------------------------------------------------


class _StubMarketService:
    """Stand-in for :class:`MarketService.overview`."""

    def overview(self) -> MarketOverview:
        return _make_market_overview()


class _StubMoneyFlowService:
    """Stand-in for :class:`MoneyFlowService.get`.

    Only the ``flow_type="north"`` path is exercised by the resource
    suite; the ``main`` / ``dragon_tiger`` branches would require
    different DTO shapes.
    """

    def get(
        self,
        symbol: str | None,
        flow_type: str,
        top_n: int,
    ) -> MoneyFlow:
        return _make_north_flow()


class _StubSymbolService:
    """Stand-in for :class:`SymbolService` -- normalize + search.

    The :func:`symbol_profile_resource` invokes both ``normalize`` and
    ``search``; the stub returns a deterministic search hit so the
    resource always renders a populated 基本信息 section.
    """

    def normalize(self, raw: str, market: str | None = None) -> str:
        # Pass-through normalization keeps the resource logic close
        # to its production behaviour for valid standardized codes.
        return raw

    def search(self, query: str, market: str = "all") -> list[SymbolHit]:
        return [
            SymbolHit(
                code=query,
                name="测试标的",
                market="a_stock",
                industry="测试行业",
            )
        ]


class _StubFundamentalService:
    """Stand-in for :class:`FundamentalService.snapshot`."""

    def snapshot(self, symbol: str) -> FundamentalSnapshot:
        return _make_snapshot(symbol)


# ---------------------------------------------------------------------------
# Monkey-patch fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_services_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Swap the lazy services tuple with stubs for the test duration.

    The :func:`_build_services` helper memoizes its result in the
    module-level ``_services_singleton`` slot. We pre-populate that
    slot with stubs *before* the resource handler runs so the
    handler's ``_, _, _, fundamental_service, _, money_flow_service,
    _, _, _, market_service = _build_services()`` unpack returns our
    stand-ins instead of forcing the real :class:`AkshareAdapter` to
    talk to the network.

    The slot order must match :func:`_build_services`:

    ``(symbol, quote, kline, fundamental, financial_report,
       money_flow, industry, fund, screen, market)``
    """

    symbol_stub = _StubSymbolService()
    fundamental_stub = _StubFundamentalService()
    money_flow_stub = _StubMoneyFlowService()
    market_stub = _StubMarketService()

    # The remaining service slots are not invoked by any of the three
    # resources, so we use ``None`` typed-as-``object`` placeholders.
    # FastMCP's resource handlers index the tuple positionally and
    # we only assert behaviour for the slots the resources use.
    services_tuple = (
        symbol_stub,
        None,  # quote_service -- not used by resources
        None,  # kline_service
        fundamental_stub,
        None,  # financial_report_service
        money_flow_stub,
        None,  # industry_service
        None,  # fund_service
        None,  # screen_service
        market_stub,
    )

    monkeypatch.setattr(
        server_module,
        "_services_singleton",
        services_tuple,
    )
    yield
    # ``monkeypatch`` restores ``_services_singleton`` automatically;
    # nothing else to clean up.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_resource_text(uri: str) -> str:
    """Synchronously read ``uri`` and return the rendered text.

    FastMCP's :meth:`FastMCP.read_resource` is async; we drive the
    coroutine to completion with :func:`asyncio.run`. The returned
    value is a :class:`fastmcp.resources.base.ResourceResult` whose
    ``contents`` list carries one or more
    :class:`fastmcp.resources.base.ResourceContent` entries; we
    concatenate their ``content`` fields so the assertion sees the
    same Markdown an MCP client would.
    """

    result = asyncio.run(mcp.read_resource(uri))
    parts: list[str] = []
    for entry in result.contents:
        # ``ResourceContent.content`` is the canonical text payload
        # in fastmcp 3.x; fall back to ``text`` to stay compatible
        # with older versions that used the previous field name.
        text = getattr(entry, "content", None) or getattr(entry, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestResourceRegistration:
    """The FastMCP server registers exactly the three documented URIs."""

    def test_static_resources_are_registered(self) -> None:
        """Both static resources are listed by ``list_resources``."""

        resources = asyncio.run(mcp.list_resources())
        # ``Resource.uri`` is a pydantic ``AnyUrl`` -- coerce to ``str``
        # for set-comparison so we are insensitive to the wrapper type.
        registered = {str(r.uri) for r in resources}

        # The two static URIs must be present; we use ``>=`` rather
        # than ``==`` so an additional debug resource added later does
        # not break the assertion. Task 21.2 only mandates these two
        # are *registered*, not that they are the only ones.
        assert _EXPECTED_STATIC_URIS <= registered, (
            f"missing static resources: "
            f"{_EXPECTED_STATIC_URIS - registered}"
        )

    def test_resource_template_is_registered(self) -> None:
        """``symbol://{code}/profile`` is listed as a URI template."""

        templates = asyncio.run(mcp.list_resource_templates())
        registered = {t.uri_template for t in templates}

        assert _EXPECTED_TEMPLATE_URI in registered, (
            f"missing resource template: {_EXPECTED_TEMPLATE_URI!r}; "
            f"registered: {sorted(registered)}"
        )

    def test_total_three_resources_registered(self) -> None:
        """Static + template count adds up to three.

        This is a weaker check than the per-URI assertions above; it
        guards against accidental duplication of a URI that happens
        to satisfy both ``list_resources`` and
        ``list_resource_templates``.
        """

        resources = asyncio.run(mcp.list_resources())
        templates = asyncio.run(mcp.list_resource_templates())
        # Filter to only the URIs we care about so future debug
        # resources do not break the count.
        static_count = sum(
            1 for r in resources if str(r.uri) in _EXPECTED_STATIC_URIS
        )
        template_count = sum(
            1 for t in templates if t.uri_template == _EXPECTED_TEMPLATE_URI
        )
        assert static_count == 2
        assert template_count == 1


# ---------------------------------------------------------------------------
# Read tests
# ---------------------------------------------------------------------------


class TestResourceRead:
    """Reading each resource returns Markdown ending with the disclaimer."""

    def test_market_overview_resource_renders_markdown_with_disclaimer(
        self,
    ) -> None:
        """``market://overview`` -- 市场总览 + disclaimer (Requirement 9.1, 12.1)."""

        text = _read_resource_text("market://overview")

        # The 市场总览 tool always renders these section markers.
        assert "**指数行情**" in text
        assert "**涨跌家数**" in text
        assert "**北向资金净流入**" in text
        # 市场热度评分 surfaces the heat_score.
        assert "市场热度评分" in text

        # Property 14 / Requirement 12.1 -- end with the canonical
        # disclaimer.
        assert text.rstrip().endswith(DISCLAIMER)

    def test_north_flow_resource_renders_markdown_with_disclaimer(
        self,
    ) -> None:
        """``flow://north`` -- 北向资金 table + disclaimer (Requirement 5.1, 12.1)."""

        text = _read_resource_text("flow://north")

        # The northbound flow tool labels the section header and
        # surfaces the snapshot timestamp.
        assert "数据时间" in text
        # Each row carries a 净流入 column populated from the stub
        # payload; verify the value or the column label is present.
        assert "净流入" in text or "北向" in text

        # Property 14 / Requirement 12.1.
        assert text.rstrip().endswith(DISCLAIMER)

    def test_symbol_profile_resource_renders_markdown_with_disclaimer(
        self,
    ) -> None:
        """``symbol://{code}/profile`` -- 标的概览 + disclaimer.

        We resolve the URI template by passing a concrete code
        (``300750.SZ``) to :meth:`FastMCP.read_resource`; FastMCP
        binds the path parameter automatically.
        """

        text = _read_resource_text("symbol://300750.SZ/profile")

        # The 标的概览 resource renders a two-section document.
        assert "## 标的概览 (300750.SZ)" in text
        assert "### 基本信息" in text
        assert "### 估值与盈利" in text

        # The stub :class:`_StubSymbolService.search` returns a
        # populated hit so the basic-info table carries real values
        # (industry / market localized labels).
        assert "300750.SZ" in text
        assert "测试标的" in text

        # Property 14 / Requirement 12.1.
        assert text.rstrip().endswith(DISCLAIMER)
