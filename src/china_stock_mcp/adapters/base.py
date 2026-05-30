"""Abstract base class for data source adapters.

The :class:`BaseAdapter` defines the uniform interface every concrete
upstream adapter (``akshare`` / ``tushare`` / ``efinance``) must
implement. Service-layer code depends on this abstraction so that the
:func:`fetch_with_fallback` helper can transparently swap a primary
source for a fallback when the primary raises ``NetworkError`` or
``RateLimitError``.

References:
    - ``design.md`` Component 4: Adapter Layer
    - Requirement 13.1: every adapter SHALL raise ``ChinaStockMCPError``
      subclasses, never bare third-party exceptions, so the error tree
      stays uniform across data sources.

Concrete subclasses are expected to:

* Set the class-level :attr:`name` attribute (e.g. ``"akshare"``) so
  log lines can disambiguate primary / fallback sources.
* Translate any third-party exception into the
  :class:`china_stock_mcp.exceptions.ChinaStockMCPError` hierarchy.
* Avoid caching, rate limiting and Markdown rendering -- those concerns
  belong to the Service / Tool layers above.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

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


class BaseAdapter(ABC):
    """Common interface for every data source adapter.

    Attributes:
        name: Short, human-readable identifier of the data source
            (e.g. ``"akshare"``, ``"tushare"``, ``"efinance"``).
            Subclasses MUST set this so logs can distinguish primary
            from fallback sources.
    """

    name: str

    @abstractmethod
    def search(self, query: str, market: str) -> list[SymbolHit]:
        """Search standardized symbols by code, name or pinyin.

        Args:
            query: Raw user query (code / Chinese name / pinyin).
            market: One of ``"a_stock"`` / ``"hk_stock"`` / ``"fund"``
                / ``"all"``.

        Returns:
            List of :class:`SymbolHit` results, possibly empty.
        """

    @abstractmethod
    def quote(self, symbols: list[str]) -> list[Quote]:
        """Fetch real-time (or delayed) quote snapshots."""

    @abstractmethod
    def kline(
        self,
        symbol: str,
        period: str,
        count: int,
        adjust: str,
    ) -> KLineSeries:
        """Fetch K-line bars for ``symbol``.

        Args:
            symbol: Standardized symbol (e.g. ``"300750.SZ"``).
            period: One of ``daily`` / ``weekly`` / ``monthly`` /
                ``60min`` / ``30min``.
            count: Maximum number of bars to return.
            adjust: One of ``qfq`` / ``hfq`` / ``none``.
        """

    @abstractmethod
    def fundamentals(self, symbol: str) -> FundamentalSnapshot:
        """Fetch the current fundamental snapshot for ``symbol``."""

    @abstractmethod
    def financial_report(
        self,
        symbol: str,
        report_type: str,
        periods: int,
    ) -> FinancialReport:
        """Fetch ``periods`` of financial statements.

        Args:
            symbol: Standardized symbol.
            report_type: ``"annual"`` or ``"quarterly"``.
            periods: Number of historical periods to return.
        """

    @abstractmethod
    def money_flow(
        self,
        symbol: str | None,
        flow_type: str,
        top_n: int,
    ) -> MoneyFlow:
        """Fetch money-flow data.

        Args:
            symbol: Standardized symbol when ``flow_type`` requires one,
                otherwise ``None``.
            flow_type: ``"north"`` / ``"main"`` / ``"dragon_tiger"``.
            top_n: Maximum number of rows to return.
        """

    @abstractmethod
    def industry_peers(
        self,
        symbol: str,
        metrics: list[str],
        top_n: int,
    ) -> PeerTable:
        """Fetch peer comparison rows for the symbol's industry."""

    @abstractmethod
    def fund_info(self, fund_code: str) -> FundInfo:
        """Fetch metadata, returns and holdings for a public fund."""

    @abstractmethod
    def market_overview(self) -> MarketOverview:
        """Fetch a snapshot of the overall A-share market."""


__all__ = ["BaseAdapter"]
