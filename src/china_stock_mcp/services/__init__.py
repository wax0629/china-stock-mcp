"""Service layer for china-stock-mcp.

This package implements *Component 3 (Service Layer)* from
``design.md``. Services compose three lower-level building blocks:

* :mod:`china_stock_mcp.normalizer`    -- canonical symbol form
* :mod:`china_stock_mcp.cache`         -- TTL-graded read-through cache
* :mod:`china_stock_mcp.rate_limiter`  -- global Token Bucket admission
* :mod:`china_stock_mcp.adapters`      -- akshare / tushare / efinance

Services are deliberately thin: they orchestrate normalization, caching
and rate limiting, then delegate to a :class:`BaseAdapter`. They never
touch the FastMCP protocol layer (that lives in ``tools/`` /
``server.py``) and never render Markdown (that lives in
``formatters.py``).

Implemented services
--------------------

- :class:`SymbolService` -- wraps :func:`normalize_symbol` and the
  adapter's ``search`` endpoint (Requirements 1.1, 1.7).
- :class:`QuoteService`  -- wraps the adapter's ``quote`` endpoint with
  per-symbol caching and a 20-symbol batch ceiling
  (Requirements 2.2, 2.3, 11.1, 11.6, 11.7).
- :class:`KLineService`  -- wraps the adapter's ``kline`` endpoint with
  per-series caching and computes MA / MACD / RSI14 / BOLL_MID
  indicators plus a ``pattern_note`` heuristic
  (Requirements 3.1, 3.2, 3.3, 3.6, 3.7).
- :class:`FundamentalService` -- wraps the adapter's ``fundamentals``
  endpoint with per-symbol caching at the ``TTL_FROZEN`` grade
  (Requirements 4.1, 4.2, 11.4).
- :class:`FinancialReportService` -- wraps the adapter's
  ``financial_report`` endpoint with per-(symbol, report_type, periods)
  caching at the ``TTL_FROZEN`` grade and a stable ascending sort by
  ``period_end`` (Requirements 4.3, 4.4, 4.5, 4.6, 11.4).
- :class:`MoneyFlowService` -- wraps the adapter's ``money_flow``
  endpoint with per-(symbol_or_aggregate, flow_type, top_n) caching at
  the ``TTL_WARM`` grade and validates ``flow_type`` /
  ``top_n`` (Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 11.4).
- :class:`IndustryService` -- wraps the adapter's ``industry_peers``
  endpoint with per-(symbol, sorted_metrics, top_n) caching at the
  ``TTL_FROZEN`` grade, validates ``metrics`` /
  ``top_n``, and annotates each row with the 0-100 行业分位 of
  every numeric metric (Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 11.4).
- :class:`FundService` -- wraps the adapter's ``fund_info`` endpoint
  with per-fund-code caching at the ``TTL_COLD`` grade and validates
  the 6-digit code shape (Requirements 7.1, 7.2, 7.3, 11.4).
- :class:`ScreenService` -- 多因子选股 (design Algorithm 5).
  Reads the A股 universe via ``ak.stock_zh_a_spot_em`` (optionally
  intersected with ``ak.stock_board_industry_cons_em``), applies
  range filters, stably sorts by the requested field and truncates
  to ``limit``. Cached at the ``TTL_WARM`` grade so screen results
  refresh intra-day (Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 11.4).
"""

from __future__ import annotations

from china_stock_mcp.services.financial_report_service import FinancialReportService
from china_stock_mcp.services.fund_service import FundService
from china_stock_mcp.services.fundamental_service import FundamentalService
from china_stock_mcp.services.industry_service import IndustryService
from china_stock_mcp.services.kline_service import KLineService
from china_stock_mcp.services.market_service import MarketService
from china_stock_mcp.services.money_flow_service import MoneyFlowService
from china_stock_mcp.services.quote_service import QuoteService
from china_stock_mcp.services.screen_service import ScreenService
from china_stock_mcp.services.symbol_service import SymbolService

__all__ = [
    "FinancialReportService",
    "FundService",
    "FundamentalService",
    "IndustryService",
    "KLineService",
    "MarketService",
    "MoneyFlowService",
    "QuoteService",
    "ScreenService",
    "SymbolService",
]
