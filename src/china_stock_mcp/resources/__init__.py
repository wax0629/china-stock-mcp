"""Resources layer for china-stock-mcp.

This package implements *Component 1 (FastMCP Server)* resources from
``design.md``. Resources expose read-only, URI-addressable views over
the same data the tool layer already serves, but framed as MCP
*resources* so AI clients can subscribe to them as live context
documents (e.g. attach ``market://overview`` to a chat session and
have it refresh on each tool turn).

Currently implemented (task 21.1, Requirements 5.1 / 9.1):

- :func:`market_overview_resource` -- ``market://overview``: a thin
  wrapper over :class:`MarketService.overview` that re-uses the same
  Markdown rendering as ``get_market_overview`` so the AI client sees
  identical content whether it goes through the tool or the resource.
- :func:`north_flow_resource` -- ``flow://north``: 北向资金 daily flow
  rendered identically to ``get_money_flow(flow_type='north', top_n=20)``.
- :func:`symbol_profile_resource` -- ``symbol://{code}/profile``: a
  compact 标的概览 combining the search-hit metadata (name / market /
  行业) with a brief 估值 / 盈利 snapshot derived from
  :class:`FundamentalService.snapshot`.

The resources are **pure** with respect to the services they receive:
callers (e.g. :mod:`china_stock_mcp.server`) construct the service
instances and pass them in via the ``services`` keyword. This keeps
the resource functions trivially testable without any FastMCP /
akshare side effects, mirroring the convention used by the prompts
package.
"""

from __future__ import annotations

from china_stock_mcp.resources.market_overview import market_overview_resource
from china_stock_mcp.resources.north_flow import north_flow_resource
from china_stock_mcp.resources.symbol_profile import symbol_profile_resource

__all__ = [
    "market_overview_resource",
    "north_flow_resource",
    "symbol_profile_resource",
]
