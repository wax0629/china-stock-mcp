"""``market://overview`` resource -- 市场总览 Markdown.

Implements the ``market_overview`` MCP resource referenced by
*design.md Component 1* and task 21.1 (Requirement 9.1).

The resource is a thin alias of the ``get_market_overview`` tool: it
invokes :meth:`MarketService.overview` and renders the same Markdown
document via :func:`tools.market_overview.get_market_overview` so an
AI client subscribing to ``market://overview`` sees byte-identical
content to a client that called the tool. Re-using the tool keeps the
two surfaces in lockstep -- any future change to the rendering (new
section, banner, etc.) flows to both entry points without
duplication.

The function is pure with respect to its services bundle: it accepts
a ``services`` dict produced by the caller (typically
:mod:`china_stock_mcp.server`) and never reaches for module-level
state. This mirrors the convention used by the prompts package and
keeps the resource trivially testable.
"""

from __future__ import annotations

from typing import TypedDict

from china_stock_mcp.services.market_service import MarketService
from china_stock_mcp.tools.market_overview import (
    get_market_overview as _render_market_overview,
)


class _Services(TypedDict):
    """Typed bundle of the service instances this resource needs."""

    market: MarketService


def market_overview_resource(*, services: _Services) -> str:
    """Return the 市场总览 Markdown document.

    Parameters
    ----------
    services:
        Bundle that must contain a pre-wired :class:`MarketService`
        under the ``"market"`` key.

    Returns
    -------
    str
        Markdown document covering 指数 / 涨跌家数 / 涨跌停 / 北向 /
        行业热度 / heat_score, ending with the standard disclaimer
        (Requirement 12.1, Property 14).

    Raises
    ------
    ChinaStockMCPError
        Any subclass raised by :class:`MarketService` is propagated
        verbatim (Requirement 13.1). The caller (the FastMCP resource
        handler in ``server.py``) translates it into a user-facing
        message via :meth:`ChinaStockMCPError.to_user_message`.
    """

    return _render_market_overview(services["market"])


__all__ = ["market_overview_resource"]
