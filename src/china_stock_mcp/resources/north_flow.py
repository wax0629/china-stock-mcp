"""``flow://north`` resource -- 北向资金 Markdown.

Implements the ``north_flow`` MCP resource referenced by *design.md
Component 1* and task 21.1 (Requirement 5.1).

The resource is a thin alias of ``get_money_flow(flow_type="north")``:
it invokes :meth:`MoneyFlowService.get` with the canonical
``(symbol=None, flow_type="north", top_n=20)`` arguments and renders
the same Markdown document via :func:`tools.money_flow.get_money_flow`
so an AI client subscribing to ``flow://north`` sees byte-identical
content to a client that called the tool. Re-using the tool keeps the
two surfaces in lockstep.

The function is pure with respect to its services bundle: it accepts
a ``services`` dict produced by the caller (typically
:mod:`china_stock_mcp.server`) and never reaches for module-level
state.
"""

from __future__ import annotations

from typing import Final, TypedDict

from china_stock_mcp.services.money_flow_service import MoneyFlowService
from china_stock_mcp.tools.money_flow import get_money_flow as _render_money_flow

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default ``top_n`` for the northbound flow resource. Pinned to 20 so
#: the resource fits within the tool layer's per-call token budget
#: (Property 13) regardless of the underlying upstream's row count.
_DEFAULT_TOP_N: Final[int] = 20


class _Services(TypedDict):
    """Typed bundle of the service instances this resource needs."""

    money_flow: MoneyFlowService


def north_flow_resource(*, services: _Services) -> str:
    """Return the 北向资金 Markdown document.

    Parameters
    ----------
    services:
        Bundle that must contain a pre-wired :class:`MoneyFlowService`
        under the ``"money_flow"`` key.

    Returns
    -------
    str
        Markdown document covering recent 北向资金 净流入 / 买入 /
        卖出 / 持股市值 rows, ending with the standard disclaimer
        (Requirement 12.1, Property 14).

    Raises
    ------
    ChinaStockMCPError
        Any subclass raised by :class:`MoneyFlowService` is propagated
        verbatim (Requirement 13.1).
    """

    return _render_money_flow(
        services["money_flow"],
        symbol=None,
        flow_type="north",
        top_n=_DEFAULT_TOP_N,
    )


__all__ = ["north_flow_resource"]
