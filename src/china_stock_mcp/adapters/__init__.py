"""Data source adapter package for the China Stock MCP server.

This package contains the :class:`BaseAdapter` abstraction together with
its concrete implementations (``akshare`` / ``tushare`` / ``efinance``).
The abstraction lets the *Service* layer remain agnostic of any single
upstream data provider so primary / fallback wiring (see
``fetch_with_fallback``) can substitute sources transparently.

See ``design.md`` Component 4 (Adapter Layer) and Requirement 13.1.
"""

from __future__ import annotations

from china_stock_mcp.adapters.base import BaseAdapter
from china_stock_mcp.adapters.fallback import fetch_with_fallback

__all__ = ["BaseAdapter", "fetch_with_fallback"]
