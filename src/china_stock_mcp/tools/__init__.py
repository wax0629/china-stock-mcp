"""Tools layer for china-stock-mcp.

This package implements *Component 2 (Tools Layer)* from ``design.md``.
Each module exposes a single MCP tool function plus its pydantic input
model. Tools are deliberately thin: they validate input, delegate to
the service layer for orchestration, and call into ``formatters`` for
Markdown rendering.

Tools never:

- Touch a third-party data SDK directly (use an :mod:`adapters` class).
- Read or write the cache (the service layer owns caching policy).
- Perform rate-limit accounting (the service layer holds the limiter).

Modules in this package register themselves with FastMCP via
``server.py``; importing this package does not register any tool.
"""

from __future__ import annotations
