"""Prompts layer for china-stock-mcp.

This package implements *Component 1 (FastMCP Server)* prompts from
``design.md``. Prompts orchestrate one or more service-layer calls
into a single Markdown research artifact, returning that artifact as
the prompt body so the AI client receives a ready-to-consume
investment-research deliverable.

Currently implemented:

- :func:`research_report` (task 20.1, design Algorithm 6,
  Requirement 10.1 / 10.2 / 10.5 / 10.6) -- 投研报告.
- :func:`valuation_compare` (task 20.2, Requirement 10.3) -- 估值对比.

The remaining prompt (``weekly_review``) lives in its own module per
task 20.3 and is *not* re-exported here yet to keep imports light.
"""

from __future__ import annotations

from china_stock_mcp.prompts.research_report import (
    ResearchReportInput,
    research_report,
)
from china_stock_mcp.prompts.valuation_compare import (
    ValuationCompareInput,
    valuation_compare,
)

__all__ = [
    "ResearchReportInput",
    "ValuationCompareInput",
    "research_report",
    "valuation_compare",
]
