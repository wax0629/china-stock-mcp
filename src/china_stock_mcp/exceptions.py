"""Unified exception tree for china-stock-mcp.

All errors raised by the server inherit from
:class:`ChinaStockMCPError` so callers can catch every domain failure
with a single ``except`` clause (Requirements 13.1). Each subclass
provides a :meth:`ChinaStockMCPError.to_user_message` implementation
that returns a plain, AI-friendly string without any Python traceback
or stack-frame information (Requirements 13.2 / Property 17).

Hierarchy::

    ChinaStockMCPError
    ├── SymbolError          -- 代码无法识别(含候选项 / market 提示)
    ├── DataSourceError      -- 第三方数据源相关
    │   ├── NetworkError     -- 连接 / DNS / 5xx
    │   └── RateLimitError   -- 限流 / 429
    ├── DataNotFoundError    -- 标的存在但目标数据为空
    └── ValidationError      -- 入参 / 参数组合非法

The hierarchy follows the design document's "Error Handling" section
and is referenced by ``fetch_with_fallback`` (only ``NetworkError`` and
``RateLimitError`` trigger fallback; ``DataNotFoundError`` does not).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Final

# Maximum number of candidate suggestions surfaced to the caller from
# :class:`SymbolError`. Bound by Requirements 13.6 ("最多 3 个候选项").
_MAX_SYMBOL_CANDIDATES: Final[int] = 3


class ChinaStockMCPError(Exception):
    """Base class for every error raised by this package.

    Parameters
    ----------
    message:
        Short human-readable summary of what went wrong. Should not
        contain stack-frame information or Python ``repr`` output.
    hint:
        Optional remediation hint appended on a new line when
        :meth:`to_user_message` is called.
    """

    #: Localized prefix used by :meth:`to_user_message` to label the
    #: error category. Subclasses override this.
    default_prefix: ClassVar[str] = "错误"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def to_user_message(self) -> str:
        """Render an AI-friendly message without any traceback.

        The returned string is safe to forward verbatim to MCP clients;
        it never embeds Python frames, file paths, or chained
        exceptions (Requirements 13.2, Property 17).
        """

        lines: list[str] = [f"{self.default_prefix}: {self.message}"]
        if self.hint:
            lines.append(self.hint)
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class SymbolError(ChinaStockMCPError):
    """Raised when a symbol cannot be normalized or resolved.

    Carries up to three candidate suggestions and an optional ``market``
    hint so callers (typically the AI client) can recover by retrying
    with a different query or market scope (Requirements 13.6).

    Parameters
    ----------
    message:
        Description of the failure (e.g. ``"无法识别的代码: '苹果'"``).
    candidates:
        Iterable of suggested symbols or names. Truncated to the first
        three entries; ``None`` and empty iterables are equivalent.
    market:
        Optional ``market`` hint that was active when normalization
        failed (``"a_stock"`` / ``"hk_stock"`` / ``"fund"`` / ``"all"``).
        Surfaced in :meth:`to_user_message` to help callers widen or
        narrow the search.
    hint:
        Free-form remediation hint.
    """

    default_prefix: ClassVar[str] = "标的代码错误"

    def __init__(
        self,
        message: str,
        *,
        candidates: Sequence[str] | None = None,
        market: str | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message, hint=hint)
        # Defensive copy + truncation so callers cannot smuggle more
        # than three candidates past the 13.6 cap.
        if candidates:
            self.candidates: tuple[str, ...] = tuple(candidates)[
                :_MAX_SYMBOL_CANDIDATES
            ]
        else:
            self.candidates = ()
        self.market = market

    def to_user_message(self) -> str:
        lines: list[str] = [f"{self.default_prefix}: {self.message}"]
        if self.candidates:
            joined = "、".join(self.candidates)
            lines.append(f"候选项(最多 3 个): {joined}")
        if self.market and self.market != "all":
            lines.append(f"当前 market 限制: {self.market}")
        if self.hint:
            lines.append(self.hint)
        return "\n".join(lines)


class DataSourceError(ChinaStockMCPError):
    """Generic failure originating from a third-party data adapter.

    Parent class for :class:`NetworkError` and :class:`RateLimitError`;
    direct instances cover residual adapter failures that do not fit
    either subclass (e.g. malformed upstream payload).
    """

    default_prefix: ClassVar[str] = "数据源错误"


class NetworkError(DataSourceError):
    """Connection, DNS, timeout or upstream 5xx failure.

    ``fetch_with_fallback`` switches to the backup adapter when this
    error is raised by the primary source (Requirements 13.3).
    """

    default_prefix: ClassVar[str] = "网络错误"


class RateLimitError(DataSourceError):
    """Local Token Bucket rejection or upstream 429 response.

    Like :class:`NetworkError`, this triggers ``fetch_with_fallback``
    to retry against the backup adapter (Requirements 13.3, 11.7).
    """

    default_prefix: ClassVar[str] = "调用频率受限"


class DataNotFoundError(ChinaStockMCPError):
    """The symbol exists but the requested data is unavailable.

    Examples include a freshly listed company without enough annual
    reports, or a money-flow query that returns an empty set. This
    error is **never** masked by ``fetch_with_fallback`` (Requirements
    13.4, Property 5).
    """

    default_prefix: ClassVar[str] = "数据不存在"


class ValidationError(ChinaStockMCPError):
    """Input validation failure (invalid combination, range, or type).

    Pydantic's own ``ValidationError`` is converted into this type at
    the protocol boundary so callers always see the unified hierarchy
    (Requirements 13.7).
    """

    default_prefix: ClassVar[str] = "参数校验失败"


__all__ = [
    "ChinaStockMCPError",
    "DataNotFoundError",
    "DataSourceError",
    "NetworkError",
    "RateLimitError",
    "SymbolError",
    "ValidationError",
]
