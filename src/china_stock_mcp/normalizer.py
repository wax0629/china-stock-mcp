"""Symbol normalization for the China Stock MCP server.

This module implements design ``Algorithm 2: normalize_symbol`` and the
companion :func:`detect_market` helper. Together they map heterogeneous
caller input (raw 6-digit codes, already-standardized codes, 5-digit HK
codes, Chinese names, pinyin, etc.) into the canonical
*Standardized_Symbol* form documented in ``requirements.md`` §Glossary
and validated by :class:`china_stock_mcp.models.SymbolHit`.

Coverage of acceptance criteria
-------------------------------

- 1.2  6-digit A 股 codes → ``.SH`` / ``.SZ`` / ``.BJ`` by prefix.
- 1.3  5-digit HK codes  → ``.HK``.
- 1.4  Already-standardized inputs are returned unchanged (idempotent,
       Property 1).
- 1.5  Every successful return matches one of the standardized shapes
       (Property 2).
- 1.6  Unrecognized input raises :class:`SymbolError` carrying up to
       three candidate suggestions and an optional ``market`` hint.

The Chinese name / pinyin path is delegated to a pluggable
:class:`SymbolIndex` so that this module stays free of I/O. A no-op
default index ships with the module; the symbol service registers a
real index at startup via :func:`set_symbol_index`.
"""

from __future__ import annotations

import re
from typing import Final, Protocol

from china_stock_mcp.exceptions import SymbolError
from china_stock_mcp.models import Market, SymbolHit

# ---------------------------------------------------------------------------
# Regular expressions
# ---------------------------------------------------------------------------

# Already-standardized A 股 (.SH / .SZ / .BJ) or HK (.HK).
_STANDARDIZED_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:\d{6}\.(?:SH|SZ|BJ)|\d{5}\.HK)$"
)

# Bare 6-digit code (A 股 candidates) and 5-digit code (HK).
_SIX_DIGITS_RE: Final[re.Pattern[str]] = re.compile(r"^\d{6}$")
_FIVE_DIGITS_RE: Final[re.Pattern[str]] = re.compile(r"^\d{5}$")

# Suffix rules from design §Data Models / Algorithm 2.
_SH_PREFIXES: Final[frozenset[str]] = frozenset({"60", "68", "90"})
_SZ_PREFIXES: Final[frozenset[str]] = frozenset({"00", "30", "20"})
_BJ_FIRST_CHAR: Final[str] = "8"

# Maximum number of candidate suggestions surfaced through SymbolError;
# matches Requirements 1.6 / 13.6 and the cap enforced by SymbolError.
_MAX_CANDIDATES: Final[int] = 3


# ---------------------------------------------------------------------------
# Symbol index abstraction
# ---------------------------------------------------------------------------


class SymbolIndex(Protocol):
    """Lookup table for Chinese names / pinyin / aliases.

    The protocol is intentionally minimal so multiple back-ends (in-memory
    dict, sqlite, akshare-derived snapshot) can satisfy it.
    """

    def lookup(self, query: str) -> SymbolHit | None:
        """Return an exact match for ``query`` or ``None``.

        ``query`` arrives already trimmed and upper-cased; implementations
        should be case-insensitive for ASCII text and pass Chinese input
        through unchanged.
        """

    def suggest(self, query: str, limit: int = _MAX_CANDIDATES) -> list[str]:
        """Return up to ``limit`` candidate symbol strings or names.

        The result is surfaced to callers via :class:`SymbolError` and
        therefore should be short, human-readable strings (e.g.
        ``"宁德时代(300750.SZ)"``). An empty list is acceptable.
        """


class _EmptySymbolIndex:
    """Fallback :class:`SymbolIndex` that never resolves anything."""

    def lookup(self, query: str) -> SymbolHit | None:
        return None

    def suggest(
        self, query: str, limit: int = _MAX_CANDIDATES
    ) -> list[str]:
        return []


_default_index: SymbolIndex = _EmptySymbolIndex()


def set_symbol_index(index: SymbolIndex) -> None:
    """Register the process-wide :class:`SymbolIndex`.

    Called once during server startup by ``SymbolService``; tests may
    invoke it to swap in a stub. Pass an :class:`_EmptySymbolIndex`
    instance to reset.
    """

    global _default_index
    _default_index = index


def get_symbol_index() -> SymbolIndex:
    """Return the currently registered :class:`SymbolIndex`."""

    return _default_index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_symbol(
    raw: str,
    *,
    index: SymbolIndex | None = None,
    market: str | None = None,
) -> str:
    """Convert ``raw`` into a Standardized_Symbol string.

    Implements design Algorithm 2 step-for-step:

    1. Trim + upper-case.
    2. Already-standardized A 股 / HK code → return as-is.
    3. 6-digit numeric input → append ``.SH`` / ``.SZ`` / ``.BJ`` per
       prefix rules.
    4. 5-digit numeric input → append ``.HK``.
    5. Otherwise consult ``index`` (or the registered default) for a
       Chinese-name / pinyin / alias hit.
    6. Raise :class:`SymbolError` listing up to three candidates and the
       active ``market`` hint when nothing matches.

    Parameters
    ----------
    raw:
        Caller-supplied input; must be a non-empty string.
    index:
        Optional override of the process-wide :class:`SymbolIndex`. When
        ``None``, the index registered via :func:`set_symbol_index` is
        used.
    market:
        Optional ``market`` filter (``"a_stock"`` / ``"hk_stock"`` /
        ``"fund"`` / ``"all"``) propagated into the resulting
        :class:`SymbolError` so callers can recover by widening or
        narrowing the search.

    Returns
    -------
    str
        A Standardized_Symbol, guaranteed to satisfy
        ``^\\d{6}\\.(SH|SZ|BJ)$``, ``^\\d{5}\\.HK$`` or ``^\\d{6}$``
        (Property 2).

    Raises
    ------
    SymbolError
        When the input cannot be classified by any rule and the index
        does not resolve it.
    """

    if raw is None or not isinstance(raw, str) or not raw.strip():
        raise SymbolError("代码不能为空", market=market)

    s = raw.strip().upper()

    # 1) Already-standardized — return verbatim (idempotency, Property 1).
    if _STANDARDIZED_RE.fullmatch(s):
        return s

    # 2) Bare 6-digit A 股 codes.
    if _SIX_DIGITS_RE.fullmatch(s):
        suffix = _suffix_for_six_digits(s)
        if suffix is not None:
            return f"{s}.{suffix}"
        # Otherwise fall through to index lookup; some 6-digit codes
        # belong to funds / B-shares / unknown universes and require
        # an index hit. Keep the literal `s` so the index sees the
        # original numeric form.

    # 3) Bare 5-digit HK codes.
    elif _FIVE_DIGITS_RE.fullmatch(s):
        return f"{s}.HK"

    # 4) Chinese name / pinyin / alias lookup.
    idx = index if index is not None else _default_index
    hit = idx.lookup(s)
    if hit is not None:
        return hit.code

    # 5) Give up — surface up to 3 candidates so the AI client can retry.
    candidates = idx.suggest(s, limit=_MAX_CANDIDATES)
    raise SymbolError(
        f"无法识别的代码: {raw!r}",
        candidates=candidates or None,
        market=market,
    )


def detect_market(symbol: str) -> Market:
    """Return the :data:`Market` of a Standardized_Symbol.

    The mapping is unambiguous because :func:`normalize_symbol` only
    produces three shapes:

    - ``\\d{6}\\.(SH|SZ|BJ)`` → ``"a_stock"``
    - ``\\d{5}\\.HK``         → ``"hk_stock"``
    - ``\\d{6}``              → ``"fund"`` (no exchange suffix)

    Parameters
    ----------
    symbol:
        A Standardized_Symbol produced by :func:`normalize_symbol`. Raw
        Chinese names or unsupported shapes are rejected.

    Raises
    ------
    SymbolError
        When ``symbol`` is not a Standardized_Symbol.
    """

    if symbol is None or not isinstance(symbol, str) or not symbol.strip():
        raise SymbolError("代码不能为空")

    s = symbol.strip().upper()

    if _STANDARDIZED_RE.fullmatch(s):
        return "hk_stock" if s.endswith(".HK") else "a_stock"
    if _SIX_DIGITS_RE.fullmatch(s):
        return "fund"

    raise SymbolError(f"无法判断市场归属: {symbol!r}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _suffix_for_six_digits(code: str) -> str | None:
    """Return the exchange suffix for a 6-digit A 股 code, or ``None``.

    Pure helper extracted so the prefix table stays in one place; called
    only after ``code`` has been verified to match ``\\d{6}``.
    """

    prefix2 = code[:2]
    if prefix2 in _SH_PREFIXES:
        return "SH"
    if prefix2 in _SZ_PREFIXES:
        return "SZ"
    if code[0] == _BJ_FIRST_CHAR:
        return "BJ"
    return None


__all__ = [
    "SymbolIndex",
    "detect_market",
    "get_symbol_index",
    "normalize_symbol",
    "set_symbol_index",
]
