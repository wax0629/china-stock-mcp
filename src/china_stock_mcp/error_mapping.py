"""Shared mapping from Pydantic ``ValidationError`` to the unified
:class:`china_stock_mcp.exceptions.ValidationError`.

Implements the protocol-layer bridge mandated by Requirement 13.7:
when Pydantic input validation fails, the tool layer must surface a
single :class:`ValidationError` whose message names the offending
field, the expected constraint and the actual value -- never a raw
Python traceback or the multi-line Pydantic default rendering.

Two entry points are exposed:

* :func:`format_pydantic_error` -- produces a flat, AI-friendly
  ``"loc: msg (实际值=...)"`` string. Joined with ``"; "`` when the
  Pydantic error contains multiple violations.
* :func:`bridge_pydantic_error` -- wraps :func:`format_pydantic_error`
  and returns a ready-to-raise :class:`ValidationError` instance with
  ``__cause__`` chaining preserved by the caller (use ``raise ... from
  exc``).

This module previously lived as a duplicated ``_format_pydantic_error``
helper inside every ``tools/*.py`` and ``prompts/*.py`` file. Centralising
it removes drift between the helpers and lets the property test for
Requirement 13.2 ("error messages contain no traceback") cover one
implementation rather than thirteen.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError as PydanticValidationError

from china_stock_mcp.exceptions import ValidationError

#: Fallback message returned when the Pydantic error somehow carries no
#: ``errors()`` rows. In practice :class:`PydanticValidationError`
#: always reports at least one entry, but the constant keeps the
#: function total.
_FALLBACK_MESSAGE = "参数校验失败"


def format_pydantic_error(exc: PydanticValidationError) -> str:
    """Render a :class:`PydanticValidationError` as a flat string.

    Each underlying error is rendered as ``"<loc>: <msg> (实际值=<value>)"``
    where:

    * ``<loc>`` -- dotted field path (e.g. ``"symbol.0"``). Empty when
      the violation is at the model root, in which case the ``loc:``
      prefix is omitted.
    * ``<msg>`` -- the validator message produced by Pydantic
      (defaulted to ``"无效输入"`` when missing).
    * ``<value>`` -- the offending input as ``repr`` so ``None`` /
      empty strings / numerics are unambiguous.

    Multiple violations are joined with ``"; "`` so the output stays a
    single line and never contains a Python traceback. This matches
    Requirement 13.7's "字段名 + 期望约束 + 实际取值" guidance.
    """

    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(item) for item in err.get("loc", ()))
        msg = err.get("msg", "无效输入")
        value: Any = err.get("input", None)
        if loc:
            parts.append(f"{loc}: {msg} (实际值={value!r})")
        else:
            parts.append(f"{msg} (实际值={value!r})")
    return "; ".join(parts) if parts else _FALLBACK_MESSAGE


def bridge_pydantic_error(exc: PydanticValidationError) -> ValidationError:
    """Convert a Pydantic ``ValidationError`` into the unified type.

    Callers should chain the original exception via ``raise ... from
    exc`` so the original Pydantic error is preserved on
    ``__cause__`` for log inspection while the user-facing message
    stays AI-friendly (Requirement 13.7 / 13.8)::

        try:
            validated = MyInput(**payload)
        except PydanticValidationError as exc:
            raise bridge_pydantic_error(exc) from exc
    """

    return ValidationError(format_pydantic_error(exc))


__all__ = ["bridge_pydantic_error", "format_pydantic_error"]
