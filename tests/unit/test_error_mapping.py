"""Unit tests for :mod:`china_stock_mcp.error_mapping`.

Covers Requirement 13.7: Pydantic ``ValidationError`` is bridged into
the unified :class:`china_stock_mcp.exceptions.ValidationError` with a
message that names (a) the offending field, (b) the expected
constraint, and (c) the actual value the caller supplied. The tests
exercise both helpers exposed by the module --
:func:`format_pydantic_error` (string output) and
:func:`bridge_pydantic_error` (ready-to-raise domain error).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from china_stock_mcp.error_mapping import (
    bridge_pydantic_error,
    format_pydantic_error,
)
from china_stock_mcp.exceptions import ValidationError

# ---------------------------------------------------------------------------
# Test fixtures: a pydantic model that exercises several constraint kinds
# ---------------------------------------------------------------------------


class _SampleInput(BaseModel):
    """Pydantic model used to provoke a variety of validator failures."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    age: int = Field(..., ge=0, le=120)
    tags: list[str] = Field(default_factory=list, max_length=3)


def _capture_pydantic_error(**kwargs: object) -> PydanticValidationError:
    """Construct ``_SampleInput`` and return the pydantic error raised."""

    with pytest.raises(PydanticValidationError) as exc_info:
        _SampleInput(**kwargs)  # type: ignore[arg-type]
    return exc_info.value


# ---------------------------------------------------------------------------
# format_pydantic_error
# ---------------------------------------------------------------------------


def test_format_includes_field_constraint_and_value() -> None:
    """Single-field violation: loc + validator msg + actual value."""

    err = _capture_pydantic_error(name="alice", age=200, tags=[])

    rendered = format_pydantic_error(err)

    assert "age" in rendered
    # Pydantic's "less than or equal to 120" message; we don't pin
    # the exact wording, just that some constraint hint surfaces.
    assert ("less than or equal" in rendered) or ("<=" in rendered)
    assert "200" in rendered
    assert "\n" not in rendered


def test_format_renders_string_value_with_repr() -> None:
    """The actual value goes through ``repr`` so empty strings show up."""

    err = _capture_pydantic_error(name="", age=30, tags=[])

    rendered = format_pydantic_error(err)

    assert "name" in rendered
    assert "''" in rendered  # repr('') -> "''"


def test_format_joins_multiple_errors_with_semicolon() -> None:
    """Multi-error pydantic failures collapse to a single line."""

    err = _capture_pydantic_error(name="", age=-5, tags=[])

    rendered = format_pydantic_error(err)

    assert "; " in rendered
    assert "name" in rendered
    assert "age" in rendered
    # Single-line output -- no embedded newlines anywhere.
    assert "\n" not in rendered


def test_format_handles_extra_fields() -> None:
    """``extra='forbid'`` violations land at the model root (empty loc)."""

    err = _capture_pydantic_error(
        name="alice", age=30, tags=[], unexpected="boom"
    )

    rendered = format_pydantic_error(err)

    assert "unexpected" in rendered


def test_format_handles_nested_path() -> None:
    """Nested locs (``tags.4``) render with dot separators."""

    err = _capture_pydantic_error(
        name="alice", age=30, tags=["a", "b", "c", "d"]
    )

    rendered = format_pydantic_error(err)

    assert "tags" in rendered


# ---------------------------------------------------------------------------
# bridge_pydantic_error
# ---------------------------------------------------------------------------


def test_bridge_returns_unified_validation_error() -> None:
    """Bridge wraps the formatted message into the unified type."""

    err = _capture_pydantic_error(name="", age=200, tags=[])

    bridged = bridge_pydantic_error(err)

    assert isinstance(bridged, ValidationError)


def test_bridge_user_message_carries_field_constraint_value() -> None:
    """``to_user_message`` surfaces field, constraint and actual value."""

    err = _capture_pydantic_error(name="alice", age=200, tags=[])

    bridged = bridge_pydantic_error(err)
    user_msg = bridged.to_user_message()

    assert user_msg.startswith("参数校验失败:")
    assert "age" in user_msg
    assert "200" in user_msg


def test_bridge_user_message_has_no_traceback() -> None:
    """The bridged message must not leak Python traceback markers."""

    try:
        _SampleInput(name="alice", age=200, tags=[])  # type: ignore[arg-type]
    except PydanticValidationError as exc:
        bridged = bridge_pydantic_error(exc)

    user_msg = bridged.to_user_message()
    for token in ('Traceback', 'File "', ", line "):
        assert token not in user_msg


def test_bridge_can_be_raised_from_pydantic() -> None:
    """End-to-end: ``raise bridge_pydantic_error(exc) from exc`` works."""

    with pytest.raises(ValidationError) as exc_info:
        try:
            _SampleInput(name="", age=200, tags=[])  # type: ignore[arg-type]
        except PydanticValidationError as exc:
            raise bridge_pydantic_error(exc) from exc

    domain_err = exc_info.value
    assert isinstance(domain_err.__cause__, PydanticValidationError)
    user_msg = domain_err.to_user_message()
    assert "name" in user_msg
    assert "age" in user_msg
