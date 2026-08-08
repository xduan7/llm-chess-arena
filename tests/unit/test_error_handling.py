"""Tests for shared error handling decorators."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from llm_chess_arena.policies import (
    config_operation,
    metrics_operation,
    move_validation,
)
from llm_chess_arena.exceptions import InvalidMoveError, MoveError


def test_move_validation_policy_preserves_move_errors() -> None:
    """MoveError subclasses should propagate unchanged."""

    @move_validation
    def validator() -> None:
        """Raise InvalidMoveError to ensure policy leaves it untouched."""
        raise InvalidMoveError("bad move")

    with pytest.raises(InvalidMoveError, match="bad move"):
        validator()


def test_move_validation_policy_converts_unexpected_errors() -> None:
    """Unexpected exceptions should convert into MoveError."""

    @move_validation
    def validator() -> None:
        """Raise RuntimeError to confirm conversion into MoveError."""
        raise RuntimeError("boom")

    with pytest.raises(MoveError, match="Move validation failed"):
        validator()


def test_config_policy_wraps_errors() -> None:
    """Configuration policy should wrap errors as ValueError."""

    @config_operation
    def builder() -> None:
        """Raise a generic runtime error to test config wrapping."""
        raise RuntimeError("bad config")

    with pytest.raises(ValueError, match="Configuration failed"):
        builder()


def test_metrics_policy_logs_and_returns_none() -> None:
    """Metrics policy should log and return None on failure."""

    @metrics_operation
    def metrics() -> None:
        """Trigger an error so the metrics policy can swallow it."""
        raise RuntimeError("stockfish unavailable")

    mock_logger = Mock()
    with patch("llm_chess_arena.policies.logger", mock_logger):
        result = metrics()

    assert result is None
    mock_logger.warning.assert_called_once()
