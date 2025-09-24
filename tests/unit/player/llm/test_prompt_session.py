"""Tests for the prompt session helper."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from llm_chess_arena.player.llm.prompting import PromptSession
from llm_chess_arena.types import PlayerDecisionContext


@pytest.fixture()
def context() -> PlayerDecisionContext:
    return PlayerDecisionContext(
        board_in_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        player_color="white",
        legal_moves_in_uci=["e2e4", "d2d4"],
        move_history_in_uci=[],
    )


def test_initial_prompt_generated_once(context: PlayerDecisionContext) -> None:
    """Prompt session should cache the first prompt it generates."""
    handler = Mock()
    handler.get_prompt.return_value = "prompt1"

    session = PromptSession(handler, context)

    assert session.ensure_initial_prompt() == "prompt1"
    assert session.ensure_initial_prompt() == "prompt1"
    handler.get_prompt.assert_called_once()


def test_retry_prompt_uses_handler(context: PlayerDecisionContext) -> None:
    """Retry prompts must be delegated to the handler."""
    handler = Mock()
    handler.get_prompt.return_value = "prompt1"
    handler.get_retry_prompt.return_value = "retry"

    session = PromptSession(handler, context)
    session.ensure_initial_prompt()

    retry_prompt = session.build_retry_prompt(
        exception_name="InvalidMoveError",
        last_response="response",
        last_attempted_move="e4",
    )

    assert retry_prompt == "retry"
    handler.get_retry_prompt.assert_called_once()
