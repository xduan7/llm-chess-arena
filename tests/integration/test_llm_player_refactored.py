"""Integration checks for the refactored LLMPlayer."""

from __future__ import annotations

from unittest.mock import Mock

from llm_chess_arena.player.llm import GameArenaLLMMoveHandler, LLMConnector, LLMPlayer
from llm_chess_arena.types import PlayerDecisionContext


def _build_context() -> PlayerDecisionContext:
    """Construct a minimal decision context for test scenarios."""
    return PlayerDecisionContext(
        board_in_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        player_color="white",
        legal_moves_in_uci=["e2e4", "d2d4", "g1f3"],
        move_history_in_uci=[],
    )


def test_refactored_player_single_vote() -> None:
    """Ensure single-vote decision returns the parsed move."""
    connector = Mock(spec=LLMConnector)
    connector.model = "test-model"
    connector.query.return_value = ["Final Answer: e4"]
    connector.get_last_usage.return_value = None
    connector.get_total_usage.return_value = Mock(
        prompt_tokens=5, completion_tokens=5, total_tokens=10, cost=0.001
    )

    player = LLMPlayer(
        player_color="white",
        connector=connector,
        handler=GameArenaLLMMoveHandler(),
        max_move_retries=1,
        num_votes=1,
    )

    decision = player._make_decision(_build_context())

    assert decision.action == "move"
    assert decision.attempted_move == "e2e4"
    assert player.last_move_attempts == 1


def test_refactored_player_voting() -> None:
    """Check that majority voting favors the most common response."""
    connector = Mock(spec=LLMConnector)
    connector.model = "test-model"
    connector.query.return_value = [
        "Final Answer: e4",
        "Final Answer: e4",
        "Final Answer: d4",
    ]
    connector.get_last_usage.return_value = None
    connector.get_total_usage.return_value = Mock(
        prompt_tokens=15, completion_tokens=15, total_tokens=30, cost=0.003
    )

    player = LLMPlayer(
        player_color="white",
        connector=connector,
        handler=GameArenaLLMMoveHandler(),
        max_move_retries=3,
        num_votes=3,
    )

    decision = player._make_decision(_build_context())

    assert decision.attempted_move == "e2e4"
    connector.query.assert_called_once()
    assert connector.query.call_args.kwargs["n"] == 3


def test_refactored_player_retry_flow() -> None:
    """Validate that retry flow recovers after an invalid first response."""
    connector = Mock(spec=LLMConnector)
    connector.model = "test-model"
    connector.query.side_effect = [
        ["Final Answer: Zz9"],
        ["Final Answer: e4"],
    ]
    connector.get_last_usage.return_value = None
    connector.get_total_usage.return_value = Mock(
        prompt_tokens=20, completion_tokens=10, total_tokens=30, cost=0.002
    )

    player = LLMPlayer(
        player_color="white",
        connector=connector,
        handler=GameArenaLLMMoveHandler(),
        max_move_retries=1,
        num_votes=1,
    )

    decision = player._make_decision(_build_context())

    assert decision.attempted_move == "e2e4"
    assert player.last_move_attempts == 2
    assert connector.query.call_count == 2
