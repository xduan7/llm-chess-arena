"""Integration tests that exercise live LLM-powered games."""

import os

import chess
import pytest

from llm_chess_arena.player.llm import (
    LLMPlayer,
    LLMConnector,
    GameArenaLLMMoveHandler,
)


pytestmark = pytest.mark.live


def get_first_available_llm_model() -> str:
    """Return the first LLM model with credentials configured or skip the test."""
    if os.getenv("OPENAI_API_KEY"):
        return "gpt-4o-mini"
    elif os.getenv("ANTHROPIC_API_KEY"):
        return "claude-haiku-4-5"
    elif os.getenv("GOOGLE_API_KEY"):
        return "gemini/gemini-2.5-flash"
    else:
        pytest.skip("No API keys available")


@pytest.mark.parametrize(
    "llm_model_name,required_env_var",
    [
        ("gpt-4o-mini", "OPENAI_API_KEY"),
        ("claude-haiku-4-5", "ANTHROPIC_API_KEY"),
        ("gemini/gemini-2.5-flash", "GOOGLE_API_KEY"),
    ],
)
def test_llm_player_generates_legal_opening_move_from_starting_position(
    llm_model_name, required_env_var
):
    """Ensure deterministic connectors return legal opening moves."""
    if not os.getenv(required_env_var):
        pytest.skip(f"{required_env_var} not set")

    deterministic_llm_connector = LLMConnector(
        model=llm_model_name,
        temperature=0.0,
        max_num_tokens=1000,
        request_timeout_in_seconds=10.0,
        max_api_request_retries=3,
    )
    game_arena_handler = GameArenaLLMMoveHandler()
    white_llm_player = LLMPlayer(
        connector=deterministic_llm_connector,
        handler=game_arena_handler,
        color="white",
        max_move_retries=3,
        num_votes=1,
    )

    starting_position_board = chess.Board()
    player_decision = white_llm_player(starting_position_board)

    assert player_decision.action == "move"
    generated_chess_move = chess.Move.from_uci(player_decision.attempted_move)
    assert generated_chess_move in starting_position_board.legal_moves
    assert generated_chess_move.from_square in range(64)
    assert generated_chess_move.to_square in range(64)
