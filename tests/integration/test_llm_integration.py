import os
import pytest
import chess

from llm_chess_arena.player.llm import (
    LLMPlayer,
    LLMConnector,
    GameArenaLLMMoveHandler,
)


pytestmark = pytest.mark.live


def get_first_available_llm_model():
    if os.getenv("OPENAI_API_KEY"):
        return "gpt-3.5-turbo"
    elif os.getenv("ANTHROPIC_API_KEY"):
        return "claude-3-haiku-20240307"
    elif os.getenv("GOOGLE_API_KEY"):
        return "gemini/gemini-2.0-flash-exp"
    else:
        pytest.skip("No API keys available")


@pytest.mark.parametrize(
    "llm_model_name,required_env_var",
    [
        ("gpt-3.5-turbo", "OPENAI_API_KEY"),
        ("claude-3-haiku-20240307", "ANTHROPIC_API_KEY"),
        ("gemini/gemini-2.0-flash-exp", "GOOGLE_API_KEY"),
    ],
)
def test_llm_player_generates_legal_opening_move_from_starting_position(
    llm_model_name, required_env_var
):
    if not os.getenv(required_env_var):
        pytest.skip(f"{required_env_var} not set")

    deterministic_llm_connector = LLMConnector(
        model=llm_model_name,
        temperature=0.0,
        max_tokens=1000,
        timeout=10.0,
    )
    game_arena_handler = GameArenaLLMMoveHandler()
    white_llm_player = LLMPlayer(
        connector=deterministic_llm_connector, handler=game_arena_handler, color="white"
    )

    starting_position_board = chess.Board()
    player_decision = white_llm_player(starting_position_board)

    assert player_decision.action == "move"
    generated_chess_move = chess.Move.from_uci(player_decision.attempted_move)
    assert generated_chess_move in starting_position_board.legal_moves
    assert generated_chess_move.from_square in range(64)
    assert generated_chess_move.to_square in range(64)


def test_llm_retry_mechanism_recovers_from_illegal_move_attempts():
    available_llm_model = get_first_available_llm_model()

    deterministic_llm_connector = LLMConnector(
        model=available_llm_model,
        temperature=0.0,
        max_tokens=1000,
        timeout=10.0,
    )

    game_arena_handler = GameArenaLLMMoveHandler()
    black_llm_player_with_retries = LLMPlayer(
        connector=deterministic_llm_connector,
        handler=game_arena_handler,
        color="black",
        max_move_retries=2,
    )

    # FEN: position after 1.e4 e5
    board_after_e4_e5 = chess.Board(
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"
    )
    player_decision = black_llm_player_with_retries(board_after_e4_e5)

    assert player_decision.action == "move"
    recovered_legal_move = chess.Move.from_uci(player_decision.attempted_move)
    assert recovered_legal_move in board_after_e4_e5.legal_moves


@pytest.mark.slow
def test_llm_plays_coherent_opening_sequence_over_five_moves():
    available_llm_model = get_first_available_llm_model()

    slightly_creative_llm_connector = LLMConnector(
        model=available_llm_model,
        temperature=0.3,
        max_tokens=1000,
        timeout=10.0,
    )
    game_arena_handler = GameArenaLLMMoveHandler()
    white_llm_player = LLMPlayer(
        connector=slightly_creative_llm_connector,
        handler=game_arena_handler,
        color="white",
    )

    evolving_game_board = chess.Board()
    white_moves_in_san_notation = []

    for move_number in range(5):
        white_player_decision = white_llm_player(evolving_game_board)
        assert white_player_decision.action == "move"

        white_move = chess.Move.from_uci(white_player_decision.attempted_move)
        assert white_move in evolving_game_board.legal_moves

        white_moves_in_san_notation.append(evolving_game_board.san(white_move))
        evolving_game_board.push(white_move)

        if not evolving_game_board.is_game_over():
            available_black_moves = list(evolving_game_board.legal_moves)
            first_available_black_move = available_black_moves[0]
            evolving_game_board.push(first_available_black_move)

    assert len(white_moves_in_san_notation) == 5

    common_opening_moves = ["e4", "d4", "Nf3", "c4"]
    first_move_is_common_opening = (
        white_moves_in_san_notation[0] in common_opening_moves
    )
    assert first_move_is_common_opening
