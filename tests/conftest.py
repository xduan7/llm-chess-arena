"""Shared test fixtures and utilities for the test suite."""

import warnings
from pathlib import Path

import chess
import pytest
from dotenv import load_dotenv

# Filter external library warnings that we can't fix
# These come from litellm, pydantic, and httpx internals
warnings.filterwarnings(
    "ignore", message="There is no current event loop", category=DeprecationWarning
)
warnings.filterwarnings(
    "ignore", message="Pydantic serializer warnings:.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="Use 'content=<...>' to upload raw bytes/text content",
    category=DeprecationWarning,
)

# Load environment variables from .env file for tests
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

from llm_chess_arena.game import Game  # noqa: E402
from llm_chess_arena.player.base_player import BasePlayer  # noqa: E402
from llm_chess_arena.player.random_player import RandomPlayer  # noqa: E402
from llm_chess_arena.player.llm.llm_move_handler import (  # noqa: E402
    BaseLLMMoveHandler,
    GameArenaLLMMoveHandler,
)
from llm_chess_arena.types import PlayerDecision  # noqa: E402
from llm_chess_arena.exceptions import (  # noqa: E402
    InvalidMoveError,
    IllegalMoveError,
    AmbiguousMoveError,
)


# ==================== BACKWARD COMPATIBILITY SHIM ====================
# This shim provides backward compatibility for tests that expect the old API
# It will be gradually removed as tests are migrated to the new DTO pattern


@pytest.fixture(autouse=True)
def backward_compatibility_shim(monkeypatch):
    """Auto-applied fixture to maintain backward compatibility for legacy tests."""

    # Save original methods
    original_parse_decision = GameArenaLLMMoveHandler.parse_decision_from_response
    original_get_prompt = GameArenaLLMMoveHandler.get_prompt

    # Wrapper for parse_decision_from_response to handle board parameter
    def parse_decision_wrapper(self, response, board=None, **kwargs):
        """Wrapper that accepts board parameter for backward compatibility."""
        # Store response for legacy tests
        self.last_response = response
        self.last_attempted_move_text = None

        # Call original to get PlayerDecision
        try:
            decision = original_parse_decision(self, response, **kwargs)
        except (InvalidMoveError, IllegalMoveError, AmbiguousMoveError) as e:
            # Convert custom exception to chess exception for legacy tests
            if board is not None:
                if isinstance(e, AmbiguousMoveError):
                    raise chess.AmbiguousMoveError(str(e))
                elif isinstance(e, IllegalMoveError):
                    raise chess.IllegalMoveError(str(e))
                else:
                    raise chess.InvalidMoveError(str(e))
            raise

        # Store attempted move for legacy tests
        if decision.attempted_move:
            self.last_attempted_move_text = decision.attempted_move

        # If board provided (legacy mode), validate and return chess.Move
        if board is not None:
            if decision.action != "move":
                raise chess.InvalidMoveError(f"No move in response: {response}")

            move_text = decision.attempted_move
            if not move_text:
                raise chess.InvalidMoveError(f"Failed to extract move from: {response}")

            # Try SAN first, then UCI
            try:
                move = board.parse_san(move_text)
                return move
            except chess.AmbiguousMoveError:
                # Re-raise ambiguous as-is
                raise
            except chess.IllegalMoveError:
                # Re-raise illegal as-is
                raise
            except chess.InvalidMoveError:
                # Try UCI fallback for invalid SAN
                try:
                    move = chess.Move.from_uci(move_text)
                    if move not in board.legal_moves:
                        raise chess.IllegalMoveError(f"Illegal move: {move_text}")
                    return move
                except (ValueError, chess.InvalidMoveError):
                    raise chess.InvalidMoveError(f"Invalid move notation: {move_text}")

        # Return PlayerDecision for new-style tests
        return decision

    # Wrapper for get_prompt to handle board parameter
    def get_prompt_wrapper(self, **kwargs):
        """Wrapper that converts board to DTO fields."""
        if "board" in kwargs:
            board = kwargs.pop("board")
            kwargs["board_in_fen"] = board.fen()
            kwargs["player_color"] = "white" if board.turn == chess.WHITE else "black"

            # Handle move_history conversion
            if "move_history" in kwargs:
                move_history_str = kwargs.pop("move_history")
                if move_history_str:
                    # Parse move history string to UCI list
                    # For simplicity, just pass empty list
                    kwargs["move_history_in_uci"] = []
                else:
                    kwargs["move_history_in_uci"] = []

        return original_get_prompt(self, **kwargs)

    # Add legacy helper methods
    @staticmethod
    def player_color(board):
        """Legacy helper to get player color from board."""
        return "White" if board.turn == chess.WHITE else "Black"

    @staticmethod
    def board_state(board):
        """Legacy helper to get board FEN."""
        return board.fen()

    # Apply patches
    monkeypatch.setattr(
        GameArenaLLMMoveHandler, "parse_decision_from_response", parse_decision_wrapper
    )
    monkeypatch.setattr(GameArenaLLMMoveHandler, "get_prompt", get_prompt_wrapper)
    # Add static methods directly to the class
    BaseLLMMoveHandler.player_color = staticmethod(player_color)
    BaseLLMMoveHandler.board_state = staticmethod(board_state)


# Common Player Fixtures
@pytest.fixture
def white_player():
    """Standard white RandomPlayer for testing."""
    return RandomPlayer(name="White", color="white", seed=42)


@pytest.fixture
def black_player():
    """Standard black RandomPlayer for testing."""
    return RandomPlayer(name="Black", color="black", seed=43)


@pytest.fixture
def game(white_player, black_player):
    """Standard game with two random players."""
    return Game(white_player, black_player)


# Common Board Positions
@pytest.fixture
def common_positions():
    """Dictionary of commonly used FEN positions for testing."""
    return {
        "stalemate": "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1",
        "king_vs_king": "k7/8/8/8/8/8/8/K7 w - - 0 1",
        "back_rank_mate": "R5k1/5ppp/8/8/8/8/8/7K b - - 0 1",
        "fools_mate": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        "queen_endgame": "8/8/8/4k3/8/3QK3/8/8 w - - 0 1",
        "spanish_opening": "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "kings_facing": "8/8/8/3k4/3K4/8/8/8 w - - 0 1",
        "black_in_check": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPPQPPP/RNB1KBNR b KQkq - 1 2",
    }


# Helper Classes for Testing
class ScriptedPlayer(BasePlayer):
    """Player that plays a predetermined sequence of moves."""

    def __init__(self, name, color, move_sequence):
        super().__init__(name, color)
        self.move_sequence = move_sequence
        self.current_move_index = 0

    def _make_decision(self, context):
        """Make decision from predetermined sequence."""
        if self.current_move_index >= len(self.move_sequence):
            raise ValueError("No more moves in sequence")
        next_move_san = self.move_sequence[self.current_move_index]
        self.current_move_index += 1

        # Convert SAN to UCI
        board = chess.Board(fen=context.board_in_fen)
        move = board.parse_san(next_move_san)
        return PlayerDecision(action="move", attempted_move=move.uci())


class RecordingPlayer(RandomPlayer):
    """RandomPlayer that records board states it observes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observed_board_fens = []

    def _make_decision(self, context):
        """Record board state and make random move."""
        self.observed_board_fens.append(context.board_in_fen)
        return super()._make_decision(context)


class FailingPlayer(RandomPlayer):
    """Player that fails after a certain number of moves."""

    def __init__(self, fail_after_moves=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_after_moves = fail_after_moves
        self.moves_requested_count = 0

    def _make_decision(self, context):
        """Fail after specified number of moves."""
        self.moves_requested_count += 1
        if self.moves_requested_count == self.fail_after_moves:
            raise RuntimeError("Simulated player error")
        return super()._make_decision(context)


class IllegalMovePlayer(BasePlayer):
    """Player that returns a specific illegal move."""

    def __init__(self, name, color, illegal_move_uci="b1e4"):
        super().__init__(name, color)
        self.illegal_move_uci = illegal_move_uci

    def _make_decision(self, context):
        """Return an illegal move."""
        return PlayerDecision(action="move", attempted_move=self.illegal_move_uci)


# Assertion Helpers
def assert_game_terminated(game, expected_termination, expected_winner=None):
    """Helper to assert game termination state."""
    assert game.finished
    assert game.outcome is not None
    assert game.outcome.termination == expected_termination
    assert game.winner == expected_winner


def assert_game_in_progress(game):
    """Helper to assert game is still in progress."""
    assert not game.finished
    assert game.outcome is None
    assert game.winner is None


def setup_game_from_fen(fen_string, white_player=None, black_player=None):
    """Create a game with a specific board position."""
    if white_player is None:
        white_player = RandomPlayer(name="White", color="white")
    if black_player is None:
        black_player = RandomPlayer(name="Black", color="black")
    game = Game(white_player, black_player)
    game.board = chess.Board(fen_string)
    return game
