"""Shared test fixtures and utilities for the test suite."""

import warnings
from collections.abc import Sequence
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
from llm_chess_arena.types import (  # noqa: E402
    Color,
    PlayerDecision,
    PlayerDecisionContext,
)


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

    def __init__(self, name: str, color: Color, move_sequence: Sequence[str]):
        """Initialize scripted player with a fixed SAN move sequence.

        Args:
            name: Display name used in logs.
            color: Player color literal.
            move_sequence: Iterable of SAN moves to execute in order.
        """
        super().__init__(name, color)
        self.move_sequence = list(move_sequence)
        self.current_move_index = 0

    def _make_decision(self, context: PlayerDecisionContext) -> PlayerDecision:
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

    def __init__(self, *args, **kwargs) -> None:
        """Track observed FEN strings while retaining RandomPlayer behavior."""
        super().__init__(*args, **kwargs)
        self.observed_board_fens = []

    def _make_decision(self, context: PlayerDecisionContext) -> PlayerDecision:
        """Record board state and make random move."""
        self.observed_board_fens.append(context.board_in_fen)
        return super()._make_decision(context)


class FailingPlayer(RandomPlayer):
    """Player that fails after a certain number of moves."""

    def __init__(
        self,
        fail_after_moves: int = 2,
        *args,
        **kwargs,
    ) -> None:
        """Set up a player that raises after a fixed number of decisions.

        Args:
            fail_after_moves: Number of decisions before raising an error.
            *args: Positional arguments forwarded to RandomPlayer.
            **kwargs: Keyword arguments forwarded to RandomPlayer.
        """
        super().__init__(*args, **kwargs)
        self.fail_after_moves = fail_after_moves
        self.moves_requested_count = 0

    def _make_decision(self, context: PlayerDecisionContext) -> PlayerDecision:
        """Fail after specified number of moves."""
        self.moves_requested_count += 1
        if self.moves_requested_count == self.fail_after_moves:
            raise RuntimeError("Simulated player error")
        return super()._make_decision(context)


class IllegalMovePlayer(BasePlayer):
    """Player that returns a specific illegal move."""

    def __init__(
        self,
        name: str,
        color: Color,
        illegal_move_uci: str = "b1e4",
    ) -> None:
        """Initialize player configured to respond with an illegal UCI move.

        Args:
            name: Display name used in diagnostics.
            color: Player color literal.
            illegal_move_uci: Always-returned illegal move in UCI notation.
        """
        super().__init__(name, color)
        self.illegal_move_uci = illegal_move_uci

    def _make_decision(self, context: PlayerDecisionContext) -> PlayerDecision:
        """Return an illegal move."""
        return PlayerDecision(action="move", attempted_move=self.illegal_move_uci)


# Assertion Helpers
def assert_game_terminated(
    game: Game,
    expected_termination: chess.Termination,
    expected_winner: BasePlayer | None = None,
) -> None:
    """Assert that a game finished with the specified termination state.

    Args:
        game: Game instance that should be finished.
        expected_termination: Expected chess.Termination enum value.
        expected_winner: Optional winning player instance.
    """
    assert game.finished
    assert game.outcome is not None
    assert game.outcome.termination == expected_termination
    assert game.winner == expected_winner


def assert_game_in_progress(game: Game) -> None:
    """Assert that a game continues without an outcome yet."""
    assert not game.finished
    assert game.outcome is None
    assert game.winner is None


def setup_game_from_fen(
    fen_string: str,
    white_player: BasePlayer | None = None,
    black_player: BasePlayer | None = None,
) -> Game:
    """Create a game whose board starts from the provided FEN.

    Args:
        fen_string: FEN string describing the desired starting position.
        white_player: Optional preconfigured white player instance.
        black_player: Optional preconfigured black player instance.

    Returns:
        Game: Newly instantiated game object with the desired board state.
    """
    if white_player is None:
        white_player = RandomPlayer(name="White", color="white")
    if black_player is None:
        black_player = RandomPlayer(name="Black", color="black")
    game = Game(white_player, black_player)
    game.board = chess.Board(fen_string)
    return game
