"""Unit tests exercising the core game orchestration logic."""

import chess
import pytest

import llm_chess_arena.renderer as renderer
from llm_chess_arena.game import Game
from llm_chess_arena.exceptions import IllegalMoveError
from llm_chess_arena.metrics import MetricsTracker, MoveMetrics, MoveQuality
from llm_chess_arena.player.base_player import BasePlayer
from llm_chess_arena.types import PlayerDecision, PlayerDecisionContext
from llm_chess_arena.utils import build_game_summary
from tests.conftest import (
    FailingPlayer,
    IllegalMovePlayer,
    ScriptedPlayer,
    assert_game_terminated,
    setup_game_from_fen,
)


class AlwaysResignPlayer(BasePlayer):
    """Player that immediately resigns."""

    def __init__(self, name: str, color: str) -> None:
        """Store identifying metadata for the stub player."""
        super().__init__(name=name, color=color)

    def _make_decision(
        self, context: PlayerDecisionContext
    ) -> PlayerDecision:  # noqa: D401
        """Return a resignation decision regardless of the board state."""
        return PlayerDecision(action="resign")


class TestGameInitialization:
    """Initialization invariants for the Game orchestration class."""

    def test_initialization__sets_players_and_initial_state_correctly(
        self, white_player, black_player
    ):
        """Game holds onto the players and starts from the initial state."""
        game = Game(
            white_player,
            black_player,
            display_board=False,
            display_summary=False,
            enable_metrics=False,
        )

        assert game.white_player == white_player
        assert game.black_player == black_player
        assert game.board == chess.Board()
        assert game.outcome is None
        assert game.winner is None

    def test_initialization__creates_standard_starting_chess_position(self, game):
        """Board should match a fresh python-chess board."""
        standard_starting_board = chess.Board()
        assert game.board == standard_starting_board
        assert len(game.board.move_stack) == 0
        assert game.board.turn == chess.WHITE

    def test_initialization__raises_value_error__when_both_players_have_same_color(
        self, white_player
    ):
        """Reject mismatched player color assignments."""
        second_white_player = white_player

        with pytest.raises(ValueError, match="wrong color"):
            Game(
                white_player,
                second_white_player,
                display_board=False,
                display_summary=False,
                enable_metrics=False,
            )


class TestGameFlow:
    """Core flow behavior such as turn order and move application."""

    def test_current_player__alternates_between_white_and_black_after_each_move(
        self, game, white_player, black_player
    ):
        """Current player should toggle after every move."""
        assert game.current_player == white_player

        game.make_move()
        assert game.current_player == black_player

        game.make_move()
        assert game.current_player == white_player

    def test_make_move__updates_board_state_and_switches_turn_to_opponent(self, game):
        """A successful move updates the board and flips the turn."""
        initial_board_state = game.board.fen()

        game.make_move()

        assert game.board.fen() != initial_board_state
        assert len(game.board.move_stack) == 1
        assert game.board.turn == chess.BLACK

    def test_make_move__raises_illegal_move_error__when_player_returns_invalid_move(
        self, black_player
    ):
        """Illegal moves from players propagate as IllegalMoveError."""
        illegal_move_player = IllegalMovePlayer(
            name="Illegal", color="white", illegal_move_uci="b1e4"
        )
        game = Game(
            illegal_move_player,
            black_player,
            display_board=False,
            display_summary=False,
            enable_metrics=False,
        )

        with pytest.raises(IllegalMoveError):
            game.make_move()


class TestGameTermination:
    """Game termination scenarios such as checkmate and stalemate."""

    def test_play__detects_checkmate__when_scholars_mate_sequence_is_played(
        self,
    ):
        """Scholars mate script should yield checkmate for white."""
        scholars_mate_white_moves = ["e4", "Bc4", "Qh5", "Qxf7#"]
        scholars_mate_black_moves = ["e5", "Nc6", "Bc5"]

        white_player = ScriptedPlayer("White", "white", scholars_mate_white_moves)
        black_player = ScriptedPlayer("Black", "black", scholars_mate_black_moves)

        game = Game(
            white_player,
            black_player,
            display_board=False,
            display_summary=False,
            enable_metrics=False,
        )
        game.play()

        assert_game_terminated(game, chess.Termination.CHECKMATE, white_player)

    def test_play__detects_stalemate__when_no_legal_moves_available_for_player_to_move(
        self, common_positions
    ):
        """Ensure stalemate positions are recognized immediately."""
        game = setup_game_from_fen(common_positions["stalemate"])

        assert_game_terminated(game, chess.Termination.STALEMATE, None)

    def test_detects_draw_by_insufficient_material__when_only_kings_remain_on_board(
        self, common_positions
    ):
        """Insufficient material should produce the proper draw outcome."""
        game = setup_game_from_fen(common_positions["king_vs_king"])

        assert_game_terminated(game, chess.Termination.INSUFFICIENT_MATERIAL, None)

    def test_board_can_claim_threefold_repetition__after_same_position_occurs_three_times(
        self, game
    ):
        """Threefold repetition should be claimable when positions repeat."""
        knight_dance_creating_repetition = [
            "Nf3",
            "Nf6",
            "Ng1",
            "Ng8",  # Return to starting position
            "Nf3",
            "Nf6",
            "Ng1",
            "Ng8",  # Second repetition
        ]

        for move_san in knight_dance_creating_repetition:
            game.board.push_san(move_san)

        assert game.board.can_claim_threefold_repetition()

    def test_board_can_claim_fifty_move_rule__after_100_halfmoves_without_pawn_move_or_capture(
        self, game
    ):
        """Fifty-move rule should be claimable once the clock reaches 100."""
        halfmove_clock_after_fifty_full_moves = 100
        game.board.halfmove_clock = halfmove_clock_after_fifty_full_moves

        assert game.board.can_claim_fifty_moves()


class TestGameResult:
    """Ensure the result reporting matches expected scoring conventions."""

    def test_back_rank_mate__results_in_white_victory_with_score_1_0(
        self, common_positions, white_player
    ):
        """Back-rank mate fixture should count as a white win."""
        game = setup_game_from_fen(common_positions["back_rank_mate"])
        game.white_player = white_player

        assert game.finished
        assert game.winner == white_player
        assert game.outcome.result() == "1-0"

    def test_fools_mate__results_in_black_victory_with_score_0_1(
        self, common_positions, black_player
    ):
        """Fool's mate fixture should count as a black win."""
        game = setup_game_from_fen(common_positions["fools_mate"])
        game.black_player = black_player

        assert game.finished
        assert game.winner == black_player
        assert game.outcome.result() == "0-1"

    def test_stalemate__results_in_draw_with_no_winner_and_score_half_half(
        self, common_positions
    ):
        """Stalemates should report a draw and no winner."""
        game = setup_game_from_fen(common_positions["stalemate"])

        assert game.finished
        assert game.winner is None
        assert game.outcome.result() == "1/2-1/2"


class TestGameResignation:
    """Resignation-specific behavior validation."""

    def test_resignation_in_summary(self):
        """Termination line should reflect resignation."""

        white_player = AlwaysResignPlayer("LLM", "white")
        black_player = ScriptedPlayer("Opponent", "black", ["e5"])
        game = Game(
            white_player,
            black_player,
            display_board=False,
            display_summary=False,
            enable_metrics=False,
        )

        game.play(max_num_moves=1)

        game_summary = build_game_summary(game)
        summary_lines = game_summary.to_cli_lines()
        termination_lines = [
            line for line in summary_lines if line.startswith("Termination:")
        ]

        assert termination_lines, "Expected a termination line in the summary"
        assert "Resignation" in termination_lines[0]


class RecordingEvaluator:
    """Test helper that records evaluation inputs and returns fixed metrics."""

    def __init__(self) -> None:
        """Initialize call tracking structures for assertions."""
        self.calls: list[tuple[str, str, str]] = []
        self.closed = False

    def evaluate_move(self, board: chess.Board, move: chess.Move) -> MoveMetrics:
        """Record the evaluation inputs and return a deterministic metric."""
        player_color = "white" if board.turn == chess.WHITE else "black"
        self.calls.append((player_color, board.fen(), move.uci()))
        return MoveMetrics(
            player_color=player_color,
            move_in_uci=move.uci(),
            best_move_in_uci=move.uci(),
            centipawn_loss=1.0,
            win_probability_delta=0.0,
            best_move_hit=True,
            quality=MoveQuality.BEST,
        )

    def close(self) -> None:
        """Mark the evaluator as closed for verification purposes."""
        self.closed = True


class TestGameMetrics:
    """Integration tests for game-level metrics tracking."""

    def test_game_records_metrics_per_move(self) -> None:
        """Game should evaluate every executed move and summarize results."""
        evaluator = RecordingEvaluator()
        tracker = MetricsTracker(evaluator)

        white_player = ScriptedPlayer(
            "White",
            "white",
            ["e4", "Nf3"],
        )
        black_player = ScriptedPlayer(
            "Black",
            "black",
            ["e5", "Nc6"],
        )

        game = Game(
            white_player,
            black_player,
            display_board=False,
            display_summary=False,
            enable_metrics=True,
            metrics_tracker=tracker,
        )
        game.play(max_num_moves=4)

        assert len(evaluator.calls) == 4
        initial_fen = chess.Board().fen()
        assert evaluator.calls[0][1] == initial_fen

        summary = tracker.summarize()
        assert summary["white"].moves_evaluated == 2
        assert summary["white"].average_centipawn_loss == 1.0
        assert summary["white"].best_move_hit_rate == 1.0
        assert summary["white"].quality_counts[MoveQuality.BEST] == 2
        assert summary["black"].moves_evaluated == 2
        assert summary["black"].quality_counts[MoveQuality.BEST] == 2
        assert evaluator.closed
        assert game._get_move_qualities_for_display() == [MoveQuality.BEST] * 4


def test_format_quality_summary_handles_empty_counts() -> None:
    """Formatting should gracefully handle games without evaluated moves."""
    assert Game._format_quality_summary({}) == "none"


def test_format_quality_summary_orders_non_zero_counts() -> None:
    """Quality summary output should respect ordering and skip empty buckets."""
    counts = {
        MoveQuality.BLUNDER: 2,
        MoveQuality.BEST: 1,
        MoveQuality.GOOD: 3,
        MoveQuality.MISTAKE: 0,
    }
    assert Game._format_quality_summary(counts) == "best:1, good:3, blunder:2"


def test_format_move_history_includes_glyphs_and_quality_annotations() -> None:
    """Rendered history shows piece glyphs and quality suffixes."""
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("e7e5"))

    rows = renderer._generate_move_history_rows(
        board,
        move_qualities=[MoveQuality.BEST, MoveQuality.MISTAKE],
    )

    assert rows, "Expected move history rows"
    move_number, white_entry, black_entry = rows[0]
    assert move_number == 1
    assert white_entry is not None
    assert white_entry.style == renderer.WHITE_PLAYER_STYLE
    assert white_entry.plain.startswith("♟ e2e4")
    assert white_entry.plain.endswith("!!")

    assert black_entry is not None
    assert black_entry.style == renderer.BLACK_PLAYER_STYLE
    assert black_entry.plain.startswith("♟ e7e5")
    assert black_entry.plain.endswith("?")


class TestGameHistory:
    """PGN export behavior."""

    def test_play_writes_pgn_to_history_output_path(self, tmp_path) -> None:
        """Completed games should be persisted to the configured record directory."""
        white_player = ScriptedPlayer(
            "White",
            "white",
            ["e4", "Qh5", "Bc4", "Qxf7#"],
        )
        black_player = ScriptedPlayer(
            "Black",
            "black",
            ["e5", "Nc6", "Nf6"],
        )
        record_dir = tmp_path / "records"

        game = Game(
            white_player,
            black_player,
            display_board=False,
            display_summary=False,
            enable_metrics=False,
            record_dir=record_dir,
            record_name="test_game",
        )

        game.play()

        # Check both PGN and JSON files are created
        pgn_path = record_dir / "test_game.pgn"
        json_path = record_dir / "test_game.json"

        assert pgn_path.exists()
        assert json_path.exists()

        pgn_contents = pgn_path.read_text(encoding="utf-8")
        assert '[Event "LLM Chess Arena"]' in pgn_contents
        assert '[Date "' in pgn_contents
        assert '[White "White (W)"]' in pgn_contents
        assert '[Black "Black (B)"]' in pgn_contents
        assert "1. e4" in pgn_contents

    def test_no_history_written_when_game_aborts(self, tmp_path) -> None:
        """Abortive games should not emit record files."""
        white_player = FailingPlayer(
            name="Failing",
            color="white",
            fail_after_moves=1,
            seed=1,
        )
        black_player = ScriptedPlayer(
            "Black",
            "black",
            ["e5"],
        )
        record_dir = tmp_path / "records"

        game = Game(
            white_player,
            black_player,
            display_board=False,
            display_summary=False,
            enable_metrics=False,
            record_dir=record_dir,
            record_name="failed_game",
        )

        with pytest.raises(RuntimeError):
            game.play()

        # Neither PGN nor JSON should exist
        pgn_path = record_dir / "failed_game.pgn"
        json_path = record_dir / "failed_game.json"
        assert not pgn_path.exists()
        assert not json_path.exists()
