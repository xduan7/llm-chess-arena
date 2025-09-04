import chess
import pytest

from llm_chess_arena.game import Game
from llm_chess_arena.player.random_player import RandomPlayer
from tests.conftest import (
    FailingPlayer,
    RecordingPlayer,
    setup_game_from_fen,
)


class TestRandomVsRandom:
    def test_game__should_terminate__before_move_limit(self, game):
        max_moves_before_timeout = 200

        game.play(max_num_moves=max_moves_before_timeout)

        game_reached_natural_end = game.finished
        game_reached_move_limit = len(game.board.move_stack) == max_moves_before_timeout

        assert game_reached_natural_end or game_reached_move_limit

        if game_reached_natural_end:
            assert game.outcome is not None
            assert game.outcome.termination in chess.Termination

    def test_games__should_produce_varied_outcomes__when_seeds_differ(
        self,
    ):
        outcomes = []
        num_games = 5
        move_limit = 200

        for game_idx in range(num_games):
            white_seed = game_idx
            black_seed = game_idx + 100

            white_player = RandomPlayer(name="White", color="white", seed=white_seed)
            black_player = RandomPlayer(name="Black", color="black", seed=black_seed)

            game = Game(white_player, black_player)
            game.play(max_num_moves=move_limit)

            outcome = game.outcome.result() if game.finished else "unfinished"
            outcomes.append(outcome)

        unique_outcomes = set(outcomes)
        assert (
            len(unique_outcomes) > 1
        ), f"Expected varied outcomes, got only: {unique_outcomes}"

    @pytest.mark.slow
    def test_games__should_achieve_checkmate__when_playing_many_games(
        self,
    ):
        termination_types = set()
        num_games = 20
        move_limit = 300

        for game_idx in range(num_games):
            white_seed = game_idx * 2
            black_seed = game_idx * 2 + 1

            white_player = RandomPlayer(name="White", color="white", seed=white_seed)
            black_player = RandomPlayer(name="Black", color="black", seed=black_seed)

            game = Game(white_player, black_player)
            game.play(max_num_moves=move_limit)

            if game.finished and game.outcome:
                termination_types.add(game.outcome.termination)

        assert (
            chess.Termination.CHECKMATE in termination_types
        ), f"Expected checkmate in {num_games} games, saw: {termination_types}"


class TestGameWithCustomBoard:
    def test_game__should_continue__from_spanish_opening(
        self, white_player, black_player
    ):
        game = Game(white_player, black_player)

        spanish_opening_moves = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"]

        for move_san in spanish_opening_moves:
            game.board.push_san(move_san)

        moves_after_setup = len(game.board.move_stack)
        assert moves_after_setup == 8
        assert game.board.turn == chess.WHITE

        max_additional_moves = 50
        game.play(max_num_moves=max_additional_moves)

        total_moves_played = len(game.board.move_stack)
        assert total_moves_played > moves_after_setup

    def test_queen_endgame_plays_until_mate_or_move_limit(self, common_positions):
        game = setup_game_from_fen(common_positions["queen_endgame"])
        initial_moves = len(game.board.move_stack)

        max_endgame_moves = 50

        game.play(max_num_moves=max_endgame_moves)

        game_ended = game.finished
        move_limit_reached = (
            len(game.board.move_stack) - initial_moves >= max_endgame_moves
        )

        assert game_ended or move_limit_reached


class TestPlayerInteraction:
    def test_each_player_sees_correct_board_states_during_play(
        self,
    ):
        white_recording_player = RecordingPlayer(name="White", color="white", seed=42)
        black_recording_player = RecordingPlayer(name="Black", color="black", seed=43)
        game = Game(white_recording_player, black_recording_player)

        moves_to_play = 6
        for move_count in range(moves_to_play):
            if not game.finished:
                game.make_move()

        assert len(white_recording_player.observed_board_fens) == 3
        assert len(black_recording_player.observed_board_fens) == 3

        unique_white_positions = set(white_recording_player.observed_board_fens)
        unique_black_positions = set(black_recording_player.observed_board_fens)

        assert (
            len(unique_white_positions) == 3
        ), "White should see 3 different positions"
        assert (
            len(unique_black_positions) == 3
        ), "Black should see 3 different positions"

    def test_game_propagates_player_exceptions(self, black_player):
        faulty_white_player = FailingPlayer(
            name="White", color="white", seed=42, fail_after_moves=2
        )
        game = Game(faulty_white_player, black_player)

        with pytest.raises(RuntimeError, match="Simulated player error"):
            game.play(max_num_moves=10)
