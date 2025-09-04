import os
import shutil

import chess
import pytest

from llm_chess_arena.game import Game
from llm_chess_arena.player.stockfish_player import StockfishPlayer


def stockfish_available():
    if os.getenv("STOCKFISH_BINARY_PATH"):
        return os.path.exists(os.getenv("STOCKFISH_BINARY_PATH"))
    return shutil.which("stockfish") is not None


@pytest.mark.skipif(not stockfish_available(), reason="Stockfish not installed")
class TestStockfishPlayer:
    def test_initialization__stores_name_color_and_engine_limits(self):
        player = StockfishPlayer(
            name="Test Stockfish", color="white", engine_limits={"depth": 5}
        )
        assert player.name == "Test Stockfish"
        assert player.color == "white"
        assert player.engine_limits["depth"] == 5
        player.close()

    def test_get_move__returns_legal_move_for_given_position(self):
        player = StockfishPlayer(
            name="Test Stockfish", color="white", engine_limits={"depth": 5}
        )

        board = chess.Board()
        decision = player(board)

        assert decision.action == "move"
        move = chess.Move.from_uci(decision.attempted_move)
        assert move in board.legal_moves

        player.close()

    def test_game_against_random_player__completes_within_move_limit(
        self, black_player
    ):
        stockfish = StockfishPlayer(
            name="Stockfish",
            color="white",
            engine_limits={"depth": 5},  # Low depth for speed
        )

        game = Game(stockfish, black_player)
        game.play(max_num_moves=100)
        assert game.finished or len(game.board.move_stack) == 100

        stockfish.close()

    def test_close__cleans_up_engine_resources_and_is_idempotent(self):
        player = StockfishPlayer(
            name="Test Stockfish", color="white", engine_limits={"depth": 5}
        )

        board = chess.Board()
        player(board)  # Lazy initialization
        assert player.engine is not None

        player.close()
        assert player.engine is None

        player.close()  # Should handle second call gracefully

    def test_time_limit_configuration__produces_valid_moves_within_constraint(self):
        player = StockfishPlayer(
            name="Fast Stockfish",
            color="white",
            engine_limits={"depth": 10, "time": 0.1},
        )

        board = chess.Board()
        decision = player(board)

        assert decision.action == "move"
        move = chess.Move.from_uci(decision.attempted_move)
        assert move in board.legal_moves

        player.close()


class TestStockfishNotAvailable:
    def test_invalid_binary_path__raises_file_not_found_error_with_descriptive_message(
        self, monkeypatch
    ):
        monkeypatch.setenv("STOCKFISH_BINARY_PATH", "/nonexistent/path")
        monkeypatch.setattr(shutil, "which", lambda x: None)

        with pytest.raises(FileNotFoundError) as exc_info:
            StockfishPlayer(
                name="Test", color="white", binary_path="/nonexistent/stockfish"
            )

        assert "Stockfish binary not found" in str(exc_info.value)

    def test_various_engine_limits__stored_without_validation(self):
        player = StockfishPlayer(
            name="Test",
            color="white",
            engine_limits={"depth": 20, "time": 5.0, "nodes": 1000000},
        )
        assert player.engine_limits["depth"] == 20
        assert player.engine_limits["time"] == 5.0
        assert player.engine_limits["nodes"] == 1000000
        player.close()

        player2 = StockfishPlayer(
            name="Test",
            color="white",
            engine_limits={"depth": 0},  # Let chess.engine handle validation
        )
        assert player2.engine_limits["depth"] == 0
        player2.close()

    def test_engine_options__stored_as_provided_in_initialization(self):
        player = StockfishPlayer(
            name="Configured Stockfish",
            color="white",
            engine_limits={"depth": 5},
            engine_options={"Hash": 256, "Threads": 2},
        )

        assert player.engine_options == {"Hash": 256, "Threads": 2}
        player.close()

    def test_default_engine_limits__produces_legal_moves(self):
        player = StockfishPlayer(
            name="Test",
            color="white",
            engine_limits={"depth": 5},
        )

        board = chess.Board()
        decision = player(board)

        assert decision.action == "move"
        move = chess.Move.from_uci(decision.attempted_move)
        assert move in board.legal_moves
        player.close()
