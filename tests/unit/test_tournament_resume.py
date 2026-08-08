"""Unit tests for tournament resume functionality.

Fixtures in this module mirror the record schema actually produced by
RecordWriter/GameSummary.to_json_dict (summary.players, hydra_config.players,
result "Unfinished" for interrupted games) - not an idealized format.
"""

from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_chess_arena.exceptions import InvalidGameRecordError
from llm_chess_arena.tournament.loader import TournamentLoader, load_game_result
from llm_chess_arena.tournament.resume import LockInfo, TournamentResumer

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

RANDOM_PLAYERS_HYDRA_CONFIG = {
    "players": {
        "white": {"kind": "random", "name": "P1", "color": "white", "seed": 1},
        "black": {"kind": "random", "name": "P2", "color": "black", "seed": 2},
    },
}


def _resumable_game_json(total_moves: int = 0) -> dict:
    """Game record as written for a network-interrupted game."""
    return {
        "summary": {
            "result": "Unfinished",
            "termination": "unfinished",
            "total_moves": total_moves,
            "winner": None,
            "winner_color": None,
            "white_player": "P1",
            "black_player": "P2",
            "players": {},
        },
        "game_outcome": {
            "result": "Unfinished",
            "total_moves": total_moves,
            "termination": "unfinished",
            "winner": None,
        },
        "termination_metadata": {
            "resumable": True,
            "error_type": "ConnectionError",
            "player_color": "white",
            "error_message": "Connection timeout",
            "halfmove_index": total_moves,
            "fullmove_number": total_moves // 2 + 1,
            "fen": STARTING_FEN,
        },
        "hydra_config": RANDOM_PLAYERS_HYDRA_CONFIG,
        "game_setup": {"initial_fen": STARTING_FEN},
        "moves": [],
    }


def _completed_game_json(result: str = "1-0") -> dict:
    """Game record as written for a normally-finished game."""
    return {
        "summary": {
            "result": result,
            "termination": "checkmate",
            "total_moves": 42,
            "winner": "P1",
            "winner_color": "white",
            "white_player": "P1",
            "black_player": "P2",
            "players": {},
        },
        "game_outcome": {
            "result": result,
            "total_moves": 42,
            "termination": "checkmate",
            "winner": "white",
        },
        "hydra_config": RANDOM_PLAYERS_HYDRA_CONFIG,
        "game_setup": {"initial_fen": STARTING_FEN},
        "moves": [],
    }


class TestGameResultDeserializer:
    """Test suite for load_game_result()."""

    @pytest.fixture
    def valid_game_json(self, tmp_path: Path) -> Path:
        """Create a game.json matching the writer's actual schema."""
        game_data = {
            "game_outcome": {
                "result": "1-0",
                "total_moves": 4,
                "termination": "checkmate",
                "end_timestamp": "2024-01-01T12:00:00.000Z",
            },
            "summary": {
                "result": "1-0",
                "termination": "checkmate",
                "total_moves": 4,
                "winner": "Player1",
                "winner_color": "white",
                "white_player": "Player1",
                "black_player": "Player2",
                "players": {
                    "white": {"thinking_time_in_sec": 5.0, "cost": 0.05},
                    "black": {"thinking_time_in_sec": 1.0, "cost": 0.04},
                },
            },
            "moves": [
                {
                    "player": "white",
                    "thinking_time_in_sec": 2.0,
                    "stockfish_evaluation": {"quality": "best", "centipawn_loss": 0.0},
                },
                {
                    "player": "black",
                    "thinking_time_in_sec": 1.0,
                    "stockfish_evaluation": {"quality": "good", "centipawn_loss": 60.0},
                },
                {
                    "player": "white",
                    "thinking_time_in_sec": 3.0,
                    "stockfish_evaluation": {
                        "quality": "excellent",
                        "centipawn_loss": 20.0,
                    },
                },
                {
                    "player": "black",
                    "stockfish_evaluation": {"quality": "good", "centipawn_loss": 40.0},
                },
            ],
        }

        json_path = tmp_path / "game.json"
        with json_path.open("w") as f:
            json.dump(game_data, f)

        # Create PGN file
        (tmp_path / "game.pgn").write_text("1. e4 e5")

        return json_path

    def test_load_from_game_json__valid_game__returns_result(
        self, valid_game_json: Path
    ) -> None:
        """Load a complete game.json successfully."""
        result = load_game_result(valid_game_json, game_id=1)

        assert result.game_id == 1
        assert result.white_player_name == "Player1"
        assert result.black_player_name == "Player2"
        assert result.result == "1-0"
        assert result.total_moves == 4
        assert result.termination_reason == "checkmate"
        # Metrics derive from per-move stockfish_evaluation entries
        assert result.white_centipawn_loss == 10.0
        assert result.black_centipawn_loss == 50.0
        assert result.white_quality_counts == {"best": 1, "excellent": 1}
        assert result.black_quality_counts == {"good": 2}
        # Costs and thinking time come from summary.players
        assert result.white_cost == 0.05
        assert result.black_cost == 0.04
        assert result.white_thinking_time_in_sec == 5.0
        assert result.black_thinking_time_in_sec == 1.0
        assert result.json_path == valid_game_json
        assert result.pgn_path == valid_game_json.parent / "game.pgn"
        assert result.was_resumed is False

    def test_load_from_game_json__thinking_time_falls_back_to_moves(
        self, tmp_path: Path
    ) -> None:
        """Without a players summary, thinking time is summed from moves."""
        game_data = {
            "game_outcome": {
                "result": "0-1",
                "total_moves": 2,
                "termination": "checkmate",
            },
            "summary": {
                "white_player": "P1",
                "black_player": "P2",
            },
            "moves": [
                {"player": "white", "thinking_time_in_sec": 1.5},
                {"player": "black", "thinking_time_in_sec": 2.5},
            ],
        }
        json_path = tmp_path / "game.json"
        json_path.write_text(json.dumps(game_data))

        result = load_game_result(json_path, game_id=3)

        assert result.white_thinking_time_in_sec == 1.5
        assert result.black_thinking_time_in_sec == 2.5

    def test_load_from_game_json__missing_required_field__raises_error(
        self, tmp_path: Path
    ) -> None:
        """Raise InvalidGameRecordError for missing fields."""
        incomplete_data = {
            "game_outcome": {"result": "1-0"},  # Missing total_moves, termination
            "summary": {"white_player": "P1", "black_player": "P2"},
        }

        json_path = tmp_path / "incomplete.json"
        with json_path.open("w") as f:
            json.dump(incomplete_data, f)

        with pytest.raises(InvalidGameRecordError, match="Missing required field"):
            load_game_result(json_path, game_id=1)

    def test_load_from_game_json__resumed_game__sets_was_resumed_flag(
        self, tmp_path: Path
    ) -> None:
        """Detect and mark games that have resumption_metadata."""
        resumed_data = {
            "game_outcome": {
                "result": "1-0",
                "total_moves": 42,
                "termination": "checkmate",
            },
            "summary": {
                "white_player": "P1",
                "black_player": "P2",
            },
            "resumption_metadata": {
                "resumed_from_file": "game.json.original",
                "resumed_at": "2024-01-01T13:00:00.000Z",
                "original_termination": {
                    "error_type": "TimeoutError",
                },
            },
        }

        json_path = tmp_path / "resumed.json"
        with json_path.open("w") as f:
            json.dump(resumed_data, f)

        result = load_game_result(json_path, game_id=1)

        assert result.was_resumed is True
        assert result.original_termination_reason == "TimeoutError"

    def test_load_from_game_json__optional_metrics__handles_none(
        self, tmp_path: Path
    ) -> None:
        """Handle games without metrics gracefully."""
        minimal_data = {
            "game_outcome": {
                "result": "1/2-1/2",
                "total_moves": 30,
                "termination": "stalemate",
            },
            "summary": {
                "white_player": "P1",
                "black_player": "P2",
                # No metrics fields
            },
        }

        json_path = tmp_path / "minimal.json"
        with json_path.open("w") as f:
            json.dump(minimal_data, f)

        result = load_game_result(json_path, game_id=2)

        assert result.white_centipawn_loss is None
        assert result.black_centipawn_loss is None
        assert result.white_cost == 0.0
        assert result.black_cost == 0.0

    def test_load_from_game_json__malformed_json__raises_error(
        self, tmp_path: Path
    ) -> None:
        """Raise InvalidGameRecordError for corrupt JSON."""
        bad_json = tmp_path / "corrupt.json"
        bad_json.write_text("not valid json {")

        with pytest.raises(InvalidGameRecordError, match="Invalid JSON"):
            load_game_result(bad_json, game_id=1)


class TestTournamentLoader:
    """Test suite for TournamentLoader."""

    @pytest.fixture
    def tournament_dir(self, tmp_path: Path) -> Path:
        """Create a mock tournament directory structure."""
        tournament = tmp_path / "test_match"
        tournament.mkdir()

        # Create results.json
        results = {
            "match_name": "test_match",
            "player1": "P1",
            "player2": "P2",
            "start_time": "2024-01-01T00:00:00",
            "total_games": 3,
        }
        (tournament / "results.json").write_text(json.dumps(results))

        # Create game directories
        for i in range(1, 4):
            game_dir = tournament / f"game_{i:03d}"
            game_dir.mkdir()

        return tournament

    def test_find_all_game_dirs__finds_and_sorts_games(
        self, tournament_dir: Path
    ) -> None:
        """Find all game_NNN directories sorted by number."""
        game_dirs = TournamentLoader.find_all_game_dirs(tournament_dir)

        assert len(game_dirs) == 3
        assert game_dirs[0].name == "game_001"
        assert game_dirs[1].name == "game_002"
        assert game_dirs[2].name == "game_003"

    def test_find_resumable_games__network_errors__found(
        self, tournament_dir: Path
    ) -> None:
        """Find games with resumable=true from network errors."""
        # Game 1: resumable (network error)
        (tournament_dir / "game_001" / "game.json").write_text(
            json.dumps(_resumable_game_json())
        )

        # Game 2: completed normally
        (tournament_dir / "game_002" / "game.json").write_text(
            json.dumps(_completed_game_json())
        )

        resumable = TournamentLoader.find_resumable_games(tournament_dir)

        assert len(resumable) == 1
        assert resumable[0][0] == 1  # game_id

    def test_find_resumable_games__completed_after_resume__ignored(
        self, tournament_dir: Path
    ) -> None:
        """A game whose resume completed (result no longer Unfinished) is done."""
        game_json = _completed_game_json()
        game_json["resumption_metadata"] = {"resumed_at": "2024-01-01T12:00:00"}
        # A completed record keeps no resumable flag, but even a leftover one
        # must not mark a finished game as resumable
        game_json["termination_metadata"] = {"resumable": True}
        (tournament_dir / "game_001" / "game.json").write_text(json.dumps(game_json))

        resumable = TournamentLoader.find_resumable_games(tournament_dir)

        assert len(resumable) == 0

    def test_find_resumable_games__interrupted_again__still_resumable(
        self, tournament_dir: Path
    ) -> None:
        """A resumed game that hit another network error stays resumable."""
        game_json = _resumable_game_json(total_moves=12)
        game_json["resumption_metadata"] = {"resumed_at": "2024-01-01T12:00:00"}
        (tournament_dir / "game_001" / "game.json").write_text(json.dumps(game_json))

        resumable = TournamentLoader.find_resumable_games(tournament_dir)

        assert len(resumable) == 1
        assert resumable[0][0] == 1

    def test_validate_game_resumable__missing_hydra_config__returns_false(
        self, tmp_path: Path
    ) -> None:
        """Validation fails for games without hydra_config."""
        game_json = _resumable_game_json()
        del game_json["hydra_config"]

        json_path = tmp_path / "game.json"
        json_path.write_text(json.dumps(game_json))

        can_resume, reason = TournamentLoader.validate_game_resumable(json_path)

        assert can_resume is False
        assert "hydra_config" in reason

    def test_validate_game_resumable__missing_players_section__returns_false(
        self, tmp_path: Path
    ) -> None:
        """Validation fails when hydra_config lacks players.white/black."""
        game_json = _resumable_game_json()
        game_json["hydra_config"] = {"players": {"white": {"kind": "random"}}}

        json_path = tmp_path / "game.json"
        json_path.write_text(json.dumps(game_json))

        can_resume, reason = TournamentLoader.validate_game_resumable(json_path)

        assert can_resume is False
        assert "white/black" in reason

    def test_validate_game_resumable__malformed_json__returns_false(
        self, tmp_path: Path
    ) -> None:
        """Validation fails for corrupt JSON."""
        json_path = tmp_path / "corrupt.json"
        json_path.write_text("not valid json {")

        can_resume, reason = TournamentLoader.validate_game_resumable(json_path)

        assert can_resume is False
        assert "malformed JSON" in reason


class TestLockManagement:
    """Test suite for lock management."""

    def test_acquire_lock__no_existing_lock__succeeds(self, tmp_path: Path) -> None:
        """Create lock file successfully."""
        lock = LockInfo(
            pid=os.getpid(),
            timestamp=datetime.now(timezone.utc),
            hostname="testhost",
            command="test command",
        )

        lock_path = tmp_path / ".resume.lock"
        lock.save(lock_path)

        assert lock_path.exists()

        # Load and verify
        loaded = LockInfo.load(lock_path)
        assert loaded is not None
        assert loaded.pid == os.getpid()
        assert loaded.hostname == "testhost"

    def test_save__existing_lock_file__raises(self, tmp_path: Path) -> None:
        """Exclusive create loses loudly when another process won the race."""
        lock_path = tmp_path / ".resume.lock"
        lock_path.write_text("{}")

        lock = LockInfo(
            pid=os.getpid(),
            timestamp=datetime.now(timezone.utc),
            hostname="testhost",
            command="test",
        )

        with pytest.raises(FileExistsError):
            lock.save(lock_path)

    def test_is_stale__dead_pid_same_host__returns_true(self) -> None:
        """Detect lock with dead PID on the same host."""
        # Use a very high PID that likely doesn't exist
        dead_pid = 999999
        lock = LockInfo(
            pid=dead_pid,
            timestamp=datetime.now(timezone.utc),
            hostname=socket.gethostname(),
            command="test",
        )

        assert lock.is_stale() is True

    def test_is_stale__live_pid_same_host_old_timestamp__returns_false(self) -> None:
        """A live process on the same host keeps its lock regardless of age.

        Multi-hour resumes are legitimate (LLM backoff); age must not evict a
        provably-running owner.
        """
        old_time = datetime.now(timezone.utc) - timedelta(hours=48)
        lock = LockInfo(
            pid=os.getpid(),
            timestamp=old_time,
            hostname=socket.gethostname(),
            command="test",
        )

        assert lock.is_stale() is False

    def test_is_stale__cross_host_old_timestamp__returns_true(self) -> None:
        """Cross-host locks cannot be PID-checked; old ones expire on age."""
        old_time = datetime.now(timezone.utc) - timedelta(hours=48)
        lock = LockInfo(
            pid=os.getpid(),
            timestamp=old_time,
            hostname="some-other-host",
            command="test",
        )

        assert lock.is_stale() is True

    def test_is_stale__cross_host_recent__returns_false(self) -> None:
        """Recent cross-host locks are honored."""
        lock = LockInfo(
            pid=os.getpid(),
            timestamp=datetime.now(timezone.utc),
            hostname="some-other-host",
            command="test",
        )

        assert lock.is_stale() is False

    def test_is_stale__current_lock__returns_false(self) -> None:
        """Active lock is not stale."""
        lock = LockInfo(
            pid=os.getpid(),
            timestamp=datetime.now(timezone.utc),
            hostname=socket.gethostname(),
            command="test",
        )

        assert lock.is_stale() is False

    def test_lock_info_load__corrupt_file__returns_none(self, tmp_path: Path) -> None:
        """Corrupt lock file returns None."""
        lock_path = tmp_path / ".resume.lock"
        lock_path.write_text("corrupt json {")

        loaded = LockInfo.load(lock_path)

        assert loaded is None


class TestTournamentResumerValidation:
    """Test suite for TournamentResumer validation."""

    @pytest.fixture
    def valid_tournament_dir(self, tmp_path: Path) -> Path:
        """Create a valid tournament directory."""
        tournament = tmp_path / "match"
        tournament.mkdir()

        # Create results.json
        results = {
            "match_name": "match",
            "player1": "P1",
            "player2": "P2",
            "start_time": "2024-01-01T00:00:00",
        }
        (tournament / "results.json").write_text(json.dumps(results))

        # Create a resumable game
        game_dir = tournament / "game_001"
        game_dir.mkdir()
        (game_dir / "game.json").write_text(json.dumps(_resumable_game_json()))

        return tournament

    def test_validate_tournament__valid__returns_no_errors(
        self, valid_tournament_dir: Path
    ) -> None:
        """Valid tournament passes validation."""
        resumer = TournamentResumer(valid_tournament_dir)

        errors = resumer.validate_tournament()

        assert errors == []

    def test_validate_tournament__missing_results__returns_error(
        self, tmp_path: Path
    ) -> None:
        """Missing results.json fails validation."""
        tournament = tmp_path / "match"
        tournament.mkdir()

        resumer = TournamentResumer(tournament)

        errors = resumer.validate_tournament()

        assert len(errors) > 0
        assert any("results.json" in err for err in errors)

    def test_validate_tournament__no_game_dirs__returns_error(
        self, tmp_path: Path
    ) -> None:
        """No game directories fails validation."""
        tournament = tmp_path / "match"
        tournament.mkdir()

        results = {"match_name": "match", "player1": "P1", "player2": "P2"}
        (tournament / "results.json").write_text(json.dumps(results))

        resumer = TournamentResumer(tournament)

        errors = resumer.validate_tournament()

        assert len(errors) > 0
        assert any("No game directories" in err for err in errors)


class TestInterruptedResume:
    """Behavior when a game is interrupted again during resume."""

    @pytest.fixture
    def tournament_with_resumable_game(self, tmp_path: Path) -> Path:
        """Create a tournament with a resumable game."""
        tournament = tmp_path / "match"
        tournament.mkdir()

        results = {
            "match_name": "match",
            "player1": "P1",
            "player2": "P2",
            "start_time": "2024-01-01T00:00:00",
            "total_games": 1,
        }
        (tournament / "results.json").write_text(json.dumps(results))

        game_dir = tournament / "game_001"
        game_dir.mkdir()
        (game_dir / "game.json").write_text(json.dumps(_resumable_game_json()))

        return tournament

    @patch("llm_chess_arena.tournament.resume.resume_game_from_file")
    def test_resume_interrupted_again__archives_and_raises(
        self, mock_resume: MagicMock, tournament_with_resumable_game: Path
    ) -> None:
        """An interrupted-again game raises, keeping its record resumable.

        The game's own auto-save persists progress, so no archive restore
        happens - the pristine original stays in game.json.original only.
        """
        mock_game = MagicMock()
        mock_game.finished = False  # Game was interrupted again
        mock_resume.return_value = mock_game

        resumer = TournamentResumer(tournament_with_resumable_game)
        game_json_path = tournament_with_resumable_game / "game_001" / "game.json"

        with pytest.raises(RuntimeError, match="interrupted again"):
            resumer._resume_single_game(1, game_json_path)

        # Pristine original was archived before the attempt
        archive_path = game_json_path.parent / "game.json.original"
        assert archive_path.exists()

        # game.json still marks the game resumable (mock game wrote nothing)
        current_data = json.loads(game_json_path.read_text())
        assert current_data["termination_metadata"]["resumable"] is True
        assert current_data["game_outcome"]["result"] == "Unfinished"

    @patch.object(TournamentResumer, "_resume_single_game")
    def test_resume__single_game_failure__continues_and_reports(
        self, mock_resume_single: MagicMock, tournament_with_resumable_game: Path
    ) -> None:
        """One failing game must not abort the resume run or leak the lock."""
        mock_resume_single.side_effect = RuntimeError("network died again")

        resumer = TournamentResumer(tournament_with_resumable_game)
        result = resumer.resume()

        # The failure was contained: aggregates regenerated, nothing resumed
        assert result.resumed_games == []
        assert result.total_games == 1
        # Lock released despite the failure
        assert not (tournament_with_resumable_game / ".resume.lock").exists()
        # Canonical results were rewritten and the original archived
        assert (tournament_with_resumable_game / "results.json").exists()
        assert (tournament_with_resumable_game / "results.json.original").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
