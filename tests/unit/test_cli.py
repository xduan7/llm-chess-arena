"""Smoke tests for the Hydra CLI entry point."""

import chess
from omegaconf import OmegaConf

from llm_chess_arena.cli.main import run_tournament_cli


class _StubPlayer:
    """Lightweight stand-in for a player used in CLI smoke tests."""

    def __init__(self, name: str, color: str) -> None:
        """Store metadata describing the fake player."""
        self.name = name
        self.color = color

    def __str__(self) -> str:  # noqa: D401
        """Return the display representation matching BasePlayer semantics."""
        return f"{self.name} ({self.color[0].upper()})"


class _StubGame:
    """Minimal game implementation for isolating the CLI entry point."""

    def __init__(self) -> None:
        """Create deterministic players and a precomputed game outcome."""
        self.white_player = _StubPlayer("WhiteBot", "white")
        self.black_player = _StubPlayer("BlackBot", "black")
        self.metrics_tracker = None
        self._rendered_metrics_summary = False
        self.board = chess.Board()
        self._outcome = chess.Outcome(
            termination=chess.Termination.CHECKMATE,
            winner=chess.WHITE,
        )
        # Provide optional overrides for summaries
        self._termination_label_override = None
        self._termination_note = None

    def play(self, max_num_moves: int | None = None) -> None:  # noqa: ARG002
        """Advance a single move so PGN export has content."""
        self.board.push(chess.Move.from_uci("e2e4"))

    @property
    def outcome(self) -> chess.Outcome:
        """Return the predetermined outcome for CLI assertions."""
        return self._outcome

    @property
    def finished(self) -> bool:
        """Signal that the stubbed game is already complete."""
        return True


def test_run_tournament_cli_smoke(monkeypatch, capsys):
    """The CLI should compose config, run tournament, and emit results."""

    from llm_chess_arena.tournament.types import TournamentResult
    from datetime import datetime, UTC

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "llm_chess_arena.cli.main.apply_env_config", lambda env_cfg: None
    )

    mock_result = TournamentResult(
        match_name="test_match",
        player1_name="Random White",
        player2_name="Random Black",
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC),
        total_games=1,
        player1_wins=0,
        player2_wins=0,
        draws=1,
    )

    class MockRunner:
        def __init__(
            self,
            tournament_cfg,
            game_cfg,
            metrics_cfg,
            white_player_cfg,
            black_player_cfg,
            hydra_cfg=None,
        ):
            captured["tournament_config"] = tournament_cfg
            captured["game_config"] = game_cfg

        def run(self):
            return mock_result

    monkeypatch.setattr("llm_chess_arena.cli.main.TournamentRunner", MockRunner)
    monkeypatch.setattr(
        "llm_chess_arena.cli.main.ResultsExporter.export_json", lambda *args: None
    )
    monkeypatch.setattr(
        "llm_chess_arena.cli.main.ResultsExporter.export_csv", lambda *args: None
    )
    monkeypatch.setattr(
        "llm_chess_arena.cli.main.display_tournament_summary", lambda **kwargs: None
    )

    cfg = OmegaConf.create(
        {
            "env": {"load_dotenv": False, "log_level": "INFO", "dotenv_path": None},
            "game": {
                "display_board": False,
                "display_summary": False,
                "enable_metrics": False,
                "max_num_moves": 12,
                "record_dir": None,
                "record_name": None,
            },
            "metrics": {
                "max_centipawn_loss_per_move": 1000,
                "stockfish_depth": 12,
                "stockfish_binary_path": None,
                "stockfish_engine_options": {},
                "quality_thresholds": {
                    "excellent": 50,
                    "good": 100,
                    "inaccuracy": 200,
                    "mistake": 300,
                },
            },
            "tournament": {
                "match_name": "test_match",
                "num_games": 1,
                "parallel_games": 1,
                "rate_limit_rpm": None,
                "alternate_colors": True,
                "display_summary": False,
                "output_dir": "output",
            },
            "players": {
                "white": {
                    "kind": "random",
                    "name": "Random White",
                    "color": "white",
                    "seed": 11,
                },
                "black": {
                    "kind": "random",
                    "name": "Random Black",
                    "color": "black",
                    "seed": 22,
                },
            },
        }
    )

    run_tournament_cli.__wrapped__(cfg)  # type: ignore[attr-defined]

    assert captured["tournament_config"].match_name == "test_match"
    assert captured["game_config"].max_num_moves == 12


def test_run_tournament_cli_with_llm_config_normalization(monkeypatch, capsys):
    """Test that Hydra config overrides work and LLM config normalization is applied."""

    from llm_chess_arena.tournament.types import TournamentResult
    from datetime import datetime, UTC

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "llm_chess_arena.cli.main.apply_env_config", lambda env_cfg: None
    )

    mock_result = TournamentResult(
        match_name="test_match",
        player1_name="Test LLM",
        player2_name="Random Black",
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC),
        total_games=1,
        player1_wins=0,
        player2_wins=0,
        draws=1,
    )

    class MockRunner:
        def __init__(
            self,
            tournament_cfg,
            game_cfg,
            metrics_cfg,
            white_player_cfg,
            black_player_cfg,
            hydra_cfg=None,
        ):
            captured["white_player_config"] = white_player_cfg

        def run(self):
            return mock_result

    monkeypatch.setattr("llm_chess_arena.cli.main.TournamentRunner", MockRunner)
    monkeypatch.setattr(
        "llm_chess_arena.cli.main.ResultsExporter.export_json", lambda *args: None
    )
    monkeypatch.setattr(
        "llm_chess_arena.cli.main.ResultsExporter.export_csv", lambda *args: None
    )
    monkeypatch.setattr(
        "llm_chess_arena.cli.main.display_tournament_summary", lambda **kwargs: None
    )

    # Mock LiteLLM model registry to test normalization logic
    monkeypatch.setattr(
        "llm_chess_arena.config.schema.resolve_model_limit",
        lambda model: (True, 16384),  # Return recognized model with 16384 token limit
    )

    # Config with LLM player that has None values and fractional max_num_tokens
    cfg = OmegaConf.create(
        {
            "env": {"load_dotenv": False, "log_level": "INFO", "dotenv_path": None},
            "game": {
                "display_board": False,
                "display_summary": False,
                "enable_metrics": False,
                "max_num_moves": 5,
                "record_dir": None,
                "record_name": None,
            },
            "metrics": {
                "max_centipawn_loss_per_move": 1000,
                "stockfish_depth": 10,
                "stockfish_binary_path": None,
                "stockfish_engine_options": {},
                "quality_thresholds": {
                    "excellent": 50,
                    "good": 100,
                    "inaccuracy": 200,
                    "mistake": 300,
                },
            },
            "tournament": {
                "match_name": "test_match",
                "num_games": 1,
                "parallel_games": 1,
                "rate_limit_rpm": None,
                "alternate_colors": True,
                "display_summary": False,
                "output_dir": "output",
            },
            "players": {
                "white": {
                    "kind": "llm",
                    "name": "Test LLM",
                    "color": "white",
                    "max_move_retries": 3,  # Must be specified (no auto-normalization)
                    "num_votes": 1,  # Must be specified (no auto-normalization)
                    "connector": {
                        "model": "gpt-4o-mini",
                        "temperature": 0.1,
                        "max_num_tokens": 0.5,  # Fractional - should be resolved to 8192
                        "request_timeout_in_seconds": 60.0,
                        "max_api_request_retries": 2,
                        "provider": None,
                        "api_base": None,
                    },
                    "handler": {"kind": "game_arena"},
                },
                "black": {
                    "kind": "random",
                    "name": "Random Black",
                    "color": "black",
                    "seed": 42,
                },
            },
        }
    )

    run_tournament_cli.__wrapped__(cfg)  # type: ignore[attr-defined]

    white_player = captured["white_player_config"]
    assert white_player.kind == "llm"
    assert white_player.max_move_retries == 3  # Passed through from config
    assert white_player.num_votes == 1  # Passed through from config
    assert white_player.connector.max_num_tokens == 8192  # 0.5 * 16384 = 8192
