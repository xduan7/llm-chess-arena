"""Smoke tests for the Hydra CLI entry point."""

import chess
from omegaconf import OmegaConf

from llm_chess_arena.cli.main import run_cli_game


class _StubPlayer:
    def __init__(self, name: str, color: str) -> None:
        self.name = name
        self.color = color

    def __str__(self) -> str:  # noqa: D401
        return f"{self.name} ({self.color[0].upper()})"


class _StubGame:
    def __init__(self) -> None:
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
        self.board.push(chess.Move.from_uci("e2e4"))

    @property
    def outcome(self) -> chess.Outcome:
        return self._outcome

    @property
    def finished(self) -> bool:
        return True


def test_run_cli_game_smoke(monkeypatch, capsys):
    """The CLI should compose config, run the game, and emit a summary."""

    stub_game = _StubGame()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "llm_chess_arena.cli.main.apply_env_config", lambda env_cfg: None
    )

    def fake_run_game(app_config):
        captured["app_config"] = app_config
        return stub_game

    monkeypatch.setattr(
        "llm_chess_arena.cli.main.run_game_from_config",
        fake_run_game,
    )

    cfg = OmegaConf.create(
        {
            "env": {"load_dotenv": False, "log_level": "INFO", "dotenv_path": None},
            "game": {
                "display_board": False,
                "enable_metrics": False,
                "max_num_moves": 12,
                "history_output_path": None,
            },
            "metrics": {
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

    run_cli_game.__wrapped__(cfg)  # type: ignore[attr-defined]

    stdout = capsys.readouterr().out
    assert "Outcome:" in stdout
    assert "Termination:" in stdout

    app_config = captured["app_config"]
    assert app_config.game.max_num_moves == 12
