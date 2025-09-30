"""Hydra-powered CLI for running chess games between configured players."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from hydra import main
from omegaconf import DictConfig

from llm_chess_arena.config import (
    app_config_from_dictconfig,
    apply_env_config,
    format_game_summary,
    run_game_from_config,
)
from llm_chess_arena.utils import is_stockfish_available


def _print_game_summary(lines: Iterable[str]) -> None:
    """Print game completion summary lines to stdout."""
    for summary_line in lines:
        print(summary_line)


HYDRA_CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


@main(version_base="1.3", config_path=HYDRA_CONFIG_DIR, config_name="config")  # type: ignore[misc, unused-ignore]
def run_cli_game(hydra_config: DictConfig) -> None:
    """Run a CLI-configured chess game via Hydra.

    Args:
        hydra_config: Composed Hydra configuration for the application.
    """

    app_config = app_config_from_dictconfig(hydra_config)
    apply_env_config(app_config.env)

    if app_config.game.enable_metrics and not is_stockfish_available():
        raise RuntimeError(
            "Stockfish not found in PATH but game.enable_metrics=true. "
            "Either install Stockfish or set game.enable_metrics=false to run without move evaluation."
        )

    game = run_game_from_config(app_config)
    _print_game_summary(format_game_summary(game))


if __name__ == "__main__":
    run_cli_game()
