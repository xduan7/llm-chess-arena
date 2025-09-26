"""Hydra-powered CLI for running chess games between configured players."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from hydra import main
from omegaconf import DictConfig
from loguru import logger

from llm_chess_arena.config import (
    app_config_from_dictconfig,
    apply_env_config,
    format_game_summary,
    run_game_from_config,
)


def _print_game_summary(lines: Iterable[str]) -> None:
    """Print game completion summary lines to stdout."""
    for line in lines:
        print(line)


# Hydra config directory relative to project root
HYDRA_CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


@main(version_base="1.3", config_path=HYDRA_CONFIG_DIR, config_name="config")  # type: ignore[misc, unused-ignore]
def run_cli_game(cfg: DictConfig) -> None:
    """Run a CLI-configured chess game via Hydra.

    Args:
        cfg: Composed Hydra configuration for the application.
    """

    app_config = app_config_from_dictconfig(cfg)
    apply_env_config(app_config.env)

    if cfg.get("metrics") and not shutil.which("stockfish"):
        logger.warning(
            "Stockfish not found in PATH - move quality metrics will be unavailable"
        )

    game = run_game_from_config(app_config)
    _print_game_summary(format_game_summary(game))


if __name__ == "__main__":
    run_cli_game()
