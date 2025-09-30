"""Runtime configuration helpers for the Chess Arena project."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

from dotenv import find_dotenv, load_dotenv
from hydra import compose, initialize_config_dir
from hydra.errors import HydraException
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from llm_chess_arena.config.schema import (
    AppConfig,
    EnvConfig,
    parse_env_config,
    parse_game_config,
    parse_metrics_config,
    parse_players_config,
)
from llm_chess_arena.factory import GameFactory
from llm_chess_arena.game import Game
from llm_chess_arena.utils import build_game_summary


_ENV_LOADED = False


def load_env(filename: str | None = None, override: bool = False) -> Path | None:
    """Load environment variables from a dotenv file if present."""

    global _ENV_LOADED

    if _ENV_LOADED and not override:
        return None

    env_file = (
        filename if filename is not None else os.environ.get("ENV_FILE") or ".env"
    )
    dotenv_path = find_dotenv(env_file, usecwd=True)

    if not dotenv_path:
        logger.debug("No environment configuration file found: {}", env_file)
        return None

    load_dotenv(dotenv_path, override=override)
    _ENV_LOADED = True
    logger.debug("Loaded environment from: {}", dotenv_path)
    return Path(dotenv_path)


def configure_logging(level: str) -> None:
    """Apply Loguru logging configuration for the application."""

    logger.remove()
    logger.add(sys.stderr, level=level.upper())


def apply_env_config(config: EnvConfig) -> None:
    """Load environment settings and configure logging."""

    if config.load_dotenv:
        load_env(config.dotenv_path)

    configure_logging(config.log_level)


def app_config_from_dictconfig(hydra_config: DictConfig) -> AppConfig:
    """Create a structured configuration from a Hydra DictConfig."""

    resolved_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(resolved_config, Mapping):
        raise ValueError("Expected mapping at root of configuration")

    env_config = parse_env_config(resolved_config.get("env", {}))
    game_config = parse_game_config(resolved_config.get("game", {}))
    metrics_config = parse_metrics_config(resolved_config.get("metrics", {}))
    players_config = parse_players_config(resolved_config.get("players", {}))

    return AppConfig(
        env=env_config,
        game=game_config,
        metrics=metrics_config,
        players=players_config,
    )


def load_app_config(
    config_name: str = "config",
    overrides: Sequence[str] | None = None,
    *,
    config_path: str | None = None,
) -> AppConfig:
    """Compose and validate the application configuration using Hydra."""

    override_list = list(overrides or [])

    if config_path is not None:
        configs_dir = Path(config_path).resolve()
    else:
        configs_dir = Path.cwd() / "configs"

    if not configs_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {configs_dir}")

    try:
        with initialize_config_dir(
            config_dir=str(configs_dir),
            version_base="1.3",
        ):
            hydra_config = compose(config_name=config_name, overrides=override_list)
    except HydraException:
        logger.exception("Failed to compose Hydra configuration")
        raise

    return app_config_from_dictconfig(hydra_config)


def run_game_from_config(app_config: AppConfig) -> Game:
    """Play a single chess game using the provided application configuration."""

    game = GameFactory.create_game(app_config)
    game.play(max_num_moves=app_config.game.max_num_moves)
    return game


def format_game_summary(game: Game) -> list[str]:
    """Format a completed game's outcome into readable summary lines."""

    if not game.finished:
        return ["Game did not finish."]

    if game._rendered_metrics_summary:
        return []

    game_summary = build_game_summary(game)
    summary_lines = game_summary.to_cli_lines()

    return summary_lines


__all__ = [
    "app_config_from_dictconfig",
    "apply_env_config",
    "configure_logging",
    "format_game_summary",
    "load_app_config",
    "load_env",
    "run_game_from_config",
]
