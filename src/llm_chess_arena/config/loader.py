"""Runtime configuration helpers for the Chess Arena project."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Sequence, cast

from dotenv import find_dotenv, load_dotenv
from hydra import compose, initialize_config_dir
from hydra.errors import HydraException
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from llm_chess_arena.config.schema import (
    AppConfig,
    EnvConfig,
    GameConfig,
    MetricsConfig,
    MoveQualityThresholdsConfig,
    parse_players_config,
)


_ENV_LOADED = False


def load_env(filename: str | None = None, override: bool = False) -> Path | None:
    """Load environment variables from a dotenv file if present.

    Args:
        filename: Path to the dotenv file. Defaults to ENV_FILE environment variable or '.env'.
        override: Whether to override existing environment variables.

    Returns:
        Path to the loaded dotenv file, or None if no file was found or loaded.
    """
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
    """Apply Loguru logging configuration for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    logger.remove()
    logger.add(sys.stderr, level=level.upper())


def apply_env_config(config: EnvConfig) -> None:
    """Load environment settings and configure logging.

    Args:
        config: Environment configuration specifying dotenv loading and log level.
    """
    if config.load_dotenv:
        load_env(config.dotenv_path)

    configure_logging(config.log_level)


def app_config_from_dictconfig(hydra_config: DictConfig) -> AppConfig:
    """Create structured config from Hydra DictConfig.

    Args:
        hydra_config: Composed Hydra configuration containing env, game, metrics, and players.

    Returns:
        Fully populated application configuration.
    """
    env_dict = cast(
        dict[str, Any], OmegaConf.to_container(hydra_config.env, resolve=True)
    )
    game_dict = cast(
        dict[str, Any], OmegaConf.to_container(hydra_config.game, resolve=True)
    )
    metrics_dict = cast(
        dict[str, Any], OmegaConf.to_container(hydra_config.metrics, resolve=True)
    )

    env_config = EnvConfig(**env_dict)
    game_config = GameConfig(**game_dict)

    thresholds_dict = metrics_dict.pop("quality_thresholds", {})
    quality_thresholds = MoveQualityThresholdsConfig(**thresholds_dict)
    metrics_config = MetricsConfig(
        quality_thresholds=quality_thresholds, **metrics_dict
    )

    players_dict = cast(
        dict[str, Any], OmegaConf.to_container(hydra_config.players, resolve=True)
    )
    players_config = parse_players_config(players_dict)

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
    """Compose and validate the application configuration using Hydra.

    Args:
        config_name: Name of the Hydra configuration file to load.
        overrides: List of Hydra override strings for parameter customization.
        config_path: Path to configuration directory. Defaults to 'configs/' in current directory.

    Returns:
        Fully composed and validated application configuration.

    Raises:
        FileNotFoundError: If the configuration directory does not exist.
        HydraException: If Hydra fails to compose the configuration.
    """
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


__all__ = [
    "app_config_from_dictconfig",
    "apply_env_config",
    "configure_logging",
    "load_app_config",
    "load_env",
]
