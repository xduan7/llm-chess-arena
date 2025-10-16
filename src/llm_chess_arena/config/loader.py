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
    parse_players_cfg,
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


def apply_env_cfg(env_cfg: EnvConfig) -> None:
    """Load environment settings and configure logging.

    Args:
        env_cfg: Environment configuration specifying dotenv loading and log level.
    """
    if env_cfg.load_dotenv:
        load_env(env_cfg.dotenv_path)

    configure_logging(env_cfg.log_level)


def app_cfg_from_dictconfig(hydra_cfg: DictConfig) -> AppConfig:
    """Create structured configuration from Hydra DictConfig.

    Args:
        hydra_cfg: Composed Hydra configuration containing env, game, metrics, and players.

    Returns:
        Fully populated application configuration.
    """
    env_dict = cast(dict[str, Any], OmegaConf.to_container(hydra_cfg.env, resolve=True))
    game_dict = cast(
        dict[str, Any], OmegaConf.to_container(hydra_cfg.game, resolve=True)
    )
    metrics_dict = cast(
        dict[str, Any], OmegaConf.to_container(hydra_cfg.metrics, resolve=True)
    )

    env_cfg = EnvConfig(**env_dict)
    game_cfg = GameConfig(**game_dict)

    thresholds_dict = metrics_dict.pop("quality_thresholds", {})
    quality_thresholds_cfg = MoveQualityThresholdsConfig(**thresholds_dict)
    metrics_cfg = MetricsConfig(
        quality_thresholds=quality_thresholds_cfg, **metrics_dict
    )

    players_dict = cast(
        dict[str, Any], OmegaConf.to_container(hydra_cfg.players, resolve=True)
    )
    players_cfg = parse_players_cfg(players_dict)

    return AppConfig(
        env=env_cfg,
        game=game_cfg,
        metrics=metrics_cfg,
        players=players_cfg,
    )


def load_app_cfg(
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
        raise FileNotFoundError(f"Configuration directory not found: {configs_dir}")

    try:
        with initialize_config_dir(
            config_dir=str(configs_dir),
            version_base="1.3",
        ):
            hydra_cfg = compose(config_name=config_name, overrides=override_list)
    except HydraException:
        logger.exception("Failed to compose Hydra configuration")
        raise

    return app_cfg_from_dictconfig(hydra_cfg)


__all__ = [
    "app_cfg_from_dictconfig",
    "apply_env_cfg",
    "configure_logging",
    "load_app_cfg",
    "load_env",
]
