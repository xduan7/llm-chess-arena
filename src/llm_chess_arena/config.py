"""Hydra-backed configuration management for the Chess Arena project."""

from __future__ import annotations

import os
import sys
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import find_dotenv, load_dotenv
from hydra import compose, initialize_config_dir
from hydra.errors import HydraException
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from llm_chess_arena.game import Game
from llm_chess_arena.metrics import MoveQualityThresholds, MetricsTracker
from llm_chess_arena.player.base_player import BasePlayer
from llm_chess_arena.player.llm import (
    GameArenaLLMMoveHandler,
    LLMConnector,
    LLMPlayer,
)
from llm_chess_arena.player.random_player import RandomPlayer
from llm_chess_arena.player.stockfish_player import StockfishPlayer
from llm_chess_arena.types import Color
from llm_chess_arena.utils import build_game_outcome_summary

# Track whether environment has been loaded
_ENV_LOADED = False


@dataclass(slots=True)
class EnvConfig:
    """Settings controlling environment preparation and logging."""

    load_dotenv: bool = True
    dotenv_path: str | None = None
    log_level: str = "INFO"


@dataclass(slots=True)
class GameConfig:
    """Configuration values for coordinating a chess game."""

    display_board: bool = False
    enable_metrics: bool = True
    max_num_moves: int | None = None
    history_output_path: str | None = None


@dataclass(slots=True)
class MoveQualityThresholdsConfig:
    """Centipawn thresholds controlling move quality categorisation."""

    excellent: float = 50.0
    good: float = 100.0
    inaccuracy: float = 200.0
    mistake: float = 300.0


@dataclass(slots=True)
class MetricsConfig:
    """Configuration for Stockfish-based metrics collection."""

    stockfish_depth: int = 10
    stockfish_binary_path: str | None = None
    stockfish_engine_options: Mapping[str, Any] = field(default_factory=dict)
    quality_thresholds: MoveQualityThresholdsConfig = field(
        default_factory=MoveQualityThresholdsConfig
    )


@dataclass(slots=True)
class PlayerConfigBase:
    """Base configuration shared by all player implementations."""

    kind: str
    color: Color = "white"
    name: str | None = None


@dataclass(slots=True)
class RandomPlayerConfig(PlayerConfigBase):
    """Random player configuration."""

    kind: str = "random"
    seed: int | None = None


@dataclass(slots=True)
class StockfishPlayerConfig(PlayerConfigBase):
    """Stockfish-backed player configuration."""

    kind: str = "stockfish"
    binary_path: str | None = None
    engine_limits: Mapping[str, Any] | None = None
    engine_options: Mapping[str, Any] | None = None


@dataclass(slots=True)
class LLMConnectorConfig:
    """Connector parameters for LiteLLM-backed players."""

    model: str
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: float = 30.0
    max_retries: int = 3


@dataclass(slots=True)
class LLMHandlerConfig:
    """Move handler configuration for LLM players."""

    kind: str = "game_arena"


@dataclass(slots=True)
class LLMPlayerConfig(PlayerConfigBase):
    """Language-model-backed player configuration."""

    kind: str = "llm"
    connector: LLMConnectorConfig | None = None
    handler: LLMHandlerConfig = field(default_factory=LLMHandlerConfig)
    max_move_retries: int = 3
    num_votes: int = 1


PlayerConfig = RandomPlayerConfig | StockfishPlayerConfig | LLMPlayerConfig


@dataclass(slots=True)
class PlayersConfig:
    """Configuration for both sides of the board."""

    white: PlayerConfig
    black: PlayerConfig


@dataclass(slots=True)
class AppConfig:
    """Top-level application configuration composed by Hydra."""

    env: EnvConfig = field(default_factory=EnvConfig)
    game: GameConfig = field(default_factory=GameConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    players: PlayersConfig = field(
        default_factory=lambda: PlayersConfig(
            white=RandomPlayerConfig(color="white", name="Random White"),
            black=RandomPlayerConfig(color="black", name="Random Black"),
        )
    )


def load_env(filename: str | None = None, override: bool = False) -> Path | None:
    """Load environment variables from .env file.

    Once loaded, subsequent calls are skipped unless ``override`` is ``True``.
    Tests should use ``override=True`` to reload different configs.

    Args:
        filename: Optional .env filename. Defaults to ``ENV_FILE`` env var or ``.env``.
        override: Whether to override existing environment variables.

    Returns:
        Path | None: Path to the loaded .env file, or ``None`` if not found.
    """
    global _ENV_LOADED

    if _ENV_LOADED and not override:
        return None

    env_file = (
        filename if filename is not None else os.environ.get("ENV_FILE") or ".env"
    )
    dotenv_path = find_dotenv(env_file, usecwd=True)

    if not dotenv_path:
        logger.debug("No .env file found: {}", env_file)
        return None

    load_dotenv(dotenv_path, override=override)
    _ENV_LOADED = True
    logger.debug("Loaded environment from: {}", dotenv_path)
    return Path(dotenv_path)


def _ensure_color(config: PlayerConfig, fallback: Color) -> PlayerConfig:
    """Ensure each player config declares a color.

    Args:
        config: Player configuration instance.
        fallback: Color to assign when not provided.

    Returns:
        PlayerConfig: Updated configuration with color set.
    """

    # Always set the color to the fallback to ensure correct assignment
    object.__setattr__(config, "color", fallback)
    return config


def _parse_player_config(raw: Mapping[str, Any], fallback_color: Color) -> PlayerConfig:
    """Convert a raw mapping into a strongly typed player configuration."""

    kind = raw.get("kind")
    config: PlayerConfig
    if kind == "random":
        config = RandomPlayerConfig(**raw)
    elif kind == "stockfish":
        config = StockfishPlayerConfig(**raw)
    elif kind == "llm":
        connector_data = raw.get("connector")
        if connector_data is None:
            raise ValueError("LLM player config requires 'connector' section")
        handler_data = raw.get("handler", {})
        connector = LLMConnectorConfig(**connector_data)
        handler = LLMHandlerConfig(**handler_data)
        config = LLMPlayerConfig(
            **{k: v for k, v in raw.items() if k not in {"connector", "handler"}},
            connector=connector,
            handler=handler,
        )
    else:
        raise ValueError(f"Unsupported player kind: {kind}")

    return _ensure_color(config, fallback_color)


def _parse_env_config(raw: Mapping[str, Any]) -> EnvConfig:
    """Return structured environment settings."""

    return EnvConfig(**raw)


def _parse_game_config(raw: Mapping[str, Any]) -> GameConfig:
    """Return game-related configuration values."""

    return GameConfig(**raw)


def _parse_metrics_config(raw: Mapping[str, Any]) -> MetricsConfig:
    """Build metrics configuration including quality thresholds."""

    thresholds_raw = raw.get("quality_thresholds", {})
    thresholds = MoveQualityThresholdsConfig(**thresholds_raw)
    metrics_kwargs = {k: v for k, v in raw.items() if k != "quality_thresholds"}
    return MetricsConfig(quality_thresholds=thresholds, **metrics_kwargs)


def _parse_players_config(raw: Mapping[str, Any]) -> PlayersConfig:
    """Return both player configurations with enforced colors."""

    white_raw = raw.get("white")
    black_raw = raw.get("black")
    if white_raw is None or black_raw is None:
        raise ValueError("Players config requires both 'white' and 'black' sections")
    white = _parse_player_config(white_raw, "white")
    black = _parse_player_config(black_raw, "black")
    return PlayersConfig(white=white, black=black)


def app_config_from_dictconfig(cfg: DictConfig) -> AppConfig:
    """Create a structured configuration from a Hydra ``DictConfig``.

    Args:
        cfg: Hydra configuration composed for the application.

    Returns:
        AppConfig: Structured configuration dataclass ready for use within the app.

    Raises:
        ValueError: If the root of the configuration is not a mapping.
    """

    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, Mapping):
        raise ValueError("Expected mapping at root of configuration")

    env_cfg = _parse_env_config(data.get("env", {}))
    game_cfg = _parse_game_config(data.get("game", {}))
    metrics_cfg = _parse_metrics_config(data.get("metrics", {}))
    players_cfg = _parse_players_config(data.get("players", {}))

    return AppConfig(
        env=env_cfg,
        game=game_cfg,
        metrics=metrics_cfg,
        players=players_cfg,
    )


def load_app_config(
    config_name: str = "config",
    overrides: Sequence[str] | None = None,
    *,
    config_path: str | None = None,
) -> AppConfig:
    """Compose and validate the application configuration using Hydra.

    Args:
        config_name: Name of the base config file (without extension).
        overrides: Optional Hydra override strings, e.g., ``["players.white=llm/gpt4"]``.
        config_path: Optional directory containing Hydra configs. Defaults to
            ``<project-root>/configs``.

    Returns:
        AppConfig: Structured configuration for the current run.

    Raises:
        HydraException: Composition errors encountered by Hydra.
        ValueError: Validation errors in the composed config structure.
    """

    overrides_list = list(overrides or [])

    if config_path is not None:
        configs_dir = Path(config_path).resolve()
    else:
        # Use absolute path to configs directory
        configs_dir = Path.cwd() / "configs"

    if not configs_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {configs_dir}")

    try:
        with initialize_config_dir(
            config_dir=str(configs_dir),
            version_base="1.3",
        ):
            cfg = compose(config_name=config_name, overrides=overrides_list)
    except HydraException:
        logger.exception("Failed to compose Hydra configuration")
        raise

    return app_config_from_dictconfig(cfg)


def apply_env_config(config: EnvConfig) -> None:
    """Load environment settings and configure logging.

    Args:
        config: Environment configuration describing dotenv loading and log level.
    """

    if config.load_dotenv:
        load_env(config.dotenv_path)

    _configure_logging(config.log_level)


def build_players(players_config: PlayersConfig) -> tuple[BasePlayer, BasePlayer]:
    """Instantiate players for both sides based on configuration.

    Args:
        players_config: Structured configuration for both white and black players.

    Returns:
        tuple[BasePlayer, BasePlayer]: Instantiated white and black players.
    """

    white_player = _build_player(players_config.white)
    black_player = _build_player(players_config.black)
    return white_player, black_player


def create_metrics_tracker(metrics_config: MetricsConfig) -> MetricsTracker:
    """Construct a metrics tracker using Stockfish settings from config.

    Args:
        metrics_config: Configuration containing engine settings and thresholds.

    Returns:
        MetricsTracker: Tracker ready to evaluate moves during gameplay.
    """

    engine_options: Mapping[str, object] | None = None
    if metrics_config.stockfish_engine_options:
        engine_options = dict(metrics_config.stockfish_engine_options)

    thresholds_cfg = metrics_config.quality_thresholds
    thresholds = MoveQualityThresholds(
        excellent=thresholds_cfg.excellent,
        good=thresholds_cfg.good,
        inaccuracy=thresholds_cfg.inaccuracy,
        mistake=thresholds_cfg.mistake,
    )

    return MetricsTracker.from_stockfish(
        depth=metrics_config.stockfish_depth,
        binary_path=metrics_config.stockfish_binary_path,
        engine_options=engine_options,
        thresholds=thresholds,
    )


def run_game_from_config(app_config: AppConfig) -> Game:
    """Play a single chess game using the provided application configuration.

    Args:
        app_config: Fully structured configuration for environment, players, and metrics.

    Returns:
        Game: Completed game instance containing outcome information.
    """

    white_player, black_player = build_players(app_config.players)
    metrics_tracker = create_metrics_tracker(app_config.metrics)

    game = Game(
        white_player=white_player,
        black_player=black_player,
        display_board=app_config.game.display_board,
        enable_metrics=True,
        metrics_tracker=metrics_tracker,
        history_output_path=app_config.game.history_output_path,
    )

    try:
        game.play(max_num_moves=app_config.game.max_num_moves)
        return game
    finally:
        _close_player(white_player)
        _close_player(black_player)
        metrics_tracker.close()


def format_game_summary(game: Game) -> list[str]:
    """Format a completed game's outcome into readable summary lines.

    Args:
        game: Finished game whose result should be summarized.

    Returns:
        list[str]: Ordered summary lines describing the outcome.
    """

    if not game.finished:
        return ["Game did not finish."]

    if getattr(game, "_rendered_metrics_summary", False):
        return []

    outcome_summary = build_game_outcome_summary(
        outcome=game.outcome,
        white_player_name=str(game.white_player),
        black_player_name=str(game.black_player),
        total_moves=len(game.board.move_stack),
    )

    lines = [
        outcome_summary.outcome_line,
        outcome_summary.termination_line,
        outcome_summary.total_moves_line,
    ]

    if outcome_summary.winner_line:
        lines.insert(2, outcome_summary.winner_line)

    return lines


def _build_player(config: PlayerConfig) -> BasePlayer:
    """Instantiate a player based on its configuration variant."""

    if isinstance(config, RandomPlayerConfig):
        name = config.name or f"Random {config.color.capitalize()}"
        return RandomPlayer(name=name, color=config.color, seed=config.seed)

    if isinstance(config, StockfishPlayerConfig):
        name = config.name or "Stockfish"
        limits = dict(config.engine_limits) if config.engine_limits else None
        options = dict(config.engine_options) if config.engine_options else None
        return StockfishPlayer(
            name=name,
            color=config.color,
            binary_path=config.binary_path,
            engine_limits=limits,
            engine_options=options,
        )

    if isinstance(config, LLMPlayerConfig):
        if config.connector is None:
            raise ValueError("LLM player configuration requires connector settings")

        connector_cfg = config.connector
        connector = LLMConnector(
            model=connector_cfg.model,
            temperature=connector_cfg.temperature,
            max_tokens=connector_cfg.max_tokens,
            timeout=connector_cfg.timeout,
            max_retries=connector_cfg.max_retries,
        )

        handler = _build_llm_handler(config.handler)
        name = config.name or connector_cfg.model
        return LLMPlayer(
            name=name,
            color=config.color,
            connector=connector,
            handler=handler,
            max_move_retries=config.max_move_retries,
            num_votes=config.num_votes,
        )

    raise ValueError(f"Unsupported player configuration: {config}")


def _build_llm_handler(config: LLMHandlerConfig) -> GameArenaLLMMoveHandler:
    """Return the configured LLM move handler implementation."""

    if config.kind == "game_arena":
        return GameArenaLLMMoveHandler()
    raise ValueError(f"Unsupported LLM handler kind: {config.kind}")


def _close_player(player: BasePlayer) -> None:
    """Silently close player resources when supported."""

    with suppress(Exception):
        close = getattr(player, "close", None)
        if close is not None:
            close()


def _configure_logging(level: str) -> None:
    """Apply Loguru logging configuration for the application."""

    normalized_level = level.upper()
    try:
        logger.remove()
    except ValueError:
        pass
    logger.add(sys.stderr, level=normalized_level)
