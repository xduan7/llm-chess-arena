"""Hydra-backed configuration management for the Chess Arena project."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import litellm

from dotenv import find_dotenv, load_dotenv
from hydra import compose, initialize_config_dir
from hydra.errors import HydraException
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from llm_chess_arena.factory import GameFactory
from llm_chess_arena.game import Game
from llm_chess_arena.types import Color
from llm_chess_arena.utils import build_game_outcome_summary


_MODEL_INFO_CACHE: dict[str, Mapping[str, Any]] = {}


def _load_default_max_tokens_ratio() -> float:
    """Read the default completion ratio from the environment with sane fallback."""

    value = os.getenv("LLM_DEFAULT_MAX_TOKENS_RATIO")
    if value is None:
        return 0.8
    try:
        ratio = float(value)
    except ValueError:
        return 0.8
    return ratio if 0 < ratio <= 1 else 0.8


DEFAULT_MAX_TOKENS_RATIO = _load_default_max_tokens_ratio()


BASE_MODEL_OUTPUT_TOKEN_LIMITS: dict[str, int] = {
    # OpenAI GPT legacy + turbo
    "gpt-3.5-turbo": 4_096,
    "gpt-3.5-turbo-16k": 4_096,
    "gpt-4": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-4-turbo": 4_096,
    "gpt-4o": 16_384,
    "gpt-4o-latest": 16_384,
    "gpt-4o-mini": 16_384,
    # OpenAI 4.1 / 5 suite
    "gpt-4.1": 16_384,
    "gpt-4.1-mini": 16_384,
    "gpt-4.1-nano": 16_384,
    "gpt-5": 128_000,
    "gpt-5-mini": 128_000,
    "gpt-5-nano": 128_000,
    # Reasoning models (max_completion_tokens)
    "gpt-o1-preview": 32_768,
    "o1-preview": 32_768,
    "gpt-o1-mini": 65_536,
    "o1-mini": 65_536,
    "gpt-o1": 100_000,
    "o1": 100_000,
    "gpt-o3": 100_000,
    "o3": 100_000,
    "gpt-o3-mini": 100_000,
    "o3-mini": 100_000,
    "gpt-o4-mini": 65_536,
    "o4-mini": 65_536,
    # Gemini
    "gemini-2.5-pro": 65_536,
    "gemini-2.5-flash": 64_536,
    # Anthropic Claude
    "claude-4-opus": 32_000,
    "claude-opus-4": 32_000,
    "claude-4-sonnet": 64_000,
    "claude-sonnet-4": 64_000,
    "claude-3.7-sonnet": 128_000,
    "claude-sonnet-3.7": 128_000,
    "claude-3.5-sonnet-v2": 8_000,
    "claude-sonnet-3.5-v2": 8_000,
    "claude-3-haiku-20240307": 4_096,
    # Embeddings
    "text-embedding-ada-002": 8_191,
    "text-embedding-3-small": 8_191,
    "text-embedding-3-large": 8_191,
}

ARGO_MODEL_CANONICAL_NAMES: dict[str, str] = {
    "argo:gpt-3.5-turbo": "gpt-3.5-turbo",
    "argo:gpt-3.5-turbo-16k": "gpt-3.5-turbo-16k",
    "argo:gpt-4": "gpt-4",
    "argo:gpt-4-32k": "gpt-4-32k",
    "argo:gpt-4-turbo": "gpt-4-turbo",
    "argo:gpt-4o": "gpt-4o",
    "argo:gpt-o1-preview": "gpt-o1-preview",
    "argo:o1-preview": "o1-preview",
    "argo:gpt-4o-latest": "gpt-4o-latest",
    "argo:gpt-o1-mini": "gpt-o1-mini",
    "argo:o1-mini": "o1-mini",
    "argo:gpt-o3-mini": "gpt-o3-mini",
    "argo:o3-mini": "o3-mini",
    "argo:gpt-o1": "gpt-o1",
    "argo:o1": "o1",
    "argo:gpt-o3": "gpt-o3",
    "argo:o3": "o3",
    "argo:gpt-o4-mini": "gpt-o4-mini",
    "argo:o4-mini": "o4-mini",
    "argo:gpt-4.1": "gpt-4.1",
    "argo:gpt-4.1-mini": "gpt-4.1-mini",
    "argo:gpt-4.1-nano": "gpt-4.1-nano",
    "argo:gpt-5": "gpt-5",
    "argo:gpt-5-mini": "gpt-5-mini",
    "argo:gpt-5-nano": "gpt-5-nano",
    "argo:gemini-2.5-pro": "gemini-2.5-pro",
    "argo:gemini-2.5-flash": "gemini-2.5-flash",
    "argo:claude-4-opus": "claude-4-opus",
    "argo:claude-opus-4": "claude-opus-4",
    "argo:claude-4-sonnet": "claude-4-sonnet",
    "argo:claude-sonnet-4": "claude-sonnet-4",
    "argo:claude-3.7-sonnet": "claude-3.7-sonnet",
    "argo:claude-sonnet-3.7": "claude-sonnet-3.7",
    "argo:claude-3.5-sonnet-v2": "claude-3.5-sonnet-v2",
    "argo:claude-sonnet-3.5-v2": "claude-sonnet-3.5-v2",
    "argo:text-embedding-ada-002": "text-embedding-ada-002",
    "argo:text-embedding-3-small": "text-embedding-3-small",
    "argo:text-embedding-3-large": "text-embedding-3-large",
}


MODEL_OUTPUT_TOKEN_LIMITS: dict[str, int] = {
    **BASE_MODEL_OUTPUT_TOKEN_LIMITS,
    **{
        argo_model: BASE_MODEL_OUTPUT_TOKEN_LIMITS[canonical]
        for argo_model, canonical in ARGO_MODEL_CANONICAL_NAMES.items()
        if canonical in BASE_MODEL_OUTPUT_TOKEN_LIMITS
    },
}

_ENV_LOADED = False


@dataclass(slots=True, frozen=True)
class EnvConfig:
    """Settings controlling environment preparation and logging."""

    load_dotenv: bool = True
    dotenv_path: str | None = None
    log_level: str = "INFO"


@dataclass(slots=True, frozen=True)
class GameConfig:
    """Configuration values for coordinating a chess game."""

    display_board: bool = False
    enable_metrics: bool = True
    max_num_moves: int | None = None
    history_output_path: str | None = None


@dataclass(slots=True, frozen=True)
class MoveQualityThresholdsConfig:
    """Centipawn thresholds controlling move quality categorisation."""

    excellent: float = 50.0
    good: float = 100.0
    inaccuracy: float = 200.0
    mistake: float = 300.0


@dataclass(slots=True, frozen=True)
class MetricsConfig:
    """Configuration for Stockfish-based metrics collection."""

    stockfish_depth: int = 10
    stockfish_binary_path: str | None = None
    stockfish_engine_options: Mapping[str, Any] = field(default_factory=dict)
    quality_thresholds: MoveQualityThresholdsConfig = field(
        default_factory=MoveQualityThresholdsConfig
    )


@dataclass(slots=True, frozen=True)
class PlayerConfigBase:
    """Base configuration shared by all player implementations."""

    kind: str
    color: Color = "white"
    name: str | None = None


@dataclass(slots=True, frozen=True)
class RandomPlayerConfig(PlayerConfigBase):
    """Random player configuration."""

    kind: str = "random"
    seed: int | None = None


@dataclass(slots=True, frozen=True)
class StockfishPlayerConfig(PlayerConfigBase):
    """Stockfish-backed player configuration."""

    kind: str = "stockfish"
    binary_path: str | None = None
    engine_limits: Mapping[str, Any] | None = None
    engine_options: Mapping[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class LLMConnectorConfig:
    """Connector parameters for LiteLLM-backed players."""

    model: str | None
    temperature: float = -1
    timeout: float = -1
    max_retries: int = -1
    max_tokens: int | float | None = None
    provider: str | None = None
    api_base: str | None = None


@dataclass(slots=True, frozen=True)
class LLMHandlerConfig:
    """Move handler configuration for LLM players."""

    kind: str = "game_arena"


@dataclass(slots=True, frozen=True)
class LLMPlayerConfig(PlayerConfigBase):
    """Language-model-backed player configuration."""

    kind: str = "llm"
    connector: LLMConnectorConfig | None = None
    handler: LLMHandlerConfig = field(default_factory=LLMHandlerConfig)
    max_move_retries: int = -1
    num_votes: int = -1


PlayerConfig = RandomPlayerConfig | StockfishPlayerConfig | LLMPlayerConfig


@dataclass(slots=True, frozen=True)
class PlayersConfig:
    """Configuration for both sides of the board."""

    white: PlayerConfig
    black: PlayerConfig


@dataclass(slots=True, frozen=True)
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

    return replace(config, color=fallback)


def _get_model_info_cached(model: str) -> Mapping[str, Any]:
    """Return cached LiteLLM model metadata to avoid repeated lookups."""

    if model not in _MODEL_INFO_CACHE:
        # LiteLLM's get_model_info is not always available across installations
        # Use defensive access since mypy cannot detect the dynamic API
        get_model_info = getattr(litellm, "get_model_info", None)
        if get_model_info is None:
            raise AttributeError("litellm.get_model_info not available")
        _MODEL_INFO_CACHE[model] = get_model_info(model)
    return _MODEL_INFO_CACHE[model]


def _resolve_model_limit(model: str | None) -> tuple[bool, Optional[int]]:
    """Identify whether ``model`` is recognised and report its output token limit."""

    if model is None:
        return False, None

    recognized = False
    candidates = [model]
    if model.startswith("argo:"):
        candidates.append(model.split(":", 1)[1])

    for candidate in candidates:
        try:
            info = _get_model_info_cached(candidate)
        except Exception:
            continue
        else:
            recognized = True
            if info is None:
                continue
            limit = info.get("max_output_tokens") or info.get("max_tokens")
            if limit is not None:
                return True, int(limit)

    for candidate in candidates:
        override = MODEL_OUTPUT_TOKEN_LIMITS.get(candidate)
        if override is not None:
            return True, int(override)

    return recognized, None


def _compute_fractional_tokens(limit: int, ratio: float) -> int:
    """Convert ``ratio`` of ``limit`` into a bounded positive integer token count."""

    tokens = int(limit * ratio)
    if tokens <= 0:
        tokens = 1
    if tokens > limit:
        tokens = limit
    return tokens


def _normalize_llm_player_config(
    player_config: PlayerConfig, player_label: str
) -> PlayerConfig:
    """Return ``player_config`` with deterministic ``max_tokens`` handling."""

    if (
        not isinstance(player_config, LLMPlayerConfig)
        or player_config.connector is None
    ):
        return player_config

    model_name = player_config.connector.model
    recognized, output_limit = _resolve_model_limit(model_name)

    if not recognized:
        logger.warning(
            "Model '%s' not recognized by LiteLLM registry; proceeding without validation (%s player)",
            model_name,
            player_label,
        )
        return player_config

    if output_limit is None:
        logger.warning(
            "Could not determine max output tokens for model '%s'; skipping max_tokens validation (%s player)",
            model_name,
            player_label,
        )
        return player_config

    connector = player_config.connector
    max_tokens = connector.max_tokens

    if max_tokens is None:
        recommended = _compute_fractional_tokens(output_limit, DEFAULT_MAX_TOKENS_RATIO)
        logger.info(
            "Setting default max_tokens=%d for model '%s' (%s player) using %.3f of limit %d",
            recommended,
            model_name,
            player_label,
            DEFAULT_MAX_TOKENS_RATIO,
            output_limit,
        )
        connector = replace(connector, max_tokens=recommended)
        return replace(player_config, connector=connector)

    if isinstance(max_tokens, float):
        if not 0 < max_tokens <= 1:
            raise ValueError(
                f"{player_label} player connector.max_tokens ({max_tokens}) must be between 0 and 1 when specified as a fraction"
            )
        resolved = _compute_fractional_tokens(output_limit, max_tokens)
        logger.info(
            "Resolved fractional max_tokens=%.3f to %d for model '%s' (%s player) with limit %d",
            max_tokens,
            resolved,
            model_name,
            player_label,
            output_limit,
        )
        connector = replace(connector, max_tokens=resolved)
        return replace(player_config, connector=connector)

    if max_tokens > output_limit:
        raise ValueError(
            f"{player_label} player connector.max_tokens ({max_tokens}) exceeds limit ({output_limit}) for model '{model_name}'"
        )

    return player_config


def _normalize_players_config(players: PlayersConfig) -> PlayersConfig:
    """Normalize both player configs so white/black share consistent defaults."""

    return PlayersConfig(
        white=_normalize_llm_player_config(players.white, "white"),
        black=_normalize_llm_player_config(players.black, "black"),
    )


def _parse_player_config(raw: Mapping[str, Any], fallback_color: Color) -> PlayerConfig:
    """Convert raw player configuration mapping into strongly typed player config.

    Args:
        raw: Raw configuration mapping from Hydra.
        fallback_color: Color to assign if not specified in config.

    Returns:
        PlayerConfig: Typed player configuration instance.

    Raises:
        ValueError: If player kind is unsupported or LLM config lacks connector.
    """

    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected mapping for player config, got {type(raw)}: {raw}")

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
    """Parse environment configuration from raw Hydra mapping."""
    return EnvConfig(**raw)


def _parse_game_config(raw: Mapping[str, Any]) -> GameConfig:
    """Parse game configuration from raw Hydra mapping."""
    return GameConfig(**raw)


def _parse_metrics_config(raw: Mapping[str, Any]) -> MetricsConfig:
    """Parse metrics configuration with quality thresholds from raw Hydra mapping."""

    thresholds_raw = raw.get("quality_thresholds", {})
    thresholds = MoveQualityThresholdsConfig(**thresholds_raw)
    metrics_kwargs = {k: v for k, v in raw.items() if k != "quality_thresholds"}
    return MetricsConfig(quality_thresholds=thresholds, **metrics_kwargs)


def _parse_players_config(raw: Mapping[str, Any]) -> PlayersConfig:
    """Parse both player configurations with enforced colors from raw Hydra mapping."""

    white_raw = raw.get("white")
    black_raw = raw.get("black")
    if white_raw is None or black_raw is None:
        raise ValueError("Players config requires both 'white' and 'black' sections")
    white = _parse_player_config(white_raw, "white")
    black = _parse_player_config(black_raw, "black")
    players = PlayersConfig(white=white, black=black)
    return _normalize_players_config(players)


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


def run_game_from_config(app_config: AppConfig) -> Game:
    """Play a single chess game using the provided application configuration.

    Args:
        app_config: Fully structured configuration for environment, players, and metrics.

    Returns:
        Game: Completed game instance containing outcome information.
    """

    game = GameFactory.create_game(app_config)

    game.play(max_num_moves=app_config.game.max_num_moves)
    return game


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
        termination_label_override=getattr(game, "_termination_label_override", None),
        termination_note=getattr(game, "_termination_note", None),
    )

    lines = [
        outcome_summary.outcome_line,
        outcome_summary.termination_line,
        outcome_summary.total_moves_line,
    ]

    if outcome_summary.winner_line:
        lines.insert(2, outcome_summary.winner_line)

    return lines


def _configure_logging(level: str) -> None:
    """Apply Loguru logging configuration for the application."""

    normalized_level = level.upper()
    logger.remove()
    logger.add(sys.stderr, level=normalized_level)
