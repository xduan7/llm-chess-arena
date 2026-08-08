"""Configuration schemas and builders for the Chess Arena project."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from loguru import logger

from llm_chess_arena.config.model_registry import (
    DEFAULT_MAX_NUM_TOKENS_RATIO,
    compute_fractional_tokens,
    resolve_model_limit,
)
from llm_chess_arena.types import PlayerColor


@dataclass(slots=True, frozen=True)
class EnvConfig:
    """Settings controlling environment preparation and logging."""

    load_dotenv: bool
    log_level: str
    dotenv_path: str | None = None


@dataclass(slots=True, frozen=True)
class GameConfig:
    """Configuration values for coordinating a chess game."""

    display_board: bool
    display_summary: bool
    enable_metrics: bool
    max_num_moves: int | None = None
    record_dir: str | None = None
    record_name: str | None = None


@dataclass(slots=True, frozen=True)
class MoveQualityThresholdsConfig:
    """Centipawn thresholds controlling move quality categorization."""

    excellent: float
    good: float
    inaccuracy: float
    mistake: float


@dataclass(slots=True, frozen=True)
class MetricsConfig:
    """Configuration for Stockfish-based metrics collection."""

    max_centipawn_loss_per_move: int | None
    stockfish_depth: int
    stockfish_engine_options: Mapping[str, Any]
    quality_thresholds: MoveQualityThresholdsConfig
    stockfish_binary_path: str | None = None


@dataclass(slots=True, frozen=True)
class PlayerConfigBase:
    """Base configuration shared by all player implementations."""

    kind: str
    color: PlayerColor | None = None
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
    temperature: float
    request_timeout_in_seconds: float
    max_api_request_retries: int
    max_num_tokens: int | float | None = None
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
    handler: LLMHandlerConfig = field(default_factory=lambda: LLMHandlerConfig())
    max_move_retries: int | None = None
    num_votes: int | None = None


PlayerConfig = RandomPlayerConfig | StockfishPlayerConfig | LLMPlayerConfig


@dataclass(slots=True, frozen=True)
class PlayersConfig:
    """Configuration for both sides of the board."""

    white: PlayerConfig
    black: PlayerConfig


@dataclass(slots=True, frozen=True)
class AppConfig:
    """Top-level application configuration composed by Hydra."""

    metrics: MetricsConfig
    env: EnvConfig
    game: GameConfig
    players: PlayersConfig


def normalize_llm_player_cfg(
    player_cfg: PlayerConfig, player_color: str
) -> PlayerConfig:
    """Return player configuration with deterministic max_num_tokens handling and defaults.

    Args:
        player_cfg: Player configuration to normalize.
        player_color: Color of the player (for error messages).

    Returns:
        Normalized player configuration with resolved token limits and defaults.

    Raises:
        ValueError: If model is not recognized or token limits cannot be determined.
    """
    if not isinstance(player_cfg, LLMPlayerConfig):
        return player_cfg

    if player_cfg.max_move_retries is None:
        raise ValueError(
            f"max_move_retries must be specified in LLM {player_color} player configuration. "
            "See configs/players/llm/default.yaml for recommended defaults."
        )
    if player_cfg.num_votes is None:
        raise ValueError(
            f"num_votes must be specified in LLM {player_color} player configuration. "
            "See configs/players/llm/default.yaml for recommended defaults."
        )

    normalized_player_cfg = player_cfg

    if normalized_player_cfg.connector is None:
        return normalized_player_cfg

    model_name = normalized_player_cfg.connector.model
    model_recognized, model_token_limit = resolve_model_limit(model_name)

    if not model_recognized:
        raise ValueError(
            f"Model '{model_name}' not recognized by LiteLLM registry for {player_color} player. "
            "Specify a model with known token limits or extend the registry to support this identifier."
        )

    if model_token_limit is None:
        raise ValueError(
            f"Could not determine max output tokens for model '{model_name}' - "
            f"LiteLLM cannot report token limits for this model. "
            f"Please specify max_num_tokens explicitly in the {player_color} player configuration "
            f"or use a model with known token limits."
        )

    connector_cfg = normalized_player_cfg.connector
    max_num_tokens = connector_cfg.max_num_tokens

    if max_num_tokens is None:
        recommended_token_count = compute_fractional_tokens(
            model_token_limit, DEFAULT_MAX_NUM_TOKENS_RATIO
        )
        logger.info(
            "Setting default max_num_tokens to {} for {} player using model '{}' (using {:.3f} of {} token limit)",
            recommended_token_count,
            player_color,
            model_name,
            DEFAULT_MAX_NUM_TOKENS_RATIO,
            model_token_limit,
        )
        connector_cfg = replace(connector_cfg, max_num_tokens=recommended_token_count)
        return replace(normalized_player_cfg, connector=connector_cfg)

    if isinstance(max_num_tokens, float):
        if not 0 < max_num_tokens <= 1:
            raise ValueError(
                f"{player_color} player connector.max_num_tokens ({max_num_tokens}) must be between 0 and 1 when specified as a fraction"
            )
        resolved_token_count = compute_fractional_tokens(
            model_token_limit, max_num_tokens
        )
        logger.info(
            "Resolved fractional max_num_tokens {:.3f} to {} tokens for {} player using model '{}' (limit: {})",
            max_num_tokens,
            resolved_token_count,
            player_color,
            model_name,
            model_token_limit,
        )
        connector_cfg = replace(connector_cfg, max_num_tokens=resolved_token_count)
        return replace(normalized_player_cfg, connector=connector_cfg)

    if max_num_tokens > model_token_limit:
        raise ValueError(
            f"{player_color} player connector.max_num_tokens ({max_num_tokens}) exceeds limit ({model_token_limit}) for model '{model_name}'"
        )

    return normalized_player_cfg


def _ensure_player_color(
    player_cfg: PlayerConfig, default_player_color: PlayerColor
) -> PlayerConfig:
    """Ensure each player configuration declares a color.

    Args:
        player_cfg: Player configuration to update.
        default_player_color: Color to assign if not already set.

    Returns:
        Player configuration with color assigned.
    """
    return replace(player_cfg, color=default_player_color)


def parse_player_cfg(
    raw_cfg: Mapping[str, Any], fallback_player_color: PlayerColor
) -> PlayerConfig:
    """Convert raw player configuration mapping into strongly typed player configuration.

    Args:
        raw_cfg: Raw configuration dictionary from Hydra.
        fallback_player_color: Color to assign if not specified in configuration.

    Returns:
        Typed player configuration (Random, Stockfish, or LLM).

    Raises:
        ValueError: If configuration is invalid or unsupported player kind.
    """
    if not isinstance(raw_cfg, Mapping):
        raise ValueError(
            f"Expected mapping for player configuration, got {type(raw_cfg)}: {raw_cfg}"
        )

    kind = raw_cfg.get("kind")
    player_cfg: PlayerConfig
    if kind == "random":
        player_cfg = RandomPlayerConfig(**raw_cfg)
    elif kind == "stockfish":
        player_cfg = StockfishPlayerConfig(**raw_cfg)
    elif kind == "llm":
        connector_cfg_dict = raw_cfg.get("connector")
        if connector_cfg_dict is None:
            raise ValueError("LLM player configuration requires 'connector' section")

        model_name = connector_cfg_dict.get("model")
        if model_name is None:
            raise ValueError(
                "LLM player connector.model cannot be null. "
                "Specify a model using Hydra overrides: "
                "players.white.connector.model=gpt-4 "
                "or players.black.connector.model=claude-3-5-sonnet-20241022"
            )

        handler_cfg_dict = raw_cfg.get("handler", {})
        connector_cfg = LLMConnectorConfig(**connector_cfg_dict)
        handler_cfg = LLMHandlerConfig(**handler_cfg_dict)
        player_cfg = LLMPlayerConfig(
            **{
                cfg_key: cfg_value
                for cfg_key, cfg_value in raw_cfg.items()
                if cfg_key not in {"connector", "handler"}
            },
            connector=connector_cfg,
            handler=handler_cfg,
        )
    else:
        raise ValueError(f"Unsupported player kind: {kind}")

    return _ensure_player_color(player_cfg, fallback_player_color)


def parse_players_cfg(raw_cfg: Mapping[str, Any]) -> PlayersConfig:
    """Parse both player configurations with enforced colors from raw Hydra mapping.

    Args:
        raw_cfg: Raw configuration dictionary containing white and black player configurations.

    Returns:
        Normalized configuration for both players.

    Raises:
        ValueError: If white or black player configuration is missing.
    """
    white_cfg_dict = raw_cfg.get("white")
    black_cfg_dict = raw_cfg.get("black")
    if white_cfg_dict is None or black_cfg_dict is None:
        raise ValueError(
            "Players configuration requires both 'white' and 'black' sections"
        )

    white_player_cfg = parse_player_cfg(white_cfg_dict, "white")
    black_player_cfg = parse_player_cfg(black_cfg_dict, "black")
    players_cfg = PlayersConfig(white=white_player_cfg, black=black_player_cfg)
    return _normalize_players_cfg(players_cfg)


def _normalize_players_cfg(players_cfg: PlayersConfig) -> PlayersConfig:
    """Apply normalization to both white and black player configurations."""
    return PlayersConfig(
        white=normalize_llm_player_cfg(players_cfg.white, "white"),
        black=normalize_llm_player_cfg(players_cfg.black, "black"),
    )


__all__ = [
    "AppConfig",
    "EnvConfig",
    "GameConfig",
    "LLMConnectorConfig",
    "LLMHandlerConfig",
    "LLMPlayerConfig",
    "MetricsConfig",
    "MoveQualityThresholdsConfig",
    "PlayerConfig",
    "PlayerConfigBase",
    "PlayersConfig",
    "RandomPlayerConfig",
    "StockfishPlayerConfig",
    "DEFAULT_MAX_NUM_TOKENS_RATIO",
    "_normalize_players_cfg",
    "compute_fractional_tokens",
    "normalize_llm_player_cfg",
    "parse_player_cfg",
    "parse_players_cfg",
    "resolve_model_limit",
]
