"""Configuration schemas and builders for the Chess Arena project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import litellm
from loguru import logger

from llm_chess_arena.types import Color


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
    color: Color | None = None
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


_MODEL_METADATA_CACHE: dict[str, Mapping[str, Any]] = {}


def _load_default_max_num_tokens_ratio() -> float:
    """Read the default completion ratio from the environment with sane fallback."""

    env_ratio_str = os.getenv("LLM_DEFAULT_MAX_NUM_TOKENS_RATIO")
    if env_ratio_str is None:
        return 0.8
    try:
        parsed_ratio = float(env_ratio_str)
    except ValueError:
        return 0.8
    return parsed_ratio if 0 < parsed_ratio <= 1 else 0.8


DEFAULT_MAX_NUM_TOKENS_RATIO = _load_default_max_num_tokens_ratio()


BASE_MODEL_OUTPUT_TOKEN_LIMITS: dict[str, int] = {
    "gpt-4.1": 16_384,
    "gpt-4.1-mini": 16_384,
    "gpt-4.1-nano": 16_384,
    "gpt-4o": 16_384,
    "gpt-4o-mini": 16_384,
    "gpt-5": 128_000,
    "gpt-5-mini": 128_000,
    "gpt-5-nano": 128_000,
    "gpt-o3": 100_000,
    "o3": 100_000,
    "gpt-o3-mini": 100_000,
    "o3-mini": 100_000,
    "gpt-o4-mini": 65_536,
    "o4-mini": 65_536,
    "claude-4-opus": 32_000,
    "claude-opus-4": 32_000,
    "claude-4-sonnet": 64_000,
    "claude-sonnet-4": 64_000,
    "claude-3.7-sonnet": 128_000,
    "claude-sonnet-3.7": 128_000,
    "claude-3.5-sonnet-v2": 8_000,
    "claude-sonnet-3.5-v2": 8_000,
    "gemini-2.5-pro": 65_536,
    "gemini-2.5-flash": 65_536,
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
        argo_model: BASE_MODEL_OUTPUT_TOKEN_LIMITS[canonical_model_name]
        for argo_model, canonical_model_name in ARGO_MODEL_CANONICAL_NAMES.items()
        if canonical_model_name in BASE_MODEL_OUTPUT_TOKEN_LIMITS
    },
}


def _get_cached_model_info(model: str) -> Mapping[str, Any]:
    """Return cached LiteLLM model metadata to avoid repeated lookups."""

    if model not in _MODEL_METADATA_CACHE:
        _MODEL_METADATA_CACHE[model] = litellm.get_model_info(model)
    return _MODEL_METADATA_CACHE[model]


def resolve_model_limit(model: str | None) -> tuple[bool, int | None]:
    """Identify whether model is recognised and report its output token limit."""

    if model is None:
        return False, None

    model_recognized = False
    model_candidates = [model]
    if model.startswith("argo:"):
        model_candidates.append(model.split(":", 1)[1])

    for candidate_model_name in model_candidates:
        try:
            model_info = _get_cached_model_info(candidate_model_name)
        except Exception:
            continue
        else:
            model_recognized = True
            if model_info is None:
                continue
            token_limit = model_info.get("max_output_tokens") or model_info.get(
                "max_tokens"
            )
            if token_limit is not None:
                logger.debug(
                    "Using LiteLLM token limit for {}: {}",
                    candidate_model_name,
                    token_limit,
                )
                return True, int(token_limit)

    for candidate_model_name in model_candidates:
        fallback_token_limit = MODEL_OUTPUT_TOKEN_LIMITS.get(candidate_model_name)
        if fallback_token_limit is not None:
            logger.debug(
                "Using fallback token limit for {}: {}",
                candidate_model_name,
                fallback_token_limit,
            )
            return True, int(fallback_token_limit)

    return model_recognized, None


def compute_fractional_tokens(token_limit: int, fractional_ratio: float) -> int:
    """Convert fractional ratio of a token limit into a bounded positive count."""

    num_tokens = int(token_limit * fractional_ratio)
    if num_tokens <= 0:
        num_tokens = 1
    if num_tokens > token_limit:
        num_tokens = token_limit
    return num_tokens


def normalize_llm_player_config(
    player_config: PlayerConfig, player_color: str
) -> PlayerConfig:
    """Return player config with deterministic max_num_tokens handling and defaults."""

    if not isinstance(player_config, LLMPlayerConfig):
        return player_config

    normalized_player_config = player_config
    if player_config.max_move_retries is None:
        normalized_player_config = replace(normalized_player_config, max_move_retries=3)
    if player_config.num_votes is None:
        normalized_player_config = replace(normalized_player_config, num_votes=1)

    if normalized_player_config.connector is None:
        return normalized_player_config

    model_name = normalized_player_config.connector.model
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

    connector = normalized_player_config.connector
    max_num_tokens = connector.max_num_tokens

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
        connector = replace(connector, max_num_tokens=recommended_token_count)
        return replace(normalized_player_config, connector=connector)

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
        connector = replace(connector, max_num_tokens=resolved_token_count)
        return replace(normalized_player_config, connector=connector)

    if max_num_tokens > model_token_limit:
        raise ValueError(
            f"{player_color} player connector.max_num_tokens ({max_num_tokens}) exceeds limit ({model_token_limit}) for model '{model_name}'"
        )

    return normalized_player_config


def _ensure_player_color(
    player_config: PlayerConfig, default_color: Color
) -> PlayerConfig:
    """Ensure each player config declares a color."""

    return replace(player_config, color=default_color)


def parse_player_config(
    raw_config: Mapping[str, Any], fallback_color: Color
) -> PlayerConfig:
    """Convert raw player configuration mapping into strongly typed player config."""

    if not isinstance(raw_config, Mapping):
        raise ValueError(
            f"Expected mapping for player config, got {type(raw_config)}: {raw_config}"
        )

    kind = raw_config.get("kind")
    player_config: PlayerConfig
    if kind == "random":
        player_config = RandomPlayerConfig(**raw_config)
    elif kind == "stockfish":
        player_config = StockfishPlayerConfig(**raw_config)
    elif kind == "llm":
        connector_config = raw_config.get("connector")
        if connector_config is None:
            raise ValueError("LLM player config requires 'connector' section")

        model_name = connector_config.get("model")
        if model_name is None:
            raise ValueError(
                "LLM player connector.model cannot be null. "
                "Specify a model using Hydra overrides: "
                "players.white.connector.model=gpt-4 "
                "or players.black.connector.model=claude-3-5-sonnet-20241022"
            )

        handler_config = raw_config.get("handler", {})
        connector = LLMConnectorConfig(**connector_config)
        handler = LLMHandlerConfig(**handler_config)
        player_config = LLMPlayerConfig(
            **{
                config_key: config_value
                for config_key, config_value in raw_config.items()
                if config_key not in {"connector", "handler"}
            },
            connector=connector,
            handler=handler,
        )
    else:
        raise ValueError(f"Unsupported player kind: {kind}")

    return _ensure_player_color(player_config, fallback_color)


def parse_players_config(raw_config: Mapping[str, Any]) -> PlayersConfig:
    """Parse both player configurations with enforced colors from raw Hydra mapping."""

    white_config = raw_config.get("white")
    black_config = raw_config.get("black")
    if white_config is None or black_config is None:
        raise ValueError("Players config requires both 'white' and 'black' sections")

    white_player_config = parse_player_config(white_config, "white")
    black_player_config = parse_player_config(black_config, "black")
    players_config = PlayersConfig(white=white_player_config, black=black_player_config)
    return _normalize_players_config(players_config)


def _normalize_players_config(players_config: PlayersConfig) -> PlayersConfig:
    """Normalize both player configs so white/black share consistent defaults."""

    return PlayersConfig(
        white=normalize_llm_player_config(players_config.white, "white"),
        black=normalize_llm_player_config(players_config.black, "black"),
    )


def register_configs() -> None:
    """No-op placeholder for backwards compatibility.

    Structured config registration removed because configs require YAML values.
    All configuration values must be provided via YAML files.
    """
    pass


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
    "_normalize_players_config",
    "compute_fractional_tokens",
    "normalize_llm_player_config",
    "parse_player_config",
    "parse_players_config",
    "register_configs",
    "resolve_model_limit",
]
