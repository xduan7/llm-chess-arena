"""Factories that build configured chess arena components."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, cast

from loguru import logger

from llm_chess_arena.config.schema import normalize_llm_player_cfg, parse_player_cfg
from llm_chess_arena.exceptions import (
    GameNotResumableError,
    InvalidGameRecordError,
)
from llm_chess_arena.game import Game
from llm_chess_arena.policies import config_operation
from llm_chess_arena.metrics import MetricsTracker, MoveQualityThresholds
from llm_chess_arena.player.base_player import BasePlayer
from llm_chess_arena.record import (
    get_record_moves_and_fen,
    load_game_record,
    replay_board_from_record,
)
from llm_chess_arena.player.random_player import RandomPlayer
from llm_chess_arena.player.stockfish_player import StockfishPlayer
from llm_chess_arena.player.llm import (
    GameArenaLLMMoveHandler,
    LLMConnector,
    LLMPlayer,
)

if TYPE_CHECKING:  # pragma: no cover
    from llm_chess_arena.config import (
        LLMConnectorConfig,
        LLMHandlerConfig,
        LLMPlayerConfig,
        MetricsConfig,
        PlayerConfig,
        RandomPlayerConfig,
        StockfishPlayerConfig,
    )
    from llm_chess_arena.rate_limiter import TokenBucketRateLimiter


class PlayerFactory:
    """Factory for instantiating player objects from configuration."""

    @staticmethod
    @config_operation
    def create_player(
        player_cfg: "PlayerConfig", rate_limiter: "TokenBucketRateLimiter | None" = None
    ) -> BasePlayer:
        """Create a player implementation from its configuration dataclass.

        Args:
            player_cfg: Player configuration containing kind and type-specific settings.
            rate_limiter: Optional tournament rate limiter for API throttling (LLM players only).

        Returns:
            BasePlayer: Configured player instance ready for gameplay.

        Raises:
            ValueError: If the player kind is unsupported or configuration is invalid.
        """
        kind = getattr(player_cfg, "kind", None)
        if kind == "random":
            return PlayerFactory._create_random_player(
                cast("RandomPlayerConfig", player_cfg)
            )
        if kind == "stockfish":
            return PlayerFactory._create_stockfish_player(
                cast("StockfishPlayerConfig", player_cfg)
            )
        if kind == "llm":
            return PlayerFactory._create_llm_player(
                cast("LLMPlayerConfig", player_cfg), rate_limiter=rate_limiter
            )
        raise ValueError(f"Unsupported player kind: {kind}")

    @staticmethod
    def _create_random_player(
        random_player_cfg: "RandomPlayerConfig",
    ) -> RandomPlayer:
        """Create random player from configuration."""
        if random_player_cfg.color is None:
            raise ValueError("Random player configuration must have a color assigned")
        name = (
            random_player_cfg.name or f"Random {random_player_cfg.color.capitalize()}"
        )
        return RandomPlayer(
            name=name,
            color=random_player_cfg.color,
            seed=random_player_cfg.seed,
        )

    @staticmethod
    def _create_stockfish_player(
        stockfish_player_cfg: "StockfishPlayerConfig",
    ) -> StockfishPlayer:
        """Create Stockfish player from configuration."""
        if stockfish_player_cfg.color is None:
            raise ValueError(
                "Stockfish player configuration must have a color assigned"
            )
        name = stockfish_player_cfg.name or "Stockfish"
        engine_limits = (
            dict(stockfish_player_cfg.engine_limits)
            if stockfish_player_cfg.engine_limits
            else None
        )
        engine_options = (
            dict(stockfish_player_cfg.engine_options)
            if stockfish_player_cfg.engine_options
            else None
        )
        return StockfishPlayer(
            name=name,
            color=stockfish_player_cfg.color,
            binary_path=stockfish_player_cfg.binary_path,
            engine_limits=engine_limits,
            engine_options=engine_options,
        )

    @staticmethod
    def _create_llm_player(
        llm_player_cfg: "LLMPlayerConfig",
        rate_limiter: "TokenBucketRateLimiter | None" = None,
    ) -> LLMPlayer:
        """Create LLM player from configuration.

        Args:
            llm_player_cfg: LLM player configuration.
            rate_limiter: Optional tournament rate limiter for API throttling.

        Returns:
            LLMPlayer: Configured LLM player instance.
        """
        if llm_player_cfg.color is None:
            raise ValueError("LLM player configuration must have a color assigned")

        connector_cfg = llm_player_cfg.connector
        if connector_cfg is None:
            raise ValueError("LLM player configuration requires connector settings")

        connector = PlayerFactory._create_llm_connector(
            connector_cfg, rate_limiter=rate_limiter
        )
        handler = PlayerFactory._create_llm_handler(llm_player_cfg.handler)
        name = llm_player_cfg.name or connector_cfg.model

        assert llm_player_cfg.max_move_retries is not None
        assert llm_player_cfg.num_votes is not None

        return LLMPlayer(
            name=name,
            color=llm_player_cfg.color,
            connector=connector,
            handler=handler,
            max_move_retries=llm_player_cfg.max_move_retries,
            num_votes=llm_player_cfg.num_votes,
        )

    @staticmethod
    def _create_llm_connector(
        connector_cfg: "LLMConnectorConfig",
        rate_limiter: "TokenBucketRateLimiter | None" = None,
    ) -> LLMConnector:
        """Create LLM connector from configuration.

        Args:
            connector_cfg: Connector configuration.
            rate_limiter: Optional tournament rate limiter for API throttling.

        Returns:
            LLMConnector: Configured connector instance.
        """
        if connector_cfg.model is None:
            raise ValueError("LLM connector requires a model to be specified")

        max_num_tokens = None
        if connector_cfg.max_num_tokens is not None:
            if (
                isinstance(connector_cfg.max_num_tokens, float)
                and connector_cfg.max_num_tokens < 1
            ):
                raise ValueError(
                    f"Fractional max_num_tokens ({connector_cfg.max_num_tokens}) cannot be converted to integer - "
                    f"this indicates a configuration error where token normalization failed"
                )
            max_num_tokens = int(connector_cfg.max_num_tokens)

        return LLMConnector(
            model=connector_cfg.model,
            temperature=connector_cfg.temperature,
            max_num_tokens=max_num_tokens,
            request_timeout_in_seconds=connector_cfg.request_timeout_in_seconds,
            max_api_request_retries=connector_cfg.max_api_request_retries,
            provider=connector_cfg.provider,
            api_base=connector_cfg.api_base,
            rate_limiter=rate_limiter,
        )

    @staticmethod
    def _create_llm_handler(
        handler_cfg: "LLMHandlerConfig" | None,
    ) -> GameArenaLLMMoveHandler:
        """Create LLM move handler from configuration."""
        kind = getattr(handler_cfg, "kind", "game_arena")
        if kind == "game_arena":
            return GameArenaLLMMoveHandler()
        raise ValueError(f"Unsupported LLM handler kind: {kind}")


class MetricsFactory:
    """Create metrics trackers based on configuration."""

    @staticmethod
    @config_operation
    def create_metrics_tracker(
        metrics_cfg: "MetricsConfig",
    ) -> MetricsTracker:
        """Create a Stockfish-based metrics tracker from configuration.

        Args:
            metrics_cfg: Metrics configuration containing Stockfish settings and quality thresholds.

        Returns:
            MetricsTracker: Configured metrics tracker for move evaluation.
        """
        engine_options: Mapping[str, Any] | None = None
        if metrics_cfg.stockfish_engine_options:
            engine_options = dict(metrics_cfg.stockfish_engine_options)

        thresholds_cfg = metrics_cfg.quality_thresholds
        thresholds = MoveQualityThresholds(
            excellent=thresholds_cfg.excellent,
            good=thresholds_cfg.good,
            inaccuracy=thresholds_cfg.inaccuracy,
            mistake=thresholds_cfg.mistake,
        )

        return MetricsTracker.from_stockfish(
            depth=metrics_cfg.stockfish_depth,
            binary_path=metrics_cfg.stockfish_binary_path,
            engine_options=engine_options,
            thresholds=thresholds,
            max_centipawn_loss=metrics_cfg.max_centipawn_loss_per_move,
        )


def resume_game_from_file(
    record_path: str | Path,
    white_player: BasePlayer | None = None,
    black_player: BasePlayer | None = None,
    display_board: bool = True,
    display_summary: bool = True,
    enable_metrics: bool = False,
    metrics_tracker: MetricsTracker | None = None,
    record_dir: str | Path | None = None,
    record_name: str | None = None,
) -> Game:
    """Rebuild a Game from a saved record file so it can continue playing.

    Args:
        record_path: Path to the JSON record file to resume from.
        white_player: Optional white player instance. If None, recreated from
            the record's hydra_config.players section.
        black_player: Optional black player instance. If None, recreated from
            the record's hydra_config.players section.
        display_board: Whether to display the board after each move.
        display_summary: Whether to display game summary at end.
        enable_metrics: Whether to compute move quality metrics.
        metrics_tracker: Optional preconfigured metrics tracker.
        record_dir: Optional directory for saving the resumed game. If None,
            uses the same directory as record_path.
        record_name: Optional custom name for the resumed game record. If
            None, generates one from the original name and a timestamp.

    Returns:
        Game: Resumed game instance ready to continue playing.

    Raises:
        FileNotFoundError: If the record file doesn't exist.
        InvalidGameRecordError: If the record file is malformed or missing required fields.
        GameNotResumableError: If the game is not marked as resumable or is already finished.
        ValueError: If players are not provided and cannot be recreated from config.
    """
    record_path = Path(record_path).expanduser()

    logger.info("Loading game record from {}", record_path)
    record_data = load_game_record(record_path)

    # Validate resumability
    termination_metadata = record_data.get("termination_metadata")
    if termination_metadata is None:
        raise InvalidGameRecordError("Game record missing termination_metadata field")

    if not termination_metadata.get("resumable", False):
        raise GameNotResumableError(
            f"Game is not marked as resumable (resumable={termination_metadata.get('resumable')})"
        )

    game_outcome = record_data.get("game_outcome", {}).get("result")
    if game_outcome is not None and game_outcome != "Unfinished":
        raise GameNotResumableError(
            f"Game is already finished with result: {game_outcome}"
        )

    moves, initial_fen = get_record_moves_and_fen(record_data)

    # Handle player creation/validation
    if white_player is None or black_player is None:
        hydra_cfg = record_data.get("hydra_config") or {}
        players_cfg_dict = hydra_cfg.get("players") or {}
        if not players_cfg_dict:
            raise ValueError(
                "Players must be provided when resuming games whose record "
                "lacks a hydra_config.players section"
            )

        logger.info("Recreating players from saved configuration")
        try:
            if white_player is None:
                white_cfg_dict = players_cfg_dict.get("white")
                if white_cfg_dict is None:
                    raise InvalidGameRecordError(
                        "Missing players.white in hydra_config"
                    )
                white_cfg = normalize_llm_player_cfg(
                    parse_player_cfg(white_cfg_dict, "white"), "white"
                )
                white_player = PlayerFactory.create_player(white_cfg)

            if black_player is None:
                black_cfg_dict = players_cfg_dict.get("black")
                if black_cfg_dict is None:
                    raise InvalidGameRecordError(
                        "Missing players.black in hydra_config"
                    )
                black_cfg = normalize_llm_player_cfg(
                    parse_player_cfg(black_cfg_dict, "black"), "black"
                )
                black_player = PlayerFactory.create_player(black_cfg)

        except Exception as e:
            raise InvalidGameRecordError(
                f"Failed to recreate players from config: {e}"
            ) from e

    logger.info(
        "Resuming game: {} (White) vs {} (Black) from move {}",
        white_player.name,
        black_player.name,
        termination_metadata.get("fullmove_number", "unknown"),
    )

    board = replay_board_from_record(initial_fen, moves)
    logger.info("Restored board position after {} moves", len(board.move_stack))

    if record_dir is None:
        record_dir = record_path.parent

    # Generate resume-specific record name
    if record_name is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        record_name = f"{record_path.stem}-resumed-{timestamp}"

    game = Game(
        white_player=white_player,
        black_player=black_player,
        display_board=display_board,
        display_summary=display_summary,
        enable_metrics=enable_metrics,
        metrics_tracker=metrics_tracker,
        record_dir=record_dir,
        record_name=record_name,
        hydra_cfg=record_data.get("hydra_config", {}),
    )

    game._restore_resumed_state(
        board=board,
        initial_fen=initial_fen,
        moves=moves,
        start_timestamp=record_data.get("environment", {}).get("timestamp_start"),
        resumed_from=record_path,
        original_termination_metadata=termination_metadata,
    )

    logger.info("Game successfully loaded and ready to resume")
    return game
