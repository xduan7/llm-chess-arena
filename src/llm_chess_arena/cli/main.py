"""Hydra-powered CLI for running chess tournaments with multiple games."""

from __future__ import annotations

from pathlib import Path

from hydra import main
from omegaconf import DictConfig, OmegaConf

from llm_chess_arena.config.loader import apply_env_config, app_config_from_dictconfig
from llm_chess_arena.tournament.types import TournamentConfig
from llm_chess_arena.tournament.executor import TournamentRunner
from llm_chess_arena.tournament.export import ResultsExporter
from llm_chess_arena.renderer import display_tournament_summary
from llm_chess_arena.utils import is_stockfish_available


HYDRA_CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


@main(version_base="1.3", config_path=HYDRA_CONFIG_DIR, config_name="config")  # type: ignore[misc, unused-ignore]
def run_tournament_cli(hydra_config: DictConfig) -> None:
    """Run a tournament with multiple games using the same player matchup.

    Args:
        hydra_config: Composed Hydra configuration for the tournament.
    """
    # Load normal app config (env, game, metrics, players)
    app_config = app_config_from_dictconfig(hydra_config)
    apply_env_config(app_config.env)

    # Check Stockfish availability if metrics enabled
    if app_config.game.enable_metrics and not is_stockfish_available():
        raise RuntimeError(
            "Stockfish not found in PATH but game.enable_metrics=true. "
            "Either install Stockfish or set game.enable_metrics=false to run without move evaluation."
        )

    # Parse tournament-specific settings
    resolved = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError("Expected dict at root of configuration")

    tournament_dict = resolved.get("tournament", {})
    if not isinstance(tournament_dict, dict):
        raise ValueError("Tournament section must be a dict")

    # Create tournament config
    tournament_config = TournamentConfig(
        match_name=tournament_dict.get("match_name", "default_match"),
        num_games=tournament_dict.get("num_games", 10),
        parallel_games=tournament_dict.get("parallel_games", 1),
        rate_limit_rpm=tournament_dict.get("rate_limit_rpm"),
        alternate_colors=tournament_dict.get("alternate_colors", True),
        output_dir=Path(tournament_dict.get("output_dir", "output")),
    )

    # Create and run tournament
    runner = TournamentRunner(
        tournament_config=tournament_config,
        game_config=app_config.game,
        metrics_config=app_config.metrics,
        white_player_config=app_config.players.white,
        black_player_config=app_config.players.black,
    )

    result = runner.run()

    # Export results
    output_dir = tournament_config.output_dir / tournament_config.match_name
    ResultsExporter.export_json(result, output_dir / "results.json")
    ResultsExporter.export_csv(result, output_dir / "results.csv")

    # Display Rich tournament summary (if enabled)
    if tournament_config.display_summary:
        display_tournament_summary(
            match_name=result.match_name,
            player1=result.player1_name,
            player2=result.player2_name,
            total_games=result.total_games,
            player1_wins=result.player1_wins,
            player2_wins=result.player2_wins,
            draws=result.draws,
            duration_seconds=result.duration_seconds,
            total_cost=result.total_cost,
            avg_game_length=result.avg_game_length,
            player1_avg_centipawn_loss=result.player1_avg_centipawn_loss,
            player2_avg_centipawn_loss=result.player2_avg_centipawn_loss,
            player1_avg_thinking_time=result.player1_avg_thinking_time,
            player2_avg_thinking_time=result.player2_avg_thinking_time,
            player1_quality_counts=result.player1_quality_counts,
            player2_quality_counts=result.player2_quality_counts,
        )


if __name__ == "__main__":
    run_tournament_cli()
