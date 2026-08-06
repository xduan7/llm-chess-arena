"""CLI for resuming interrupted tournaments."""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import main
from loguru import logger
from omegaconf import DictConfig

from llm_chess_arena.tournament.loader import TournamentLoader
from llm_chess_arena.tournament.resume import TournamentResumer

HYDRA_CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


@main(version_base="1.3", config_path=HYDRA_CONFIG_DIR, config_name="resume")  # type: ignore[misc, unused-ignore]
def resume_tournament_cli(hydra_cfg: DictConfig) -> None:
    """Resume an interrupted tournament by completing network-failed games.

    This command scans a tournament directory for games that were interrupted
    by network errors, resumes them sequentially, and regenerates tournament
    results.

    Usage:
        python -m llm_chess_arena.cli.resume tournament_dir=/path/to/match
        python -m llm_chess_arena.cli.resume tournament_dir=/path force_unlock=true
        python -m llm_chess_arena.cli.resume tournament_dir=/path validate_only=true

    Args:
        hydra_cfg: Hydra configuration containing:
            - tournament_dir: Path to tournament directory
            - force_unlock: Override stale lock protection (default: False)
            - validate_only: Only validate, don't resume (default: False)
    """
    # Extract configuration
    tournament_dir_str = hydra_cfg.get("tournament_dir")
    if not tournament_dir_str:
        logger.error("tournament_dir parameter is required")
        sys.exit(1)

    tournament_dir = Path(tournament_dir_str).expanduser().resolve()
    force_unlock = hydra_cfg.get("force_unlock", False)
    validate_only = hydra_cfg.get("validate_only", False)

    # Validation
    if not tournament_dir.exists():
        logger.error(f"Tournament directory not found: {tournament_dir}")
        sys.exit(1)

    if not tournament_dir.is_dir():
        logger.error(f"Path is not a directory: {tournament_dir}")
        sys.exit(1)

    # Log configuration
    logger.info(f"Tournament directory: {tournament_dir}")
    if force_unlock:
        logger.warning("Force unlock enabled - will override active locks")
    if validate_only:
        logger.info("Validation mode - will not resume games")

    # Execute resume
    try:
        resumer = TournamentResumer(
            tournament_dir=tournament_dir,
            force_unlock=force_unlock,
            validate_only=validate_only,
        )

        if validate_only:
            # Just validate, don't resume
            errors = resumer.validate_tournament()
            if errors:
                logger.error("Validation failed:")
                for error in errors:
                    logger.error(f"  - {error}")
                sys.exit(1)
            else:
                logger.info("✓ Tournament is valid")
                # Count resumable games
                resumable = TournamentLoader.find_resumable_games(tournament_dir)
                if resumable:
                    logger.info(f"Found {len(resumable)} resumable game(s)")
                    for game_id, _ in resumable:
                        logger.info(f"  - Game {game_id}")
                else:
                    logger.info("No resumable games found (tournament may be complete)")
                return

        # Resume tournament; regenerated results.json/results.csv are written
        # by the resumer itself
        result = resumer.resume()

        logger.info("")
        logger.info("=" * 60)
        logger.info("Resume Complete")
        logger.info("=" * 60)
        if result.resumed_games:
            logger.info(f"Games resumed: {len(result.resumed_games)}")
        else:
            logger.info("No games needed resuming")
        logger.info(f"Total games: {result.total_games}")
        logger.info("")

        # Display results summary
        logger.info(f"Match: {result.match_name}")
        logger.info(f"  {result.player1_name}: {result.player1_wins} wins")
        logger.info(f"  {result.player2_name}: {result.player2_wins} wins")
        logger.info(f"  Draws: {result.draws}")

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Resume error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error during resume: {e}")
        sys.exit(1)


if __name__ == "__main__":
    resume_tournament_cli()
