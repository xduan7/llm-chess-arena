"""Tournament resume execution with lock management and archival."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from llm_chess_arena.exceptions import InvalidGameRecordError
from llm_chess_arena.factory import resume_game_from_file
from llm_chess_arena.metrics import MetricsTracker, MoveQualityThresholds
from llm_chess_arena.tournament.aggregator import aggregate_tournament_results
from llm_chess_arena.tournament.export import ResultsExporter
from llm_chess_arena.tournament.loader import TournamentLoader, load_game_result
from llm_chess_arena.tournament.types import GameResult, TournamentResult

# Cross-host locks cannot be PID-checked, so they only expire on age. Chess
# games with LLM backoff can legitimately run for hours - keep this generous
# and rely on force_unlock for manual override.
CROSS_HOST_LOCK_MAX_AGE_HOURS = 24


def _parse_utc_timestamp(timestamp_str: str | None) -> datetime | None:
    """Parse an ISO timestamp, normalizing naive values to UTC.

    Stored results may carry naive timestamps; mixing them with aware
    datetimes makes duration arithmetic raise TypeError.
    """
    if not timestamp_str:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp_str)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass
class LockInfo:
    """Resume lock metadata for preventing concurrent operations."""

    pid: int
    timestamp: datetime
    hostname: str
    command: str

    def is_stale(self, max_age_hours: int = CROSS_HOST_LOCK_MAX_AGE_HOURS) -> bool:
        """Check if lock is stale.

        On the same host the process check is authoritative: a lock is stale
        iff its PID is dead, regardless of age (long-running resumes with LLM
        backoff can legitimately exceed any fixed age). For locks created on a
        different host the PID cannot be checked, so age is the only signal.

        Args:
            max_age_hours: Maximum cross-host lock age before considering stale.

        Returns:
            True if lock is stale and can be removed.
        """
        if self.hostname == socket.gethostname():
            try:
                os.kill(self.pid, 0)  # Signal 0 checks existence without killing
            except OSError:
                logger.debug(f"Lock PID {self.pid} is dead")
                return True
            return False

        age = datetime.now(timezone.utc) - self.timestamp
        if age > timedelta(hours=max_age_hours):
            logger.debug(
                f"Cross-host lock is {age.total_seconds() / 3600:.1f} hours old "
                f"(> {max_age_hours})"
            )
            return True

        return False

    def save(self, path: Path) -> None:
        """Write lock file as JSON, failing if the file already exists.

        Args:
            path: Path to write lock file.

        Raises:
            FileExistsError: If another process created the lock first.
        """
        lock_data = {
            "pid": self.pid,
            "timestamp": self.timestamp.isoformat(),
            "hostname": self.hostname,
            "command": self.command,
        }

        # Exclusive create makes the check-then-create race lose loudly
        with path.open("x", encoding="utf-8") as f:
            json.dump(lock_data, f, indent=2)

        logger.debug(f"Created lock file: {path}")

    @staticmethod
    def load(path: Path) -> LockInfo | None:
        """Read lock file, return None if corrupt/missing.

        Args:
            path: Path to lock file.

        Returns:
            LockInfo if valid, None otherwise.
        """
        if not path.exists():
            return None

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            return LockInfo(
                pid=data["pid"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                hostname=data["hostname"],
                command=data["command"],
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Corrupt lock file {path}: {e}")
            return None


class TournamentResumer:
    """Resume interrupted tournament games with safety guarantees."""

    def __init__(
        self,
        tournament_dir: Path,
        force_unlock: bool = False,
        validate_only: bool = False,
    ):
        """Initialize tournament resumer.

        Args:
            tournament_dir: Path to tournament directory (contains game_*/, results.json).
            force_unlock: Override stale lock protection.
            validate_only: Only validate, don't actually resume.
        """
        self.tournament_dir = tournament_dir
        self.force_unlock = force_unlock
        self.validate_only = validate_only
        self.lock_path = tournament_dir / ".resume.lock"

    def resume(self) -> TournamentResult:
        """Execute resume process.

        Steps:
        1. Validate tournament structure
        2. Acquire lock
        3. Find resumable games
        4. Resume each game (with archival); failures leave the game resumable
        5. Regenerate aggregates from ALL game.json files
        6. Write updated results.json / results.csv
        7. Release lock

        Returns:
            Updated TournamentResult with resumed games.

        Raises:
            ValueError: If tournament validation fails.
            RuntimeError: If lock cannot be acquired.
        """
        # Step 1: Validate tournament
        validation_errors = self.validate_tournament()
        if validation_errors:
            error_msg = "Tournament validation failed:\n" + "\n".join(
                f"  - {err}" for err in validation_errors
            )
            raise ValueError(error_msg)

        if self.validate_only:
            logger.info("Validation passed, skipping resume (validate_only=True)")
            # Return empty result for validation mode
            return TournamentResult(
                match_name="validation",
                player1_name="",
                player2_name="",
                start_time=datetime.now(timezone.utc),
            )

        # Step 2: Acquire lock
        self._acquire_lock()

        try:
            # Step 3: Find resumable games
            resumable_games = TournamentLoader.find_resumable_games(self.tournament_dir)
            logger.info(f"Found {len(resumable_games)} resumable games")

            results_path = self.tournament_dir / "results.json"
            original_data = TournamentLoader.load_results(results_path)

            if not resumable_games:
                logger.info("No games to resume, tournament is complete")
                return self._parse_tournament_result(original_data)

            # Step 4: Resume each game; one failure must not abandon the rest
            resumed_game_ids: list[int] = []
            failed_game_ids: list[int] = []
            for idx, (game_id, game_json_path) in enumerate(resumable_games, start=1):
                logger.info(
                    f"Resuming game {game_id} ({idx}/{len(resumable_games)})..."
                )
                try:
                    self._resume_single_game(game_id, game_json_path)
                except Exception as resume_error:
                    logger.error(
                        f"Game {game_id} resume failed: {resume_error} - "
                        "continuing with remaining games"
                    )
                    failed_game_ids.append(game_id)
                    continue
                resumed_game_ids.append(game_id)

            if failed_game_ids:
                logger.warning(
                    f"{len(failed_game_ids)} game(s) could not be completed this "
                    f"run: {failed_game_ids}. Their records remain resumable - "
                    "rerun the resume command to retry."
                )

            # Step 5: Regenerate aggregates from ALL game.json files
            logger.info("Regenerating tournament aggregates from all games...")
            updated_result = self._regenerate_aggregates(
                original_data, resumed_game_ids
            )

            # Step 6: Persist regenerated results canonically
            self._write_results(updated_result)

            return updated_result

        finally:
            # Step 7: Release lock
            self._release_lock()

    def _acquire_lock(self) -> None:
        """Acquire lock with stale detection.

        Raises:
            RuntimeError: If lock cannot be acquired.
        """
        existing_lock = LockInfo.load(self.lock_path)

        if existing_lock:
            if self.force_unlock:
                logger.warning(
                    f"Force unlock enabled, removing lock (PID {existing_lock.pid})"
                )
                self.lock_path.unlink()
            elif existing_lock.is_stale():
                logger.warning(
                    f"Found stale lock (PID {existing_lock.pid} on "
                    f"{existing_lock.hostname}, "
                    f"age {(datetime.now(timezone.utc) - existing_lock.timestamp).total_seconds() / 3600:.1f}h), "
                    "removing..."
                )
                self.lock_path.unlink()
            else:
                raise RuntimeError(
                    f"Resume already in progress (PID {existing_lock.pid} on {existing_lock.hostname}). "
                    "Wait for completion or use --force-unlock if stale."
                )

        lock = LockInfo(
            pid=os.getpid(),
            timestamp=datetime.now(timezone.utc),
            hostname=socket.gethostname(),
            command=" ".join(sys.argv),
        )
        try:
            lock.save(self.lock_path)
        except FileExistsError as lock_race_error:
            raise RuntimeError(
                "Another resume process acquired the lock concurrently. "
                "Wait for it to finish and rerun."
            ) from lock_race_error
        logger.info("Lock acquired")

    def _release_lock(self) -> None:
        """Remove lock file."""
        if self.lock_path.exists():
            self.lock_path.unlink()
            logger.debug("Lock released")

    def validate_tournament(self) -> list[str]:
        """Validate tournament can be resumed.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        if not self.tournament_dir.is_dir():
            errors.append(f"Tournament directory not found: {self.tournament_dir}")
            return errors

        results_path = self.tournament_dir / "results.json"
        if not results_path.exists():
            errors.append("Missing results.json - not a valid tournament directory")

        game_dirs = TournamentLoader.find_all_game_dirs(self.tournament_dir)
        if not game_dirs:
            errors.append("No game directories found")

        return errors

    def _resume_single_game(self, game_id: int, game_json_path: Path) -> GameResult:
        """Resume one game with archival.

        Steps:
        1. Archive the pristine original game.json to game.json.original
           (kept from the first resume attempt onward)
        2. Rebuild the metrics tracker from the stored configuration
        3. Resume the game (players auto-created from hydra_config.players)
           and play to completion; the game auto-saves game.json whether it
           finishes or is interrupted again, so progress is never discarded
        4. Return the new GameResult loaded from the rewritten game.json

        Args:
            game_id: Game identifier.
            game_json_path: Path to game.json file.

        Returns:
            Completed GameResult.

        Raises:
            GameNotResumableError: If game cannot be resumed.
            InvalidGameRecordError: If game record is invalid.
            RuntimeError: If the resumed game is interrupted again.
            OSError: If filesystem operations fail.
        """
        # Step 1: Archive the pristine original exactly once
        archive_path = game_json_path.parent / "game.json.original"
        if not archive_path.exists():
            try:
                shutil.copy2(game_json_path, archive_path)
                logger.debug(f"Archived game {game_id} to {archive_path.name}")
            except OSError as e:
                raise OSError(
                    f"Failed to archive game {game_id} to {archive_path}: {e}. "
                    "This may be due to insufficient disk space or permissions."
                ) from e

        # Step 2: Recover run settings from the stored configuration
        with game_json_path.open("r", encoding="utf-8") as f:
            record_data = json.load(f)
        hydra_config = record_data.get("hydra_config") or {}
        max_num_moves = (hydra_config.get("game") or {}).get("max_num_moves")
        metrics_tracker = self._build_metrics_tracker(hydra_config)

        # Step 3: Resume and play. record_dir/record_name target game.json
        # itself, so Game auto-saves the updated record on completion AND on a
        # repeat interruption (keeping it resumable with the new moves).
        resumed_game = resume_game_from_file(
            record_path=game_json_path,
            white_player=None,  # Recreated from hydra_config.players
            black_player=None,  # Recreated from hydra_config.players
            display_board=False,
            display_summary=False,
            enable_metrics=False,
            metrics_tracker=metrics_tracker,
            record_dir=game_json_path.parent,
            record_name=game_json_path.stem,
        )

        resumed_game.play(max_num_moves=max_num_moves)

        if not resumed_game.finished:
            # The updated record was auto-saved with resumable metadata; the
            # next resume run picks up from the extended move list.
            raise RuntimeError(
                f"Game {game_id} was interrupted again before completion; "
                "progress was saved and the game remains resumable."
            )

        logger.info(f"Game {game_id} resumed and completed")

        # Step 4: Load the new game result from the rewritten record
        return load_game_result(game_json_path, game_id)

    @staticmethod
    def _build_metrics_tracker(hydra_config: dict[str, Any]) -> MetricsTracker | None:
        """Rebuild the metrics tracker from a stored hydra_config, if enabled.

        Returns None (metrics disabled) when the stored config did not enable
        metrics, lacks a metrics section, or Stockfish is unavailable.
        """
        game_cfg = hydra_config.get("game") or {}
        if not game_cfg.get("enable_metrics"):
            return None

        metrics_dict = hydra_config.get("metrics")
        if not isinstance(metrics_dict, dict):
            logger.warning(
                "Stored config enables metrics but has no metrics section - "
                "resuming without move evaluation"
            )
            return None

        try:
            thresholds_dict = metrics_dict.get("quality_thresholds") or {}
            thresholds = (
                MoveQualityThresholds(**thresholds_dict) if thresholds_dict else None
            )
            return MetricsTracker.from_stockfish(
                depth=metrics_dict.get("stockfish_depth", 10),
                binary_path=metrics_dict.get("stockfish_binary_path"),
                engine_options=metrics_dict.get("stockfish_engine_options"),
                thresholds=thresholds,
                max_centipawn_loss=metrics_dict.get("max_centipawn_loss_per_move"),
                require_stockfish=False,
            )
        except Exception as metrics_error:
            logger.warning(
                f"Could not rebuild metrics tracker for resume: {metrics_error} - "
                "resuming without move evaluation"
            )
            return None

    def _regenerate_aggregates(
        self, original_results: dict[str, Any], resumed_game_ids: list[int]
    ) -> TournamentResult:
        """Rebuild TournamentResult from ALL game.json files.

        Preserves match_name, player names, and start_time from the original
        results; delegates the statistics to the shared tournament aggregator
        so resumed and freshly-run tournaments are counted identically.

        Args:
            original_results: Original results.json data.
            resumed_game_ids: List of game IDs that were resumed.

        Returns:
            New TournamentResult with updated aggregates.
        """
        match_name = original_results.get("match_name", "unknown")
        player1_name = original_results.get("player1", "Player1")
        player2_name = original_results.get("player2", "Player2")

        start_time = _parse_utc_timestamp(
            original_results.get("start_time")
        ) or datetime.now(timezone.utc)

        games: list[GameResult] = []
        for game_dir in TournamentLoader.find_all_game_dirs(self.tournament_dir):
            game_json = game_dir / "game.json"
            if not game_json.exists():
                continue

            match = re.search(r"game_(\d+)", game_dir.name)
            if not match:
                continue
            game_id = int(match.group(1))

            try:
                games.append(load_game_result(game_json, game_id))
            except InvalidGameRecordError as e:
                logger.warning(f"Failed to load game {game_id}: {e}")
                continue

        updated_result = aggregate_tournament_results(
            match_name=match_name,
            results=games,
            start_time=start_time,
            player1_name=player1_name,
            player2_name=player2_name,
        )
        updated_result.resumed_games = resumed_game_ids
        updated_result.resume_timestamp = datetime.now(timezone.utc)
        return updated_result

    def _write_results(self, result: TournamentResult) -> None:
        """Persist regenerated results to the canonical files.

        The pre-resume results.json is archived once as results.json.original
        so the initial aggregates remain inspectable.
        """
        results_path = self.tournament_dir / "results.json"
        results_archive_path = self.tournament_dir / "results.json.original"
        if results_path.exists() and not results_archive_path.exists():
            try:
                shutil.copy2(results_path, results_archive_path)
            except OSError as archive_error:
                logger.warning(
                    f"Could not archive original results.json: {archive_error}"
                )

        ResultsExporter.export_json(result, results_path)
        ResultsExporter.export_csv(result, self.tournament_dir / "results.csv")

    def _parse_tournament_result(self, data: dict[str, Any]) -> TournamentResult:
        """Parse existing results.json into TournamentResult.

        Args:
            data: Parsed results.json dictionary.

        Returns:
            TournamentResult object.
        """
        # Extract basic metadata
        match_name = data.get("match_name", "unknown")
        player1_name = data.get("player1", "Player1")
        player2_name = data.get("player2", "Player2")

        start_time = _parse_utc_timestamp(data.get("start_time")) or datetime.now(
            timezone.utc
        )
        end_time = _parse_utc_timestamp(data.get("end_time"))

        # Extract results
        results = data.get("results", {})
        total_games = data.get("total_games", 0)
        player1_wins = results.get("player1_wins", 0)
        player2_wins = results.get("player2_wins", 0)
        draws = results.get("draws", 0)

        # Extract metrics
        total_cost = data.get("total_cost", 0.0)
        avg_game_length = data.get("avg_game_length", 0.0)

        avg_metrics = data.get("avg_metrics", {})
        player1_avg_centipawn_loss = avg_metrics.get("player1_centipawn_loss")
        player2_avg_centipawn_loss = avg_metrics.get("player2_centipawn_loss")

        thinking_time = data.get("thinking_time", {})
        player1_total_thinking = thinking_time.get("player1_total", 0.0)
        player2_total_thinking = thinking_time.get("player2_total", 0.0)
        player1_avg_per_game = thinking_time.get("player1_avg_per_game", 0.0)
        player2_avg_per_game = thinking_time.get("player2_avg_per_game", 0.0)
        player1_avg_per_move = thinking_time.get("player1_avg_per_move", 0.0)
        player2_avg_per_move = thinking_time.get("player2_avg_per_move", 0.0)

        move_quality = data.get("move_quality", {})
        player1_quality_counts = move_quality.get("player1_counts", {})
        player2_quality_counts = move_quality.get("player2_counts", {})

        return TournamentResult(
            match_name=match_name,
            player1_name=player1_name,
            player2_name=player2_name,
            start_time=start_time,
            end_time=end_time,
            total_games=total_games,
            player1_wins=player1_wins,
            player2_wins=player2_wins,
            draws=draws,
            total_cost=total_cost,
            avg_game_length=avg_game_length,
            player1_avg_centipawn_loss=player1_avg_centipawn_loss,
            player2_avg_centipawn_loss=player2_avg_centipawn_loss,
            player1_total_thinking_time_in_sec=player1_total_thinking,
            player2_total_thinking_time_in_sec=player2_total_thinking,
            player1_avg_thinking_time_per_game_in_sec=player1_avg_per_game,
            player2_avg_thinking_time_per_game_in_sec=player2_avg_per_game,
            player1_avg_thinking_time_per_move_in_sec=player1_avg_per_move,
            player2_avg_thinking_time_per_move_in_sec=player2_avg_per_move,
            player1_quality_counts=player1_quality_counts,
            player2_quality_counts=player2_quality_counts,
        )
