"""Tournament state loading and validation for resume functionality."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from llm_chess_arena.exceptions import InvalidGameRecordError

GAME_DIR_PATTERN = re.compile(r"game_(\d+)")


class TournamentLoader:
    """Load tournament state from disk for resume operations."""

    @staticmethod
    def load_results(results_path: Path) -> dict[str, Any]:
        """Load and parse results.json file.

        Args:
            results_path: Path to results.json file.

        Returns:
            Parsed results dictionary.

        Raises:
            FileNotFoundError: If results file doesn't exist.
            InvalidGameRecordError: If JSON is malformed.
        """
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")

        try:
            with results_path.open("r", encoding="utf-8") as f:
                result: dict[str, Any] = json.load(f)
                return result
        except json.JSONDecodeError as e:
            raise InvalidGameRecordError(
                f"Invalid JSON in results file {results_path}: {e}"
            ) from e

    @staticmethod
    def find_all_game_dirs(tournament_dir: Path) -> list[Path]:
        """Find all game_NNN directories in tournament.

        Args:
            tournament_dir: Path to tournament directory.

        Returns:
            List of game directory paths, sorted by game number.
        """
        if not tournament_dir.is_dir():
            return []

        game_dirs = []
        for item in tournament_dir.iterdir():
            if item.is_dir() and GAME_DIR_PATTERN.match(item.name):
                game_dirs.append(item)

        # Sort by game number
        def sort_key(p: Path) -> int:
            match = GAME_DIR_PATTERN.search(p.name)
            return int(match.group(1)) if match else -1

        game_dirs.sort(key=sort_key)
        return game_dirs

    @staticmethod
    def find_resumable_games(tournament_dir: Path) -> list[tuple[int, Path]]:
        """Find games that need to be resumed.

        Criteria for resumable games:
        - Has termination_metadata.resumable=true
        - Result is still "Unfinished" (a completed resume rewrites the result)
        - Has valid JSON structure and player configs for recreation

        Args:
            tournament_dir: Path to tournament directory.

        Returns:
            List of (game_id, game_json_path) tuples for resumable games.
        """
        resumable_games = []
        game_dirs = TournamentLoader.find_all_game_dirs(tournament_dir)

        for game_dir in game_dirs:
            # Extract game ID from directory name (e.g., "game_001" -> 1)
            match = re.search(r"game_(\d+)", game_dir.name)
            if not match:
                continue
            game_id = int(match.group(1))

            game_json = game_dir / "game.json"
            can_resume, reason = TournamentLoader.validate_game_resumable(game_json)

            if can_resume:
                resumable_games.append((game_id, game_json))
            elif reason:
                # Log why game is not resumable (for debugging)
                logger.debug(f"Game {game_id} not resumable: {reason}")

        return resumable_games

    @staticmethod
    def validate_game_resumable(game_json_path: Path) -> tuple[bool, str]:
        """Validate whether a single game can be resumed.

        Checks:
        - File exists and is valid JSON
        - Has termination_metadata.resumable=true
        - game_outcome.result is still "Unfinished" (a game that was resumed
          and interrupted again stays resumable; a completed one does not)
        - Has hydra_config.players.white/black (needed for player recreation)

        Args:
            game_json_path: Path to game.json file.

        Returns:
            Tuple of (can_resume: bool, reason_if_not: str).
            If can_resume is True, reason will be empty string.
        """
        if not game_json_path.exists():
            return False, "file does not exist"

        try:
            with game_json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return False, f"malformed JSON: {e}"
        except Exception as e:
            return False, f"failed to read file: {e}"

        if "termination_metadata" not in data:
            return False, "missing termination_metadata"

        term_meta = data["termination_metadata"]
        if not isinstance(term_meta, dict):
            return False, "termination_metadata is not a dictionary"

        if not term_meta.get("resumable", False):
            error_type = term_meta.get("error_type", "unknown")
            return False, f"not marked as resumable (error: {error_type})"

        result = data.get("game_outcome", {}).get("result")
        if result != "Unfinished":
            return False, f"already finished (result: {result})"

        hydra_config = data.get("hydra_config")
        if not isinstance(hydra_config, dict):
            return False, "missing hydra_config (cannot recreate players)"

        players_cfg = hydra_config.get("players")
        if not isinstance(players_cfg, dict):
            return False, "hydra_config missing players section"

        if "white" not in players_cfg or "black" not in players_cfg:
            return False, "hydra_config.players missing white/black configurations"

        return True, ""
