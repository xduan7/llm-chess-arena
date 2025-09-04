import re
from abc import ABC, abstractmethod
from typing import Optional

from llm_chess_arena.exceptions import ParseMoveError
from llm_chess_arena.types import PlayerDecision


class BaseLLMMoveHandler(ABC):
    """Abstract handler for extracting chess moves from LLM responses."""

    prompt_template: str
    retry_prompt_templates: dict[str, str]

    def parse_decision_from_response(self, response: str, **kwargs) -> PlayerDecision:
        """Parse LLM response into a PlayerDecision without validation.

        This method only extracts the move text from the raw LLM response.
        Move legality validation is handled upstream by the game engine.

        Args:
            response: Raw text response from the LLM.
            **kwargs: Additional context for parsing (unused in base implementation).

        Returns:
            PlayerDecision with action='move' and the extracted move text.

        Raises:
            ParseMoveError: Failed to extract move from response.
        """
        decision_text = self._extract_decision_text(response, **kwargs)
        if decision_text is None or len(decision_text) == 0:
            raise ParseMoveError(
                f"Failed to extract decision from LLM response: {response}"
            )

        return PlayerDecision(
            action="move",
            attempted_move=decision_text,
            response=response,
        )

    def get_prompt(self, **kwargs) -> str:
        """Generate move request prompt with provided context."""
        return self._fill_prompt_template(self.prompt_template, **kwargs)

    def get_retry_prompt(
        self,
        exception_name: str,
        **kwargs,
    ) -> str:
        """Generate retry prompt based on the specific parsing error."""
        if exception_name not in self.retry_prompt_templates:
            raise ValueError(
                f"No retry prompt defined for exception type: {exception_name}"
            )

        retry_prompt_template = self.retry_prompt_templates[exception_name]
        return self._fill_prompt_template(
            retry_prompt_template,
            **kwargs,
        )

    def _fill_prompt_template(self, template: str, **kwargs) -> str:
        """Fill in prompt template with dynamic values.

        Args:
            template: String template with {field} placeholders.
            **kwargs: Values to fill into the template.

        Returns:
            Formatted string with placeholders replaced.

        Raises:
            KeyError: If template requires a field not in kwargs.
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise KeyError(
                f"Template requires field {e} which was not provided. "
                f"Available fields: {list(kwargs.keys())}"
            ) from e

    @abstractmethod
    def _extract_decision_text(self, response: str, **kwargs) -> str | None:
        """Extract move text from LLM response.

        Args:
            response: The full LLM response text.
            **kwargs: Additional context that may be needed for extraction.

        Returns:
            Extracted move text, or None if no valid move found.
        """
        pass


GAME_ARENA_PROMPT_TEMPLATE = """Let's play chess. The current game state in FEN is:
{board_in_fen}
The moves played so far are:
{flattened_move_history_in_uci}
You are playing as player {player_color}.
It is now your turn. Play your strongest move. The move MUST be legal. Reason step by step to come up with your move, then output your final answer in the format "Final Answer: X" where X is your chosen move in standard algebraic notation (e.g., e4, Nf3, O-O)."""

GAME_ARENA_INVALID_MOVE_TEMPLATE = """{last_prompt}

Your previously suggested move was not parsable.
Please think carefully and generate a new and legal move. Your previous response was:
{last_response}"""

GAME_ARENA_ILLEGAL_MOVE_TEMPLATE = """{last_prompt}

Your previously suggested move was: {last_attempted_move}, which is an illegal move.
Please think carefully and generate a new and legal move."""

GAME_ARENA_AMBIGUOUS_MOVE_TEMPLATE = """{last_prompt}

Your previously suggested move was: {last_attempted_move}, which is ambiguous (multiple pieces can make this move).
Please think carefully and generate a new and unambiguous move."""


class GameArenaLLMMoveHandler(BaseLLMMoveHandler):
    """Handler using Game Arena's 'Final Answer: X' pattern.

    See: https://github.com/google-deepmind/game_arena
    """

    prompt_template = GAME_ARENA_PROMPT_TEMPLATE
    retry_prompt_templates = {
        "InvalidMoveError": GAME_ARENA_INVALID_MOVE_TEMPLATE,
        "IllegalMoveError": GAME_ARENA_ILLEGAL_MOVE_TEMPLATE,
        "AmbiguousMoveError": GAME_ARENA_AMBIGUOUS_MOVE_TEMPLATE,
    }

    def get_prompt(self, **kwargs) -> str:
        """Generate move request prompt with provided context.

        Overrides base to add flattened_move_history_in_uci field.
        Follows exactly Game Arena format for reproducibility.
        """
        if "move_history_in_uci" in kwargs:
            move_history_in_uci = kwargs["move_history_in_uci"]
            kwargs["flattened_move_history_in_uci"] = self._flatten_move_history_in_uci(
                move_history_in_uci
            )
        return super().get_prompt(**kwargs)

    def _extract_decision_text(self, response: str, **kwargs) -> str:
        """Extract and sanitize move text from LLM response."""
        raw_move_text = self._extract_raw_move_text(response)
        move_text = self._sanitize_move_text(raw_move_text)
        return move_text

    @staticmethod
    def _extract_raw_move_text(response: str) -> Optional[str]:
        """Extract move text after 'Final Answer:' marker or simple move notation."""
        if not response:
            return None

        # Try multiple variations of the final answer marker
        markers = [
            "Final Answer:",
            "final answer:",
            "Final answer:",
            "The final answer is",
            "the final answer is",
            "My final answer is",
            "my final answer is",
        ]

        index = -1
        marker_len = 0
        response_lower = response.lower()

        for marker in markers:
            marker_lower = marker.lower()
            found_index = response_lower.rfind(marker_lower)
            if found_index != -1:
                index = found_index
                marker_len = len(marker)
                break

        if index == -1:
            # Fallback: If response is very short (< 10 chars) and looks like a move, use it directly
            # This handles models like Gemini that sometimes just return "e4" or "Nf3"
            response_stripped = response.strip()
            if (
                len(response_stripped) <= 10
                and response_stripped
                and " " not in response_stripped
            ):
                # Likely just a move notation like "e4", "Nf3", "O-O", "O-O-O", "exd5", etc.
                if any(c in response_stripped for c in "abcdefghNBRQKO12345678x=+-#"):
                    return response_stripped
            return None

        text_after_marker = response[index + marker_len :]

        # Remove common LLM formatting artifacts (LaTeX, markdown, HTML)
        # Keeping exact Game Arena escape sequences for reproducibility,
        # even though "\boxed{" creates \b control char (should be raw string).
        # Works in practice as LLMs don't output literal \b or \t in chess moves.
        raw_move_text = (
            text_after_marker.strip(" .")
            .replace("$", "")
            .replace("\\boxed{", "")
            .replace("\\text{", "")
            .replace("\boxed{", "")
            .replace("\text{", "")
            .replace("}", "")
            .replace("*", "")
            .replace("`", "")
            .replace("\n", " ")
        )

        raw_move_text = re.sub(r"<.*?>", "", raw_move_text)

        # Handle castling notation with spaces first (e.g., "O - O" or "O - O - O")
        if raw_move_text.strip().upper().replace(" ", "").replace("-", "") in [
            "OO",
            "OOO",
        ]:
            # Keep the castling notation intact but remove spaces
            raw_move_text = raw_move_text.strip().replace(" ", "")
        else:
            # For non-castling moves, take only the first word (prevents 'e4 therefore...' misparsing)
            # Split on whitespace and common punctuation that might follow a move
            parts = re.split(r"[\s,;.!]", raw_move_text.strip())
            raw_move_text = parts[0] if parts else ""

        return raw_move_text

    @staticmethod
    def _sanitize_move_text(raw_move_text: str) -> Optional[str]:
        """Remove move numbers and non-chess punctuation."""
        if not raw_move_text:
            return None

        sanitized_move_text = raw_move_text.strip()

        if sanitized_move_text and sanitized_move_text[0].isdigit():
            match = re.match(r"(\d+)(\.+)\s*(.*)", sanitized_move_text)
            if match:
                sanitized_move_text = match.group(3)
            else:
                return None

        # Strip symbols that python-chess rejects to increase parse success
        for char in [
            ":",
            ".",
            "*",
            ",",
            "&",
            "^",
            "\\",
            "<",
            ">",
            "{",
            "}",
            "[",
            "]",
            "?",
            "!",
        ]:
            sanitized_move_text = sanitized_move_text.replace(char, "")

        # LLMs sometimes output "exd6ep" but python-chess expects just "exd6"
        if sanitized_move_text.endswith("ep"):
            sanitized_move_text = sanitized_move_text[:-2]

        return sanitized_move_text.strip() if sanitized_move_text else None

    @staticmethod
    def _flatten_move_history_in_uci(move_history_in_uci: list[str]) -> str:
        """Format move history into Game Arena style for reproducibility.

        Args:
            move_history_in_uci: List of moves in UCI format.

        Returns:
            String with numbered moves (e.g., "1. e2e4 e7e5 2. g1h3").
        """
        flattened_move_history_in_uci = []
        for i, move_in_uci in enumerate(move_history_in_uci):
            move_num = (i // 2) + 1
            if i % 2 == 0:
                flattened_move_history_in_uci.append(f"{move_num}.")
            flattened_move_history_in_uci.append(move_in_uci)

        return " ".join(flattened_move_history_in_uci)
