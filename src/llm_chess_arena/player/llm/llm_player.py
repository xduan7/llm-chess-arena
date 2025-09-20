"""LLM-backed player that queries models and applies voting heuristics."""

from __future__ import annotations

from collections import Counter
from loguru import logger

from llm_chess_arena.player.base_player import BasePlayer
from llm_chess_arena.player.llm.llm_connector import LLMConnector
from llm_chess_arena.player.llm.llm_move_handler import BaseLLMMoveHandler
from llm_chess_arena.utils import parse_attempted_move_to_uci
from llm_chess_arena.types import PlayerDecisionContext, PlayerDecision, Color
from llm_chess_arena.exceptions import (
    InvalidMoveError,
    IllegalMoveError,
    AmbiguousMoveError,
    ParseMoveError,
)


class LLMPlayer(BasePlayer):
    """Chess player that queries an LLM and applies optional majority voting.

    The player delegates transport responsibility to the connector and relies on
    the move handler for prompt rendering and move extraction.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        color: Color,
        connector: LLMConnector,
        handler: BaseLLMMoveHandler,
        max_move_retries: int = 3,
        num_votes: int = 1,
    ) -> None:
        """Initialize the LLM-backed player.

        Args:
            name: Optional player name override.
            color: Chess side this player controls.
            connector: Interface responsible for communicating with the LLM.
            handler: Component that renders prompts and parses responses.
            max_move_retries: Maximum retries after invalid or illegal moves.
            num_votes: Number of LLM samples to request for majority voting.

        Raises:
            ValueError: If ``num_votes`` is less than one.
        """
        if num_votes < 1:
            raise ValueError(f"`num_votes` must be >= 1, got {num_votes}")

        super().__init__(name or connector.model, color)
        self.connector = connector
        self.handler = handler
        self.max_move_retries = max_move_retries
        self.num_votes = num_votes
        self.last_move_attempts: int = 0
        self.last_move_decision: PlayerDecision | None = None

    def _make_decision(self, context: PlayerDecisionContext) -> PlayerDecision:
        """Produce a move via the LLM, retrying and voting before resigning."""
        # Must initialize prompt outside loop to preserve retry context across iterations
        prompt = self.handler.get_prompt(**context.model_dump())

        self.last_move_attempts = 0
        self.last_move_decision = None

        max_attempts = self.max_move_retries + 1

        for attempt in range(max_attempts):
            fen_preview = f"{context.board_in_fen[:30]}..."
            logger.debug(
                "LLM player {} move attempt {}/{} for position FEN: {}",
                self,
                attempt + 1,
                max_attempts,
                fen_preview,
            )
            try:
                decision = self._get_most_voted_player_decision_from_llm(prompt)
                logger.debug(
                    "LLM returned decision: action={}, move={}",
                    decision.action,
                    decision.attempted_move if decision.action == "move" else "N/A",
                )
                self.last_move_attempts = attempt + 1
                self.last_move_decision = decision

            except (TimeoutError, ConnectionError) as e:
                # Network errors should propagate up - the game/tournament
                # manager should decide how to handle network failures
                logger.error(
                    "LLM player {} network error after {} attempts: {}: {}",
                    self,
                    attempt + 1,
                    e.__class__.__name__,
                    e,
                )
                raise

            try:
                self._validate_player_decision_from_llm(decision, context.board_in_fen)
                logger.info(
                    "LLM player {} successfully generated valid move {} after {} attempt(s)",
                    self,
                    decision.attempted_move,
                    attempt + 1,
                )
                return decision
            except (InvalidMoveError, IllegalMoveError, AmbiguousMoveError) as e:
                # These three exceptions indicate issues with the move (must
                # contain a move and the action must be 'move')
                retry_status = (
                    "Will retry with context"
                    if attempt < max_attempts - 1
                    else "No retries left"
                )
                logger.warning(
                    "LLM player {} attempt {} failed with {}: {}. Invalid move: '{}'. {}",
                    self,
                    attempt + 1,
                    e.__class__.__name__,
                    e,
                    decision.attempted_move,
                    retry_status,
                )
                if attempt < max_attempts - 1:
                    prompt = self.handler.get_retry_prompt(
                        exception_name=e.__class__.__name__,
                        last_prompt=prompt,
                        last_response=getattr(decision, "response", None),
                        last_attempted_move=decision.attempted_move,
                        **context.model_dump(),
                    )
                    logger.debug(
                        "Generated retry prompt with error context for {}",
                        e.__class__.__name__,
                    )
            except NotImplementedError:
                # Unsupported action - no point in retrying
                logger.error(
                    "LLM player {} returned unsupported action '{}', resigning.",
                    self,
                    decision.action,
                )
                resign_decision = PlayerDecision(action="resign")
                self.last_move_decision = resign_decision
                return resign_decision

        # All attempts exhausted - fall back to deterministic move or resign
        self.last_move_attempts = max_attempts
        logger.warning(
            "LLM player {} failed to produce a valid move after {} attempts, resigning.",
            self,
            max_attempts,
        )
        self.last_move_decision = PlayerDecision(action="resign")
        return self.last_move_decision

    def _validate_player_decision_from_llm(
        self,
        decision: PlayerDecision,
        board_in_fen: str,
    ) -> PlayerDecision:
        """Normalize the decision and ensure the move is legal for the position.

        Args:
            decision: Decision returned by the language model handler.
            board_in_fen: FEN string for the position the decision applies to.

        Returns:
            PlayerDecision: Decision with ``attempted_move`` normalized to UCI.

        Raises:
            NotImplementedError: When the decision action is unsupported.
            InvalidMoveError: When the move text is missing or malformed.
            IllegalMoveError: When the move violates the chess rules for the position.
            AmbiguousMoveError: When the move text remains ambiguous after parsing.
        """

        if decision.action == "resign":
            return decision

        if decision.action != "move":
            raise NotImplementedError(
                f"LLM currently only supports 'move' and 'resign' actions, "
                f"got '{decision.action}'"
            )

        attempted_move = decision.attempted_move
        if attempted_move is None:
            raise InvalidMoveError("LLM move decision missing attempted_move text")

        # Validate and convert move to UCI
        valid_attempted_move = parse_attempted_move_to_uci(attempted_move, board_in_fen)
        decision.attempted_move = valid_attempted_move
        return decision

    def _get_most_voted_player_decision_from_llm(
        self,
        prompt: str,
    ) -> PlayerDecision:
        """Query the LLM and derive a decision using majority voting."""
        responses = self.connector.query(prompt, n=self.num_votes)
        logger.debug(
            "Requested {} response(s) from LLM for majority voting", self.num_votes
        )
        decisions = []

        # Handle parsing errors per response to avoid breaking the entire voting
        for idx, response in enumerate(responses):
            try:
                decision = self.handler.parse_decision_from_response(response)
                if decision is not None:
                    decisions.append(decision)
                    logger.debug(
                        "Vote {}/{}: Parsed move '{}' from response",
                        idx + 1,
                        len(responses),
                        decision.attempted_move,
                    )
            except ParseMoveError as e:
                logger.warning(
                    "Vote {}/{}: Failed to parse LLM response: {}",
                    idx + 1,
                    len(responses),
                    e,
                )
                # Continue with other responses instead of crashing

        logger.debug(
            "Successfully parsed {}/{} responses for voting",
            len(decisions),
            len(responses),
        )

        if not decisions:
            # All responses failed to parse, in which case we create a
            # fake decision to trigger a retry
            logger.error("All LLM responses failed to parse, triggering retry ...")
            return PlayerDecision(
                action="move",
                attempted_move="???",  # Invalid move to trigger retry
                response=responses[0],  # Use first response for context for retry
            )

        # Majority voting implementation:
        # 1. Convert decisions to tuples (action, attempted_move) for counting
        #    since Pydantic models aren't hashable
        # 2. Count occurrences of each unique decision
        # 3. Select the most common decision
        # 4. Ties are broken by first occurrence (Counter preserves insertion order)
        decision_tuples = [(d.action, d.attempted_move) for d in decisions]
        vote_counts = Counter(decision_tuples)
        most_voted_tuple, vote_count = vote_counts.most_common(1)[0]

        # Check for ties and log them explicitly
        ties = [t for t, c in vote_counts.items() if c == vote_count]
        if len(ties) > 1:
            logger.debug(
                "Tie in voting between {} options with {} votes each. Selecting first occurrence: {}",
                len(ties),
                vote_count,
                most_voted_tuple,
            )
        else:
            logger.debug(
                "Majority voting: {}/{} votes for {}",
                vote_count,
                len(decisions),
                most_voted_tuple,
            )

        # Find all decisions that match the winning tuple to preserve metadata
        winning_decisions = [
            d for d in decisions if (d.action, d.attempted_move) == most_voted_tuple
        ]

        # Return the first matching decision, preserving its original metadata
        most_voted_decision = winning_decisions[0]

        # Log response texts from all winning votes for debugging
        response_value = getattr(most_voted_decision, "response", None)
        if response_value is not None:
            all_responses: list[str] = []
            for decision_candidate in winning_decisions:
                candidate_response = getattr(decision_candidate, "response", None)
                if candidate_response:
                    all_responses.append(str(candidate_response))
            if all_responses:
                logger.debug(
                    "Winning decision had {} supporting responses",
                    len(all_responses),
                )

        return most_voted_decision

    def close(self) -> None:
        """Release connector resources when the player is torn down."""
        # Connectors may expose an optional close hook; guard the call.
        if hasattr(self.connector, "close"):
            try:
                self.connector.close()
                logger.debug("LLM connector closed successfully")
            except Exception as e:
                logger.warning("Error closing LLM connector: {}", e)
