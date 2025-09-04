from collections import Counter
from typing import Optional
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
    """Chess player that uses an LLM to generate moves.

    Orchestrates between:
    - LLMConnector: Handles communication with the LLM service
    - LLMMoveHandler: Handles prompt generation and response parsing
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        color: Color,
        connector: LLMConnector,
        handler: BaseLLMMoveHandler,
        max_move_retries: int = 3,
        num_votes: int = 1,
    ):
        """Initialize LLM player.

        Args:
            connector: LLM service connector
            handler: Move prompt/parse handler
            color: Chess piece color ('white' or 'black')
            name: Optional player name
            max_move_retries: Maximum retry attempts for invalid/illegal moves
            num_votes: Number of samples of LLM queries for majority voting
        """
        if num_votes < 1:
            raise ValueError(f"`num_votes` must be >= 1, got {num_votes}")

        super().__init__(name or connector.model, color)
        self.connector = connector
        self.handler = handler
        self.max_move_retries = max_move_retries
        self.num_votes = num_votes

    def _make_decision(self, context: PlayerDecisionContext) -> PlayerDecision:
        """Get next move from the LLM with retry logic and majority voting.

        If the move is invalid after all retries, returns resignation.
        This function should handle retries internally and not raise.

        Args:
            context: Current game context

        Returns:
            Player decision (move or resignation)
        """
        # Must initialize prompt outside loop to preserve retry context across iterations
        prompt = self.handler.get_prompt(**context.model_dump())

        for attempt in range(self.max_move_retries + 1):
            logger.debug(
                f"LLM player {self} move attempt {attempt + 1}/{self.max_move_retries + 1} "
                f"for position FEN: {context.board_in_fen[:30]}..."
            )
            try:
                decision = self._get_most_voted_player_decision_from_llm(prompt)
                logger.debug(
                    f"LLM returned decision: action={decision.action}, "
                    f"move={decision.attempted_move if decision.action == 'move' else 'N/A'}"
                )

            except (TimeoutError, ConnectionError) as e:
                # Network errors should propagate up - the game/tournament
                # manager should decide how to handle network failures
                logger.error(
                    f"LLM player {self} network error after {attempt + 1} attempts: "
                    f"{e.__class__.__name__}: {e}"
                )
                raise

            try:
                self._validate_player_decision_from_llm(decision, context.board_in_fen)
                logger.info(
                    f"LLM player {self} successfully generated valid move: {decision.attempted_move} "
                    f"after {attempt + 1} attempt(s)"
                )
                return decision
            except (InvalidMoveError, IllegalMoveError, AmbiguousMoveError) as e:
                # These three exceptions indicate issues with the move (must
                # contains a move and the action must be 'move')
                logger.warning(
                    f"LLM player {self} attempt {attempt + 1} failed with {e.__class__.__name__}: {e}. "
                    f"Invalid move: '{decision.attempted_move}'. "
                    f"{'Will retry with context' if attempt < self.max_move_retries else 'No retries left'}"
                )
                if attempt < self.max_move_retries:
                    prompt = self.handler.get_retry_prompt(
                        exception_name=e.__class__.__name__,
                        last_prompt=prompt,
                        last_response=decision.response,
                        last_attempted_move=decision.attempted_move,
                        **context.model_dump(),
                    )
                    logger.debug(
                        f"Generated retry prompt with error context for {e.__class__.__name__}"
                    )
            except NotImplementedError:
                # Unsupported action - no point in retrying
                logger.error(
                    f"LLM player {self} returned unsupported action '{decision.action}', resigning."
                )
                return PlayerDecision(action="resign")

        # All attempts exhausted - resign
        logger.warning(
            f"LLM player {self} failed to produce a valid move after "
            f"{self.max_move_retries + 1} attempts, resigning."
        )
        return PlayerDecision(action="resign")

    def _validate_player_decision_from_llm(
        self,
        decision: PlayerDecision,
        board_in_fen: str,
    ) -> PlayerDecision:
        """Get the player's action from the LLM.

        If the action is a move, we validate it and convert to UCI format.

        Args:
            prompt: The prompt to send to the LLM
            context: Current game context for validation

        Returns:
            Validated PlayerDecision with move in UCI format if applicable, or resignation.

        Raises:
            NotImplementedError: If LLM returned unsupported action
            InvalidMoveError: If move format is invalid
            IllegalMoveError: If move is illegal in current position
            AmbiguousMoveError: If SAN move is ambiguous in current position

        """

        if decision.action == "resign":
            return decision

        if decision.action != "move":
            raise NotImplementedError(
                f"LLM currently only supports 'move' and 'resign' actions, "
                f"got '{decision.action}'"
            )

        # Validate and convert move to UCI
        valid_attempted_move = parse_attempted_move_to_uci(
            decision.attempted_move, board_in_fen
        )
        decision.attempted_move = valid_attempted_move
        return decision

    def _get_most_voted_player_decision_from_llm(
        self,
        prompt: str,
    ) -> PlayerDecision:
        """Query LLM and parse response into PlayerDecision. (could be invalid or None)

        Raises:
            TimeoutError: If LLM request times out
            ConnectionError: If LLM service is unreachable

        """
        responses = self.connector.query(prompt, n=self.num_votes)
        logger.debug(
            f"Requested {self.num_votes} response(s) from LLM for majority voting"
        )
        decisions = []

        # Handle parsing errors per response to avoid breaking the entire voting
        for idx, response in enumerate(responses):
            try:
                decision = self.handler.parse_decision_from_response(response)
                if decision is not None:
                    decisions.append(decision)
                    logger.debug(
                        f"Vote {idx + 1}/{len(responses)}: Parsed move '{decision.attempted_move}' "
                        f"from response"
                    )
            except ParseMoveError as e:
                logger.warning(
                    f"Vote {idx + 1}/{len(responses)}: Failed to parse LLM response: {e}"
                )
                # Continue with other responses instead of crashing

        logger.debug(
            f"Successfully parsed {len(decisions)}/{len(responses)} responses for voting"
        )

        if not decisions:
            # All responses failed to parse, in which case we create a
            # fake desicion to trigger a retry
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
                f"Tie in voting between {len(ties)} options with {vote_count} votes each. "
                f"Selecting first occurrence: {most_voted_tuple}"
            )
        else:
            logger.debug(
                f"Majority voting: {vote_count}/{len(decisions)} votes for {most_voted_tuple}"
            )

        # Find all decisions that match the winning tuple to preserve metadata
        winning_decisions = [
            d for d in decisions if (d.action, d.attempted_move) == most_voted_tuple
        ]

        # Return the first matching decision, preserving its original metadata
        most_voted_decision = winning_decisions[0]

        # Log response texts from all winning votes for debugging
        if hasattr(most_voted_decision, "response"):
            all_responses = [
                d.response
                for d in winning_decisions
                if hasattr(d, "response") and d.response
            ]
            if all_responses:
                logger.debug(
                    f"Winning decision had {len(all_responses)} supporting responses"
                )

        return most_voted_decision

    def close(self) -> None:
        """Clean up LLM connector resources if needed."""
        if hasattr(self.connector, "close"):
            try:
                self.connector.close()
                logger.debug("LLM connector closed successfully")
            except Exception as e:
                logger.warning(f"Error closing LLM connector: {e}")
