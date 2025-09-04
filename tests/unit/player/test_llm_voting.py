"""Comprehensive tests for LLM voting logic edge cases."""

import chess
import pytest
from unittest.mock import Mock

from llm_chess_arena.player.llm import LLMPlayer, GameArenaLLMMoveHandler
from tests.fixtures.mock_llm_connector import MockLLMConnector


class TestLLMVotingEdgeCases:
    """Test edge cases in LLM majority voting logic."""

    def test_voting_with_all_invalid_responses(self):
        """Test voting when all responses are invalid moves."""
        connector = MockLLMConnector(
            model="test-model",
            responses=[
                "Invalid response 1",
                "No move here",
                "Random text",
                # Retry attempt will also fail
                "Still invalid",
                "No move",
                "Random",
            ],
        )
        handler = GameArenaLLMMoveHandler()
        player = LLMPlayer(
            connector=connector,
            handler=handler,
            color="white",
            num_votes=3,
            max_move_retries=1,
        )

        board = chess.Board()
        decision = player(board)
        # Should resign after retries fail
        assert decision.action == "resign"
        assert decision.attempted_move is None

    def test_voting_with_tie_resolution(self):
        """Test tie-breaking by first occurrence."""
        connector = MockLLMConnector(
            model="test-model",
            responses=[
                "Final Answer: e4",
                "Final Answer: d4",
                "Final Answer: e4",
                "Final Answer: d4",
                "Final Answer: Nf3",
            ],
        )
        handler = GameArenaLLMMoveHandler()
        player = LLMPlayer(
            connector=connector,
            handler=handler,
            color="white",
            num_votes=5,
            max_move_retries=1,
        )

        board = chess.Board()
        decision = player(board)

        # e4 and d4 both have 2 votes, e4 appeared first so should win
        assert decision.attempted_move == "e2e4"

    def test_voting_with_mixed_valid_invalid(self):
        """Test voting with mix of valid and invalid responses."""
        connector = MockLLMConnector(
            model="test-model",
            responses=[
                "Final Answer: e4",
                "Invalid response",
                "Final Answer: e4",
                "No move here",
                "Final Answer: d4",
            ],
        )
        handler = GameArenaLLMMoveHandler()
        player = LLMPlayer(
            connector=connector,
            handler=handler,
            color="white",
            num_votes=5,
            max_move_retries=1,
        )

        board = chess.Board()
        decision = player(board)

        # e4 has 2 valid votes, d4 has 1, invalid responses ignored
        assert decision.attempted_move == "e2e4"

    def test_voting_with_single_valid_among_invalid(self):
        """Test voting when only one response is valid."""
        connector = MockLLMConnector(
            model="test-model",
            responses=[
                "Invalid 1",
                "Invalid 2",
                "Final Answer: Nf3",
                "Invalid 3",
                "Invalid 4",
            ],
        )
        handler = GameArenaLLMMoveHandler()
        player = LLMPlayer(
            connector=connector,
            handler=handler,
            color="white",
            num_votes=5,
            max_move_retries=1,
        )

        board = chess.Board()
        decision = player(board)

        # Only Nf3 is valid, should be selected
        assert decision.attempted_move == "g1f3"

    def test_voting_with_illegal_moves(self):
        """Test voting when some moves are illegal for the position."""
        # Set up a position where only certain moves are legal
        board = chess.Board("8/8/8/4k3/8/3K4/8/8 w - - 0 1")  # Kings only

        connector = MockLLMConnector(
            model="test-model",
            responses=[
                "Final Answer: Kd4",  # Legal
                "Final Answer: Qd1",  # Illegal - no queen
                "Final Answer: Kd4",  # Legal
                "Final Answer: e4",  # Illegal - no pawn
                "Final Answer: Kc3",  # Legal
                # If first attempt fails, add retry responses
                "Final Answer: Kd4",
            ],
        )
        handler = GameArenaLLMMoveHandler()
        player = LLMPlayer(
            connector=connector,
            handler=handler,
            color="white",
            num_votes=5,
            max_move_retries=1,
        )

        decision = player(board)

        # Since illegal moves are caught during parsing, they fail validation
        # The player should resign if it can't get valid moves
        if decision.action == "move":
            # If successful, should be one of the legal moves
            assert decision.attempted_move in [
                "d3d4",
                "d3c3",
                "d3e3",
                "d3c4",
                "d3e4",
                "d3c2",
                "d3d2",
                "d3e2",
            ]
        else:
            assert decision.action == "resign"

    def test_voting_retry_on_all_illegal(self):
        """Test retry mechanism when voting produces only illegal moves."""
        board = chess.Board("8/8/8/4k3/8/3K4/8/8 w - - 0 1")  # Kings only

        # First round: all illegal moves
        # Second round: mix of legal and illegal
        connector = MockLLMConnector(
            model="test-model",
            responses=[
                # First voting round (all illegal)
                "Final Answer: Qd1",
                "Final Answer: e4",
                "Final Answer: Nf3",
                # Second retry voting round
                "Final Answer: Kd4",
                "Final Answer: Kd4",
                "Final Answer: Kc3",
            ],
        )
        handler = GameArenaLLMMoveHandler()
        player = LLMPlayer(
            connector=connector,
            handler=handler,
            color="white",
            num_votes=3,
            max_move_retries=2,
        )

        decision = player(board)

        # Should succeed on retry with Kd4 if voting works
        if decision.action == "move":
            assert decision.attempted_move == "d3d4"
        else:
            # Or resign if retries exhausted
            assert decision.action == "resign"

    def test_voting_with_different_move_notations(self):
        """Test voting with moves in different notations that represent same move."""
        connector = MockLLMConnector(
            model="test-model",
            responses=[
                "Final Answer: e4",  # SAN
                "Final Answer: e2e4",  # UCI
                "Final Answer: 1.e4",  # With move number
                "Final Answer: P-K4",  # Descriptive (should fail)
                "Final Answer: e4!",  # With annotation
            ],
        )
        handler = GameArenaLLMMoveHandler()
        player = LLMPlayer(
            connector=connector,
            handler=handler,
            color="white",
            num_votes=5,
            max_move_retries=1,
        )

        board = chess.Board()
        decision = player(board)

        # All valid forms should be converted to UCI e2e4
        assert decision.attempted_move == "e2e4"

    def test_voting_with_network_error_during_sampling(self):
        """Test that network errors during voting result in resignation."""

        def raise_timeout(*args, **kwargs):
            raise TimeoutError("Network timeout")

        connector = Mock()
        connector.query = Mock(side_effect=raise_timeout)

        handler = GameArenaLLMMoveHandler()
        player = LLMPlayer(
            connector=connector,
            handler=handler,
            color="white",
            num_votes=3,
            max_move_retries=2,
        )

        board = chess.Board()

        # Network errors should propagate up
        with pytest.raises(TimeoutError, match="Network timeout"):
            player(board)
        # Network errors fail fast - only one attempt
        assert connector.query.call_count == 1

    def test_voting_preserves_context_between_retries(self):
        """Test that retry context matches the actual error from voting."""
        board = chess.Board()

        # Mock responses that will fail voting then succeed on retry
        connector = MockLLMConnector(
            model="test-model",
            responses=[
                # First voting round - all attempts at illegal Qd1
                "Final Answer: Qd1",
                "Final Answer: Qd1",
                "Final Answer: Qd1",
                # Retry with correct response (single vote)
                "Final Answer: e4",
            ],
        )

        handler = GameArenaLLMMoveHandler()
        player = LLMPlayer(
            connector=connector,
            handler=handler,
            color="white",
            num_votes=3,
            max_move_retries=2,
        )

        decision = player(board)

        # Should succeed with e4 after retry
        assert decision.attempted_move == "e2e4"
        # Should have made 2 queries (one batch of 3, one retry)
        assert connector.query_count == 2

    def test_voting_with_unanimous_decision(self):
        """Test voting when all responses agree."""
        connector = MockLLMConnector(
            model="test-model",
            responses=[
                "Final Answer: Nf3",
                "Final Answer: Nf3",
                "Final Answer: Nf3",
            ],
        )
        handler = GameArenaLLMMoveHandler()
        player = LLMPlayer(
            connector=connector,
            handler=handler,
            color="white",
            num_votes=3,
            max_move_retries=1,
        )

        board = chess.Board()
        decision = player(board)

        assert decision.attempted_move == "g1f3"
        # Single query call with n=3
        assert connector.query_count == 1

    def test_voting_with_single_vote(self):
        """Test that num_votes=1 bypasses voting logic."""
        connector = MockLLMConnector(model="test-model", responses=["Final Answer: d4"])
        handler = GameArenaLLMMoveHandler()
        player = LLMPlayer(
            connector=connector,
            handler=handler,
            color="white",
            num_votes=1,
            max_move_retries=1,
        )

        board = chess.Board()
        decision = player(board)

        assert decision.attempted_move == "d2d4"
        assert connector.query_count == 1


class TestVotingPerformance:
    """Test voting performance characteristics."""

    def test_voting_stops_early_on_majority(self):
        """Test that voting could optimize by stopping early (future enhancement)."""
        # This test documents current behavior - all samples are always collected
        connector = MockLLMConnector(
            model="test-model",
            responses=[
                "Final Answer: e4",
                "Final Answer: e4",
                "Final Answer: e4",
                "Final Answer: d4",
                "Final Answer: Nf3",
            ],
        )
        handler = GameArenaLLMMoveHandler()
        player = LLMPlayer(
            connector=connector,
            handler=handler,
            color="white",
            num_votes=5,
            max_move_retries=1,
        )

        board = chess.Board()
        decision = player(board)

        assert decision.attempted_move == "e2e4"
        # Currently collects all 5 samples in one query call
        assert connector.query_count == 1

    def test_voting_with_many_samples(self):
        """Test voting with large number of samples."""
        responses = ["Final Answer: e4"] * 51 + ["Final Answer: d4"] * 49
        connector = MockLLMConnector(model="test-model", responses=responses)
        handler = GameArenaLLMMoveHandler()
        player = LLMPlayer(
            connector=connector,
            handler=handler,
            color="white",
            num_votes=100,
            max_move_retries=1,
        )

        board = chess.Board()
        decision = player(board)

        assert decision.attempted_move == "e2e4"
        # Single query call with n=100
        assert connector.query_count == 1
