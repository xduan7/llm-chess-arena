import chess
import pytest
from unittest.mock import Mock

from llm_chess_arena.player.llm import (
    LLMPlayer,
    GameArenaLLMMoveHandler,
)
from tests.fixtures.mock_llm_connector import MockLLMConnector


@pytest.fixture
def mock_connector():
    return MockLLMConnector(
        model="test-model",
        responses=["Final Answer: e4", "Final Answer: Nf3", "Final Answer: d4"],
    )


@pytest.fixture
def handler():
    return GameArenaLLMMoveHandler()


@pytest.fixture
def llm_player(mock_connector, handler):
    return LLMPlayer(
        connector=mock_connector,
        handler=handler,
        color="white",
        name="TestPlayer",
        max_move_retries=3,
        num_votes=1,
    )


class TestLLMPlayerInitialization:
    def test_player_stores_all_configuration_parameters_provided_at_initialization(
        self,
    ):
        test_connector = MockLLMConnector()
        game_arena_handler = GameArenaLLMMoveHandler()

        configured_player = LLMPlayer(
            connector=test_connector,
            handler=game_arena_handler,
            color="black",
            name="CustomName",
            max_move_retries=5,
            num_votes=3,
        )

        assert configured_player.connector == test_connector
        assert configured_player.handler == game_arena_handler
        assert configured_player.color == "black"
        assert configured_player.name == "CustomName"
        assert configured_player.max_move_retries == 5
        assert configured_player.num_votes == 3

    def test_initialization_raises_value_error_when_num_votes_is_zero_or_negative(self):
        test_connector = MockLLMConnector()
        game_arena_handler = GameArenaLLMMoveHandler()

        with pytest.raises(ValueError, match="`num_votes` must be >= 1"):
            LLMPlayer(
                connector=test_connector,
                handler=game_arena_handler,
                color="white",
                num_votes=0,
            )

        with pytest.raises(ValueError, match="`num_votes` must be >= 1"):
            LLMPlayer(
                connector=test_connector,
                handler=game_arena_handler,
                color="white",
                num_votes=-1,
            )


class TestLLMPlayerMoveGeneration:
    def test_player_returns_valid_move_decision_from_llm_response(self, llm_player):
        starting_board = chess.Board()
        move_decision = llm_player(starting_board)

        assert move_decision.action == "move"
        assert move_decision.attempted_move == "e2e4"
        assert llm_player.connector.query_count == 1

    def test_player_consumes_llm_responses_in_sequential_order_across_multiple_calls(
        self, llm_player
    ):
        game_board = chess.Board()

        first_move_decision = llm_player(game_board)
        assert first_move_decision.action == "move"
        assert first_move_decision.attempted_move == "e2e4"

        second_move_decision = llm_player(game_board)
        assert second_move_decision.action == "move"
        assert second_move_decision.attempted_move == "g1f3"

        third_move_decision = llm_player(game_board)
        assert third_move_decision.action == "move"
        assert third_move_decision.attempted_move == "d2d4"


class TestLLMPlayerRetryLogic:
    def test_player_resigns_after_exceeding_maximum_retry_attempts(self):
        failing_connector = MockLLMConnector(
            responses=["invalid", "also invalid", "still invalid"]
        )
        game_arena_handler = GameArenaLLMMoveHandler()

        player_with_limited_retries = LLMPlayer(
            connector=failing_connector,
            handler=game_arena_handler,
            color="white",
            max_move_retries=2,
        )

        starting_board = chess.Board()
        resignation_decision = player_with_limited_retries(starting_board)
        assert resignation_decision.action == "resign"

        # Verify retry attempts: initial + 2 retries = 3 queries
        assert failing_connector.query_count == 3

    def test_player_successfully_recovers_on_second_attempt_after_initial_invalid_move(
        self,
    ):
        connector_with_retry_scenario = MockLLMConnector(
            responses=["invalid move", "Final Answer: e4"]
        )
        game_arena_handler = GameArenaLLMMoveHandler()

        player_with_retry_capability = LLMPlayer(
            connector=connector_with_retry_scenario,
            handler=game_arena_handler,
            color="white",
            max_move_retries=3,
        )

        starting_board = chess.Board()
        recovered_move_decision = player_with_retry_capability(starting_board)
        assert recovered_move_decision.action == "move"
        assert recovered_move_decision.attempted_move == "e2e4"
        assert connector_with_retry_scenario.query_count == 2

    def test_retry_prompt_includes_previous_invalid_move_attempt_context(self):
        initial_invalid_response = "I'll play knight to e5"

        connector_with_invalid_first_response = MockLLMConnector(
            responses=[initial_invalid_response, "Final Answer: Nf3"]
        )
        game_arena_handler = GameArenaLLMMoveHandler()

        player_needing_retry = LLMPlayer(
            connector=connector_with_invalid_first_response,
            handler=game_arena_handler,
            color="white",
            max_move_retries=3,
        )

        starting_board = chess.Board()
        corrected_move_decision = player_needing_retry(starting_board)

        assert corrected_move_decision.action == "move"
        assert corrected_move_decision.attempted_move == "g1f3"

        retry_prompt_sent_to_llm = connector_with_invalid_first_response.query_history[
            -1
        ]["prompt"]
        assert (
            "previously suggested move" in retry_prompt_sent_to_llm.lower()
            or "previous response" in retry_prompt_sent_to_llm.lower()
        )
        assert initial_invalid_response in retry_prompt_sent_to_llm

    def test_illegal_move_triggers_retry_with_specific_illegal_move_context(self):
        illegal_move_response = "Final Answer: e5"  # Illegal for white from start
        legal_move_response = "Final Answer: e4"

        connector_with_illegal_then_legal = MockLLMConnector(
            responses=[illegal_move_response, legal_move_response]
        )
        game_arena_handler = GameArenaLLMMoveHandler()

        player_recovering_from_illegal = LLMPlayer(
            connector=connector_with_illegal_then_legal,
            handler=game_arena_handler,
            color="white",
            max_move_retries=3,
        )

        starting_board = chess.Board()
        corrected_move_decision = player_recovering_from_illegal(starting_board)

        assert corrected_move_decision.action == "move"
        assert corrected_move_decision.attempted_move == "e2e4"

        retry_prompt_after_illegal = connector_with_illegal_then_legal.query_history[
            -1
        ]["prompt"]
        assert "illegal" in retry_prompt_after_illegal.lower()
        assert "e5" in retry_prompt_after_illegal


class TestLLMPlayerNetworkErrors:
    def test_network_timeout_error_propagates_immediately_without_chess_retry(self):
        timeout_connector = MockLLMConnector()
        timeout_connector.query = Mock(side_effect=TimeoutError("API timeout"))
        game_arena_handler = GameArenaLLMMoveHandler()

        player_experiencing_timeout = LLMPlayer(
            connector=timeout_connector,
            handler=game_arena_handler,
            color="white",
            max_move_retries=3,
        )

        starting_board = chess.Board()
        with pytest.raises(TimeoutError, match="API timeout"):
            player_experiencing_timeout(starting_board)

        # Network error should fail immediately, not retry
        assert timeout_connector.query.call_count == 1

    def test_connection_error_propagates_immediately_without_chess_retry(self):
        connection_error_connector = MockLLMConnector()
        connection_error_connector.query = Mock(
            side_effect=ConnectionError("Network unavailable")
        )
        game_arena_handler = GameArenaLLMMoveHandler()

        player_with_connection_issue = LLMPlayer(
            connector=connection_error_connector,
            handler=game_arena_handler,
            color="black",
            max_move_retries=3,
        )

        mid_game_board = chess.Board()
        with pytest.raises(ConnectionError, match="Network unavailable"):
            player_with_connection_issue(mid_game_board)

        assert connection_error_connector.query.call_count == 1


class TestLLMPlayerMajorityVoting:
    def test_majority_voting_selects_most_frequent_move_from_multiple_samples(self):
        # 2 votes for e4, 1 for Nf3
        voting_responses = ["Final Answer: e4", "Final Answer: Nf3", "Final Answer: e4"]

        voting_connector = MockLLMConnector(responses=voting_responses)
        game_arena_handler = GameArenaLLMMoveHandler()

        player_using_majority_vote = LLMPlayer(
            connector=voting_connector,
            handler=game_arena_handler,
            color="white",
            num_votes=3,
        )

        starting_board = chess.Board()
        majority_decision = player_using_majority_vote(starting_board)

        assert majority_decision.action == "move"
        assert majority_decision.attempted_move == "e2e4"
        assert voting_connector.query_count == 1

    def test_voting_tie_resolved_by_selecting_first_occurrence(self):
        # Each move gets 1 vote - tie
        tied_voting_responses = [
            "Final Answer: e4",
            "Final Answer: d4",
            "Final Answer: Nf3",
        ]

        tie_scenario_connector = MockLLMConnector(responses=tied_voting_responses)
        game_arena_handler = GameArenaLLMMoveHandler()

        player_with_tied_votes = LLMPlayer(
            connector=tie_scenario_connector,
            handler=game_arena_handler,
            color="white",
            num_votes=3,
        )

        starting_board = chess.Board()
        tiebreaker_decision = player_with_tied_votes(starting_board)

        assert tiebreaker_decision.action == "move"
        assert tiebreaker_decision.attempted_move == "e2e4"  # First in list wins

    def test_invalid_samples_excluded_from_vote_counting(self):
        # 2 valid e4, 1 invalid
        samples_with_invalid = [
            "Final Answer: e4",
            "gibberish response",
            "Final Answer: e4",
        ]

        mixed_validity_connector = MockLLMConnector(responses=samples_with_invalid)
        game_arena_handler = GameArenaLLMMoveHandler()

        player_filtering_invalid_votes = LLMPlayer(
            connector=mixed_validity_connector,
            handler=game_arena_handler,
            color="white",
            num_votes=3,
        )

        starting_board = chess.Board()
        filtered_vote_decision = player_filtering_invalid_votes(starting_board)

        assert filtered_vote_decision.action == "move"
        assert filtered_vote_decision.attempted_move == "e2e4"

    def test_all_samples_invalid_triggers_single_sample_retry(self):
        all_invalid_samples = ["invalid1", "invalid2", "invalid3"]
        valid_retry_response = "Final Answer: e4"

        connector_needing_full_retry = MockLLMConnector(
            responses=all_invalid_samples + [valid_retry_response]
        )
        game_arena_handler = GameArenaLLMMoveHandler()

        player_recovering_from_all_invalid = LLMPlayer(
            connector=connector_needing_full_retry,
            handler=game_arena_handler,
            color="white",
            num_votes=3,
            max_move_retries=2,
        )

        starting_board = chess.Board()
        retry_after_voting_failure = player_recovering_from_all_invalid(starting_board)

        assert retry_after_voting_failure.action == "move"
        assert retry_after_voting_failure.attempted_move == "e2e4"
        # Initial batch query + single sample retry
        assert connector_needing_full_retry.query_count == 2

    def test_voting_converts_san_notation_to_uci_for_comparison(self):
        # Same move in different notations
        different_notation_samples = [
            "Final Answer: e4",
            "Final Answer: e2e4",
            "Final Answer: e4",
        ]

        notation_mixing_connector = MockLLMConnector(
            responses=different_notation_samples
        )
        game_arena_handler = GameArenaLLMMoveHandler()

        player_handling_notation_variance = LLMPlayer(
            connector=notation_mixing_connector,
            handler=game_arena_handler,
            color="white",
            num_votes=3,
        )

        starting_board = chess.Board()
        normalized_notation_decision = player_handling_notation_variance(starting_board)

        assert normalized_notation_decision.action == "move"
        assert normalized_notation_decision.attempted_move == "e2e4"

    def test_network_error_during_voting_propagates_immediately(self):
        error_during_voting_connector = MockLLMConnector()
        error_during_voting_connector.query = Mock(
            side_effect=ConnectionError("API down")
        )
        game_arena_handler = GameArenaLLMMoveHandler()

        player_with_voting_network_error = LLMPlayer(
            connector=error_during_voting_connector,
            handler=game_arena_handler,
            color="white",
            num_votes=3,
        )

        starting_board = chess.Board()
        with pytest.raises(ConnectionError, match="API down"):
            player_with_voting_network_error(starting_board)

        assert error_during_voting_connector.query.call_count == 1


class TestLLMPlayerIntegration:
    def test_player_generates_moves_throughout_complete_game_sequence(self):
        game_sequence_responses = [
            "Final Answer: e4",
            "Final Answer: Nf3",
            "Final Answer: Bc4",
            "Final Answer: O-O",
            "Final Answer: d3",
        ]

        full_game_connector = MockLLMConnector(responses=game_sequence_responses)
        game_arena_handler = GameArenaLLMMoveHandler()

        white_player_in_game = LLMPlayer(
            connector=full_game_connector,
            handler=game_arena_handler,
            color="white",
        )

        evolving_board = chess.Board()

        # Move 1: e4
        white_move_1 = white_player_in_game(evolving_board)
        assert white_move_1.attempted_move == "e2e4"
        evolving_board.push(chess.Move.from_uci("e2e4"))
        evolving_board.push(chess.Move.from_uci("e7e5"))  # Black response

        # Move 2: Nf3
        white_move_2 = white_player_in_game(evolving_board)
        assert white_move_2.attempted_move == "g1f3"
        evolving_board.push(chess.Move.from_uci("g1f3"))
        evolving_board.push(chess.Move.from_uci("b8c6"))  # Black response

        # Move 3: Bc4
        white_move_3 = white_player_in_game(evolving_board)
        assert white_move_3.attempted_move == "f1c4"
        evolving_board.push(chess.Move.from_uci("f1c4"))
        evolving_board.push(chess.Move.from_uci("g8f6"))  # Black response

        # Move 4: O-O (castling)
        white_move_4 = white_player_in_game(evolving_board)
        assert white_move_4.attempted_move == "e1g1"
        evolving_board.push(chess.Move.from_uci("e1g1"))
        evolving_board.push(chess.Move.from_uci("d7d6"))  # Black response

        # Move 5: d3
        white_move_5 = white_player_in_game(evolving_board)
        assert white_move_5.attempted_move == "d2d3"

    def test_player_name_defaults_to_model_name_when_not_specified(self):
        test_connector = MockLLMConnector(model="gpt-4-turbo")
        game_arena_handler = GameArenaLLMMoveHandler()

        player_without_custom_name = LLMPlayer(
            connector=test_connector,
            handler=game_arena_handler,
            color="white",
        )

        assert player_without_custom_name.name == "gpt-4-turbo"

    def test_player_correctly_handles_black_perspective_moves(self):
        black_move_responses = ["Final Answer: e5", "Final Answer: Nc6"]

        black_perspective_connector = MockLLMConnector(responses=black_move_responses)
        game_arena_handler = GameArenaLLMMoveHandler()

        black_player = LLMPlayer(
            connector=black_perspective_connector,
            handler=game_arena_handler,
            color="black",
        )

        # Position after 1.e4
        board_after_white_e4 = chess.Board()
        board_after_white_e4.push(chess.Move.from_uci("e2e4"))

        # Black plays e5
        black_move_1 = black_player(board_after_white_e4)
        assert black_move_1.attempted_move == "e7e5"
        board_after_white_e4.push(chess.Move.from_uci("e7e5"))
        board_after_white_e4.push(chess.Move.from_uci("g1f3"))  # White Nf3

        # Black plays Nc6
        black_move_2 = black_player(board_after_white_e4)
        assert black_move_2.attempted_move == "b8c6"
