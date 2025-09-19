"""Unit tests for the move handler that parses LLM outputs."""

import chess
import pytest

from llm_chess_arena.player.llm.llm_move_handler import GameArenaLLMMoveHandler


@pytest.fixture
def game_arena_move_handler():
    """Provide a reusable Game Arena move handler instance."""
    return GameArenaLLMMoveHandler()


@pytest.fixture
def test_board_positions():
    """Return a collection of boards covering specific scenarios."""
    return {
        "starting_position": chess.Board(),
        "both_sides_can_castle": chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
        "white_pawn_ready_to_promote": chess.Board("8/4P3/8/8/8/8/8/8 w - - 0 1"),
        "two_knights_can_reach_same_square": chess.Board(
            "8/8/8/8/4N1N1/8/8/8 w - - 0 1"
        ),
        "en_passant_capture_available": chess.Board("8/8/8/3pP3/8/8/8/8 w - d6 0 1"),
    }


class TestGameArenaPromptGeneration:
    """Prompt construction expectations for the Game Arena handler."""

    def test_prompt_includes_board_state_player_color_and_move_history_from_kwargs(
        self,
    ):
        move_handler = GameArenaLLMMoveHandler()
        starting_board = chess.Board()

        generated_prompt = move_handler.get_prompt(
            board_in_fen=starting_board.fen(),
            player_color="white",
            move_history_in_uci=["e2e4", "e7e5"],
        )

        assert "FEN" in generated_prompt
        assert "white" in generated_prompt
        assert starting_board.fen() in generated_prompt
        assert "e2e4 e7e5" in generated_prompt

    def test_prompt_correctly_identifies_black_as_player_color_after_white_moves(self):
        move_handler = GameArenaLLMMoveHandler()
        board_after_white_e4 = chess.Board()
        board_after_white_e4.push_san("e4")

        black_player_prompt = move_handler.get_prompt(
            board_in_fen=board_after_white_e4.fen(),
            player_color="black",
            move_history_in_uci=["e2e4"],
        )

        assert "black" in black_player_prompt
        # Ensure "white" doesn't appear before "black" in the prompt
        assert "white" not in black_player_prompt.split("black")[0]

    def test_prompt_template_cannot_access_private_attributes_starting_with_underscore(
        self,
    ):
        move_handler = GameArenaLLMMoveHandler()
        move_handler.prompt_template = "Access {_private_attr}"
        move_handler._private_attr = "secret_data"

        with pytest.raises(KeyError, match="_private_attr"):
            move_handler.get_prompt()

    def test_prompt_generation_raises_error_when_template_references_nonexistent_field(
        self,
    ):
        move_handler = GameArenaLLMMoveHandler()
        move_handler.prompt_template = "Missing {nonexistent_field}"

        with pytest.raises(KeyError, match="nonexistent_field"):
            move_handler.get_prompt()

    def test_prompt_generation_passes_kwargs_to_template_fields_correctly(self):
        move_handler = GameArenaLLMMoveHandler()
        starting_board = chess.Board()

        prompt_with_kwargs = move_handler.get_prompt(
            board_in_fen=starting_board.fen(),
            player_color="white",
            move_history_in_uci=[],
        )

        assert "white" in prompt_with_kwargs
        assert starting_board.fen() in prompt_with_kwargs

    def test_prompt_template_raises_error_when_required_field_not_provided_in_kwargs(
        self,
    ):
        move_handler = GameArenaLLMMoveHandler()
        move_handler.prompt_template = "{player_color}"

        with pytest.raises(KeyError, match="player_color"):
            move_handler.get_prompt()


class TestMoveExtractionFromLLMResponse:
    """Validation for raw move extraction heuristics."""

    @pytest.mark.parametrize(
        "response,expected",
        [
            ("Final Answer: e4", "e4"),
            ("final answer: Nf3", "Nf3"),
            ("I think e4 is good. Final Answer: Nf3", "Nf3"),
            ("Final Answer: $\\boxed{e4}$", "e4"),
            ("Final Answer: `Nf3`", "Nf3"),
            ("Final Answer: <b>e4</b>", "e4"),
            ("Final Answer: \\text{Nf3}", "Nf3"),
            ("Final Answer: e4. Final Answer: d4", "d4"),
            ("I think e4", None),
            ("", None),
            ("Final Answer: e4 is best", "e4"),  # Now correctly extracts just the move
            # Real LLM response - now correctly extracts just the move
            ("Final Answer: d4\n\nThis move opens up lines.", "d4"),
        ],
    )
    def test_extracts_move_text_after_final_answer_marker_in_various_formats(
        self, response, expected
    ):
        move_handler = GameArenaLLMMoveHandler()
        assert move_handler._extract_raw_move_text(response) == expected


class TestMoveTextSanitization:
    """Ensure sanitized move text is compatible with downstream parsing."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("1. e4", "e4"),
            ("1...Nf6", "Nf6"),
            ("25.e4", "e4"),
            ("e4!", "e4"),
            ("Nf3??", "Nf3"),
            ("O-O!!", "O-O"),
            ("exd6ep", "exd6"),
            ("e8=Q", "e8=Q"),
            ("e4+", "e4+"),  # Check/mate markers preserved
            ("e4#", "e4#"),
            ("", None),
            ("   ", None),
        ],
    )
    def test_removes_move_numbers_and_evaluation_symbols_while_preserving_check_notation(
        self, text, expected
    ):
        move_handler = GameArenaLLMMoveHandler()
        assert move_handler._sanitize_move_text(text) == expected


class TestMoveParsingFromSanitizedText:
    """End-to-end parsing checks once moves are sanitized."""

    def test_returns_player_decision_with_san_move(self):
        move_handler = GameArenaLLMMoveHandler()

        decision = move_handler.parse_decision_from_response("Final Answer: e4")

        assert decision.action == "move"
        assert decision.attempted_move == "e4"

    def test_returns_player_decision_with_uci_move(self):
        move_handler = GameArenaLLMMoveHandler()

        decision = move_handler.parse_decision_from_response("Final Answer: e2e4")

        assert decision.action == "move"
        assert decision.attempted_move == "e2e4"

    def test_normalizes_castling_variations(self):
        move_handler = GameArenaLLMMoveHandler()

        compact = move_handler.parse_decision_from_response("Final Answer: O-O")
        spaced = move_handler.parse_decision_from_response("Final Answer: O - O")

        assert compact.attempted_move == "O-O"
        assert spaced.attempted_move == "O-O"

    def test_preserves_promotion_notation(self):
        move_handler = GameArenaLLMMoveHandler()

        decision = move_handler.parse_decision_from_response("Final Answer: e8=Q")

        assert decision.attempted_move == "e8=Q"

    def test_returns_raw_text_for_later_validation(self):
        move_handler = GameArenaLLMMoveHandler()

        decision = move_handler.parse_decision_from_response("Final Answer: xyz")

        assert decision.attempted_move == "xyz"

    def test_raises_parse_error_when_marker_missing(self):
        move_handler = GameArenaLLMMoveHandler()

        from llm_chess_arena.exceptions import ParseMoveError

        with pytest.raises(ParseMoveError):
            move_handler.parse_decision_from_response(
                "I think e4 is the best move here"
            )


class TestRetryPromptGeneration:
    def test_generates_retry_prompt_explaining_parse_failure_with_original_response(
        self,
    ):
        move_handler = GameArenaLLMMoveHandler()

        retry_prompt_after_parse_error = move_handler.get_retry_prompt(
            "InvalidMoveError", last_response="I play e4", last_prompt="Initial prompt"
        )

        assert (
            "previously suggested move was not parsable"
            in retry_prompt_after_parse_error
        )
        assert "I play e4" in retry_prompt_after_parse_error

    def test_includes_attempted_illegal_move_in_retry_prompt_for_context(self):
        move_handler = GameArenaLLMMoveHandler()

        retry_prompt_after_illegal = move_handler.get_retry_prompt(
            "IllegalMoveError", last_attempted_move="e5", last_prompt="Initial prompt"
        )

        assert "illegal move" in retry_prompt_after_illegal
        assert "e5" in retry_prompt_after_illegal

    def test_requests_disambiguation_when_move_could_refer_to_multiple_pieces(self):
        move_handler = GameArenaLLMMoveHandler()

        retry_prompt_for_ambiguous = move_handler.get_retry_prompt(
            "AmbiguousMoveError",
            last_attempted_move="Nf3",
            last_prompt="Initial prompt",
        )

        assert "ambiguous" in retry_prompt_for_ambiguous
        assert "Nf3" in retry_prompt_for_ambiguous

    def test_handles_missing_fields_gracefully_by_substituting_none(self):
        move_handler = GameArenaLLMMoveHandler()

        retry_prompt_with_missing_response = move_handler.get_retry_prompt(
            "InvalidMoveError", last_response=None, last_prompt="Initial prompt"
        )

        assert retry_prompt_with_missing_response is not None
        assert "None" in retry_prompt_with_missing_response

    def test_raises_error_for_unknown_exception_type_without_retry_template(self):
        move_handler = GameArenaLLMMoveHandler()

        with pytest.raises(ValueError, match="No retry prompt defined"):
            move_handler.get_retry_prompt("UnknownError", last_prompt="Initial prompt")


class TestSpecialMoveHandling:
    def test_correctly_parses_en_passant_capture_notation(self):
        move_handler = GameArenaLLMMoveHandler()

        en_passant_response = "Final Answer: exd6"
        decision = move_handler.parse_decision_from_response(en_passant_response)

        assert decision.attempted_move == "exd6"

    def test_parses_uci_format_promotion_with_lowercase_piece_indicator(self):
        move_handler = GameArenaLLMMoveHandler()

        uci_promotion_response = "Final Answer: e7e8q"
        decision = move_handler.parse_decision_from_response(uci_promotion_response)

        assert decision.attempted_move == "e7e8q"

    def test_successfully_falls_back_from_invalid_san_to_valid_uci_parsing(self):
        move_handler = GameArenaLLMMoveHandler()

        # "e2e4" is UCI notation, not SAN
        uci_formatted_response = "Final Answer: e2e4"
        decision = move_handler.parse_decision_from_response(uci_formatted_response)

        assert decision.attempted_move == "e2e4"

    def test_preserves_check_notation_while_parsing_move_correctly(self):
        move_handler = GameArenaLLMMoveHandler()

        check_move_response = "Final Answer: Re2+"
        decision = move_handler.parse_decision_from_response(check_move_response)

        assert decision.attempted_move == "Re2+"

    def test_preserves_checkmate_notation_while_parsing_move_correctly(self):
        move_handler = GameArenaLLMMoveHandler()

        checkmate_response = "Final Answer: Qh5#"

        decision = move_handler.parse_decision_from_response(checkmate_response)
        assert decision.attempted_move == "Qh5#"

    def test_extracts_move_correctly_even_with_trailing_explanation_text(self):
        move_handler = GameArenaLLMMoveHandler()

        response_with_reasoning = "Final Answer: e4 because it controls the center"

        decision = move_handler.parse_decision_from_response(response_with_reasoning)
        assert decision.attempted_move == "e4"
