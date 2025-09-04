import chess
import pytest

from llm_chess_arena.player.llm.llm_move_handler import (
    BaseLLMMoveHandler,
    GameArenaLLMMoveHandler,
)


@pytest.fixture
def game_arena_move_handler():
    return GameArenaLLMMoveHandler()


@pytest.fixture
def test_board_positions():
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
    def test_parses_standard_algebraic_notation_move_to_chess_move_object(self):
        move_handler = GameArenaLLMMoveHandler()
        starting_board = chess.Board()

        llm_response = "Final Answer: e4"
        parsed_move = move_handler.parse_decision_from_response(
            llm_response, starting_board
        )

        assert parsed_move == chess.Move.from_uci("e2e4")

    def test_falls_back_to_uci_notation_parsing_when_san_parsing_fails(self):
        move_handler = GameArenaLLMMoveHandler()
        starting_board = chess.Board()

        uci_format_response = "Final Answer: e2e4"
        parsed_move = move_handler.parse_decision_from_response(
            uci_format_response, starting_board
        )

        assert parsed_move == chess.Move.from_uci("e2e4")

    def test_correctly_parses_kingside_castling_notation_to_king_move(self):
        move_handler = GameArenaLLMMoveHandler()
        castling_ready_position = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")

        castling_response = "Final Answer: O-O"
        parsed_castling = move_handler.parse_decision_from_response(
            castling_response, castling_ready_position
        )

        assert parsed_castling == chess.Move.from_uci("e1g1")

    def test_handles_castling_notation_with_spaces_between_zeros(self):
        move_handler = GameArenaLLMMoveHandler()
        castling_ready_position = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")

        spaced_castling_response = "Final Answer: O - O"
        parsed_castling = move_handler.parse_decision_from_response(
            spaced_castling_response, castling_ready_position
        )

        assert parsed_castling == chess.Move.from_uci("e1g1")

    def test_parses_pawn_promotion_move_with_piece_specification(self):
        move_handler = GameArenaLLMMoveHandler()
        pawn_on_seventh_rank = chess.Board("8/4P3/8/8/8/8/8/8 w - - 0 1")

        promotion_response = "Final Answer: e8=Q"
        parsed_promotion = move_handler.parse_decision_from_response(
            promotion_response, pawn_on_seventh_rank
        )

        assert parsed_promotion == chess.Move.from_uci("e7e8q")

    def test_raises_parse_error_when_response_lacks_final_answer_marker(self):
        move_handler = GameArenaLLMMoveHandler()
        starting_board = chess.Board()

        response_without_marker = "I think e4 is the best move here"

        from llm_chess_arena.exceptions import ParseMoveError

        with pytest.raises(ParseMoveError):
            move_handler.parse_decision_from_response(
                response_without_marker, starting_board
            )

    def test_returns_unparseable_move_text_as_is_for_later_validation(self):
        move_handler = GameArenaLLMMoveHandler()

        nonsense_move_response = "Final Answer: xyz"

        # Handler returns unparseable text without validation
        player_decision = move_handler.parse_decision_from_response(
            nonsense_move_response
        )
        assert player_decision.action == "move"
        assert player_decision.attempted_move == "xyz"

    def test_returns_illegal_move_without_validation_when_board_not_provided(self):
        move_handler = GameArenaLLMMoveHandler()

        illegal_opening_response = "Final Answer: e5"  # White can't play e5 on turn 1

        # Handler returns move without board validation
        player_decision = move_handler.parse_decision_from_response(
            illegal_opening_response
        )
        assert player_decision.action == "move"
        assert player_decision.attempted_move == "e5"

    def test_returns_potentially_ambiguous_move_without_board_context_validation(self):
        move_handler = GameArenaLLMMoveHandler()

        knight_move_response = "Final Answer: Nf3"

        # Handler returns move without checking for ambiguity
        player_decision = move_handler.parse_decision_from_response(
            knight_move_response
        )
        assert player_decision.action == "move"
        assert player_decision.attempted_move == "Nf3"

    def test_stores_extracted_move_text_for_retry_context_after_parsing(self):
        move_handler = GameArenaLLMMoveHandler()
        starting_board = chess.Board()

        standard_move_response = "Final Answer: e4"
        move_handler.parse_decision_from_response(
            standard_move_response, starting_board
        )

        assert move_handler.last_attempted_move_text == "e4"

    def test_stores_response_but_not_move_text_when_extraction_fails(self):
        move_handler = GameArenaLLMMoveHandler()

        incomplete_response = "I think e4"
        from llm_chess_arena.exceptions import ParseMoveError

        with pytest.raises(ParseMoveError):
            move_handler.parse_decision_from_response(incomplete_response)

        assert move_handler.last_response == "I think e4"
        assert move_handler.last_attempted_move_text is None

    def test_stores_both_response_and_move_text_when_move_is_illegal(self):
        move_handler = GameArenaLLMMoveHandler()
        starting_board = chess.Board()

        illegal_white_opening = "Final Answer: e5"

        with pytest.raises(chess.IllegalMoveError):
            move_handler.parse_decision_from_response(
                illegal_white_opening, starting_board
            )

        assert move_handler.last_response == "Final Answer: e5"
        assert move_handler.last_attempted_move_text == "e5"


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
        en_passant_position = chess.Board("8/8/8/3pP3/8/8/8/8 w - d6 0 1")

        en_passant_response = "Final Answer: exd6"
        parsed_en_passant = move_handler.parse_decision_from_response(
            en_passant_response, en_passant_position
        )

        assert parsed_en_passant == chess.Move.from_uci("e5d6")

    def test_parses_uci_format_promotion_with_lowercase_piece_indicator(self):
        move_handler = GameArenaLLMMoveHandler()
        promotion_position = chess.Board("8/4P3/8/8/8/8/8/8 w - - 0 1")

        uci_promotion_response = "Final Answer: e7e8q"
        parsed_uci_promotion = move_handler.parse_decision_from_response(
            uci_promotion_response, promotion_position
        )

        assert parsed_uci_promotion == chess.Move.from_uci("e7e8q")

    def test_successfully_falls_back_from_invalid_san_to_valid_uci_parsing(self):
        move_handler = GameArenaLLMMoveHandler()
        starting_board = chess.Board()

        # "e2e4" is UCI notation, not SAN
        uci_formatted_response = "Final Answer: e2e4"
        parsed_via_uci_fallback = move_handler.parse_decision_from_response(
            uci_formatted_response, starting_board
        )

        assert parsed_via_uci_fallback == chess.Move.from_uci("e2e4")

    def test_preserves_check_notation_while_parsing_move_correctly(self):
        move_handler = GameArenaLLMMoveHandler()
        # Rook on e1 can check king on e7 by moving to e2
        check_position = chess.Board("8/4k3/8/8/8/8/3K4/4R3 w - - 0 1")

        check_move_response = "Final Answer: Re2+"
        parsed_check_move = move_handler.parse_decision_from_response(
            check_move_response, check_position
        )

        assert parsed_check_move == chess.Move.from_uci("e1e2")

    def test_preserves_checkmate_notation_while_parsing_move_correctly(self):
        move_handler = GameArenaLLMMoveHandler()
        # Scholar's mate position where Qh5# is checkmate
        checkmate_position = chess.Board(
            "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1"
        )

        checkmate_response = "Final Answer: Qh5#"

        parsed_checkmate = move_handler.parse_decision_from_response(
            checkmate_response, checkmate_position
        )
        assert parsed_checkmate == chess.Move.from_uci("f3h5")

    def test_extracts_move_correctly_even_with_trailing_explanation_text(self):
        move_handler = GameArenaLLMMoveHandler()
        starting_board = chess.Board()

        response_with_reasoning = "Final Answer: e4 because it controls the center"

        extracted_move = move_handler.parse_decision_from_response(
            response_with_reasoning, starting_board
        )
        assert extracted_move == chess.Move.from_uci("e2e4")

    def test_static_method_returns_white_when_white_to_move(self):
        starting_board = chess.Board()
        assert BaseLLMMoveHandler.player_color(starting_board) == "White"

    def test_static_method_returns_black_when_black_to_move(self):
        board_after_e4 = chess.Board()
        board_after_e4.push_san("e4")
        assert BaseLLMMoveHandler.player_color(board_after_e4) == "Black"

    def test_static_method_returns_board_fen_notation_as_state(self):
        any_board_position = chess.Board()
        assert (
            BaseLLMMoveHandler.board_state(any_board_position)
            == any_board_position.fen()
        )


class TestEndToEndMoveHandlerWorkflow:
    def test_complete_workflow_from_prompt_generation_to_move_parsing(self):
        move_handler = GameArenaLLMMoveHandler()
        starting_board = chess.Board()

        initial_prompt = move_handler.get_prompt(board=starting_board, move_history="")
        assert "white" in initial_prompt
        assert starting_board.fen() in initial_prompt

        llm_response = "After analysis, Final Answer: e4"
        parsed_decision = move_handler.parse_decision_from_response(
            llm_response, starting_board
        )
        assert parsed_decision == chess.Move.from_uci("e2e4")

    def test_retry_workflow_recovers_from_invalid_response_with_proper_context(self):
        move_handler = GameArenaLLMMoveHandler()

        # First attempt fails due to missing marker
        invalid_first_response = "I think e4 is best"
        from llm_chess_arena.exceptions import ParseMoveError

        with pytest.raises(ParseMoveError):
            move_handler.parse_decision_from_response(invalid_first_response)

        # Generate retry prompt with context
        starting_board = chess.Board()
        original_prompt = move_handler.get_prompt(
            board_in_fen=starting_board.fen(),
            flattened_move_history_in_uci="",
            flattened_move_history_in_san="",
            num_halfmoves=0,
            player_color="white",
        )
        retry_prompt_with_error = move_handler.get_retry_prompt(
            exception_name="InvalidMoveError",
            last_response=invalid_first_response,
            last_prompt=original_prompt,
            board_in_fen=starting_board.fen(),
            flattened_move_history_in_uci="",
            flattened_move_history_in_san="",
            num_halfmoves=0,
            player_color="white",
        )
        assert "not parsable" in retry_prompt_with_error

        # Second attempt succeeds with proper format
        valid_second_response = "Final Answer: e4"
        successful_decision = move_handler.parse_decision_from_response(
            valid_second_response
        )
        assert successful_decision.action == "move"
        assert successful_decision.attempted_move == "e4"

    def test_retry_workflow_handles_ambiguous_move_by_requesting_clarification(self):
        move_handler = GameArenaLLMMoveHandler()
        # Two knights (g1 and e5) can both reach f3
        two_knights_position = chess.Board(
            "rnbqkb1r/pppppppp/8/4N3/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )

        # First attempt is ambiguous
        ambiguous_knight_move = "Final Answer: Nf3"
        with pytest.raises(chess.AmbiguousMoveError):
            move_handler.parse_decision_from_response(
                ambiguous_knight_move, two_knights_position
            )

        # Simulate state after failed parse
        move_handler.last_attempted_move_text = "Nf3"
        initial_context_prompt = move_handler.get_prompt(
            board=two_knights_position, move_history=""
        )
        disambiguation_prompt = move_handler.get_retry_prompt(
            "AmbiguousMoveError",
            last_attempted_move="Nf3",
            last_prompt=initial_context_prompt,
        )
        assert "ambiguous" in disambiguation_prompt
        assert "Nf3" in disambiguation_prompt

        # Second attempt specifies which knight
        specific_knight_move = "Final Answer: Ngf3"
        disambiguated_move = move_handler.parse_decision_from_response(
            specific_knight_move, two_knights_position
        )
        assert disambiguated_move == chess.Move.from_uci("g1f3")
