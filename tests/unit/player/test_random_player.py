import chess

from llm_chess_arena.player.random_player import RandomPlayer


class TestRandomPlayerInitialization:
    def test_stores_provided_name_and_color_as_player_attributes(self):
        random_player_with_custom_name = RandomPlayer(name="Test", color="white")

        assert random_player_with_custom_name.name == "Test"
        assert random_player_with_custom_name.color == "white"

    def test_accepts_both_white_and_black_as_valid_color_strings(self):
        white_random_player = RandomPlayer(name="White", color="white")
        black_random_player = RandomPlayer(name="Black", color="black")

        assert white_random_player.color == "white"
        assert black_random_player.color == "black"

    def test_players_with_same_seed_generate_identical_moves_from_same_position(self):
        player_with_seed_42_first = RandomPlayer(name="Player1", color="white", seed=42)
        player_with_seed_42_second = RandomPlayer(
            name="Player2", color="white", seed=42
        )
        player_with_different_seed = RandomPlayer(
            name="Player3", color="white", seed=99
        )

        starting_position = chess.Board()

        first_player_move = player_with_seed_42_first(starting_position)
        second_player_move = player_with_seed_42_second(starting_position)
        different_seed_move = player_with_different_seed(starting_position)

        assert first_player_move.attempted_move == second_player_move.attempted_move
        assert first_player_move.attempted_move != different_seed_move.attempted_move

    def test_seed_defaults_to_none_when_not_explicitly_provided(self):
        unseeded_random_player = RandomPlayer(name="Test", color="white")

        assert unseeded_random_player.seed is None


class TestRandomPlayerMoveGeneration:
    def test_generates_legal_move_from_standard_starting_position(self, white_player):
        standard_starting_board = chess.Board()

        player_decision = white_player(standard_starting_board)

        assert player_decision.action == "move"
        assert (
            chess.Move.from_uci(player_decision.attempted_move)
            in standard_starting_board.legal_moves
        )

    def test_two_players_with_identical_seeds_select_same_move(self):
        first_seeded_player = RandomPlayer(name="Test1", color="white", seed=42)
        second_seeded_player = RandomPlayer(name="Test2", color="white", seed=42)
        test_board = chess.Board()

        first_player_decision = first_seeded_player(test_board)
        second_player_decision = second_seeded_player(test_board)

        assert (
            first_player_decision.attempted_move
            == second_player_decision.attempted_move
        )

    def test_generates_legal_moves_from_multiple_positions_during_game(
        self, black_player
    ):
        game_board = chess.Board()
        game_board.push_san("e4")  # White opens with e4

        black_first_move_decision = black_player(game_board)
        black_first_move = chess.Move.from_uci(black_first_move_decision.attempted_move)
        assert black_first_move in game_board.legal_moves

        game_board.push(black_first_move)
        game_board.push_san("Nf3")  # White develops knight

        black_second_move_decision = black_player(game_board)
        black_second_move = chess.Move.from_uci(
            black_second_move_decision.attempted_move
        )
        assert black_second_move in game_board.legal_moves

    def test_selects_legal_move_even_with_limited_endgame_options(
        self, common_positions, white_player
    ):
        kings_facing_endgame_position = chess.Board(common_positions["kings_facing"])

        endgame_move_decision = white_player(kings_facing_endgame_position)

        assert (
            chess.Move.from_uci(endgame_move_decision.attempted_move)
            in kings_facing_endgame_position.legal_moves
        )

    def test_selects_legal_escape_move_when_king_is_in_check(
        self, common_positions, black_player
    ):
        black_king_in_check_position = chess.Board(common_positions["black_in_check"])

        available_escape_moves = list(black_king_in_check_position.legal_moves)

        check_escape_decision = black_player(black_king_in_check_position)
        selected_escape_move = chess.Move.from_uci(check_escape_decision.attempted_move)

        assert (
            len(available_escape_moves) > 0
        ), "Position must have legal moves to escape check"
        assert selected_escape_move in available_escape_moves


class TestRandomPlayerReproducibility:
    def test_identical_seeds_produce_identical_move_sequences_across_multiple_turns(
        self,
    ):
        shared_seed_value = 12345

        first_player = RandomPlayer(
            name="Player1", color="white", seed=shared_seed_value
        )
        first_game_board = chess.Board()
        first_player_move_sequence = []

        for turn in range(5):
            if first_game_board.legal_moves:
                first_player_decision = first_player(first_game_board)
                selected_move = chess.Move.from_uci(
                    first_player_decision.attempted_move
                )
                first_player_move_sequence.append(first_player_decision.attempted_move)
                first_game_board.push(selected_move)
                if first_game_board.legal_moves:
                    first_game_board.push(list(first_game_board.legal_moves)[0])

        second_player = RandomPlayer(
            name="Player2", color="white", seed=shared_seed_value
        )
        second_game_board = chess.Board()
        second_player_move_sequence = []

        for turn in range(5):
            if second_game_board.legal_moves:
                second_player_decision = second_player(second_game_board)
                selected_move = chess.Move.from_uci(
                    second_player_decision.attempted_move
                )
                second_player_move_sequence.append(
                    second_player_decision.attempted_move
                )
                second_game_board.push(selected_move)
                if second_game_board.legal_moves:
                    second_game_board.push(list(second_game_board.legal_moves)[0])

        assert first_player_move_sequence == second_player_move_sequence

    def test_different_seeds_produce_different_move_selections_from_same_position(self):
        shared_starting_position = chess.Board()

        player_with_seed_100 = RandomPlayer(name="Player1", color="white", seed=100)
        player_with_seed_200 = RandomPlayer(name="Player2", color="white", seed=200)

        first_player_moves = [
            player_with_seed_100(shared_starting_position).attempted_move
            for _ in range(3)
        ]
        second_player_moves = [
            player_with_seed_200(shared_starting_position).attempted_move
            for _ in range(3)
        ]

        assert first_player_moves != second_player_moves
