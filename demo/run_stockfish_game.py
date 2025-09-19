#!/usr/bin/env python3
"""Demo script to run a game between strong and weak Stockfish players."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_chess_arena.config import load_env
from llm_chess_arena.game import Game
from llm_chess_arena.player.stockfish_player import StockfishPlayer

# Load environment variables
load_env()


def main() -> None:
    """Run a demo game between strong and weak Stockfish players."""

    # Create a strong Stockfish player (roughly 2800 ELO)
    strong_stockfish = StockfishPlayer(
        name="Stockfish Master (2800 ELO)",
        color="white",
        engine_limits={
            "depth": 20,  # Deep search
            "time": 1.0,  # 1 second per move
        },
        engine_options={
            "Hash": 256,  # 256 MB hash table
            "Threads": 2,  # Use 2 threads
            "UCI_LimitStrength": False,  # Disable to use full strength
            "UCI_Elo": 2800,  # This is used only when UCI_LimitStrength is True
        },
    )

    # Create a weak Stockfish player (roughly 1320 ELO - beginner level)
    weak_stockfish = StockfishPlayer(
        name="Stockfish Beginner (1320 ELO)",
        color="black",
        engine_limits={
            "depth": 8,  # Shallow search
            "time": 0.1,  # 100ms per move
        },
        engine_options={
            "Hash": 16,  # Small hash table
            "Threads": 1,  # Single thread
            "UCI_LimitStrength": True,  # Enable ELO limiting
            "UCI_Elo": 1320,  # Minimum ELO rating (weakest possible)
        },
    )

    # Create and run game with beautiful board display
    game = Game(strong_stockfish, weak_stockfish, display_board=True)

    print(f"Starting game: {strong_stockfish.name} vs {weak_stockfish.name}")
    print("-" * 60)

    # Play the game with a move limit for demo
    game.play(max_num_moves=100)

    # Display results
    if game.finished and game.outcome:
        print(f"\nResult: {game.outcome.result()}")
        print(f"Termination: {game.outcome.termination.name}")
        if game.winner:
            print(f"Winner: {game.winner.name}")
        else:
            print("Winner: Draw")
        print(f"Total moves: {len(game.board.move_stack)}")


if __name__ == "__main__":
    main()
