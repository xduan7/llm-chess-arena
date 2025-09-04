#!/usr/bin/env python3
"""Demo script to run a chess game between two random players."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_chess_arena.config import load_env
from llm_chess_arena.game import Game
from llm_chess_arena.player.random_player import RandomPlayer

# Load environment variables
load_env()


def main():
    """Run a demo game between two random players."""
    # Create players
    white_player = RandomPlayer(name="Random White", color="white", seed=42)
    black_player = RandomPlayer(name="Random Black", color="black", seed=43)

    # Create and run game
    game = Game(white_player, black_player)

    print(f"Starting game: {white_player.name} vs {black_player.name}")
    print("-" * 60)

    # Play the game
    game.play()

    # Display results
    if game.finished:
        print(f"\nResult: {game.outcome.result()}")
        print(f"Termination: {game.outcome.termination.name}")
        if game.winner:
            print(f"Winner: {game.winner.name}")
        else:
            print("Winner: Draw")
        print(f"Total moves: {len(game.board.move_stack)}")
        print("\nFinal board:")
        print(game.board)


if __name__ == "__main__":
    main()
