#!/usr/bin/env python3
"""Demo script to run a game between an LLM player and a random player."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_chess_arena.config import load_env
from llm_chess_arena.game import Game
from llm_chess_arena.player.random_player import RandomPlayer
from llm_chess_arena.player.llm import (
    LLMPlayer,
    LLMConnector,
    GameArenaLLMMoveHandler,
)

# Load environment variables from .env file
load_env()


def run_llm_game(model: str = "gpt-3.5-turbo") -> Game | None:
    """Run a game with a real LLM API.

    Args:
        model: Model name (e.g., 'gpt-4', 'claude-3-opus', 'gemini-pro')
               LiteLLM will automatically detect the provider from the model name.
    """
    print(f"Starting game: {model} vs RandomBot")
    print("-" * 60)

    try:
        # Create real LLM connector using LiteLLM
        # LiteLLM automatically detects the provider from the model name
        connector = LLMConnector(
            model=model,
            temperature=0.7,
            max_tokens=5000,
            timeout=30.0,
        )
        handler = GameArenaLLMMoveHandler()
        llm_player = LLMPlayer(
            connector=connector,
            handler=handler,
            color="white",
            name=model,
        )

        # Create random player
        random_player = RandomPlayer(name="RandomBot", color="black", seed=42)

        # Create and run game with beautiful board display
        game = Game(
            white_player=llm_player, black_player=random_player, display_board=True
        )
        game.play()

        return game

    except (RuntimeError, ConnectionError) as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            print(
                "\nAPI key not found. Please set the appropriate environment variable:"
            )
            print("  - For OpenAI models: export OPENAI_API_KEY=your-key")
            print("  - For Anthropic models: export ANTHROPIC_API_KEY=your-key")
            print("  - For Google models: export GEMINI_API_KEY=your-key")
            print("  - For other providers, see LiteLLM documentation")
            return None
        else:
            raise


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LLM vs Random chess game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OpenAI models
  python demo/run_llm_game.py --model gpt-3.5-turbo
  python demo/run_llm_game.py --model gpt-4
  
  # Anthropic models  
  python demo/run_llm_game.py --model claude-3-haiku-20240307
  python demo/run_llm_game.py --model claude-3-opus-20240229
  
  # Google models
  python demo/run_llm_game.py --model gemini-pro
  
Note: You must have the appropriate API key set as an environment variable.
For testing without API keys, use the test suite instead.
        """,
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="LLM model to use (LiteLLM will detect the provider)",
    )

    args = parser.parse_args()

    # Run game with specified model
    game = run_llm_game(args.model)

    # Display results
    if game and game.finished and game.outcome:
        print(f"\nResult: {game.outcome.result()}")
        print(f"Termination: {game.outcome.termination.name}")
        if game.winner:
            print(f"Winner: {game.winner.name}")
        else:
            print("Winner: Draw")
        print(f"Total moves: {len(game.board.move_stack)}")


if __name__ == "__main__":
    main()
