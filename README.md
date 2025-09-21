# LLM Chess Arena

A clean, modular platform for Large Language Models (LLMs) to play chess against each other to test out different models, reasoning strategies, and learning techniques.

This project heavily inspired by [google-deepmind/game_arena](https://github.com/google-deepmind/game_arena). Please check out their repo and the [Kaggle AI Chess Exhibition](https://www.kaggle.com/game-arena) for the original implementation and the matches.

For development roadmap and TODO list, see [AGENTS.md](AGENTS.md).

---

## Quick Start

### Installation

A working Python environment with Python 3.12 or higher is required.
To install the required dependencies, run:
```bash
pip install -e .
```

### Set API Keys

Create a `.env` file in the root directory and add your API keys:
```bash
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
...
```

### Run Demo Games

```bash
# Random vs Random players
bash demo/run_random_game.sh

# Stockfish vs Stockfish (requires stockfish installed)
bash demo/run_stockfish_game.sh

# LLM vs Random player (requires API keys in .env)
bash demo/run_llm_game.sh gpt-4
```

### Hydra-Powered CLI

The new Hydra configuration system enables composable experiments. Use the CLI entry point to run games with arbitrary player combinations and overrides:

```bash
# Default random vs random
python -m llm_chess_arena.cli.play

# Stockfish vs GPT-4
python -m llm_chess_arena.cli.play \
  players@players.white=stockfish/elo_2800 \
  players@players.black=llm/gpt4

# Override individual parameters
python -m llm_chess_arena.cli.play \
  players.white.engine_limits.depth=18 \
  players.black.connector.temperature=0.2

# Parameter sweep across temperatures
python -m llm_chess_arena.cli.play --multirun \
  players@players.black=llm/gpt4 \
  players.black.connector.temperature=0.2,0.6,0.9
```

Hydra configs live under `configs/` with groups for `game/`, `players/`, and `metrics/`. Stockfish presets cover ELO 1320/1600/2000/2400/2800 (e.g. `players@players.white=stockfish/elo_2000`); mix and match these or author new YAML files to capture research settings.
Move-quality thresholds are configurable via `configs/metrics/default.yaml`, and metrics are always enabled by default.


---

## License

See [LICENSE](LICENSE) for details.


---

## Author

Xiaotian Duan (xduan7 at gmail.com)
