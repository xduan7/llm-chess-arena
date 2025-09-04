# LLM Chess Arena

A clean, modular platform for Large Language Models (LLMs) to play chess against each other to test out different models, reasoning strategies, and learning techniques.

This project heavily inspired by [google-deepmind/game_arena](https://github.com/google-deepmind/game_arena). Please check out their repo and the [Kaggle AI Chess Exhibition](https://www.kaggle.com/game-arena) for the original implementation and the matches.

To extend the original implementation, we are planning to add the following features: 
- [ ] **Human Player**: Allow human players to play against LLMs.
- [ ] **Move/Game Evaluations**: Add annotations or metrics to moves and games for better analysis.
- [ ] **Better Visualizations**: Improve the display of game states and reasoning traces.
- [ ] **Learning Techniques**: Improve LLM performance over time using techniques like different reasoning strategies, in-context learning, and more.


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
python demo/run_game.py

# Stockfish vs Stockfish (requires stockfish installed)
python demo/run_stockfish_game.py

# LLM vs Random player (requires API keys in .env)
python demo/run_llm_game.py
```


---

## License

See [LICENSE](LICENSE) for details.


---

## Author

Xiaotian Duan (xduan7 at gmail.com)
