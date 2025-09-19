# LLM Chess Arena

A clean, modular platform for Large Language Models (LLMs) to play chess against each other to test out different models, reasoning strategies, and learning techniques.

This project heavily inspired by [google-deepmind/game_arena](https://github.com/google-deepmind/game_arena). Please check out their repo and the [Kaggle AI Chess Exhibition](https://www.kaggle.com/game-arena) for the original implementation and the matches.

## Roadmap

Primary priorities:

- [ ] **Metrics & Logging Surface** — add structured events around move validation and game flow so we can record prompts, responses, retry causes, and eventually evaluation scores.
    - Potential targets: hooks in `LLMPlayer._validate_player_decision_from_llm`, callbacks from `Game.make_move`, emitter interface for downstream storage (files, DB, analytics).
- [ ] **Context Enrichment Pipeline** — compute optional signals (engine evals, opening tags, tactical motifs) and pass them via `PlayerDecisionContext` extras for prompts or analytics.
    - Options: integrate Stockfish eval snapshots, reuse python-chess heuristics, log derived features for offline learning.
- [ ] **Prompt & Decision Strategies** — introduce a strategy layer above `LLMPlayer` so multi-turn prompting, self-critique, or ICL recipes can plug in without rewiring the player loop.
    - Options: strategy objects managing message history, few-shot exemplar loaders, self-evaluation retries.
- [ ] **Reusable Board/Context Serialization** — expose serialization helpers (ASCII boards, annotated histories) that both prompts and UIs can consume.
    - Options: pure-text renders, Rich tables, JSON payloads for future front-ends.
- [ ] **Template Modularization** — move current Game Arena strings into a lightweight templating helper once multiple strategies exist.
    - Options: simple format fragments, string.Template, Jinja2 hierarchy.
- [ ] **Deferred Handler/Connector Factory** — create factories only after we have several handler or connector variants that share configuration knobs.
    - Options: registry-based factory, Hydra/OmegaConf-driven configs.

Second-tier initiatives — stay visible for future sprints:

- [ ] **Human Player Support** — CLI or UI for humans to play against LLM/engine opponents.
- [ ] **Move/Game Evaluations** — richer post-game reports built on top of the metrics surface.
- [ ] **Better Visualizations** — improved terminal or web views leveraging the new serialization helpers.
- [ ] **Learning Techniques** — experimentation playground for different prompting/ICL strategies once the strategy layer is ready.


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
