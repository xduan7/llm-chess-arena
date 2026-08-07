# LLM Chess Arena - Coding Agent Instructions

## Project Overview

This project creates a clean, modular platform for Large Language Models (LLMs) to play chess against each other. See [README](README.md) for detailed project information.


---

## Development Workflow

Follow this 6-step workflow for EVERY task:

### Step 1: Understand Requirements
- Read the TODO list section below thoroughly
- Ask the user for clarification on any unclear requirements BEFORE starting implementation
- Break down complex tasks into smaller subtasks if needed
- If user doesn't respond to clarification requests, keep asking

### Step 2: Implement
- Write clean, modular code following the coding standards (see below)
- Focus on simplicity and clarity over cleverness
- Ensure proper error handling and edge cases are covered
- If stuck or blocked, ask the user for guidance

### Step 3: Code Review
- Use Zen MCP to review your implementation
- Request feedback from ALL 5 models (GPT-5, O3-mini, O4-mini, Gemini 2.5 Pro, XAI Grok 4)
- Provide clear context about what the code does and what feedback you need
- For simple fixes/refactors, one model review may be sufficient

### Step 4: Iterate
- Address all CRITICAL feedback (security, correctness, major design flaws)
- Synthesize suggestions from models and make informed decisions
- Do not proceed until critical issues are resolved
- If models strongly disagree on critical issues, ask user for direction
- Minor style/optimization suggestions can be addressed or documented for later

### Step 5: Update TODO List
- Mark completed tasks as DONE
- Update remaining tasks based on learnings and feedback
- NEVER add new tasks without user approval
- Break large tasks into smaller subtasks when appropriate
- Log important design decisions in the Architecture Key Design Decisions section

### Step 6: Continue
- Move to the next task in the TODO list in CLAUDE.md
- Repeat the entire workflow from Step 1


---

## Core Principles

### 1. Model Collaboration
- Actively use Zen MCP to consult other models when you need:
  - Domain expertise you lack
  - Alternative implementation approaches
  - Code review and quality checks
  - Complex problem-solving assistance
- Recommended models: GPT-5, O3-mini, O4-mini, Gemini 2.5 Pro, XAI Grok 4

### 2. Code Quality Standards
- **Naming**: Use descriptive, self-documenting names for all variables, functions, and classes
- **Structure**: Keep functions small and focused on a single responsibility
- **Docstrings**: Use google-style docstrings for all public functions and classes
- **Comments**: Only add comments to explain WHY (business logic, design decisions), never WHAT
- **Formatting**: Follow Python PEP 8 style guidelines consistently
- **Type Hints**: Use type hints for all function signatures and class attributes

### 3. Testing Requirements
- Write unit tests for all critical functionality
- Focus on testing edge cases and error conditions
- Tests should be simple and focused
- Aim for reasonable coverage, not perfection (this is not production code)

### 4. Documentation
- Each module must have a clear docstring explaining its purpose
- Public functions need docstrings with parameter and return descriptions
- Keep documentation concise and practical
- Update documentation when code changes


---

## Architecture

```
src/
└── llm_chess_arena/
    ├── __init__.py
    ├── cli/
    │   ├── main.py             # Hydra CLI: run games/tournaments
    │   └── resume.py           # Hydra CLI: resume interrupted tournaments
    ├── config/
    │   ├── loader.py           # Hydra composition & env setup
    │   └── schema.py           # Dataclass config schemas & normalization
    ├── core/
    │   └── policies.py         # Centralized error-handling decorators
    ├── exceptions.py
    ├── factory/                # Object construction helpers
    │   ├── metrics_factory.py
    │   └── player_factory.py
    ├── game.py                 # Game loop, recording, resume-from-record
    ├── metrics.py              # Stockfish-based move evaluation
    ├── record.py               # Game record collection & JSON serialization
    ├── renderer.py             # Rich terminal board visualization
    ├── tournament/
    │   ├── aggregator.py       # Pure aggregation of game results
    │   ├── executor.py         # TournamentRunner (sequential/parallel)
    │   ├── export.py           # results.json / results.csv export
    │   ├── loader.py           # Tournament state loading & resumability checks
    │   ├── resume.py           # TournamentResumer with locking & archival
    │   └── types.py            # TournamentConfig, GameResult, TournamentResult
    ├── types.py
    ├── utils.py
    └── player/
        ├── base_player.py
        ├── random_player.py
        ├── stockfish_player.py
        └── llm/
            ├── connector.py       # LiteLLM wrapper for testing isolation
            ├── player.py          # Orchestrates prompting, voting, retries
            ├── types.py           # Vote/decision artifact dataclasses
            ├── prompting/
            │   ├── handlers.py    # Move parsing and templating
            │   └── session.py     # Prompt generation & retry context
            └── decision/
                ├── retry.py       # Retry budgeting & resignation helpers
                └── voting.py      # Majority voting with tie-breaking

configs/
├── config.yaml               # Hydra composition root
├── resume.yaml               # Config for the resume CLI
├── env/default.yaml
├── game/default.yaml
├── hydra/default.yaml
├── metrics/
│   ├── default.yaml
│   ├── high.yaml
│   └── ultra.yaml
├── players/
│   ├── random.yaml
│   ├── stockfish/
│   │   ├── elo_1320.yaml ... elo_2800.yaml
│   └── llm/
│       ├── default.yaml
│       └── reasoning.yaml
└── tournament/default.yaml

demo/
├── run_random_game.sh
├── run_stockfish_game.sh
└── run_llm_game.sh

tests/
├── conftest.py
├── test_demos.py
├── fixtures/mock_llm_connector.py
├── unit/                      # Per-module unit tests (game, record, config,
│                              # metrics, players, tournament, resume, ...)
└── integration/               # Golden master, VCR recordings, snapshots,
                               # env-gated live API tests, end-to-end resume
```

### Key Design Decisions

1. **Chess Logic**: Using `python-chess` library for all chess rules, board state, and move validation. No need to reinvent the wheel.

2. **LLM Integration**: Using LiteLLM as a unified interface to 100+ LLM providers. This provides:
   - Single API for OpenAI, Anthropic, Google, and many others
   - Built-in retry logic and error handling
   - Automatic parameter translation between providers
   - Thin wrapper (`LLMConnector`) for testing isolation

3. **Testing Strategy**:
   - Mock LiteLLM at the boundary (`litellm.completion`) for unit tests
   - Environment-gated integration tests for real API calls
   - Test fixtures separated from production code
   - Comprehensive test coverage without requiring API keys

4. **Configuration Management**:
   - Environment variables loaded from `.env` file via `python-dotenv`
   - Future: Hydra for configuration composition to allow flexible experimentation

5. **Componentized LLM Player Architecture**:
   - LLM player logic decomposed into dedicated collaborators (`PromptSession`, `VoteAggregator`, `RetryController`).
   - Prompting utilities live under `player.llm.prompting`, while decision-making utilities reside in `player.llm.decision` for clearer navigation.
   - Each component has focused responsibilities and targeted unit tests.
   - Public API of `LLMPlayer` remains unchanged for backwards compatibility.

6. **Standardized Error Handling Policies**:
   - Decorator-based policies (`move_validation`, `network_operation`, `config_operation`, `metrics_operation`) capture consistent behavior.
   - Move parsing always raises typed `MoveError` subclasses, network errors bubble to experiment orchestration, metrics degrade gracefully.

7. **Factory-Based Configuration Assembly**:
   - Player, metrics, and game instantiation moved to `llm_chess_arena.factory` package.
   - `config.py` now focuses on schema composition and Hydra wiring, improving readability and testability.

8. **Separation of Concerns**:
   - Core chess logic remains independent of player implementations.
   - Prompt building, voting, parsing, and retry budgeting live in distinct modules.
   - Configuration parsing separated from object construction.

9. **Testing Guard Rails**:
   - Integration snapshots plus golden-master tests protect current LLM player behaviour.
   - Component-level unit tests exercise voting, prompt sessions, retry controller, and move parsing.
   - Hydra smoke tests ensure key configurations compose successfully.

10. **Metrics System Implementation**:
   - **Move-level metrics**: Stockfish-backed evaluation with centipawn loss, win probability delta, and best-move hits
   - **Quality classification**: Six-tier system (best, excellent, good, inaccuracy, mistake, blunder) using centipawn thresholds
   - **Real-time evaluation**: Tracks metrics during gameplay with per-player aggregation
   - **Post-game summaries**: Quality count distribution and average metrics for each player
   - **Graceful degradation**: Functions without Stockfish for development environments
   - **Future extensibility**: Designed for additional LLM-specific metrics (legal move rate, retry count, prompt efficiency)

11. **Hydra Configuration System**:
    - **Structured schema**: Dataclass-backed config parsing in `config.py` with runtime helpers colocated in `config.py` to instantiate players and metrics safely.
    - **Composable YAML groups**: Presets in `configs/` for game modes, players (including LLM connectors and Stockfish ELO tiers (1320/1400/1600/2000/2400/2800)), metrics defaults (with configurable thresholds), and Hydra runtime settings.
    - **Unified execution**: Hydra CLI runner (`python -m llm_chess_arena.cli.main`) and demo wrappers share the same configuration pipeline with override support.
    - **Sweep readiness**: Supports Hydra multirun parameter sweeps and reproducible output directories.

---

## TODO List

### Completed

**Foundation & Core:**
- Project setup, core game loop, base player abstraction
- Random/Stockfish/LLM players with majority voting
- Testing infrastructure, demo scripts

**Architecture Improvements:**
- DTOs for decoupling (PlayerDecisionContext/PlayerDecision)
- Standardized error taxonomy (MoveError hierarchy)
- Pre-commit hooks (ruff, black, mypy)
- Property-based tests with Hypothesis

**Metrics & Evaluation:**
- Stockfish-based move evaluation with centipawn loss
- Move quality classification (best, excellent, good, inaccuracy, mistake, blunder)
- Real-time move metrics tracking during gameplay
- Per-player averages and post-game summaries

**Visualization & Testing:**
- Rich terminal board visualization with color-coded moves
- Move quality annotations in game display
- VCR-based HTTP recording for reliable LLM integration tests
- Comprehensive test coverage with mocking and fixtures

**Major Refactoring:**
- LLM player decomposed into prompt, voting, parsing, and retry collaborators with focused unit tests
- Configuration factories extracted to `llm_chess_arena.factory` for cleaner Hydra integration
- Standardized error-handling decorators applied across parsing, factories, connector, and metrics
- Integration snapshots, golden-master baseline, and component tests protect behaviour during future changes

---

### Core Research Experiments

- [x] **Add Hydra configuration system**
  - Why: Enable massive batch experimentation and parameter sweeps essential for research
  - How: Replace current config.py, create yaml configs for players/games, CLI integration
  - Scope: Transform from single-game tool to research platform

- [ ] ** Opening move diversity of LLM players**
  - Why: Tests whether LLMs can generate a variety of reasonable moves or just converge to a few safe options
  - How: Measure entropy of move distributions across many games from same position
  - Scope: Could reveal fundamental limitations in LLM strategic flexibility

- [ ] **Opening-specific in-context learning experiment**
  - Why: Tests whether LLMs genuinely learn strategic patterns vs statistical mimicry - core question about in-context learning mechanisms
  - How: Show LLM 10 London System games vs 10 random games, measure counter-play improvement
  - Scope: Could reveal fundamental differences in how transformers acquire strategic knowledge

- [ ] **LLM tokenization effects on chess move processing**
  - Why: How transformers internally tokenize chess moves could fundamentally limit reasoning - affects ALL other experiments
  - How: Analyze how different LLMs tokenize chess notation, test if tokenization patterns correlate with performance
  - Scope: Foundational finding about transformer limitations in structured reasoning tasks

- [ ] **Information richness effects on LLM chess performance**
  - Why: Tests whether giving LLMs richer context (opponent reasoning, board visualizations, position analysis) dramatically improves play vs basic move strings
  - How: Compare LLM performance with: (1) just moves, (2) + opponent thoughts, (3) + board visualizations, (4) + position analysis
  - Scope: Could reveal how much LLM chess limitations are due to information poverty vs reasoning capacity

- [ ] **Human vs LLM learning curve comparison**
  - Why: Direct test of whether LLMs learn like humans or just pattern-match differently
  - How: Compare LLM improvement after seeing N games vs human improvement after N games
  - Scope: Could reveal fundamental differences between transformer and human strategic knowledge acquisition

### Supporting Infrastructure

- [x] **Stockfish move evaluation metrics**
  - Why: Provides objective ground truth for move quality assessment
  - How: Centipawn loss, win probability delta, move quality classification
  - Scope: Implemented in metrics.py with graceful degradation

- [ ] **Track legality and retry counters**
  - Why: Essential diagnostic metrics for LLM reliability
  - How: Per-game counters for illegal/parsing errors, emit summary at game_end
  - Scope: Updates to game.py and metrics.py

- [ ] **Batch experiment runner**
  - Why: Execute hundreds of games with different configurations automatically
  - How: CLI interface with Hydra configs, parallel execution, results aggregation
  - Scope: New cli/experiment.py with statistical analysis output

- [x] **Configurable PGN history export**
  - Why: Persist completed games for downstream analysis and replay
  - How: Config-driven PGN output path written after successful games
  - Scope: Game loop integration, Hydra config update, and regression tests

---

### Implementation Philosophy

**This is lab equipment for research, not a product.** Every feature should directly enable learning about LLM chess performance or accelerate the experiment cycle.

**Key Principles:**
- **Metrics first:** If we're not measuring it, we're not learning from it
- **Reproducibility:** Every experiment must be exactly repeatable
- **Rapid iteration:** Minimize time from hypothesis to results
- **Modular design:** Components should be independently testable and replaceable
- **Cost awareness:** Track API usage aggressively to enable large-scale experiments

**What we're NOT building:**
- Production-grade web interfaces or APIs
- CI/CD pipelines or deployment automation
- User authentication or multi-tenancy
- Real-time gameplay or spectator features
- Enterprise error handling (basic retry/timeout is sufficient)

---

### Future Work (Post-MVP)

<details>
<summary>Advanced Features</summary>

#### Tournament System
- Round-robin and Swiss tournaments
- ELO rating persistence
- Match scheduling and results database

#### Time Controls
- Classical, rapid, blitz time controls
- Fischer increment support
- Time pressure analysis

#### Web Interface
- FastAPI backend
- Real-time game streaming
- Tournament management UI
- Game analysis and replay

#### Advanced Analysis
- Opening book compliance
- Endgame tablebase integration
- Real-time commentary generation
- Positional complexity metrics

#### Visualization
- matplotlib/plotly charts
- Performance trends
- Head-to-head comparisons
- Rating progression graphs
</details>
