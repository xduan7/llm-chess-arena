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
    ├── config.py
    ├── exceptions.py
    ├── game.py
    ├── metrics.py            # Stockfish-based move evaluation
    ├── renderer.py           # Rich terminal board visualization
    ├── types.py
    ├── utils.py
    └── player/
        ├── base_player.py
        ├── random_player.py
        ├── stockfish_player.py
        └── llm/
            ├── __init__.py
            ├── llm_player.py
            ├── llm_connector.py    # LiteLLM wrapper for testing isolation
            └── llm_move_handler.py # Move parsing and templating

configs/
├── config.yaml
├── env/
│   └── default.yaml
├── game/
│   └── classical.yaml
├── metrics/
│   └── default.yaml
├── players/
│   ├── random.yaml
│   ├── stockfish.yaml
│   ├── stockfish/
│   │   ├── elo_2800.yaml
│   │   ├── elo_2400.yaml
│   │   ├── elo_2000.yaml
│   │   ├── elo_1800.yaml
│   │   ├── elo_1600.yaml
│   │   ├── elo_1400.yaml
│   │   └── elo_1200.yaml
│   └── llm/
│       ├── gpt4.yaml
│       ├── claude.yaml
│       └── gemini.yaml
└── hydra/
    └── default.yaml

demo/
├── run_random_game.sh
├── run_stockfish_game.sh
└── run_llm_game.sh

tests/
├── __init__.py
├── conftest.py
├── test_demos.py
├── fixtures/
│   └── mock_llm_connector.py
├── unit/
│   ├── test_config.py
│   ├── test_game.py
│   ├── test_metrics.py
│   ├── test_types.py
│   ├── test_utils.py
│   ├── test_utils_property.py  # Property-based tests with Hypothesis
│   └── player/
│       ├── test_base_player.py
│       ├── test_random_player.py
│       ├── test_stockfish_player.py
│       ├── test_llm_connector.py
│       ├── test_llm_player.py
│       ├── test_llm_move_handler.py
│       └── test_llm_voting.py
└── integration/
    ├── test_chess_edge_cases.py
    ├── test_game_scenarios.py
    ├── test_llm_integration.py       # Environment-gated real API tests
    ├── test_llm_integration_vcr.py   # VCR-based tests with recordings
    └── cassettes/                     # VCR HTTP recordings for testing
        ├── llm_complex_position.yaml
        ├── llm_endgame_position.yaml
        ├── llm_majority_voting.yaml
        └── [other VCR recordings...]

.env                 # Local environment (gitignored)
.env.example         # Template for API keys
.pre-commit-config.yaml
pyproject.toml
README.md
AGENTS.md
CLAUDE.md -> AGENTS.md    # Symlink to AGENTS.md
LICENSE
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

5. **Separation of Concerns**:
   - Core chess logic independent of player implementations
   - Move parsing/templating separated from LLM communication
   - Voting logic separated from retry logic (critical for context preservation)

6. **Synchronous Architecture**: Starting with synchronous, single-game execution for simplicity. This avoids the complexity of async/concurrent code while we validate the core functionality.

7. **LLM Player Retry Strategy** (Critical Design Decision):
   - **Problem**: When using majority voting (n_samples > 1), retry context could mismatch the actual error
   - **Solution**: Separated voting from retry logic - `_try_get_move_with_voting()` is pure voting
   - **Implementation**:
     - Initial prompt generated ONCE outside retry loop (preserves context)
     - Voting attempts are separate from retry attempts
     - On voting failure, capture single sample for accurate error context
     - Network errors propagate immediately (not recoverable via chess retries)
   - **Rationale**: Following Game Arena's Option 2 - clean separation prevents state confusion

8. **Majority Voting Implementation**:
   - Use UCI notation for unambiguous move comparison
   - Tie-breaking by first occurrence (deterministic)
   - Invalid samples excluded from voting (not counted as votes)
   - Network errors during voting fail immediately (affect all samples)

9. **Error Handling Philosophy**:
   - Chess errors (invalid/illegal moves) trigger retries with context
   - Network errors fail fast (no point retrying with chess prompts)
   - Clear error messages with actionable information
   - API key errors caught and displayed with setup instructions

10. **Metrics System Implementation**:
   - **Move-level metrics**: Stockfish-backed evaluation with centipawn loss, win probability delta, and best-move hits
   - **Quality classification**: Six-tier system (best, excellent, good, inaccuracy, mistake, blunder) using centipawn thresholds
   - **Real-time evaluation**: Tracks metrics during gameplay with per-player aggregation
   - **Post-game summaries**: Quality count distribution and average metrics for each player
   - **Graceful degradation**: Functions without Stockfish for development environments
   - **Future extensibility**: Designed for additional LLM-specific metrics (legal move rate, retry count, prompt efficiency)

11. **Hydra Configuration System**:
    - **Structured schema**: Dataclass-backed config parsing in `config.py` with runtime helpers colocated in `config.py` to instantiate players and metrics safely.
    - **Composable YAML groups**: Presets in `configs/` for game modes, players (including LLM connectors and Stockfish ELO tiers (1320/1600/2000/2400/2800) (1320/1600/2000/2400/2800)), metrics defaults (with configurable thresholds), and Hydra runtime settings.
    - **Unified execution**: Hydra CLI runner (`python -m llm_chess_arena.cli.play`) and demo wrappers share the same configuration pipeline with override support.
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

---

### Core Research Experiments

- [x] **Add Hydra configuration system**
  - Why: Enable massive batch experimentation and parameter sweeps essential for research
  - How: Replace current config.py, create yaml configs for players/games, CLI integration
  - Scope: Transform from single-game tool to research platform

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

