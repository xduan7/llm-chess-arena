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
llm_chess_arena/
├── __init__.py
├── config.py
├── exceptions.py
├── game.py
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

configs/              # Empty - Hydra configs to be implemented
├── config.yaml
├── player/
│   ├── random.yaml
│   ├── stockfish.yaml
│   ├── gpt4.yaml
│   └── claude.yaml
└── game/
    ├── classical.yaml
    └── blitz.yaml

demo/
├── run_game.py
├── run_stockfish_game.py
└── run_llm_game.py

tests/
├── conftest.py
├── test_demos.py
├── fixtures/
│   └── mock_llm_connector.py
├── unit/
│   ├── test_config.py
│   ├── test_game.py
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
    ├── test_llm_integration_vcr.py   # VCR-based tests (future)
    └── cassettes/                     # Empty - VCR recordings (future)

.env                 # Local environment (gitignored)
.env.example         # Template for API keys
.pre-commit-config.yaml
pyproject.toml
README.md
CLAUDE.md
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

10. **Metrics System Design** (Planned):
   - **Move-level metrics**: Quality scoring using Stockfish as ground truth
   - **Centipawn loss**: Standard chess metric for move quality
   - **LLM-specific metrics**: Legal move rate, retry count, prompt efficiency
   - **Game phases**: Separate evaluation for opening, middlegame, endgame
   - **Comparison framework**: ELO-style ratings for relative strength
   - **Real-time evaluation**: Track metrics during gameplay, not just post-game
   - **Extensibility**: Plugin architecture for custom metrics

11. **Move Metrics Evaluation**:
    - Stockfish-backed evaluator computes centipawn loss, win probability delta, and best-move hits after every move.
    - Aggregates per-player averages for centipawn loss and best-move hit rate that surface in post-game summaries.
    - Gracefully disables when Stockfish is unavailable so development environments without the engine still function.


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

---

### Foundation - Metrics & Instrumentation

- [ ] **Add MetricsEvent and MetricsSink interface**
  - Why: Establish structured stream for all research metrics without entangling logic
  - How: Define MetricsEvent(dataclass) with type, timestamp, game_id, ply, player, payload. Inject emit points in LLMPlayer and Game for move_request, model_response, move_selected, move_applied, game_end
  - Scope: New metrics.py, small changes to llm_player.py and game.py

- [ ] **Implement JSONL metrics sink**
  - Why: JSONL is standard research logging format for analysis and replay
  - How: JsonlSink(file_path) appends structured events as lines, enable via CLI flag
  - Scope: New sinks/jsonl_sink.py, wire-up in game bootstrap

- [ ] **Add token/cost/latency capture from LiteLLM**
  - Why: Cost/efficiency tracking essential for comparing strategies across runs
  - How: Parse tokens, cost, latency, retry count from LiteLLM response/headers
  - Scope: Update llm_connector.py and llm_player.py

- [x] **Add Stockfish centipawn evaluation hook**
  - Why: Centipawn loss is core objective metric for rapid model comparisons
  - How: Depth-limited eval (configurable depth=10) after each move, emit engine_eval events
  - Scope: Extend metrics.py with Stockfish evaluator and integrate hooks in game.py

- [ ] **Track legality and retry counters**
  - Why: Legality rate is key LLM capability metric, retries affect cost/latency
  - How: Maintain per-game counters for illegal/parsing errors, emit summary at game_end
  - Scope: Updates to game.py and metrics.py

### Experimentation Infrastructure

- [ ] **Create DecisionStrategy interface above LLMPlayer**
  - Why: Cleanly compare prompting strategies without refactoring core player logic
  - How: Protocol with decide(board, context) -> MoveDecision. Implement SingleShot and VoteMajority strategies
  - Scope: New strategies/base.py, adapt llm_player.py to delegate decisions

- [ ] **Build minimal round-robin tournament CLI**
  - Why: Repeatable head-to-head experiments and batching essential for research
  - How: CLI args for players, games count, seeds, output path. Produces JSONL and CSV summary
  - Scope: New cli/arena.py, extend existing tournament structure

- [ ] **Add game lifecycle callbacks**
  - Why: Decouple metrics, evaluation, and visualization from core game rules
  - How: on_pre_move, on_post_move, on_game_end callback lists for extensibility
  - Scope: Update game.py, optional new callbacks.py

- [ ] **Create board/context serialization helpers**
  - Why: Reliable replay and analysis require stable serialization formats
  - How: to_ascii(board), to_json(game_state, context), leverage python-chess FEN/PGN
  - Scope: New utils/serialization.py, minor calls in game.py

### Analysis & Visualization

- [ ] **Implement Rich terminal board visualization**
  - Why: Fast human QA during research runs, better debugging experience
  - How: render(board, last_move, centipawn?) with color, toggle via --visualize-every N
  - Scope: New viz/rich_board.py, optional integration in CLI

- [ ] **Build JSONL game viewer for replay**
  - Why: Post-hoc debugging and qualitative assessment of games
  - How: Stream JSONL events, reconstruct board state, step through with keypress
  - Scope: New cli/view.py, reuse rich_board.py

- [ ] **Add cost/metrics post-run summarizer**
  - Why: Quick feedback loop on spend vs strength for experiment planning
  - How: Read JSONL, aggregate tokens, cost, latency, legality, centipawn stats
  - Scope: New analysis/cost_report.py, optional call from arena CLI

### Modularity & Configuration

- [ ] **Modularize prompt templates**
  - Why: Enable clean A/B testing of prompts without code changes
  - How: Load templates by name, support .format or Jinja2, expose via CLI
  - Scope: New prompts/templates/ directory, update llm_player.py

- [ ] **Create registry-based factory for players/strategies**
  - Why: Avoid conditional explosions, simplify adding new baselines
  - How: Simple dict name->callable registry for LLMPlayer, RandomPlayer, StockfishPlayer, strategies
  - Scope: New registry.py, update arena CLI

- [ ] **Add optional response cache for determinism**
  - Why: Enable controlled experiments and reduced API spend during development
  - How: Hash key = model+template+prompt+FEN+ply, file-backed cache with TTL
  - Scope: New utils/cache.py, integration in llm_player.py

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













