#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}/src"

python -m llm_chess_arena.cli.main \
  tournament.output_dir="${ROOT_DIR}/output/demo" \
  tournament.match_name="stockfish_game" \
  tournament.num_games=1 \
  game.display_board=false \
  game.max_num_moves=120 \
  metrics.stockfish_depth=12 \
  metrics.stockfish_engine_options.Threads=2 \
  metrics.stockfish_engine_options.Hash=256 \
  players@players.white=stockfish/elo_2800 \
  players@players.black=stockfish/elo_1320
