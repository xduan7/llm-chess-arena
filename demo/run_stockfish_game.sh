#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}/src"

OUTPUT_ROOT="${ROOT_DIR}/output/demo/stockfish_game"
mkdir -p "${OUTPUT_ROOT}"

python -m llm_chess_arena.cli.main \
  "game.record_dir=${OUTPUT_ROOT}" \
  game.display_board=false \
  game.max_num_moves=120 \
  metrics.stockfish_depth=12 \
  metrics.stockfish_engine_options.Threads=2 \
  metrics.stockfish_engine_options.Hash=256 \
  players@players.white=stockfish/elo_2800 \
  players@players.black=stockfish/elo_1320
