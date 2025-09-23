#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}/src"

OUTPUT_ROOT="${ROOT_DIR}/output/demo/random_game"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_FILE="${OUTPUT_ROOT}/$(date -u +"%Y%m%dT%H%M%SZ").pgn"

python -m llm_chess_arena.cli.main \
  "game.history_output_path=${OUTPUT_FILE}" \
  game.max_num_moves=100 \
  game.enable_metrics=false \
  players@players.white=random \
  players.white.seed=42 \
  players.white.name="RandomBot 42" \
  players@players.black=random \
  players.black.seed=43 \
  players.black.name="RandomBot 43"
