#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}/src"

OUTPUT_ROOT="${ROOT_DIR}/output/demo/stockfish_game"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_FILE="${OUTPUT_ROOT}/$(date -u +"%Y%m%dT%H%M%SZ").pgn"

python -m llm_chess_arena.cli.main \
  "game.history_output_path=${OUTPUT_FILE}" \
  players@players.white=stockfish/elo_2800 \
  players@players.black=stockfish/elo_1320
