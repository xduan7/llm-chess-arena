#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}/src"

python -m llm_chess_arena.cli.main \
  tournament.output_dir="${ROOT_DIR}/output/demo" \
  tournament.match_name="random_game" \
  tournament.num_games=1 \
  game.max_num_moves=100 \
  game.enable_metrics=false \
  players@players.white=random \
  players.white.seed=42 \
  players.white.name="RandomBot 42" \
  players@players.black=random \
  players.black.seed=43 \
  players.black.name="RandomBot 43"
