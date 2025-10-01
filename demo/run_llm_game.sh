#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}/src"

python -m llm_chess_arena.cli.main \
  tournament.output_dir="${ROOT_DIR}/output/demo" \
  tournament.match_name="llm_game" \
  tournament.num_games=1 \
  game.enable_metrics=false \
  players@players.white=llm/default \
  players.white.connector.model=gpt-4o-mini \
  players.white.name="GPT-4o Mini" \
  players@players.black=random
