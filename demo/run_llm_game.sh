#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}/src"

OUTPUT_ROOT="${ROOT_DIR}/output/demo/llm_game"
mkdir -p "${OUTPUT_ROOT}"

python -m llm_chess_arena.cli.main \
  "game.record_dir=${OUTPUT_ROOT}" \
  game.enable_metrics=false \
  players@players.white=llm/default \
  players.white.connector.model=gpt-4o-mini \
  players.white.name="GPT-4o Mini" \
  players@players.black=random
