#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}/src"

python -m llm_chess_arena.cli.main \
  players@players.white=random \
  players.white.seed=42 \
  players.white.name="RandomBot 42" \
  players@players.black=random \
  players.black.seed=43 \
  players.black.name="RandomBot 43"
