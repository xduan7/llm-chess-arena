"""Smoke tests covering the demo entry points."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).parent.parent / "demo"


def _demo_env() -> dict[str, str]:
    """Environment for demo subprocesses with this interpreter on PATH.

    The demo scripts invoke bare ``python``; when pytest runs from a venv
    without activation, that binary is not on PATH.
    """
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    return env


@pytest.mark.smoke
def test_demo__should_complete_successfully__when_running_random_game():
    """Smoke test for the random-vs-random demo script."""
    script_path = DEMO_DIR / "run_random_game.sh"

    result = subprocess.run(
        ["bash", str(script_path)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=script_path.parent.parent,
        env=_demo_env(),
    )

    assert result.returncode == 0, (
        f"Random game demo failed with exit code {result.returncode}.\n"
        f"STDERR:\n{result.stderr}"
    )

    assert "Tournament Results" in result.stdout
    assert "Total Games:" in result.stdout
    assert "W/D/L:" in result.stdout


@pytest.mark.smoke
@pytest.mark.requires_stockfish
def test_demo__should_complete_successfully__when_running_stockfish_game():
    """Smoke test for the Stockfish demo script."""
    script_path = DEMO_DIR / "run_stockfish_game.sh"

    result = subprocess.run(
        ["bash", str(script_path)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=script_path.parent.parent,
        env=_demo_env(),
    )

    assert result.returncode == 0, (
        f"Stockfish demo failed with exit code {result.returncode}.\n"
        f"STDERR:\n{result.stderr}"
    )

    assert "Tournament Results" in result.stdout


@pytest.mark.smoke
@pytest.mark.requires_llm
def test_demo__should_complete_successfully__when_running_llm_game():
    """Smoke test for the LLM demo script."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not configured")

    script_path = DEMO_DIR / "run_llm_game.sh"

    try:
        result = subprocess.run(
            ["bash", str(script_path)],
            capture_output=True,
            text=True,
            timeout=300,  # LLM calls with retries can take longer
            cwd=script_path.parent.parent,
            env=_demo_env(),  # Current environment plus this interpreter on PATH
        )
    except subprocess.TimeoutExpired:
        pytest.skip("LLM demo timed out - API may be slow or unavailable")

    assert result.returncode == 0, (
        f"LLM game demo failed with exit code {result.returncode}.\n"
        f"STDERR:\n{result.stderr}"
    )

    assert "Tournament Results" in result.stdout
    assert "GPT-4o Mini" in result.stdout  # Player name from demo script
