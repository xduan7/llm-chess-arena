import subprocess
import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).parent.parent / "demo"


@pytest.mark.smoke
def test_demo__should_complete_successfully__when_running_random_game():
    script_path = DEMO_DIR / "run_game.py"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=script_path.parent.parent,
    )

    assert result.returncode == 0, (
        f"Random game demo failed with exit code {result.returncode}.\n"
        f"STDERR:\n{result.stderr}"
    )

    assert "Result:" in result.stdout
    assert "Termination:" in result.stdout
    assert "Total moves:" in result.stdout


@pytest.mark.smoke
@pytest.mark.requires_stockfish
def test_demo__should_complete_successfully__when_running_stockfish_game():
    script_path = DEMO_DIR / "run_stockfish_game.py"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=script_path.parent.parent,
    )

    assert result.returncode == 0, (
        f"Stockfish demo failed with exit code {result.returncode}.\n"
        f"STDERR:\n{result.stderr}"
    )

    assert "Result:" in result.stdout or "Winner:" in result.stdout
