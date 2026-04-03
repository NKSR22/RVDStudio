from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def venv_python() -> Path:
    if sys.platform.startswith("win"):
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a project virtual environment and install requirements.")
    return parser.parse_args()


def main() -> int:
    parse_args()
    run([sys.executable, "-m", "venv", str(VENV_DIR)])
    python_path = str(venv_python())
    run([python_path, "-m", "pip", "install", "--upgrade", "pip"])
    run([python_path, "-m", "pip", "install", "-r", "requirements.txt"])
    print(f"Virtual environment ready at: {VENV_DIR}")
    if sys.platform.startswith("win"):
        print(r"Activate with: .venv\Scripts\activate")
    else:
        print("Activate with: source .venv/bin/activate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
