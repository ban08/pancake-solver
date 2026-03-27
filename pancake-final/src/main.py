"""
Main entry point for the Pancake Solver project.

Provides a simple terminal menu:
    0 -> Exit
    1 -> Launch GUI
    2 -> Run benchmark

Works with:
    python -m src.main
    python src/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.gui import main as run_gui
from src.core.benchmark import run_benchmark


def show_menu() -> None:
    print("\n=== Pancake Solver ===")
    print("0. Exit")
    print("1. Run GUI")
    print("2. Run Benchmark")


def main() -> None:
    while True:
        show_menu()

        choice = input("Choose an option: ").strip()

        if choice == "0":
            print("Exiting...")
            break

        elif choice == "1":
            print("Launching GUI...")
            run_gui()

        elif choice == "2":
            print("Running benchmark...")
            try:
                run_benchmark(PROJECT_ROOT / "games")
            except Exception as e:
                print(f"Error running benchmark: {e}")

        else:
            print("Invalid option. Try again.")


if __name__ == "__main__":
    main()