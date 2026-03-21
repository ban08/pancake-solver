from __future__ import annotations

"""
Main entry point for the Pancake Solver (checkpoint version).

This file provides a simple terminal interface to:
- Select a predefined puzzle (easy / medium / hard)
- Choose a search algorithm
- Execute it and display results

Design notes:
- Games are embedded directly (no file I/O for checkpoint)
- Algorithms are mapped in a registry for clean selection
- Heuristics are injected only where needed (Greedy, A*, Weighted A*)
"""

import sys
from pathlib import Path
from typing import Callable

# Ensure imports work whether running as a module or script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.heuristics import gap_heuristic
from src.search.bfs import bfs
from src.search.dfs import dfs
from src.search.ucs import ucs
from src.search.ids import ids
from src.search.greedy import greedy
from src.search.astar import astar
from src.search.weighted_astar import weighted_astar
from src.core.state import PancakeState


SearchFunction = Callable[[object], object]


# Predefined puzzle instances (checkpoint simplification: no external files)
GAMES: dict[str, tuple[str, PancakeState]] = {
    "1": ("easy", PancakeState([2, 1, 3, 4])),
    "2": ("medium", PancakeState([3, 1, 4, 2, 5])),
    "3": ("hard", PancakeState([4, 1, 3, 6, 2, 5])),
}

# Algorithm registry: maps menu options to (name, function)
# Lambdas are used where additional parameters (heuristic, weight) are required
ALGORITHMS: dict[str, tuple[str, SearchFunction]] = {
    "1": ("BFS", bfs),
    "2": ("DFS", dfs),
    "3": ("UCS", ucs),
    "4": ("IDS", ids),
    "5": ("Greedy", lambda state: greedy(state, gap_heuristic)),
    "6": ("A*", lambda state: astar(state, gap_heuristic)),
    "7": (
        "Weighted A*",
        lambda state: weighted_astar(state, gap_heuristic, weight=1.5),
    ),
}


# Prompt user to choose one of the predefined games
def select_game_key() -> str:
    print("\nAvailable games:")
    for key, (label, _) in GAMES.items():
        print(f"  {key}. {label}")

    while True:
        choice = input("\nChoose a game number: ").strip()
        if choice in GAMES:
            return choice
        print("Please enter a valid number from 1 to 3.")


def select_algorithm() -> str:
    """
    Prompt the user to choose one algorithm or all algorithms.
    """
    print("\nAvailable algorithms:")
    for key, (label, _) in ALGORITHMS.items():
        print(f"  {key}. {label}")
    print("  8. Run all")

    while True:
        choice = input("\nChoose an algorithm number: ").strip()
        if choice in ALGORITHMS or choice == "8":
            return choice
        print("Please enter a valid number from 1 to 8.")


# Format and display the result returned by a search algorithm
def print_result(result) -> None:
    """
    Print a search result in a clean terminal format.
    """
    print("\n--- Result ---")
    print(result.summary())
    print(f"Solution moves: {result.solution_moves}")
    print("Solution states:")
    for state in result.solution_states:
        print(f"  {state}")


# Execute a single algorithm selected by the user
def run_single_algorithm(algorithm_key: str, initial_state) -> None:
    """
    Run one selected algorithm and print its result.
    """
    algorithm_name, algorithm_function = ALGORITHMS[algorithm_key]
    print(f"\nRunning {algorithm_name}...")
    result = algorithm_function(initial_state)
    print_result(result)


# Execute all algorithms sequentially (useful for comparison)
def run_all_algorithms(initial_state) -> None:
    """
    Run all available algorithms and print each result.
    """
    for algorithm_key in ALGORITHMS:
        run_single_algorithm(algorithm_key, initial_state)


def main() -> None:
    """
    Terminal entry point for the Pancake Solver checkpoint delivery.
    """
    print("Pancake Solver")

    # Step 1: choose problem instance
    game_key = select_game_key()
    selected_game_name, initial_state = GAMES[game_key]

    print(f"\nSelected game: {selected_game_name}")
    print(f"Initial state: {initial_state}")

    # Step 2: choose algorithm(s)
    selected_algorithm = select_algorithm()

    # Step 3: execute selection
    if selected_algorithm == "8":
        run_all_algorithms(initial_state)
    else:
        run_single_algorithm(selected_algorithm, initial_state)


if __name__ == "__main__":
    main()