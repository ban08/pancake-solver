from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from src.core.state import PancakeState
from src.core.search import SearchResult


# -----------------------------------------------------
# Puzzle loading
# -----------------------------------------------------

# Utilities for loading pancake problem instances from files


def load_puzzle(file_path: str | Path) -> PancakeState:
    """
    Load a pancake instance from a text file.

    Supported formats:
    - "3 1 2 4"
    - "4\n3 1 2 4"
    """

    # Convert input to Path object and validate existence
    path = Path(file_path)

    # Read non-empty lines from file
    if not path.exists():
        raise FileNotFoundError(f"Puzzle file not found: {path}")

    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]

    if not lines:
        raise ValueError(f"Empty puzzle file: {path}")

    # Case 1: single-line format (just pancake sequence)
    if len(lines) == 1:
        pancakes = _parse_pancakes(lines[0])

    # Case 2: first line is size, second line is pancake sequence
    elif len(lines) == 2:
        size = int(lines[0])
        pancakes = _parse_pancakes(lines[1])

        if len(pancakes) != size:
            raise ValueError("Size does not match pancake list.")

    # Invalid format (too many lines or unexpected structure)
    else:
        raise ValueError("Invalid puzzle format.")

    # Create and return PancakeState object
    return PancakeState(pancakes)


def load_puzzles(directory: str | Path) -> List[PancakeState]:
    """
    Load all .txt puzzles from a directory.
    """

    # Convert input to Path object and validate directory
    path = Path(directory)

    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    # Load all .txt files in sorted order
    return [load_puzzle(f) for f in sorted(path.glob("*.txt"))]


# -----------------------------------------------------
# CSV writing
# -----------------------------------------------------

# Utilities for exporting search results to CSV format


def write_results(
    results: Iterable[SearchResult],
    output_file: str | Path,
    instance_name: str | None = None,
) -> None:
    """
    Write results to CSV.
    """

    # Ensure output directory exists
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    # CSV column definitions
    fields = [
        "instance",
        "algorithm",
        "heuristic",
        "solved",
        "solution_cost",
        "solution_length",
        "nodes_expanded",
        "nodes_generated",
        "max_frontier_size",
        "runtime_seconds",
    ]

    # Write header only if file does not exist yet
    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        # Initialize CSV writer
        writer = csv.DictWriter(f, fieldnames=fields)

        # Write header row if needed
        if write_header:
            writer.writeheader()

        for r in results:
            # Write one row per SearchResult
            writer.writerow(
                {
                    "instance": instance_name,
                    "algorithm": r.algorithm_name,
                    "heuristic": r.heuristic_name,
                    "solved": r.solved,
                    "solution_cost": r.solution_cost,
                    "solution_length": len(r.solution_moves),
                    "nodes_expanded": r.nodes_expanded,
                    "nodes_generated": r.nodes_generated,
                    "max_frontier_size": r.max_frontier_size,
                    "runtime_seconds": r.runtime_seconds,
                }
            )


# -----------------------------------------------------
# Helpers
# -----------------------------------------------------

# Internal helper functions


def _parse_pancakes(line: str) -> list[int]:
    # Convert space-separated string into list of integers
    try:
        values = [int(x) for x in line.split()]
    except ValueError:
        raise ValueError(f"Invalid pancake line: {line}")

    # Ensure the list is not empty
    if not values:
        raise ValueError("Empty pancake line.")

    return values