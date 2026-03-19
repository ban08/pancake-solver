from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from src.core.state import PancakeState
from src.core.search import SearchResult


# ---------------------------------------------------------
# Puzzle Loading
# ---------------------------------------------------------

def load_puzzle(file_path: str | Path) -> PancakeState:
    """
    Load a pancake puzzle instance from a text file.

    Supported formats:

    1) Single-line format
       Example:
           3 1 2 4

    2) Two-line format
       Example:
           4
           3 1 2 4
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Puzzle file not found: {path}")

    content = path.read_text(encoding="utf-8").strip()

    if not content:
        raise ValueError(f"Puzzle file is empty: {path}")

    lines = [line.strip() for line in content.splitlines() if line.strip()]

    if len(lines) == 1:
        pancakes = _parse_pancake_line(lines[0])

    elif len(lines) == 2:
        expected_size = _parse_size_line(lines[0])
        pancakes = _parse_pancake_line(lines[1])

        if len(pancakes) != expected_size:
            raise ValueError(
                f"Puzzle size mismatch in file {path}: "
                f"expected {expected_size} pancakes, got {len(pancakes)}"
            )

    else:
        raise ValueError(
            f"Invalid puzzle format in file {path}. "
            "Expected either 1 line or 2 lines."
        )

    return PancakeState(pancakes)


def load_multiple_puzzles(directory: str | Path) -> List[PancakeState]:
    """
    Load all puzzle instances from a directory.

    Useful for benchmarking.
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    puzzles: List[PancakeState] = []

    for file in sorted(directory.glob("*.txt")):
        puzzles.append(load_puzzle(file))

    return puzzles


# ---------------------------------------------------------
# CSV Result Writing
# ---------------------------------------------------------

def write_results_csv(
    results: Iterable[SearchResult],
    output_file: str | Path,
    instance_name: str | None = None,
) -> None:
    """
    Write benchmark results to a CSV file.
    """

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
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

    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "instance": instance_name,
                    "algorithm": result.algorithm_name,
                    "heuristic": result.heuristic_name,
                    "solved": result.solved,
                    "solution_cost": result.solution_cost,
                    "solution_length": len(result.solution_moves),
                    "nodes_expanded": result.nodes_expanded,
                    "nodes_generated": result.nodes_generated,
                    "max_frontier_size": result.max_frontier_size,
                    "runtime_seconds": result.runtime_seconds,
                }
            )


# ---------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------

def _parse_size_line(line: str) -> int:
    """
    Parse the size line from a two-line puzzle file.
    """
    try:
        size = int(line)
    except ValueError as exc:
        raise ValueError(f"Invalid puzzle size: {line!r}") from exc

    if size <= 0:
        raise ValueError(f"Puzzle size must be positive, got {size}")

    return size


def _parse_pancake_line(line: str) -> list[int]:
    """
    Parse a line containing pancake values separated by whitespace.
    """
    try:
        pancakes = [int(value) for value in line.split()]
    except ValueError as exc:
        raise ValueError(f"Invalid pancake values in line: {line!r}") from exc

    if not pancakes:
        raise ValueError("Pancake line cannot be empty.")

    return pancakes