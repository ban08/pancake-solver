from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from src.core.state import PancakeState
from src.core.search import SearchResult


# -----------------------------------------------------
# Puzzle loading
# -----------------------------------------------------


def load_puzzle(file_path: str | Path) -> PancakeState:
    """
    Load a pancake instance from a text file.

    Supported formats:
    - "3 1 2 4"
    - "4\n3 1 2 4"
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Puzzle file not found: {path}")

    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]

    if not lines:
        raise ValueError(f"Empty puzzle file: {path}")

    if len(lines) == 1:
        pancakes = _parse_pancakes(lines[0])

    elif len(lines) == 2:
        size = int(lines[0])
        pancakes = _parse_pancakes(lines[1])

        if len(pancakes) != size:
            raise ValueError("Size does not match pancake list.")

    else:
        raise ValueError("Invalid puzzle format.")

    return PancakeState(pancakes)


def load_puzzles(directory: str | Path) -> List[PancakeState]:
    """
    Load all .txt puzzles from a directory.
    """

    path = Path(directory)

    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    return [load_puzzle(f) for f in sorted(path.glob("*.txt"))]


# -----------------------------------------------------
# CSV writing
# -----------------------------------------------------


def write_results(
    results: Iterable[SearchResult],
    output_file: str | Path,
    instance_name: str | None = None,
) -> None:
    """
    Write results to CSV.
    """

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

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

    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        if write_header:
            writer.writeheader()

        for r in results:
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


def _parse_pancakes(line: str) -> list[int]:
    try:
        values = [int(x) for x in line.split()]
    except ValueError:
        raise ValueError(f"Invalid pancake line: {line}")

    if not values:
        raise ValueError("Empty pancake line.")

    return values