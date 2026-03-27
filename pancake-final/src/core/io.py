from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Sequence

from src.core.search import SearchResult
from src.core.state import PancakeState


# ---------------------------------------------------------
# Puzzle Loading
# ---------------------------------------------------------


def load_puzzle(file_path: str | Path) -> PancakeState:
    """
    Load a pancake puzzle from a text file.

    Supported formats:

    1) Single-line format
       Example:
           3 1 2 4

    2) Two-line format
       Example:
           4
           3 1 2 4

    Blank lines and lines starting with '#' are ignored.
    """
    path = Path(file_path)
    _ensure_existing_file(path)

    lines = _read_relevant_lines(path)

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



def list_puzzle_files(
    directory: str | Path,
    recursive: bool = False,
) -> List[Path]:
    """
    Return all puzzle files in a directory, sorted by name.

    Parameters
    ----------
    directory : str | Path
        Directory containing puzzle files.

    recursive : bool
        If True, also search subdirectories.
    """
    base_dir = Path(directory)
    _ensure_existing_directory(base_dir)

    pattern = "**/*.txt" if recursive else "*.txt"
    return sorted(path for path in base_dir.glob(pattern) if path.is_file())



def load_multiple_puzzles(
    directory: str | Path,
    recursive: bool = False,
) -> List[PancakeState]:
    """
    Load all puzzle files from a directory.

    Useful for benchmarking when only the states are needed.
    """
    return [load_puzzle(file_path) for file_path in list_puzzle_files(directory, recursive)]



def load_named_puzzles(
    directory: str | Path,
    recursive: bool = False,
) -> List[tuple[str, PancakeState]]:
    """
    Load all puzzle files from a directory, preserving a readable name.

    Useful for benchmarking and reporting, where both the puzzle name and
    puzzle state are needed.
    """
    base_dir = Path(directory)
    puzzle_files = list_puzzle_files(base_dir, recursive)

    named_puzzles: List[tuple[str, PancakeState]] = []

    for file_path in puzzle_files:
        if recursive:
            name = str(file_path.relative_to(base_dir))
        else:
            name = file_path.name
        named_puzzles.append((name, load_puzzle(file_path)))

    return named_puzzles



def save_puzzle(
    puzzle: PancakeState | Sequence[int],
    output_file: str | Path,
    include_size_line: bool = False,
) -> None:
    """
    Save a puzzle to a text file.

    Parameters
    ----------
    puzzle : PancakeState | Sequence[int]
        Puzzle to save.

    output_file : str | Path
        Destination file.

    include_size_line : bool
        If True, write the size on the first line and the puzzle on the second.
    """
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    values = list(puzzle.pancakes if isinstance(puzzle, PancakeState) else puzzle)
    line = " ".join(str(value) for value in values)

    if include_size_line:
        content = f"{len(values)}\n{line}\n"
    else:
        content = f"{line}\n"

    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------
# CSV Result Writing
# ---------------------------------------------------------


def write_results_csv(
    results: Iterable[SearchResult],
    output_file: str | Path,
    game_name: str | None = None,
    append: bool = True,
) -> None:
    """
    Write search results to a CSV file.

    Parameters
    ----------
    results : Iterable[SearchResult]
        Results to write.

    output_file : str | Path
        Destination CSV file.

    game_name : str | None
        Optional puzzle name to associate with every result.

    append : bool
        If True, append to the file. If False, overwrite it.
    """
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "game",
        "algorithm",
        "heuristic",
        "solved",
        "timed_out",
        "solution_cost",
        "solution_length",
        "nodes_expanded",
        "nodes_generated",
        "max_frontier_size",
        "runtime_seconds",
    ]

    mode = "a" if append else "w"
    should_write_header = not append or not path.exists()

    with path.open(mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if should_write_header:
            writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "game": game_name,
                    "algorithm": result.algorithm_name,
                    "heuristic": result.heuristic_name,
                    "solved": result.solved,
                    "timed_out": getattr(result, "timed_out", False),
                    "solution_cost": result.solution_cost,
                    "solution_length": len(result.solution_moves),
                    "nodes_expanded": result.nodes_expanded,
                    "nodes_generated": result.nodes_generated,
                    "max_frontier_size": result.max_frontier_size,
                    "runtime_seconds": result.runtime_seconds,
                }
            )



def write_named_results_csv(
    named_results: Iterable[tuple[str, SearchResult]],
    output_file: str | Path,
    append: bool = True,
) -> None:
    """
    Write (game_name, result) pairs to a CSV file.

    This is useful for benchmarking pipelines where the puzzle name varies per row.
    """
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "game",
        "algorithm",
        "heuristic",
        "solved",
        "timed_out",
        "solution_cost",
        "solution_length",
        "nodes_expanded",
        "nodes_generated",
        "max_frontier_size",
        "runtime_seconds",
    ]

    mode = "a" if append else "w"
    should_write_header = not append or not path.exists()

    with path.open(mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if should_write_header:
            writer.writeheader()

        for game_name, result in named_results:
            writer.writerow(
                {
                    "game": game_name,
                    "algorithm": result.algorithm_name,
                    "heuristic": result.heuristic_name,
                    "solved": result.solved,
                    "timed_out": getattr(result, "timed_out", False),
                    "solution_cost": result.solution_cost,
                    "solution_length": len(result.solution_moves),
                    "nodes_expanded": result.nodes_expanded,
                    "nodes_generated": result.nodes_generated,
                    "max_frontier_size": result.max_frontier_size,
                    "runtime_seconds": result.runtime_seconds,
                }
            )


# ---------------------------------------------------------
# Validation / parsing helpers
# ---------------------------------------------------------


def _ensure_existing_file(path: Path) -> None:
    """Ensure that a given path exists and is a file."""
    if not path.exists():
        raise FileNotFoundError(f"Puzzle file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected a puzzle file, got: {path}")



def _ensure_existing_directory(path: Path) -> None:
    """Ensure that a given path exists and is a directory."""
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {path}")



def _read_relevant_lines(path: Path) -> list[str]:
    """
    Read puzzle file lines, ignoring blanks and comment lines.
    """
    content = path.read_text(encoding="utf-8").strip()

    if not content:
        raise ValueError(f"Puzzle file is empty: {path}")

    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not lines:
        raise ValueError(f"Puzzle file contains no puzzle data: {path}")

    return lines



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