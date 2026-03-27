from __future__ import annotations

import time
from pathlib import Path
from statistics import mean
from typing import Iterable, List

from src.core.io import load_named_puzzles, write_named_results_csv
from src.core.search import SearchResult, solve
from src.core.state import PancakeState


# ---------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------

DEFAULT_ALGORITHMS: list[tuple[str, str | None, float | None]] = [
    ("bfs", None, None),
    ("dfs", None, None),
    ("ucs", None, None),
    ("ids", None, None),
    ("greedy", "gap", None),
    ("astar", "gap", None),
    ("weighted_astar", "gap", 1.5),
]


# ---------------------------------------------------------
# Run a single algorithm
# ---------------------------------------------------------


def run_algorithm(
    algorithm_name: str,
    initial_state: PancakeState,
    heuristic_name: str | None = None,
    weight: float | None = None,
) -> SearchResult:
    """
    Run one search algorithm on a given initial state.

    The benchmark uses the shared `solve()` dispatcher so that CLI, GUI,
    and benchmark execution all follow the same code path.
    """
    start_time = time.perf_counter()

    if heuristic_name is not None:
        from src.core.heuristics import get_heuristic

        heuristic_fn = get_heuristic(heuristic_name)

        if weight is not None:
            result = solve(
                initial_state,
                algorithm=algorithm_name,
                heuristic=heuristic_fn,
                weight=weight,
            )
        else:
            result = solve(
                initial_state,
                algorithm=algorithm_name,
                heuristic=heuristic_fn,
            )
    else:
        result = solve(initial_state, algorithm=algorithm_name)

    end_time = time.perf_counter()

    result.algorithm_name = algorithm_name
    result.heuristic_name = heuristic_name
    result.runtime_seconds = end_time - start_time

    return result


# ---------------------------------------------------------
# Run all algorithms on one puzzle
# ---------------------------------------------------------


def run_benchmark_on_state(
    initial_state: PancakeState,
    algorithms: list[tuple[str, str | None, float | None]] | None = None,
) -> List[SearchResult]:
    """
    Run a benchmark over a single puzzle state.
    """
    benchmark_algorithms = algorithms or DEFAULT_ALGORITHMS
    results: List[SearchResult] = []

    for algorithm_name, heuristic_name, weight in benchmark_algorithms:
        print(f"Running {algorithm_name}...")
        result = run_algorithm(
            algorithm_name=algorithm_name,
            initial_state=initial_state,
            heuristic_name=heuristic_name,
            weight=weight,
        )
        results.append(result)

    return results



def run_benchmark_on_named_puzzle(
    puzzle_name: str,
    initial_state: PancakeState,
    algorithms: list[tuple[str, str | None, float | None]] | None = None,
) -> List[SearchResult]:
    """
    Run all benchmark algorithms on one named puzzle and print a summary.
    """
    print(f"\ngame: {puzzle_name}")
    print(f"Initial state: {list(initial_state.pancakes)}")

    results = run_benchmark_on_state(initial_state, algorithms)
    print_results_table(results)

    return results


# ---------------------------------------------------------
# Pretty console output
# ---------------------------------------------------------


def print_results_table(results: Iterable[SearchResult]) -> None:
    """
    Print result rows in a readable fixed-width table.
    """
    header = (
        f"{'Algorithm':<18}"
        f"{'Solved':<8}"
        f"{'Cost':<8}"
        f"{'Expanded':<12}"
        f"{'Generated':<12}"
        f"{'Frontier':<12}"
        f"{'Time(s)':<12}"
    )

    print("\n" + header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result.algorithm_name:<18}"
            f"{str(result.solved):<8}"
            f"{str(result.solution_cost):<8}"
            f"{str(result.nodes_expanded):<12}"
            f"{str(result.nodes_generated):<12}"
            f"{str(result.max_frontier_size):<12}"
            f"{result.runtime_seconds:<12.6f}"
        )



def print_overall_summary(all_results: list[tuple[str, SearchResult]]) -> None:
    """
    Print an aggregate summary across all benchmarked puzzles.
    """
    if not all_results:
        return

    print("\nOVERALL SUMMARY")
    print("-" * 80)

    algorithm_names = sorted({result.algorithm_name for _, result in all_results})

    summary_header = (
        f"{'Algorithm':<18}"
        f"{'Solved':<12}"
        f"{'Avg Cost':<12}"
        f"{'Avg Expanded':<16}"
        f"{'Avg Time(s)':<12}"
    )
    print(summary_header)
    print("-" * len(summary_header))

    for algorithm_name in algorithm_names:
        subset = [result for _, result in all_results if result.algorithm_name == algorithm_name]
        solved_count = sum(1 for result in subset if result.solved)

        avg_cost = mean(result.solution_cost for result in subset if result.solved) if any(result.solved for result in subset) else 0.0
        avg_expanded = mean(result.nodes_expanded for result in subset)
        avg_time = mean(result.runtime_seconds for result in subset)

        print(
            f"{algorithm_name:<18}"
            f"{f'{solved_count}/{len(subset)}':<12}"
            f"{avg_cost:<12.2f}"
            f"{avg_expanded:<16.2f}"
            f"{avg_time:<12.6f}"
        )


# ---------------------------------------------------------
# Benchmark directory runner
# ---------------------------------------------------------


def run_benchmark(
    games_dir: str | Path,
    save_csv: bool = True,
    recursive: bool = True,
    output_file: str | Path = "results/benchmark_results.csv",
    algorithms: list[tuple[str, str | None, float | None]] | None = None,
) -> None:
    """
    Run benchmarks for all puzzle files inside a directory.

    Parameters
    ----------
    games_dir : str | Path
        Directory containing puzzle files.

    save_csv : bool
        If True, export results to CSV.

    recursive : bool
        If True, also search subdirectories.

    output_file : str | Path
        CSV path used when `save_csv=True`.

    algorithms : list[...] | None
        Optional custom algorithm configuration.
    """
    named_puzzles = load_named_puzzles(games_dir, recursive=recursive)

    if not named_puzzles:
        raise ValueError(f"No puzzle games found in {games_dir}")

    output_path = Path(output_file)
    if save_csv and output_path.exists():
        output_path.unlink()

    all_results: list[tuple[str, SearchResult]] = []

    for puzzle_name, state in named_puzzles:
        results = run_benchmark_on_named_puzzle(
            puzzle_name=puzzle_name,
            initial_state=state,
            algorithms=algorithms,
        )

        named_results = [(puzzle_name, result) for result in results]
        all_results.extend(named_results)

        if save_csv:
            write_named_results_csv(
                named_results=named_results,
                output_file=output_path,
                append=True,
            )

    print_overall_summary(all_results)


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------


if __name__ == "__main__":
    run_benchmark("games")