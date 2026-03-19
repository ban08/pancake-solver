from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterable, List

from src.core.state import PancakeState
from src.core.io import load_puzzle, write_results_csv
from src.core.heuristics import gap_heuristic
from src.core.search import SearchResult, solve


# ---------------------------------------------------------
# Algorithm definition
# ---------------------------------------------------------

Algorithm = Callable[[PancakeState], SearchResult]


# ---------------------------------------------------------
# Run a single algorithm
# ---------------------------------------------------------

def run_algorithm(
    algorithm_name: str,
    initial_state: PancakeState,
    heuristic_name: str | None = None,
    weight: float = 1.5,
) -> SearchResult:
    """
    Run a search algorithm and measure runtime.

    This uses the central `solve()` dispatcher so that
    benchmark, GUI, and CLI share the same execution path.
    """

    start = time.perf_counter()

    if heuristic_name:
        from src.core.heuristics import get_heuristic
        heuristic = get_heuristic(heuristic_name)
        result = solve(initial_state, algorithm=algorithm_name, heuristic=heuristic, weight=weight)
        result.heuristic_name = heuristic_name
    else:
        result = solve(initial_state, algorithm=algorithm_name)

    end = time.perf_counter()

    result.runtime_seconds = end - start
    result.algorithm_name = algorithm_name

    return result


# ---------------------------------------------------------
# Run all algorithms on one instance
# ---------------------------------------------------------

def run_benchmark_on_instance(instance_path: Path) -> List[SearchResult]:
    """
    Run all algorithms on a single puzzle instance.
    """

    print(f"\nINSTANCE: {instance_path}")

    initial_state = load_puzzle(instance_path)

    algorithms = [
        ("bfs", None),
        ("dfs", None),
        ("ucs", None),
        ("ids", None),
        ("greedy", "gap"),
        ("astar", "gap"),
        ("weighted_astar", "gap"),
    ]

    results: List[SearchResult] = []

    for algo_name, heuristic in algorithms:
        print(f"Running {algo_name}...")

        if algo_name == "weighted_astar":
            result = run_algorithm(
                algo_name,
                initial_state,
                heuristic_name=heuristic,
                weight=1.5,
            )
        else:
            result = run_algorithm(
                algo_name,
                initial_state,
                heuristic_name=heuristic,
            )

        results.append(result)

    print_results(results)

    return results


# ---------------------------------------------------------
# Pretty console output
# ---------------------------------------------------------

def print_results(results: Iterable[SearchResult]) -> None:
    """
    Print benchmark results in a readable table.
    """

    header = (
        f"{'Algorithm':<15}"
        f"{'Solved':<8}"
        f"{'Cost':<8}"
        f"{'Expanded':<12}"
        f"{'Generated':<12}"
        f"{'Frontier':<12}"
        f"{'Time(s)':<10}"
    )

    print("\n" + header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r.algorithm_name:<15}"
            f"{str(r.solved):<8}"
            f"{r.solution_cost:<8}"
            f"{r.nodes_expanded:<12}"
            f"{r.nodes_generated:<12}"
            f"{r.max_frontier_size:<12}"
            f"{r.runtime_seconds:<10.6f}"
        )


# ---------------------------------------------------------
# Run benchmark across instance directory
# ---------------------------------------------------------

def run_benchmark(instances_dir: str | Path, save_csv: bool = True) -> None:
    """
    Run benchmarks on all instances inside a directory tree.
    """

    base_path = Path(instances_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"Instances directory not found: {base_path}")

    instance_files = sorted(base_path.rglob("*.txt"))

    if not instance_files:
        raise ValueError(f"No instance files found in {base_path}")

    output_file = Path("results") / "benchmark_results.csv"

    if save_csv and output_file.exists():
        output_file.unlink()

    for instance_file in instance_files:

        results = run_benchmark_on_instance(instance_file)

        if save_csv:
            instance_name = str(instance_file.relative_to(base_path))
            write_results_csv(results, output_file, instance_name=instance_name)


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    run_benchmark("instances")