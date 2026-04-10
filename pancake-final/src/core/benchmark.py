from __future__ import annotations
import sys

import multiprocessing as mp
import queue
import time
from pathlib import Path
from statistics import mean
from typing import Iterable, List

import psutil

# Ensure the project root is available on sys.path when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

# A safer default for real benchmarking.
DEFAULT_TIMEOUT_SECONDS = 10.0


# ---------------------------------------------------------
# Internal worker helpers
# ---------------------------------------------------------


def _solve_in_worker(
    result_queue,
    algorithm_name: str,
    initial_state: PancakeState,
    heuristic_name: str | None,
    weight: float | None,
) -> None:
    """
    Run one search algorithm inside a dedicated worker process.

    The result (or error) is pushed through a multiprocessing queue so the
    parent process can enforce a timeout and terminate the worker cleanly.
    """
    try:
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

        result_queue.put(("ok", result))
    except Exception as exc:
        result_queue.put(("error", str(exc)))


def _build_timeout_result(
    algorithm_name: str,
    heuristic_name: str | None,
    runtime_seconds: float,
) -> SearchResult:
    """
    Create a fallback SearchResult when a benchmark run times out.
    """
    return SearchResult(
        solved=False,
        solution_moves=[],
        solution_states=[],
        solution_cost=-1,
        nodes_expanded=0,
        nodes_generated=0,
        max_frontier_size=0,
        memory_used_bytes=0,
        runtime_seconds=runtime_seconds,
        algorithm_name=algorithm_name,
        heuristic_name=heuristic_name,
        timed_out=True,
    )


def _build_error_result(
    algorithm_name: str,
    heuristic_name: str | None,
    runtime_seconds: float,
) -> SearchResult:
    """
    Create a fallback SearchResult when a worker process fails.
    """
    return SearchResult(
        solved=False,
        solution_moves=[],
        solution_states=[],
        solution_cost=-1,
        nodes_expanded=0,
        nodes_generated=0,
        max_frontier_size=0,
        memory_used_bytes=0,
        runtime_seconds=runtime_seconds,
        algorithm_name=algorithm_name,
        heuristic_name=heuristic_name,
        timed_out=False,
    )


# ---------------------------------------------------------
# Run a single algorithm
# ---------------------------------------------------------


def run_algorithm(
    algorithm_name: str,
    initial_state: PancakeState,
    heuristic_name: str | None = None,
    weight: float | None = None,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
) -> SearchResult:
    """
    Run one search algorithm on a given initial state.

    The benchmark uses the shared `solve()` dispatcher so that CLI, GUI,
    and benchmark execution all follow the same code path.

    If `timeout_seconds` is provided, the algorithm runs in a dedicated
    child process and is terminated if it exceeds the limit.
    """
    start_time = time.perf_counter()

    def _peak_memory_of_process_tree(pid: int) -> int:
        """
        Return the current RSS of the worker process plus any live children.
        """
        try:
            root = psutil.Process(pid)
        except psutil.Error:
            return 0

        total_rss = 0
        processes = [root]

        try:
            processes.extend(root.children(recursive=True))
        except psutil.Error:
            pass

        for proc in processes:
            try:
                total_rss += proc.memory_info().rss
            except psutil.Error:
                continue

        return total_rss

    if timeout_seconds is None:
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
        current_process = psutil.Process()
        try:
            result.memory_used_bytes = current_process.memory_info().rss
        except psutil.Error:
            result.memory_used_bytes = 0
        result.algorithm_name = algorithm_name
        result.heuristic_name = heuristic_name
        result.runtime_seconds = end_time - start_time
        result.timed_out = False
        return result

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_solve_in_worker,
        args=(result_queue, algorithm_name, initial_state, heuristic_name, weight),
    )

    process.start()

    max_memory_bytes = 0
    deadline = None if timeout_seconds is None else start_time + timeout_seconds

    while process.is_alive():
        max_memory_bytes = max(max_memory_bytes, _peak_memory_of_process_tree(process.pid))

        if deadline is not None and time.perf_counter() >= deadline:
            process.terminate()
            process.join()
            elapsed = time.perf_counter() - start_time
            result_queue.close()
            timeout_result = _build_timeout_result(
                algorithm_name=algorithm_name,
                heuristic_name=heuristic_name,
                runtime_seconds=elapsed,
            )
            timeout_result.memory_used_bytes = max_memory_bytes
            return timeout_result

        time.sleep(0.01)

    process.join()
    max_memory_bytes = max(max_memory_bytes, _peak_memory_of_process_tree(process.pid))
    elapsed = time.perf_counter() - start_time

    try:
        status, payload = result_queue.get_nowait()
    except queue.Empty:
        result_queue.close()
        error_result = _build_error_result(
            algorithm_name=algorithm_name,
            heuristic_name=heuristic_name,
            runtime_seconds=elapsed,
        )
        error_result.memory_used_bytes = max_memory_bytes
        return error_result
    finally:
        result_queue.close()

    if status == "ok":
        result: SearchResult = payload
        result.memory_used_bytes = max_memory_bytes
        result.algorithm_name = algorithm_name
        result.heuristic_name = heuristic_name
        result.runtime_seconds = elapsed
        result.timed_out = False
        return result

    error_result = _build_error_result(
        algorithm_name=algorithm_name,
        heuristic_name=heuristic_name,
        runtime_seconds=elapsed,
    )
    error_result.memory_used_bytes = max_memory_bytes
    print(f"Worker error in {algorithm_name}: {payload}")
    return error_result


# ---------------------------------------------------------
# Run all algorithms on one puzzle
# ---------------------------------------------------------


def run_benchmark_on_state(
    initial_state: PancakeState,
    algorithms: list[tuple[str, str | None, float | None]] | None = None,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
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
            timeout_seconds=timeout_seconds,
        )
        results.append(result)

    return results


def run_benchmark_on_named_puzzle(
    puzzle_name: str,
    initial_state: PancakeState,
    algorithms: list[tuple[str, str | None, float | None]] | None = None,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
) -> List[SearchResult]:
    """
    Run all benchmark algorithms on one named puzzle and print a summary.
    """
    print(f"\nGame: {puzzle_name}")
    print(f"Initial state: {list(initial_state.pancakes)}")

    results = run_benchmark_on_state(
        initial_state=initial_state,
        algorithms=algorithms,
        timeout_seconds=timeout_seconds,
    )
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
        f"{'Timeout':<10}"
        f"{'Cost':<8}"
        f"{'Expanded':<12}"
        f"{'Generated':<12}"
        f"{'Frontier':<12}"
        f"{'Memory(MB)':<14}"
        f"{'Time(s)':<12}"
    )

    print("\n" + header)
    print("-" * len(header))

    for result in results:
        memory_mb = getattr(result, "memory_used_bytes", 0) / (1024 * 1024)
        print(
            f"{result.algorithm_name:<18}"
            f"{str(result.solved):<8}"
            f"{str(getattr(result, 'timed_out', False)):<10}"
            f"{str(result.solution_cost):<8}"
            f"{str(result.nodes_expanded):<12}"
            f"{str(result.nodes_generated):<12}"
            f"{str(result.max_frontier_size):<12}"
            f"{memory_mb:<14.6f}"
            f"{result.runtime_seconds:<12.6f}"
        )


def print_overall_summary(all_results: list[tuple[str, SearchResult]]) -> None:
    """
    Print an aggregate summary across all benchmarked puzzles.
    """
    if not all_results:
        return

    print("\nOVERALL SUMMARY")
    print("-" * 110)

    algorithm_names = sorted({result.algorithm_name for _, result in all_results})

    summary_header = (
        f"{'Algorithm':<18}"
        f"{'Solved':<12}"
        f"{'Timeouts':<12}"
        f"{'Avg Cost':<12}"
        f"{'Avg Expanded':<16}"
        f"{'Avg Memory(MB)':<18}"
        f"{'Avg Time(s)':<12}"
    )
    print(summary_header)
    print("-" * len(summary_header))

    for algorithm_name in algorithm_names:
        subset = [result for _, result in all_results if result.algorithm_name == algorithm_name]
        solved_count = sum(1 for result in subset if result.solved)
        timeout_count = sum(1 for result in subset if getattr(result, "timed_out", False))

        solved_subset = [result for result in subset if result.solved]
        avg_cost = mean(result.solution_cost for result in solved_subset) if solved_subset else 0.0
        avg_expanded = mean(result.nodes_expanded for result in subset)
        avg_memory_mb = mean(getattr(result, "memory_used_bytes", 0) for result in subset) / (1024 * 1024)
        avg_time = mean(result.runtime_seconds for result in subset)

        print(
            f"{algorithm_name:<18}"
            f"{f'{solved_count}/{len(subset)}':<12}"
            f"{timeout_count:<12}"
            f"{avg_cost:<12.2f}"
            f"{avg_expanded:<16.2f}"
            f"{avg_memory_mb:<18.6f}"
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
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
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

    timeout_seconds : float | None
        Maximum runtime allowed per algorithm per game.
        Use None to disable timeouts.
    """
    named_puzzles = load_named_puzzles(games_dir, recursive=recursive)

    if not named_puzzles:
        raise ValueError(f"No puzzle games found in {games_dir}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if save_csv and output_path.exists():
        output_path.unlink()

    all_results: list[tuple[str, SearchResult]] = []

    for puzzle_name, state in named_puzzles:
        results = run_benchmark_on_named_puzzle(
            puzzle_name=puzzle_name,
            initial_state=state,
            algorithms=algorithms,
            timeout_seconds=timeout_seconds,
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
    run_benchmark(
        PROJECT_ROOT / "games",
        output_file=PROJECT_ROOT / "results" / "benchmark_results.csv",
    )