from __future__ import annotations

import csv
from pathlib import Path

from src.core.benchmark import run_algorithm
from src.core.io import load_named_puzzles


HEURISTICS = ["gap", "misplaced", "zero"]


def main() -> None:
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)
    output = analysis_dir / "astar_heuristics_all_games.csv"

    rows: list[dict[str, object]] = []

    for game_name, state in load_named_puzzles("games", recursive=True):
        print(f"GAME {game_name}: {list(state.pancakes)}")

        for heuristic in HEURISTICS:
            result = run_algorithm(
                algorithm_name="astar",
                initial_state=state,
                heuristic_name=heuristic,
                timeout_seconds=10.0,
            )

            row = {
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
                "memory_mb": result.memory_used_bytes / (1024 * 1024),
                "runtime_seconds": result.runtime_seconds,
            }
            rows.append(row)

            print(
                f"  astar/{heuristic:<10} solved={row['solved']:<5} "
                f"timeout={row['timed_out']:<5} cost={row['solution_cost']:<4} "
                f"expanded={row['nodes_expanded']:<8} time={row['runtime_seconds']:.4f}s"
            )

    with output.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
