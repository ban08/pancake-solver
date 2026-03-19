from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.core.state import PancakeState


@dataclass(slots=True)
class SearchResult:
    """
    Standardized result returned by all search algorithms.
    """

    # --- status ---
    solved: bool = False

    # --- solution data ---
    solution_moves: List[int] = field(default_factory=list)
    solution_states: List[PancakeState] = field(default_factory=list)
    solution_cost: int = 0

    # --- search metrics ---
    nodes_expanded: int = 0
    nodes_generated: int = 0
    max_frontier_size: int = 0

    # --- performance metrics ---
    runtime_seconds: float = 0.0
    memory_bytes: Optional[int] = None

    # --- algorithm metadata ---
    algorithm_name: str = ""
    heuristic_name: Optional[str] = None

    def has_solution(self) -> bool:
        return self.solved

    def solution_length(self) -> int:
        return len(self.solution_moves)

    def summary(self) -> str:
        lines = [f"Algorithm: {self.algorithm_name}"]

        if self.heuristic_name:
            lines.append(f"Heuristic: {self.heuristic_name}")

        lines.extend([
            f"Solved: {self.solved}",
            f"Solution cost: {self.solution_cost}",
            f"Solution length: {self.solution_length()}",
            f"Nodes expanded: {self.nodes_expanded}",
            f"Nodes generated: {self.nodes_generated}",
            f"Max frontier size: {self.max_frontier_size}",
            f"Runtime (seconds): {self.runtime_seconds:.6f}",
        ])

        if self.memory_bytes is not None:
            lines.append(f"Memory (bytes): {self.memory_bytes}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            "SearchResult("
            f"solved={self.solved}, "
            f"cost={self.solution_cost}, "
            f"expanded={self.nodes_expanded}, "
            f"generated={self.nodes_generated}, "
            f"time={self.runtime_seconds:.6f}"
            ")"
        )


# ------------------------------------------------------------------
# Solver Dispatcher
# ------------------------------------------------------------------

def solve(
    initial_state: PancakeState,
    algorithm: str,
    heuristic=None,
    weight: float = 1.5,
) -> SearchResult:
    """
    Central solver dispatcher used by GUI, CLI and benchmark code.
    """

    algorithm = algorithm.lower()

    # Lazy imports to avoid circular dependencies
    from src.search.bfs import bfs
    from src.search.dfs import dfs
    from src.search.ucs import ucs
    from src.search.ids import ids
    from src.search.greedy import greedy
    from src.search.astar import astar
    from src.search.weighted_astar import weighted_astar

    if algorithm == "bfs":
        return bfs(initial_state)

    if algorithm == "dfs":
        return dfs(initial_state)

    if algorithm == "ucs":
        return ucs(initial_state)

    if algorithm == "ids":
        return ids(initial_state)

    if algorithm == "greedy":
        if heuristic is None:
            raise ValueError("Greedy search requires a heuristic.")
        return greedy(initial_state, heuristic)

    if algorithm == "astar":
        if heuristic is None:
            raise ValueError("A* requires a heuristic.")
        return astar(initial_state, heuristic)

    if algorithm == "weighted_astar":
        if heuristic is None:
            raise ValueError("Weighted A* requires a heuristic.")
        return weighted_astar(initial_state, heuristic, weight)

    raise ValueError(f"Unknown algorithm: {algorithm}")