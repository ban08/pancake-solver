from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.core.state import PancakeState


@dataclass(slots=True)
class SearchResult:
    """
    Standard result returned by all search algorithms.

    Notes:
    - Used as a unified output format for all search algorithms
    - Stores both solution information and performance metrics
    """

    # Solution information
    solved: bool = False

    solution_moves: List[int] = field(default_factory=list)
    solution_states: List[PancakeState] = field(default_factory=list)
    solution_cost: int = 0

    # Performance metrics
    nodes_expanded: int = 0
    nodes_generated: int = 0
    max_frontier_size: int = 0

    # Execution time
    runtime_seconds: float = 0.0

    # Metadata about the algorithm used
    algorithm_name: str = ""
    heuristic_name: Optional[str] = None

    # Return number of moves in the solution
    def solution_length(self) -> int:
        return len(self.solution_moves)

    # Build a human-readable summary of the result
    def summary(self) -> str:
        # Start with algorithm name
        lines = [f"Algorithm: {self.algorithm_name}"]

        # Include heuristic name if applicable
        if self.heuristic_name is not None:
            lines.append(f"Heuristic: {self.heuristic_name}")

        # Append main metrics and results
        lines.extend(
            [
                f"Solved: {self.solved}",
                f"Solution cost: {self.solution_cost}",
                f"Solution length: {self.solution_length()}",
                f"Nodes expanded: {self.nodes_expanded}",
                f"Nodes generated: {self.nodes_generated}",
                f"Max frontier size: {self.max_frontier_size}",
                f"Runtime (seconds): {self.runtime_seconds:.6f}",
            ]
        )

        return "\n".join(lines)

    # Compact debug representation
    def __repr__(self) -> str:
        return (
            f"SearchResult(solved={self.solved}, cost={self.solution_cost}, "
            f"expanded={self.nodes_expanded}, generated={self.nodes_generated}, "
            f"time={self.runtime_seconds:.6f})"
        )


# -----------------------------------------------------
# Solver dispatcher
# -----------------------------------------------------
# Routes a request to the appropriate search algorithm
# based on a string identifier.


def solve(
    initial_state: PancakeState,
    algorithm: str,
    heuristic: Optional[Callable[[PancakeState], int]] = None,
    weight: float = 1.5,
) -> SearchResult:
    """
    Central solver dispatcher used by the terminal interface.
    """

    # Normalize algorithm name for case-insensitive matching
    algorithm = algorithm.lower()

    # Lazy imports to avoid circular dependencies between modules
    from src.search.bfs import bfs
    from src.search.dfs import dfs
    from src.search.ucs import ucs
    from src.search.ids import ids
    from src.search.greedy import greedy
    from src.search.astar import astar
    from src.search.weighted_astar import weighted_astar

    # Breadth-First Search
    if algorithm == "bfs":
        return bfs(initial_state)

    # Depth-First Search
    if algorithm == "dfs":
        return dfs(initial_state)

    # Uniform Cost Search
    if algorithm == "ucs":
        return ucs(initial_state)

    # Iterative Deepening Search
    if algorithm == "ids":
        return ids(initial_state)

    # Greedy Best-First Search
    if algorithm == "greedy":
        if heuristic is None:
            raise ValueError("Greedy search requires a heuristic.")
        return greedy(initial_state, heuristic)

    # A* Search
    if algorithm == "astar":
        if heuristic is None:
            raise ValueError("A* requires a heuristic.")
        return astar(initial_state, heuristic)

    # Weighted A* Search
    if algorithm == "weighted_astar":
        if heuristic is None:
            raise ValueError("Weighted A* requires a heuristic.")
        return weighted_astar(initial_state, heuristic, weight)

    # Invalid algorithm name
    raise ValueError(f"Unknown algorithm: {algorithm}")