from __future__ import annotations

import heapq
import time
from typing import Callable

from src.core.node import Node
from src.core.state import PancakeState
from src.core.search import SearchResult


def greedy(
    initial_state: PancakeState,
    heuristic: Callable[[PancakeState], int],
) -> SearchResult:
    """
    Greedy Best-First Search.

    Uses:
        f(n) = h(n)

    Fast, but not optimal.

    Notes:
    - Uses a priority queue ordered only by heuristic h(n)
    - Ignores path cost g(n), so it may find suboptimal solutions
    """

    # Start timer for performance measurement
    start_time = time.perf_counter()

    # Initialize root node with heuristic value
    root = Node(
        state=initial_state,
        g=0,
        h=heuristic(initial_state),
    )

    # Handle trivial case where initial state is already the goal
    if initial_state.is_goal():
        return SearchResult(
            solved=True,
            solution_moves=[],
            solution_states=[initial_state],
            solution_cost=0,
            nodes_expanded=0,
            nodes_generated=1,
            max_frontier_size=1,
            runtime_seconds=time.perf_counter() - start_time,
            algorithm_name="Greedy",
            heuristic_name=heuristic.__name__,
        )

    # Frontier is a priority queue ordered by h(n) (via Node.f)
    frontier: list[Node] = [root]
    heapq.heapify(frontier)

    # Track visited states to avoid revisiting
    visited: set[PancakeState] = set()

    # Tracking performance metrics
    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while frontier:
        # Extract node with lowest heuristic value
        node = heapq.heappop(frontier)

        # Skip already visited states
        if node.state in visited:
            continue

        # Mark state as visited
        visited.add(node.state)
        # Count node expansion
        nodes_expanded += 1

        # Check if goal is reached
        if node.state.is_goal():
            moves = node.solution_moves()
            states = node.solution_states()

            return SearchResult(
                solved=True,
                solution_moves=moves,
                solution_states=states,
                solution_cost=len(moves),
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                max_frontier_size=max_frontier_size,
                runtime_seconds=time.perf_counter() - start_time,
                algorithm_name="Greedy",
                heuristic_name=heuristic.__name__,
            )

        # Expand successors (all possible flips)
        for move, successor_state in node.state.get_successors():
            # Skip successors already visited
            if successor_state in visited:
                continue

            # Create node for successor (heuristic-driven)
            child = Node(
                state=successor_state,
                parent=node,
                move=move,
                g=node.g + 1,
                h=heuristic(successor_state),
            )

            # Add successor to frontier
            heapq.heappush(frontier, child)
            # Count generated node
            nodes_generated += 1

        # Track maximum frontier size
        max_frontier_size = max(max_frontier_size, len(frontier))

    # Return failure if no solution is found
    return SearchResult(
        solved=False,
        solution_moves=[],
        solution_states=[],
        solution_cost=0,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier_size=max_frontier_size,
        runtime_seconds=time.perf_counter() - start_time,
        algorithm_name="Greedy",
        heuristic_name=heuristic.__name__,
    )