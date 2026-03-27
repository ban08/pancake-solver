from __future__ import annotations

import heapq
import time
from typing import Callable, Dict

from src.core.node import Node
from src.core.search import SearchResult
from src.core.state import PancakeState


def astar(
    initial_state: PancakeState,
    heuristic: Callable[[PancakeState], int],
) -> SearchResult:
    """
    A* search for the pancake puzzle.

    Uses:
        f(n) = g(n) + h(n)

    Notes:
    - Uses a priority queue ordered by f = g + h
    - Tracks best known cost per state to avoid revisiting worse paths
    """

    # Start timer for performance measurement
    start_time = time.perf_counter()

    # Initialize root node with g=0 and heuristic value
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
            algorithm_name="A*",
            heuristic_name=heuristic.__name__,
        )

    # Frontier is a priority queue (min-heap) ordered by node.f
    frontier: list[Node] = [root]
    heapq.heapify(frontier)

    # Stores the best (lowest) g value found for each state
    best_g: Dict[PancakeState, int] = {initial_state: 0}

    # Tracking performance metrics
    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while frontier:
        # Extract node with lowest f(n)
        node = heapq.heappop(frontier)

        # Skip if this node has a worse g than the best known
        if node.g > best_g.get(node.state, float("inf")):
            continue

        # Count node expansion
        nodes_expanded += 1

        # Check if goal is reached
        if node.state.is_goal():
            return SearchResult(
                solved=True,
                solution_moves=node.solution_moves(),
                solution_states=node.solution_states(),
                solution_cost=len(node.solution_moves()),
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                max_frontier_size=max_frontier_size,
                runtime_seconds=time.perf_counter() - start_time,
                algorithm_name="A*",
                heuristic_name=heuristic.__name__,
            )

        # Expand successors (all possible flips)
        for move, successor_state in node.state.get_successors():
            # Cost of reaching successor
            new_g = node.g + 1

            # Skip if we already have a better path to this state
            if new_g >= best_g.get(successor_state, float("inf")):
                continue

            # Update best cost for this state
            best_g[successor_state] = new_g

            # Create new node for successor
            child = Node(
                state=successor_state,
                parent=node,
                move=move,
                g=new_g,
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
        algorithm_name="A*",
        heuristic_name=heuristic.__name__,
    )