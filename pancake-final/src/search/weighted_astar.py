from __future__ import annotations

import heapq
import time
from typing import Callable, Dict

from src.core.node import Node
from src.core.search import SearchResult
from src.core.state import PancakeState


def weighted_astar(
    initial_state: PancakeState,
    heuristic: Callable[[PancakeState], int],
    weight: float = 1.5,
) -> SearchResult:
    """
    Weighted A* search.

    Uses:
        f(n) = g(n) + w * h(n)

    Faster than standard A*, but not guaranteed to be optimal.

    Notes:
    - Uses a priority queue ordered by f = g + w*h
    - Biases the search towards the heuristic (higher w = more greedy)
    """

    # Start timer for performance measurement
    start_time = time.perf_counter()

    # Initialize root node with heuristic value
    root = Node(
        state=initial_state,
        g=0,
        h=heuristic(initial_state),
    )
    # Compute weighted evaluation function for root
    root.f = root.g + weight * root.h

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
            algorithm_name="Weighted A*",
            heuristic_name=heuristic.__name__,
        )

    # Frontier is a priority queue ordered by f(n)
    frontier: list[Node] = [root]
    heapq.heapify(frontier)

    # Stores best (lowest) f value seen for each state (pruning)
    best_f: Dict[PancakeState, float] = {initial_state: root.f}

    # Tracking performance metrics
    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while frontier:
        # Extract node with lowest f(n)
        node = heapq.heappop(frontier)

        # Skip nodes with outdated (worse) f values
        if node.f > best_f.get(node.state, float("inf")):
            continue

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
                algorithm_name="Weighted A*",
                heuristic_name=heuristic.__name__,
            )

        # Expand successors (all possible flips)
        for move, successor_state in node.state.get_successors():
            # Compute path cost to successor
            g = node.g + 1
            # Compute heuristic value
            h = heuristic(successor_state)
            # Compute weighted evaluation f = g + w*h
            f = g + weight * h

            # Skip if a better f value is already known for this state
            if f >= best_f.get(successor_state, float("inf")):
                continue

            # Update best f value for this state
            best_f[successor_state] = f

            # Create node for successor
            child = Node(
                state=successor_state,
                parent=node,
                move=move,
                g=g,
                h=h,
            )
            # Override default f with weighted value
            child.f = f

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
        algorithm_name="Weighted A*",
        heuristic_name=heuristic.__name__,
    )