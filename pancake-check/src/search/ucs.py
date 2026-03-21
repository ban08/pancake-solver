from __future__ import annotations

import heapq
import time
from typing import Dict

from src.core.node import Node
from src.core.state import PancakeState
from src.core.search import SearchResult


def ucs(initial_state: PancakeState) -> SearchResult:
    """
    Uniform Cost Search (UCS).

    Equivalent to BFS for unit-cost problems,
    but implemented with a priority queue.

    Notes:
    - Uses a priority queue ordered by path cost g(n)
    - Equivalent to BFS when all step costs are equal (unit-cost)
    """

    # Start timer for performance measurement
    start_time = time.perf_counter()

    # Initialize root node with zero path cost
    root = Node(state=initial_state, g=0)

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
            algorithm_name="UCS",
        )

    # Frontier is a priority queue ordered by g(n) (lowest cost first)
    frontier: list[Node] = [root]
    heapq.heapify(frontier)

    # Stores the best (lowest) cost found for each state
    best_g: Dict[PancakeState, int] = {initial_state: 0}

    # Tracking performance metrics
    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while frontier:
        # Extract node with lowest path cost
        node = heapq.heappop(frontier)

        # Skip outdated paths (we already found a cheaper way to this state)
        if node.g > best_g.get(node.state, float("inf")):
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
                algorithm_name="UCS",
            )

        # Expand successors (all possible flips)
        for move, successor_state in node.state.get_successors():
            # Cost to reach successor
            new_g = node.g + 1

            # Skip if a better path to this state is already known
            if new_g >= best_g.get(successor_state, float("inf")):
                continue

            # Update best known cost for this state
            best_g[successor_state] = new_g

            # Create node for successor
            child = Node(
                state=successor_state,
                parent=node,
                move=move,
                g=new_g,
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
        algorithm_name="UCS",
    )