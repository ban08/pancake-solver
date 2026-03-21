from __future__ import annotations

import time
from collections import deque
from typing import Set

from src.core.state import PancakeState
from src.core.node import Node
from src.core.search import SearchResult


def bfs(initial_state: PancakeState) -> SearchResult:
    """
    Breadth-First Search (BFS).

    Notes:
    - Explores nodes level by level using a FIFO queue
    - Guarantees shortest path when all step costs are equal

    Guarantees optimal solution in number of flips.
    """

    # Start timer for performance measurement
    start_time = time.perf_counter()

    # Initialize root node (starting state)
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
            algorithm_name="BFS",
        )

    # Frontier is a FIFO queue (breadth-first exploration)
    frontier = deque([root])
    # Track visited states to avoid revisiting
    visited: Set[PancakeState] = {initial_state}

    # Tracking performance metrics
    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while frontier:
        # Dequeue next node (FIFO order)
        node = frontier.popleft()
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
                algorithm_name="BFS",
            )

        # Expand successors (all possible flips)
        for move, successor_state in node.state.get_successors():
            # Skip already visited states
            if successor_state in visited:
                continue

            # Mark state as visited
            visited.add(successor_state)

            # Create node for successor
            child = Node(
                state=successor_state,
                parent=node,
                move=move,
                g=node.g + 1,
            )

            # Add successor to frontier (end of queue)
            frontier.append(child)
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
        algorithm_name="BFS",
    )