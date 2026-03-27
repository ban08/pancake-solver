from __future__ import annotations

import time

from src.core.node import Node
from src.core.state import PancakeState
from src.core.search import SearchResult


def _dls(node: Node, limit: int, path: set[PancakeState]):
    """
    Depth-limited DFS used by IDS.

    Notes:
    - Recursive depth-limited DFS used by IDS
    - Returns goal node, flags (found/cutoff), and performance metrics
    """

    # Check if current node is goal
    if node.state.is_goal():
        return node, True, False, 0, 0, 1  # goal, found, cutoff, metrics

    # If depth limit reached, signal cutoff
    if limit == 0:
        return None, False, True, 0, 0, 1

    # Tracking metrics for this subtree
    nodes_expanded = 1
    nodes_generated = 0
    max_frontier = 1
    cutoff_occurred = False

    # Explore successors (depth-first)
    for move, successor_state in node.state.get_successors():
        # Avoid revisiting states in current path (cycle prevention)
        if successor_state in path:
            continue

        # Create child node for successor
        child = Node(
            state=successor_state,
            parent=node,
            move=move,
            g=node.g + 1,
        )

        nodes_generated += 1

        # Add to current path before recursion
        path.add(successor_state)
        # Recursive depth-limited search on child
        goal, found, cutoff, e, g, f = _dls(child, limit - 1, path)
        # Remove from path after recursion (backtracking)
        path.remove(successor_state)

        # Accumulate metrics from subtree
        nodes_expanded += e
        nodes_generated += g
        max_frontier = max(max_frontier, 1 + f)

        # If goal found, propagate result upward
        if found:
            return goal, True, False, nodes_expanded, nodes_generated, max_frontier

        # Track if any branch hit the depth limit
        if cutoff:
            cutoff_occurred = True

    # Return whether cutoff occurred if no solution found
    return None, False, cutoff_occurred, nodes_expanded, nodes_generated, max_frontier


def ids(initial_state: PancakeState) -> SearchResult:
    """
    Iterative Deepening Search (IDS).

    Finds optimal solutions for unit-cost problems.

    Notes:
    - Repeatedly applies depth-limited search with increasing depth
    - Combines DFS space efficiency with BFS optimality
    """

    # Start timer for performance measurement
    start_time = time.perf_counter()

    # Initialize root node
    root = Node(state=initial_state, g=0)

    # Handle trivial case where initial state is already goal
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
            algorithm_name="IDS",
        )

    # Aggregate metrics across all depth iterations
    total_expanded = 0
    total_generated = 1
    max_frontier_size = 1

    # Initial depth limit
    depth = 0

    while True:
        # Perform depth-limited search with current limit

        # Track current path to prevent cycles
        path = {initial_state}

        # Run depth-limited search
        goal, found, cutoff, expanded, generated, frontier = _dls(
            root, depth, path
        )

        # Accumulate metrics from this iteration
        total_expanded += expanded
        total_generated += generated
        max_frontier_size = max(max_frontier_size, frontier)

        # If solution found, return result
        if found and goal is not None:
            moves = goal.solution_moves()
            states = goal.solution_states()

            return SearchResult(
                solved=True,
                solution_moves=moves,
                solution_states=states,
                solution_cost=len(moves),
                nodes_expanded=total_expanded,
                nodes_generated=total_generated,
                max_frontier_size=max_frontier_size,
                runtime_seconds=time.perf_counter() - start_time,
                algorithm_name="IDS",
            )

        # If no cutoff occurred, search space fully explored
        if not cutoff:
            return SearchResult(
                solved=False,
                solution_moves=[],
                solution_states=[],
                solution_cost=0,
                nodes_expanded=total_expanded,
                nodes_generated=total_generated,
                max_frontier_size=max_frontier_size,
                runtime_seconds=time.perf_counter() - start_time,
                algorithm_name="IDS",
            )

        # Increase depth limit and repeat
        depth += 1