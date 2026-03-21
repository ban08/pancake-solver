from __future__ import annotations

import time

from src.core.node import Node
from src.core.state import PancakeState
from src.core.search import SearchResult


def _dls(node: Node, limit: int, path: set[PancakeState]):
    """
    Depth-limited DFS used by IDS.
    """

    if node.state.is_goal():
        return node, True, False, 0, 0, 1  # goal, found, cutoff, metrics

    if limit == 0:
        return None, False, True, 0, 0, 1

    nodes_expanded = 1
    nodes_generated = 0
    max_frontier = 1
    cutoff_occurred = False

    for move, successor_state in node.state.get_successors():
        if successor_state in path:
            continue

        child = Node(
            state=successor_state,
            parent=node,
            move=move,
            g=node.g + 1,
        )

        nodes_generated += 1

        path.add(successor_state)
        goal, found, cutoff, e, g, f = _dls(child, limit - 1, path)
        path.remove(successor_state)

        nodes_expanded += e
        nodes_generated += g
        max_frontier = max(max_frontier, 1 + f)

        if found:
            return goal, True, False, nodes_expanded, nodes_generated, max_frontier

        if cutoff:
            cutoff_occurred = True

    return None, False, cutoff_occurred, nodes_expanded, nodes_generated, max_frontier


def ids(initial_state: PancakeState) -> SearchResult:
    """
    Iterative Deepening Search (IDS).

    Finds optimal solutions for unit-cost problems.
    """

    start_time = time.perf_counter()

    root = Node(state=initial_state, g=0)

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

    total_expanded = 0
    total_generated = 1
    max_frontier_size = 1

    depth = 0

    while True:
        path = {initial_state}

        goal, found, cutoff, expanded, generated, frontier = _dls(
            root, depth, path
        )

        total_expanded += expanded
        total_generated += generated
        max_frontier_size = max(max_frontier_size, frontier)

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

        depth += 1