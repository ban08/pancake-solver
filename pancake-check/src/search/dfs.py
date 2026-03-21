from __future__ import annotations

import time

from src.core.node import Node
from src.core.state import PancakeState
from src.core.search import SearchResult


def dfs(initial_state: PancakeState) -> SearchResult:
    """
    Depth-First Search (DFS).

    Explores deepest nodes first using a stack.
    Does not guarantee optimal solutions.
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
            algorithm_name="DFS",
        )

    frontier: list[Node] = [root]  # stack
    visited: set[PancakeState] = set()

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while frontier:
        node = frontier.pop()

        if node.state in visited:
            continue

        visited.add(node.state)
        nodes_expanded += 1

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
                algorithm_name="DFS",
            )

        for move, successor_state in node.state.get_successors():
            if successor_state in visited:
                continue

            child = Node(
                state=successor_state,
                parent=node,
                move=move,
                g=node.g + 1,
            )

            frontier.append(child)
            nodes_generated += 1

        max_frontier_size = max(max_frontier_size, len(frontier))

    return SearchResult(
        solved=False,
        solution_moves=[],
        solution_states=[],
        solution_cost=0,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier_size=max_frontier_size,
        runtime_seconds=time.perf_counter() - start_time,
        algorithm_name="DFS",
    )