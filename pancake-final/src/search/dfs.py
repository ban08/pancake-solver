from __future__ import annotations

import time

from src.core.node import Node
from src.core.state import PancakeState
from src.core.search import SearchResult


def dfs(initial_state: PancakeState) -> SearchResult:
    """
    Depth-First Search for the Pancake Puzzle.

    DFS explores the deepest nodes first using a stack.
    It does not guarantee an optimal solution.
    """

    start_time = time.perf_counter()

    root = Node(
        state=initial_state,
        g=0,
        h=0,
        f=0,
    )

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
            heuristic_name=None,
        )

    frontier: list[Node] = [root]   # stack
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
            solution_moves = node.build_solution_moves()
            solution_states = node.build_solution_states()

            return SearchResult(
                solved=True,
                solution_moves=solution_moves,
                solution_states=solution_states,
                solution_cost=len(solution_moves),
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                max_frontier_size=max_frontier_size,
                runtime_seconds=time.perf_counter() - start_time,
                algorithm_name="DFS",
                heuristic_name=None,
            )

        for move, successor_state in node.state.get_successors():
            if successor_state in visited:
                continue

            child = Node(
                state=successor_state,
                parent=node,
                move=move,
                g=node.g + 1,
                h=0,
                f=node.g + 1,
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
        heuristic_name=None,
    )