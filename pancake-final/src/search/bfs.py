from __future__ import annotations

import time
from collections import deque
from typing import Set

from src.core.state import PancakeState
from src.core.node import Node
from src.core.search import SearchResult


def bfs(initial_state: PancakeState) -> SearchResult:
    """
    Breadth-First Search for the pancake puzzle.

    BFS guarantees the shortest solution (minimum number of flips)
    because all actions have equal cost.
    """

    start_time = time.perf_counter()

    root = Node(state=initial_state)

    if initial_state.is_goal():
        return SearchResult(
            solved=True,
            solution_moves=[],
            solution_states=[initial_state],
            solution_cost=0,
            nodes_expanded=0,
            nodes_generated=0,
            max_frontier_size=1,
            runtime_seconds=time.perf_counter() - start_time,
            algorithm_name="BFS"
        )

    frontier = deque([root])
    visited: Set[PancakeState] = {initial_state}

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while frontier:
        node = frontier.popleft()
        nodes_expanded += 1

        for move, successor_state in node.state.get_successors():

            if successor_state in visited:
                continue

            visited.add(successor_state)

            child = Node(
                state=successor_state,
                parent=node,
                move=move,
                g=node.g + 1
            )

            nodes_generated += 1

            if successor_state.is_goal():

                solution_moves = child.build_solution_moves()
                solution_states = child.build_solution_states()

                return SearchResult(
                    solved=True,
                    solution_moves=solution_moves,
                    solution_states=solution_states,
                    solution_cost=len(solution_moves),
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    max_frontier_size=max_frontier_size,
                    runtime_seconds=time.perf_counter() - start_time,
                    algorithm_name="BFS"
                )

            frontier.append(child)

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
        algorithm_name="BFS"
    )