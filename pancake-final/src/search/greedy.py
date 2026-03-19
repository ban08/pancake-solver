from __future__ import annotations

import heapq
import time
from typing import Callable, Dict

from src.core.node import Node
from src.core.state import PancakeState
from src.core.search import SearchResult


def greedy(
    initial_state: PancakeState,
    heuristic: Callable[[PancakeState], int],
) -> SearchResult:
    """
    Greedy Best-First Search for the pancake puzzle.

    Greedy search evaluates nodes using:
        f(n) = h(n)

    It is usually faster than BFS, but it does not guarantee
    an optimal solution.
    """

    start_time = time.perf_counter()

    root = Node(
        state=initial_state,
        g=0,
        h=heuristic(initial_state),
        f=heuristic(initial_state),
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
            algorithm_name="Greedy",
            heuristic_name=heuristic.__name__,
        )

    frontier: list[Node] = [root]
    heapq.heapify(frontier)

    visited: set[PancakeState] = set()
    best_h: Dict[PancakeState, int] = {initial_state: heuristic(initial_state)}

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while frontier:
        node = heapq.heappop(frontier)

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
                algorithm_name="Greedy",
                heuristic_name=heuristic.__name__,
            )

        for move, successor_state in node.state.get_successors():
            if successor_state in visited:
                continue

            h_value = heuristic(successor_state)

            if h_value >= best_h.get(successor_state, float("inf")):
                continue

            best_h[successor_state] = h_value

            child = Node(
                state=successor_state,
                parent=node,
                move=move,
                g=node.g + 1,
                h=h_value,
                f=h_value,
            )

            heapq.heappush(frontier, child)
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
        algorithm_name="Greedy",
        heuristic_name=heuristic.__name__,
    )