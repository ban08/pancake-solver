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
    """

    start_time = time.perf_counter()

    root = Node(
        state=initial_state,
        g=0,
        h=heuristic(initial_state),
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
            algorithm_name="A*",
            heuristic_name=heuristic.__name__,
        )

    frontier: list[Node] = [root]
    heapq.heapify(frontier)

    best_g: Dict[PancakeState, int] = {initial_state: 0}

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while frontier:
        node = heapq.heappop(frontier)

        if node.g > best_g.get(node.state, float("inf")):
            continue

        nodes_expanded += 1

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

        for move, successor_state in node.state.get_successors():
            new_g = node.g + 1

            if new_g >= best_g.get(successor_state, float("inf")):
                continue

            best_g[successor_state] = new_g

            child = Node(
                state=successor_state,
                parent=node,
                move=move,
                g=new_g,
                h=heuristic(successor_state),
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
        algorithm_name="A*",
        heuristic_name=heuristic.__name__,
    )