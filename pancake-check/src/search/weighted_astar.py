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
    """

    start_time = time.perf_counter()

    root = Node(
        state=initial_state,
        g=0,
        h=heuristic(initial_state),
    )
    root.f = root.g + weight * root.h

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

    frontier: list[Node] = [root]
    heapq.heapify(frontier)

    best_f: Dict[PancakeState, float] = {initial_state: root.f}

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while frontier:
        node = heapq.heappop(frontier)

        if node.f > best_f.get(node.state, float("inf")):
            continue

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
                algorithm_name="Weighted A*",
                heuristic_name=heuristic.__name__,
            )

        for move, successor_state in node.state.get_successors():
            g = node.g + 1
            h = heuristic(successor_state)
            f = g + weight * h

            if f >= best_f.get(successor_state, float("inf")):
                continue

            best_f[successor_state] = f

            child = Node(
                state=successor_state,
                parent=node,
                move=move,
                g=g,
                h=h,
            )
            child.f = f

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
        algorithm_name="Weighted A*",
        heuristic_name=heuristic.__name__,
    )