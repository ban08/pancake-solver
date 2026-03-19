from __future__ import annotations

import heapq
import time
from typing import Dict

from src.core.node import Node
from src.core.state import PancakeState
from src.core.search import SearchResult


def ucs(initial_state: PancakeState) -> SearchResult:
    """
    Uniform Cost Search for the Pancake Puzzle.

    f(n) = g(n)

    Since all pancake flips cost 1, UCS behaves similarly to BFS,
    but it is implemented with a priority queue ordered by path cost.
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
            algorithm_name="UCS",
            heuristic_name=None,
        )

    frontier: list[Node] = [root]
    heapq.heapify(frontier)

    visited: set[PancakeState] = set()
    best_g: Dict[PancakeState, int] = {initial_state: 0}

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
                algorithm_name="UCS",
                heuristic_name=None,
            )

        for move, successor_state in node.state.get_successors():
            g = node.g + 1

            if successor_state in visited:
                continue

            if g >= best_g.get(successor_state, float("inf")):
                continue

            best_g[successor_state] = g

            child = Node(
                state=successor_state,
                parent=node,
                move=move,
                g=g,
                h=0,
                f=g,
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
        algorithm_name="UCS",
        heuristic_name=None,
    )