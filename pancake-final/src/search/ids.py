from __future__ import annotations

import time
from dataclasses import dataclass

from src.core.node import Node
from src.core.state import PancakeState
from src.core.search import SearchResult


@dataclass(slots=True)
class DLSResult:
    goal_node: Node | None
    found: bool
    cutoff: bool
    nodes_expanded: int
    nodes_generated: int
    max_frontier_size: int


def _recursive_dls(
    node: Node,
    limit: int,
    path_states: set[PancakeState],
) -> DLSResult:
    """
    Recursive depth-limited search used by IDS.

    Uses path-based cycle prevention instead of a global visited set,
    so IDS preserves optimality on unit-cost problems.
    """

    if node.state.is_goal():
        return DLSResult(
            goal_node=node,
            found=True,
            cutoff=False,
            nodes_expanded=0,
            nodes_generated=0,
            max_frontier_size=1,
        )

    if limit == 0:
        return DLSResult(
            goal_node=None,
            found=False,
            cutoff=True,
            nodes_expanded=0,
            nodes_generated=0,
            max_frontier_size=1,
        )

    nodes_expanded = 1
    nodes_generated = 0
    max_frontier_size = 1
    cutoff_occurred = False

    for move, successor_state in node.state.get_successors():
        if successor_state in path_states:
            continue

        child = Node(
            state=successor_state,
            parent=node,
            move=move,
            g=node.g + 1,
            h=0,
            f=node.g + 1,
        )

        nodes_generated += 1

        path_states.add(successor_state)
        result = _recursive_dls(child, limit - 1, path_states)
        path_states.remove(successor_state)

        nodes_expanded += result.nodes_expanded
        nodes_generated += result.nodes_generated
        max_frontier_size = max(max_frontier_size, 1 + result.max_frontier_size)

        if result.found:
            return DLSResult(
                goal_node=result.goal_node,
                found=True,
                cutoff=False,
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                max_frontier_size=max_frontier_size,
            )

        if result.cutoff:
            cutoff_occurred = True

    return DLSResult(
        goal_node=None,
        found=False,
        cutoff=cutoff_occurred,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier_size=max_frontier_size,
    )


def ids(initial_state: PancakeState) -> SearchResult:
    """
    Iterative Deepening Search for the Pancake Puzzle.

    Repeatedly performs depth-limited DFS with increasing depth limits.
    For unit-cost problems, IDS should return an optimal solution.
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
            algorithm_name="IDS",
            heuristic_name=None,
        )

    total_nodes_expanded = 0
    total_nodes_generated = 1
    overall_max_frontier_size = 1
    depth_limit = 0

    while True:
        path_states = {initial_state}

        result = _recursive_dls(root, depth_limit, path_states)

        total_nodes_expanded += result.nodes_expanded
        total_nodes_generated += result.nodes_generated
        overall_max_frontier_size = max(
            overall_max_frontier_size,
            result.max_frontier_size,
        )

        if result.found and result.goal_node is not None:
            solution_moves = result.goal_node.build_solution_moves()
            solution_states = result.goal_node.build_solution_states()

            return SearchResult(
                solved=True,
                solution_moves=solution_moves,
                solution_states=solution_states,
                solution_cost=len(solution_moves),
                nodes_expanded=total_nodes_expanded,
                nodes_generated=total_nodes_generated,
                max_frontier_size=overall_max_frontier_size,
                runtime_seconds=time.perf_counter() - start_time,
                algorithm_name="IDS",
                heuristic_name=None,
            )

        if not result.cutoff:
            return SearchResult(
                solved=False,
                solution_moves=[],
                solution_states=[],
                solution_cost=0,
                nodes_expanded=total_nodes_expanded,
                nodes_generated=total_nodes_generated,
                max_frontier_size=overall_max_frontier_size,
                runtime_seconds=time.perf_counter() - start_time,
                algorithm_name="IDS",
                heuristic_name=None,
            )

        depth_limit += 1