from __future__ import annotations

from typing import Callable, Dict

from src.core.state import PancakeState


# -----------------------------------------------------
# Heuristics
# -----------------------------------------------------


def gap_heuristic(state: PancakeState) -> int:
    """
    Counts gaps between adjacent pancakes.
    Also includes the gap with the plate (n at the bottom).
    """

    pancakes = state.pancakes
    n = len(pancakes)

    gaps = 0

    for i in range(n - 1):
        if abs(pancakes[i] - pancakes[i + 1]) != 1:
            gaps += 1

    if pancakes[-1] != n:
        gaps += 1

    return gaps


def misplaced_heuristic(state: PancakeState) -> int:
    """
    Counts pancakes not in their goal position.
    """

    return sum(1 for i, v in enumerate(state.pancakes, start=1) if v != i)


def zero_heuristic(state: PancakeState) -> int:
    """
    Always returns 0 (equivalent to UCS when used with A*).
    """

    return 0


# -----------------------------------------------------
# Registry
# -----------------------------------------------------


HEURISTICS: Dict[str, Callable[[PancakeState], int]] = {
    "gap": gap_heuristic,
    "misplaced": misplaced_heuristic,
    "zero": zero_heuristic,
}


def get_heuristic(name: str) -> Callable[[PancakeState], int]:
    name = name.lower()

    if name not in HEURISTICS:
        raise ValueError(
            f"Unknown heuristic '{name}'. Available: {list(HEURISTICS.keys())}"
        )

    return HEURISTICS[name]


def list_heuristics() -> list[str]:
    return list(HEURISTICS.keys())