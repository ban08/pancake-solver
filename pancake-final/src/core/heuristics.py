from __future__ import annotations

from typing import Callable, Dict

from src.core.state import PancakeState


# ---------------------------------------------------------
# Gap Heuristic
# ---------------------------------------------------------

def gap_heuristic(state: PancakeState) -> int:
    """
    Gap heuristic for the pancake puzzle.

    A gap exists between two adjacent pancakes if their sizes
    are not consecutive numbers.

    Also counts the gap between the bottom pancake and the
    virtual plate (n + 1).

    Example:
        (3, 1, 2, 4)

        |3 - 1| = 2  -> gap
        |1 - 2| = 1  -> no gap
        |2 - 4| = 2  -> gap
        bottom check -> 4 vs plate -> no gap

        heuristic = 2
    """

    pancakes = state.pancakes
    n = len(pancakes)

    gaps = 0

    for i in range(n - 1):
        if abs(pancakes[i] - pancakes[i + 1]) != 1:
            gaps += 1

    # bottom pancake vs plate
    if pancakes[-1] != n:
        gaps += 1

    return gaps


# ---------------------------------------------------------
# Misplaced Pancakes
# ---------------------------------------------------------

def misplaced_heuristic(state: PancakeState) -> int:
    """
    Counts how many pancakes are not in their goal position.

    Example:
        (3,1,2,4)

        position 1 -> 3 (wrong)
        position 2 -> 1 (wrong)
        position 3 -> 2 (wrong)
        position 4 -> 4 (correct)

        heuristic = 3
    """

    pancakes = state.pancakes

    misplaced = 0
    for i, value in enumerate(pancakes, start=1):
        if value != i:
            misplaced += 1

    return misplaced


# ---------------------------------------------------------
# Zero Heuristic
# ---------------------------------------------------------

def zero_heuristic(state: PancakeState) -> int:
    """
    Always returns 0.

    Using this with A* makes it behave exactly like
    Uniform Cost Search.
    """
    return 0


# ---------------------------------------------------------
# Heuristic Registry
# ---------------------------------------------------------

HEURISTICS: Dict[str, Callable[[PancakeState], int]] = {
    "gap": gap_heuristic,
    "misplaced": misplaced_heuristic,
    "zero": zero_heuristic,
}


def get_heuristic(name: str) -> Callable[[PancakeState], int]:
    """
    Returns heuristic function by name.

    Example:
        h = get_heuristic("gap")
        value = h(state)
    """

    name = name.lower()

    if name not in HEURISTICS:
        raise ValueError(
            f"Unknown heuristic '{name}'. "
            f"Available: {list(HEURISTICS.keys())}"
        )

    return HEURISTICS[name]


def list_heuristics() -> list[str]:
    """
    Returns available heuristic names.
    """
    return list(HEURISTICS.keys())