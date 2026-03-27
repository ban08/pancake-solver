from __future__ import annotations

from typing import Callable, Dict

from src.core.state import PancakeState


# -----------------------------------------------------
# Heuristics
# -----------------------------------------------------

# Each heuristic estimates the distance to the goal.
# Lower values should indicate states closer to the sorted order.

def gap_heuristic(state: PancakeState) -> int:
    """
    Counts gaps between adjacent pancakes.
    Also includes the gap with the plate (n at the bottom).
    """

    # Access pancake sequence and its size
    pancakes = state.pancakes
    n = len(pancakes)

    gaps = 0

    # Count "gaps" where adjacent pancakes are not consecutive numbers
    for i in range(n - 1):
        if abs(pancakes[i] - pancakes[i + 1]) != 1:
            gaps += 1

    # Check if the largest pancake is not at the bottom (plate gap)
    if pancakes[-1] != n:
        gaps += 1

    return gaps


def misplaced_heuristic(state: PancakeState) -> int:
    """
    Counts pancakes not in their goal position.
    """

    # Count how many pancakes are not in their correct position
    return sum(1 for i, v in enumerate(state.pancakes, start=1) if v != i)


def zero_heuristic(state: PancakeState) -> int:
    """
    Always returns 0 (equivalent to UCS when used with A*).
    """

    # No heuristic information (uninformed search)
    return 0


# -----------------------------------------------------
# Registry
# -----------------------------------------------------

# Maps heuristic names to their corresponding functions

HEURISTICS: Dict[str, Callable[[PancakeState], int]] = {
    "gap": gap_heuristic,
    "misplaced": misplaced_heuristic,
    "zero": zero_heuristic,
}


def get_heuristic(name: str) -> Callable[[PancakeState], int]:
    # Normalize input for case-insensitive lookup
    name = name.lower()

    # Validate that the heuristic exists
    if name not in HEURISTICS:
        raise ValueError(
            f"Unknown heuristic '{name}'. Available: {list(HEURISTICS.keys())}"
        )

    # Return the selected heuristic function
    return HEURISTICS[name]


def list_heuristics() -> list[str]:
    # List available heuristic names
    return list(HEURISTICS.keys())