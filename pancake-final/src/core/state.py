from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple


@dataclass(frozen=True, slots=True)
class PancakeState:
    """
    Immutable representation of a pancake puzzle state.

    A state is a permutation of integers 1..N.

    Notes:
    - States are immutable and hashable (safe for sets/dicts)
    - The tuple `pancakes` stores the current order (top -> bottom)
    - Goal state is the sorted sequence 1..N
    """

    # Internal representation (immutable tuple for hashing)
    pancakes: Tuple[int, ...]

    # -----------------------------------------------------
    # Construction
    # -----------------------------------------------------

    # Convert input iterable to tuple and validate it
    def __init__(self, pancakes: Iterable[int]) -> None:
        values = tuple(pancakes)
        self._validate(values)
        object.__setattr__(self, "pancakes", values)

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    @staticmethod
    def _validate(values: Tuple[int, ...]) -> None:
        # Ensure non-empty, integer-only permutation of 1..N
        if not values:
            raise ValueError("Pancake state cannot be empty.")

        if not all(isinstance(v, int) for v in values):
            raise TypeError("All pancakes must be integers.")

        n = len(values)
        if set(values) != set(range(1, n + 1)):
            raise ValueError(
                "State must be a permutation of integers 1..N. "
                f"Received: {values}"
            )

    # -----------------------------------------------------
    # Core properties
    # -----------------------------------------------------

    # Number of pancakes in the stack
    @property
    def size(self) -> int:
        return len(self.pancakes)

    # Check if pancakes are in sorted (goal) order
    def is_goal(self) -> bool:
        return self.pancakes == tuple(range(1, self.size + 1))

    # -----------------------------------------------------
    # Moves
    # -----------------------------------------------------

    # Valid flips: reversing the first k pancakes (k >= 2)
    def get_legal_moves(self) -> List[int]:
        return list(range(2, self.size + 1))

    # Apply a flip of size k (reverse first k pancakes)
    def flip(self, k: int) -> PancakeState:
        if not isinstance(k, int):
            raise TypeError("Flip must be an integer.")

        if k < 2 or k > self.size:
            raise ValueError(f"Flip must satisfy 2 <= k <= {self.size}.")

        # Reverse prefix [0:k) and keep the rest unchanged
        new_values = self.pancakes[:k][::-1] + self.pancakes[k:]
        return PancakeState(new_values)

    # -----------------------------------------------------
    # Successors
    # -----------------------------------------------------

    # Generate all successor states as (move, new_state)
    def get_successors(self) -> List[Tuple[int, PancakeState]]:
        return [(k, self.flip(k)) for k in self.get_legal_moves()]

    # -----------------------------------------------------
    # Utilities
    # -----------------------------------------------------

    # Allow len(state)
    def __len__(self) -> int:
        return self.size

    # Allow iteration over pancakes
    def __iter__(self) -> Iterator[int]:
        return iter(self.pancakes)

    # Hash based on tuple (enables use in sets/dicts)
    def __hash__(self) -> int:
        return hash(self.pancakes)

    # -----------------------------------------------------
    # Representation
    # -----------------------------------------------------

    # Human-readable format (e.g., "3 1 2 4")
    def __str__(self) -> str:
        return " ".join(map(str, self.pancakes))

    # Debug representation
    def __repr__(self) -> str:
        return f"PancakeState({self.pancakes})"