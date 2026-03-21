from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple


@dataclass(frozen=True, slots=True)
class PancakeState:
    """
    Immutable representation of a pancake puzzle state.

    A state is a permutation of integers 1..N.
    """

    pancakes: Tuple[int, ...]

    # -----------------------------------------------------
    # Construction
    # -----------------------------------------------------

    def __init__(self, pancakes: Iterable[int]) -> None:
        values = tuple(pancakes)
        self._validate(values)
        object.__setattr__(self, "pancakes", values)

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    @staticmethod
    def _validate(values: Tuple[int, ...]) -> None:
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

    @property
    def size(self) -> int:
        return len(self.pancakes)

    def is_goal(self) -> bool:
        return self.pancakes == tuple(range(1, self.size + 1))

    # -----------------------------------------------------
    # Moves
    # -----------------------------------------------------

    def get_legal_moves(self) -> List[int]:
        return list(range(2, self.size + 1))

    def flip(self, k: int) -> PancakeState:
        if not isinstance(k, int):
            raise TypeError("Flip must be an integer.")

        if k < 2 or k > self.size:
            raise ValueError(f"Flip must satisfy 2 <= k <= {self.size}.")

        new_values = self.pancakes[:k][::-1] + self.pancakes[k:]
        return PancakeState(new_values)

    # -----------------------------------------------------
    # Successors
    # -----------------------------------------------------

    def get_successors(self) -> List[Tuple[int, PancakeState]]:
        return [(k, self.flip(k)) for k in self.get_legal_moves()]

    # -----------------------------------------------------
    # Utilities
    # -----------------------------------------------------

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[int]:
        return iter(self.pancakes)

    def __hash__(self) -> int:
        return hash(self.pancakes)

    # -----------------------------------------------------
    # Representation
    # -----------------------------------------------------

    def __str__(self) -> str:
        return " ".join(map(str, self.pancakes))

    def __repr__(self) -> str:
        return f"PancakeState({self.pancakes})"