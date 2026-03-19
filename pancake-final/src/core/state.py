from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple


@dataclass(frozen=True, slots=True)
class PancakeState:
    """
    Immutable representation of a pancake puzzle state.

    A state is a permutation of integers 1..N where each number
    represents a pancake size.

    Legal action:
        flip(k)  for 2 <= k <= N

    Example:
        (3,1,4,2) -> flip(3) -> (4,1,3,2)
    """

    pancakes: Tuple[int, ...]

    # -------------------------------------------------------
    # Constructor
    # -------------------------------------------------------

    def __init__(self, pancakes: Iterable[int]) -> None:
        pancakes_tuple = tuple(pancakes)
        self._validate(pancakes_tuple)
        object.__setattr__(self, "pancakes", pancakes_tuple)

    # -------------------------------------------------------
    # Validation
    # -------------------------------------------------------

    @staticmethod
    def _validate(pancakes: Tuple[int, ...]) -> None:
        if len(pancakes) == 0:
            raise ValueError("Pancake state cannot be empty.")

        if not all(isinstance(p, int) for p in pancakes):
            raise TypeError("All pancakes must be integers.")

        n = len(pancakes)
        expected = set(range(1, n + 1))

        if set(pancakes) != expected:
            raise ValueError(
                "State must be a permutation of integers 1..N. "
                f"Received: {pancakes}"
            )

    # -------------------------------------------------------
    # Properties
    # -------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self.pancakes)

    @property
    def goal(self) -> Tuple[int, ...]:
        return tuple(range(1, self.size + 1))

    # -------------------------------------------------------
    # Goal test
    # -------------------------------------------------------

    def is_goal(self) -> bool:
        return self.pancakes == self.goal

    # -------------------------------------------------------
    # Moves
    # -------------------------------------------------------

    def get_legal_moves(self) -> List[int]:
        """
        All valid flip sizes.
        """
        return list(range(2, self.size + 1))

    def flip(self, k: int) -> PancakeState:
        """
        Flip the top k pancakes.
        """
        if not isinstance(k, int):
            raise TypeError("Flip value must be an integer.")

        if k < 2 or k > self.size:
            raise ValueError(
                f"Flip must satisfy 2 <= k <= {self.size}. Received {k}"
            )

        flipped = self.pancakes[:k][::-1] + self.pancakes[k:]
        return PancakeState(flipped)

    # -------------------------------------------------------
    # Successors
    # -------------------------------------------------------

    def get_successors(self) -> List[Tuple[int, PancakeState]]:
        """
        Returns list of (move, successor_state)
        """
        return [(k, self.flip(k)) for k in self.get_legal_moves()]

    # -------------------------------------------------------
    # Utilities
    # -------------------------------------------------------

    def to_list(self) -> List[int]:
        return list(self.pancakes)

    @classmethod
    def from_list(cls, values: List[int]) -> PancakeState:
        return cls(values)

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[int]:
        return iter(self.pancakes)

    def __hash__(self) -> int:
        return hash(self.pancakes)

    # -------------------------------------------------------
    # String representations
    # -------------------------------------------------------

    def __str__(self) -> str:
        return " ".join(map(str, self.pancakes))

    def __repr__(self) -> str:
        return f"PancakeState({self.pancakes})"