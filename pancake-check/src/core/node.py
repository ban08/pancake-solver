from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.core.state import PancakeState


@dataclass(slots=True)
class Node:
    """
    Represents a node in the search tree.

    Each node stores the current state and the information
    required to reconstruct the solution path.
    """

    state: PancakeState
    parent: Optional["Node"] = None
    move: Optional[int] = None

    g: int = 0  # path cost
    h: int = 0  # heuristic value
    f: int = 0  # evaluation function (g + h)
    depth: int = 0

    def __post_init__(self) -> None:
        """
        Initialize derived values (depth and f).
        """
        if self.parent is not None:
            self.depth = self.parent.depth + 1

        self.f = self.g + self.h

    # -----------------------------------------------------
    # Solution reconstruction
    # -----------------------------------------------------

    def solution_moves(self) -> List[int]:
        """
        Return the sequence of moves from root to this node.
        """
        moves: List[int] = []
        current: Optional[Node] = self

        while current is not None:
            if current.move is not None:
                moves.append(current.move)
            current = current.parent

        return list(reversed(moves))

    def solution_states(self) -> List[PancakeState]:
        """
        Return the sequence of states from root to this node.
        """
        states: List[PancakeState] = []
        current: Optional[Node] = self

        while current is not None:
            states.append(current.state)
            current = current.parent

        return list(reversed(states))

    # -----------------------------------------------------
    # Priority queue support
    # -----------------------------------------------------

    def __lt__(self, other: "Node") -> bool:
        """
        Comparison for priority queues (heapq).
        """
        return (self.f, self.h, self.g) < (other.f, other.h, other.g)

    # -----------------------------------------------------
    # Representation
    # -----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Node(state={self.state}, move={self.move}, "
            f"g={self.g}, h={self.h}, f={self.f}, depth={self.depth})"
        )