from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.core.state import PancakeState


@dataclass(slots=True)
class Node:
    """
    Represents a node in the search tree.

    Attributes
    ----------
    state : PancakeState
        Current pancake puzzle configuration.

    parent : Node | None
        Parent node in the search tree.

    move : int | None
        Flip size used to reach this node.

    g : int
        Path cost from root to this node.

    h : int
        Heuristic estimate to goal.

    f : int
        Evaluation value used by informed searches.

    depth : int
        Depth of node in search tree.
    """

    state: PancakeState
    parent: Optional["Node"] = None
    move: Optional[int] = None

    g: int = 0
    h: int = 0
    f: int = 0
    depth: int = 0

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------

    def __post_init__(self) -> None:
        """
        Automatically compute f and depth when node is created.
        """

        if self.parent is not None:
            self.depth = self.parent.depth + 1

        self.f = self.g + self.h

    # ---------------------------------------------------------
    # Path reconstruction
    # ---------------------------------------------------------

    def build_path(self) -> List["Node"]:
        """
        Return the path from the root node to this node.
        """
        path: List[Node] = []
        current: Optional[Node] = self

        while current is not None:
            path.append(current)
            current = current.parent

        path.reverse()
        return path

    def build_solution_moves(self) -> List[int]:
        """
        Return the sequence of flip moves from the root to this node.
        """
        moves: List[int] = []
        current: Optional[Node] = self

        while current is not None:
            if current.move is not None:
                moves.append(current.move)
            current = current.parent

        moves.reverse()
        return moves

    def build_solution_states(self) -> List[PancakeState]:
        """
        Return the sequence of states from the root to this node.
        """
        states: List[PancakeState] = []
        current: Optional[Node] = self

        while current is not None:
            states.append(current.state)
            current = current.parent

        states.reverse()
        return states

    # ---------------------------------------------------------
    # Priority queue ordering
    # ---------------------------------------------------------

    def __lt__(self, other: "Node") -> bool:
        """
        Ordering for heapq priority queues.

        Priority order:
        1) f
        2) h
        3) g
        """
        return (self.f, self.h, self.g) < (other.f, other.h, other.g)

    # ---------------------------------------------------------
    # Debugging
    # ---------------------------------------------------------

    def __repr__(self) -> str:
        return (
            "Node("
            f"state={self.state!r}, "
            f"move={self.move}, "
            f"g={self.g}, "
            f"h={self.h}, "
            f"f={self.f}, "
            f"depth={self.depth}"
            ")"
        )