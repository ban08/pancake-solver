from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.core.state import PancakeState


@dataclass(slots=True)
class Node:
    """
    Represents a node in the search tree.

    Notes:
    - Nodes form a linked structure via the `parent` reference
    - `move` represents the flip applied to reach this state
    - Cost values (g, h, f) support different search strategies

    Each node stores the current state and the information
    required to reconstruct the solution path.
    """

    # Core node data
    state: PancakeState
    parent: Optional["Node"] = None
    move: Optional[int] = None

    # Search-related values
    # g = path cost from root
    # h = heuristic estimate to goal
    # f = evaluation function (typically g + h)
    # depth = number of steps from root
    g: int = 0  # path cost
    h: int = 0  # heuristic value
    f: int = 0  # evaluation function (g + h)
    depth: int = 0

    def __post_init__(self) -> None:
        """
        Initialize derived values (depth and f).
        """
        # Compute depth based on parent
        if self.parent is not None:
            self.depth = self.parent.depth + 1

        # Default evaluation (can be overridden externally if needed, e.g., Weighted A*)
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

        # Traverse back through parents collecting moves
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

        # Traverse back through parents collecting states
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
        # Compare nodes based on priority (f first, then h, then g)
        return (self.f, self.h, self.g) < (other.f, other.h, other.g)

    # -----------------------------------------------------
    # Representation
    # -----------------------------------------------------

    def __repr__(self) -> str:
        # Debug-friendly representation of the node
        return (
            f"Node(state={self.state}, move={self.move}, "
            f"g={self.g}, h={self.h}, f={self.f}, depth={self.depth})"
        )