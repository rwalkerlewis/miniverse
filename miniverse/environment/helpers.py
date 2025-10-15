"""Utilities for environment graphs and grids."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .graph import EnvironmentGraph
from .grid import EnvironmentGrid


class GraphOccupancy:
    """Tracks agent occupancy per node and enforces capacities.

    Maintains real-time count of agents at each location and enforces node capacity limits.
    Used by simulation rules to prevent overcrowding and by world engine to validate moves.
    """

    def __init__(self, graph: EnvironmentGraph):
        self.graph = graph
        # Maps node_id -> set of agent_ids currently at that node. Using sets ensures
        # each agent counted at most once per location and provides O(1) membership checks.
        self._counts: Dict[str, Set[str]] = {}

    def occupants(self, node_id: str) -> Set[str]:
        """Get set of agent IDs currently at node. Creates empty set if node unvisited."""
        # setdefault ensures node has entry even if never visited (avoids KeyError)
        return self._counts.setdefault(node_id, set())

    def can_enter(self, node_id: str, agent_id: str) -> bool:
        """Check if agent can enter node without exceeding capacity.

        Returns True if: (1) node exists, (2) agent already there (allow stay),
        (3) node has no capacity limit, or (4) node has room for one more agent.
        """
        # Check node exists in graph. Non-existent nodes can't be entered.
        node = self.graph.nodes.get(node_id)
        if node is None:
            return False
        occupants = self.occupants(node_id)
        # Agent already at node - allow stay (not counted as new entry)
        if agent_id in occupants:
            return True
        # No capacity limit - allow entry
        if node.capacity is None:
            return True
        # Check if node has room (current count < capacity limit)
        return len(occupants) < node.capacity

    def enter(self, node_id: str, agent_id: str) -> bool:
        """Attempt to add agent to node. Returns True if successful, False if at capacity."""
        # Check capacity before entering. Prevents silent capacity violations.
        if not self.can_enter(node_id, agent_id):
            return False
        # Add agent to occupancy set. Set.add is idempotent (no duplicates).
        self.occupants(node_id).add(agent_id)
        return True

    def leave(self, node_id: str, agent_id: str) -> None:
        """Remove agent from node's occupancy list. Safe to call even if agent not present."""
        # discard is safer than remove - doesn't raise KeyError if agent not in set.
        # This handles cases where agent leaves before formally "entering" (e.g., during init).
        self.occupants(node_id).discard(agent_id)


def shortest_path(graph: EnvironmentGraph, start: str, goal: str) -> Optional[List[str]]:
    """Return a list of node ids from start to goal using BFS.

    Uses breadth-first search to find shortest unweighted path. Returns None if no path
    exists (disconnected graph). Path includes both start and goal nodes.
    """

    # Trivial case: already at goal. Return single-node path.
    if start == goal:
        return [start]
    # Track visited nodes to avoid cycles. Start node is already visited.
    visited = {start}
    # Queue stores (current_node, path_to_current_node) tuples. BFS explores layer by layer.
    queue: deque[Tuple[str, List[str]]] = deque([(start, [start])])

    while queue:
        # Process nodes in FIFO order (breadth-first). This ensures we find shortest path
        # before exploring longer alternatives.
        node, path = queue.popleft()
        # Explore all neighbors of current node. Graph.neighbors() returns connected nodes.
        for neighbor in graph.neighbors(node):
            # Skip already-visited nodes to avoid cycles and duplicate work
            if neighbor in visited:
                continue
            # Mark neighbor as visited before adding to queue (prevents duplicate queue entries)
            visited.add(neighbor)
            # Extend path to include neighbor
            new_path = path + [neighbor]
            # Goal found - return path immediately (BFS guarantees this is shortest)
            if neighbor == goal:
                return new_path
            # Goal not found yet - add neighbor to queue for exploration
            queue.append((neighbor, new_path))
    # Queue exhausted without finding goal - no path exists
    return None


def grid_shortest_path(grid: EnvironmentGrid, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """Return a path of (row, col) coordinates avoiding collisions.

    Uses BFS to find shortest path on 2D grid with obstacles. Only explores walkable cells
    (no walls, obstacles). Returns None if goal unreachable. Path includes start and goal.
    """

    # Trivial case: already at goal
    if start == goal:
        return [start]

    # Four-directional movement (up, down, left, right). Diagonal movement not supported.
    # Order doesn't matter for BFS correctness but affects tie-breaking for equal-length paths.
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    visited = {start}
    queue: deque[Tuple[Tuple[int, int], List[Tuple[int, int]]]] = deque([(start, [start])])

    def neighbors(coord: Tuple[int, int]) -> Iterable[Tuple[int, int]]:
        """Generate valid neighbor coordinates (within bounds, walkable)."""
        r, c = coord
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            # Check bounds (within grid) and walkability (no obstacles). Grid.is_walkable()
            # consults collision map to determine if cell is passable.
            if 0 <= nr < grid.height and 0 <= nc < grid.width and grid.is_walkable(nr, nc):
                yield nr, nc

    while queue:
        # Process cells in FIFO order (BFS). First path to reach goal is shortest.
        coord, path = queue.popleft()
        for nb in neighbors(coord):
            # Skip visited cells to avoid cycles
            if nb in visited:
                continue
            visited.add(nb)
            new_path = path + [nb]
            # Goal reached - return path immediately
            if nb == goal:
                return new_path
            # Continue exploring from this cell
            queue.append((nb, new_path))
    # No path exists - goal blocked or disconnected
    return None
