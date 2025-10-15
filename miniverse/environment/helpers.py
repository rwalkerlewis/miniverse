"""Utilities for environment graphs and grids."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .graph import EnvironmentGraph
from .grid import EnvironmentGrid


class GraphOccupancy:
    """Tracks agent occupancy per node and enforces capacities."""

    def __init__(self, graph: EnvironmentGraph):
        self.graph = graph
        self._counts: Dict[str, Set[str]] = {}

    def occupants(self, node_id: str) -> Set[str]:
        return self._counts.setdefault(node_id, set())

    def can_enter(self, node_id: str, agent_id: str) -> bool:
        node = self.graph.nodes.get(node_id)
        if node is None:
            return False
        occupants = self.occupants(node_id)
        if agent_id in occupants:
            return True
        if node.capacity is None:
            return True
        return len(occupants) < node.capacity

    def enter(self, node_id: str, agent_id: str) -> bool:
        if not self.can_enter(node_id, agent_id):
            return False
        self.occupants(node_id).add(agent_id)
        return True

    def leave(self, node_id: str, agent_id: str) -> None:
        self.occupants(node_id).discard(agent_id)


def shortest_path(graph: EnvironmentGraph, start: str, goal: str) -> Optional[List[str]]:
    """Return a list of node ids from start to goal using BFS."""

    if start == goal:
        return [start]
    visited = {start}
    queue: deque[Tuple[str, List[str]]] = deque([(start, [start])])

    while queue:
        node, path = queue.popleft()
        for neighbor in graph.neighbors(node):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            new_path = path + [neighbor]
            if neighbor == goal:
                return new_path
            queue.append((neighbor, new_path))
    return None


def grid_shortest_path(grid: EnvironmentGrid, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """Return a path of (row, col) coordinates avoiding collisions."""

    if start == goal:
        return [start]

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    visited = {start}
    queue: deque[Tuple[Tuple[int, int], List[Tuple[int, int]]]] = deque([(start, [start])])

    def neighbors(coord: Tuple[int, int]) -> Iterable[Tuple[int, int]]:
        r, c = coord
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.height and 0 <= nc < grid.width and grid.is_walkable(nr, nc):
                yield nr, nc

    while queue:
        coord, path = queue.popleft()
        for nb in neighbors(coord):
            if nb in visited:
                continue
            visited.add(nb)
            new_path = path + [nb]
            if nb == goal:
                return new_path
            queue.append((nb, new_path))
    return None
