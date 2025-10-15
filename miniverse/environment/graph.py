"""Logical environment graph scaffolding.

Tier 1 environments map locations/rooms/teams to adjacency relationships.
This module will eventually expose graph data structures, capacity checks,
and routing helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class LocationNode:
    """Placeholder graph node definition."""

    name: str
    capacity: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class EnvironmentGraph:
    """Lightweight graph container until full helper set is implemented."""

    nodes: Dict[str, LocationNode] = field(default_factory=dict)
    adjacency: Dict[str, List[str]] = field(default_factory=dict)

    def neighbors(self, node_id: str) -> List[str]:
        return self.adjacency.get(node_id, [])

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes
