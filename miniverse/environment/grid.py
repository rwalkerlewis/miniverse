"""Spatial grid scaffolding (Tier 2 environments).

The Stanford Reverie engine uses tile maps exported from Tiled. We will add
helpers for pathfinding, collision checks, and object interactions in this
module. For now we define placeholder containers so scenarios can begin to
reference the forthcoming API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class GridTile:
    """Metadata about a single tile in the environment grid."""

    world: str | None = None
    sector: str | None = None
    arena: str | None = None
    game_object: str | None = None
    collision: bool = False
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class EnvironmentGrid:
    """Placeholder 2D grid representation."""

    width: int
    height: int
    tiles: Dict[Tuple[int, int], GridTile] = field(default_factory=dict)

    def get_tile(self, row: int, col: int) -> GridTile | None:
        return self.tiles.get((row, col))

    def is_walkable(self, row: int, col: int) -> bool:
        tile = self.get_tile(row, col)
        if tile is None:
            return True  # treat missing tiles as empty space for now
        return not tile.collision
