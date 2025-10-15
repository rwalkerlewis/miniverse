"""Scratchpad scaffolding for agent working memory.

The real implementation will track medium-term plans, active commitments,
and any temporary state the planner/executor/reflection modules need to
share. We start with a lightweight dataclass so downstream modules have a
place to attach data structures while we design the full shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Scratchpad:
    """Placeholder working-memory container.

    Notes
    -----
    * ``state`` should eventually store structured plan metadata (e.g.
      daily agendas, remaining steps, timestamps).
    * Keep the structure flexible so custom planners can add arbitrary keys
      without modifying the core class.
    * As we implement plan/execute/reflect, expect helper methods for
      reading/updating specific slots (current task, commitments, etc.).
    """

    state: Dict[str, Any] = field(default_factory=dict)

    def clear(self) -> None:
        """Reset the scratchpad completely (placeholder implementation)."""

        self.state.clear()

    # TODO: Add typed accessors once plan schema is finalized.
