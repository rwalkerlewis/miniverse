"""Reflection module scaffolding.

Reflection converts accumulated experiences into higher-level insights and
feeds them back into memory. This placeholder captures the planned surface
area without committing to an implementation yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

from miniverse.schemas import AgentMemory

from .context import PromptContext

from .scratchpad import Scratchpad


@dataclass
class ReflectionResult:
    """Represents a reflection output awaiting persistence."""

    content: str
    importance: int = 5
    metadata: dict[str, Any] = field(default_factory=dict)


class ReflectionEngine(Protocol):
    """Protocol for reflection strategies."""

    async def maybe_reflect(
        self,
        agent_id: str,
        scratchpad: Scratchpad,
        recent_memories: Iterable[AgentMemory],
        *,
        trigger_context: dict[str, Any] | None = None,
        context: PromptContext | None = None,
    ) -> list[ReflectionResult]:
        """Produce zero or more reflections for the agent.

        ``trigger_context`` can carry metrics such as poignancy checks,
        planned/actual deltas, etc.
        """

        ...


class SimpleReflectionEngine:
    """Fallback reflection engine producing no output."""

    async def maybe_reflect(
        self,
        agent_id: str,
        scratchpad: Scratchpad,
        recent_memories: Iterable[AgentMemory],
        *,
        trigger_context: dict[str, Any] | None = None,
        context: PromptContext | None = None,
    ) -> list[ReflectionResult]:
        return []


# TODO: integrate with memory strategy once importance scoring is available.
