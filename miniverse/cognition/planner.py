"""Planning module scaffolding.

Defines placeholder interfaces for planner outputs so implementers can
start drafting prompts/logic. Real implementations will convert agent
profiles + scratchpad + world context into structured plans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Protocol

from .scratchpad import Scratchpad
from .context import PromptContext


@dataclass
class PlanStep:
    """Represents a single step in an agent plan (placeholder).

    Fields are intentionally loose for now; we will firm up the schema when
    plan prompts are finalized.
    """

    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    """Container for multi-step plans (daily agenda, projects, etc.).

    ``steps`` are ordered; ``metadata`` can carry schedule info (timestamps,
    priorities, goals). The planner should populate this structure in the
    future.
    """

    steps: List[PlanStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Planner(Protocol):
    """Protocol for plan generators.

    Implementations may call LLMs, procedural logic, or both.
    """

    async def generate_plan(
        self,
        agent_id: str,
        scratchpad: Scratchpad,
        *,
        world_context: Any,
        context: PromptContext,
    ) -> Plan:
        """Produce/update a plan for the agent.

        Parameters
        ----------
        agent_id:
            Identifier for the agent requesting a plan.
        scratchpad:
            Current scratchpad (may contain previous plan state).
        world_context:
            Arbitrary context object (world state snapshot, scenario config,
            etc.). Type kept broad until we finalize the pipeline.
        """

        ...


class SimplePlanner:
    """Placeholder planner returning an empty plan.

    Acts as a temporary default so the orchestrator can require a planner
    reference before the real implementation is ready. Agents using SimplePlanner
    rely entirely on executor logic (reactive behavior) rather than multi-step plans.
    Useful for testing, deterministic simulations, or purely reactive agents.
    """

    async def generate_plan(
        self,
        agent_id: str,
        scratchpad: Scratchpad,
        *,
        world_context: Any,
        context: PromptContext,
    ) -> Plan:
        # Return empty plan. Orchestrator will provide None plan_step to executor,
        # signaling executor to use fallback logic (rest, wander, reactive actions).
        return Plan()


# TODO: add helper functions for merging new plan outputs into scratchpad.
