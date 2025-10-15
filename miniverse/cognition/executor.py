"""Execution module scaffolding.

Responsible for turning plan steps + perceptions into concrete AgentAction
objects. This version keeps compatibility with the legacy `get_agent_action`
flow so existing simulations continue to run while we layer in structured
prompts.
"""

from __future__ import annotations

from typing import Any, Protocol

from miniverse.schemas import AgentAction, AgentPerception
from miniverse.llm_calls import get_agent_action

from .planner import Plan, PlanStep
from .scratchpad import Scratchpad
from .context import PromptContext
from .prompts import DEFAULT_PROMPTS
from .renderers import render_prompt


class Executor(Protocol):
    """Protocol for execution strategies."""

    async def choose_action(
        self,
        agent_id: str,
        perception: AgentPerception,
        scratchpad: Scratchpad,
        *,
        plan: Plan,
        plan_step: PlanStep | None,
        context: PromptContext,
    ) -> AgentAction:
        """Select an action for the current tick.

        ``plan_step`` may be ``None`` when plans are empty; implementations
        should fallback to heuristics in that case (e.g., rest or observe).
        The context object carries prompts, provider/model metadata, and
        recent memory summaries.
        """

        ...


class SimpleExecutor:
    """Fallback executor delegating to the legacy `get_agent_action` call.

    Bridges new cognition stack (Plan, PlanStep, PromptContext) with legacy LLM call.
    Renders prompt template using context, then delegates to get_agent_action. This
    maintains backward compatibility while allowing gradual migration to new prompt system.
    """

    async def choose_action(
        self,
        agent_id: str,
        perception: AgentPerception,
        scratchpad: Scratchpad,
        *,
        plan: Plan,
        plan_step: PlanStep | None,
        context: PromptContext,
    ) -> AgentAction:
        # Extract base agent prompt (personality, role definition) from context
        base_prompt = context.extra.get("base_agent_prompt", "")
        # Extract LLM credentials. If not configured, return default rest action.
        llm_provider = context.extra.get("llm_provider")
        llm_model = context.extra.get("llm_model")
        # Resolve prompt library with fallback to system defaults
        prompt_library = context.extra.get("prompt_library") or DEFAULT_PROMPTS
        # Allow dynamic template selection via scratchpad (e.g., switch to crisis mode)
        template_name = context.extra.get("execute_prompt_template", "execute_tick")

        # Look up execution template by name. Template contains instructions for converting
        # plan step + perception into concrete action (move, interact, communicate, rest).
        try:
            template = prompt_library.get(template_name)
        except KeyError:
            # Template not found - fall back to default tick execution template
            template = DEFAULT_PROMPTS.get("execute_tick")

        # Render template with context. Replaces placeholders with agent profile, perception,
        # memories, plan state, etc. Returns system and user prompt components.
        rendered = render_prompt(template, context)
        # Combine base agent prompt (personality) with rendered system prompt (instructions).
        # Filter empty blocks to avoid extra whitespace.
        system_blocks = [block for block in (base_prompt, rendered.system) if block]
        system_prompt = "\n\n".join(system_blocks)
        user_prompt = rendered.user

        # If no LLM configured, return deterministic rest action. This allows simulations
        # to run without LLM (testing, physics-only mode).
        if not llm_provider or not llm_model:
            return AgentAction(
                agent_id=agent_id,
                tick=perception.tick,
                action_type="rest",
                target=None,
                parameters={},
                reasoning="No LLM provider configured; defaulting to rest.",
                communication=None,
            )

        # Delegate to legacy get_agent_action. This function makes LLM call with retry
        # logic and validates response matches AgentAction schema. user_prompt_override
        # allows us to inject rendered prompt instead of building prompt internally.
        return await get_agent_action(
            system_prompt,
            perception,
            llm_provider,
            llm_model,
            user_prompt_override=user_prompt,
        )
