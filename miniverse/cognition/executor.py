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
    """Fallback executor delegating to the legacy `get_agent_action` call."""

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
        base_prompt = context.extra.get("base_agent_prompt", "")
        llm_provider = context.extra.get("llm_provider")
        llm_model = context.extra.get("llm_model")
        prompt_library = context.extra.get("prompt_library") or DEFAULT_PROMPTS
        template_name = context.extra.get("execute_prompt_template", "execute_tick")

        try:
            template = prompt_library.get(template_name)
        except KeyError:
            template = DEFAULT_PROMPTS.get("execute_tick")

        rendered = render_prompt(template, context)
        system_blocks = [block for block in (base_prompt, rendered.system) if block]
        system_prompt = "\n\n".join(system_blocks)
        user_prompt = rendered.user

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

        return await get_agent_action(
            system_prompt,
            perception,
            llm_provider,
            llm_model,
            user_prompt_override=user_prompt,
        )
