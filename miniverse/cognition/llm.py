"""LLM-backed implementations for planner, executor, and reflection engines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from miniverse.llm_calls import call_llm_with_retries
from miniverse.schemas import AgentAction, AgentPerception

from .context import PromptContext
from .executor import Executor
from .planner import Plan, PlanStep, Planner
from .prompts import DEFAULT_PROMPTS, PromptLibrary
from .reflection import ReflectionEngine, ReflectionResult
from .renderers import render_prompt
from .scratchpad import Scratchpad


class LLMPlanStepModel(BaseModel):
    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMPlanResponse(BaseModel):
    steps: List[LLMPlanStepModel] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMReflectionItem(BaseModel):
    content: str
    importance: int = Field(default=5, ge=1, le=10)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMReflectionResponse(BaseModel):
    reflections: List[LLMReflectionItem] = Field(default_factory=list)


class LLMPlanner(Planner):
    """Planner that delegates plan generation to an LLM.

    Uses LLM to generate multi-step plans based on agent personality, current world state,
    recent memories, and existing plan. Supports custom prompt templates for different
    planning styles (daily routine, project planning, reactive vs proactive).
    """

    def __init__(
        self,
        *,
        template_name: str = "plan_daily",
        prompt_library: Optional[PromptLibrary] = None,
    ) -> None:
        # Template name identifies which prompt to use from library (e.g., "plan_daily",
        # "plan_project", "plan_crisis"). Allows per-agent or per-scenario customization.
        self.template_name = template_name
        # Custom prompt library for scenario-specific planning styles. If None, uses
        # context-provided library or system defaults.
        self.prompt_library = prompt_library

    async def generate_plan(
        self,
        agent_id: str,
        scratchpad: Scratchpad,
        *,
        world_context: Any,
        context: PromptContext,
    ) -> Plan:
        # Extract LLM provider and model from context. These come from orchestrator config.
        # If not configured, return empty plan (agent will use executor fallback logic).
        provider = context.extra.get("llm_provider")
        model = context.extra.get("llm_model")
        if not provider or not model:
            # No LLM available - return empty plan rather than failing
            return Plan()

        # Resolve prompt library with three-level fallback: instance config -> context
        # config -> system defaults. This allows maximum flexibility in prompt customization.
        library = self.prompt_library or context.extra.get("prompt_library") or DEFAULT_PROMPTS
        try:
            # Look up template by name. Template contains system and user prompt components
            # with placeholders for context injection.
            template = library.get(self.template_name)
        except KeyError:
            # Template not found - fall back to default daily planning template
            template = DEFAULT_PROMPTS.get("plan_daily")

        # Render template with context. Replaces placeholders like {{perception_json}},
        # {{memories_text}} with actual context data. Returns system and user prompts.
        rendered = render_prompt(template, context, include_default=False)

        # Call LLM with retry logic. LLM generates structured response matching
        # LLMPlanResponse schema (list of steps with descriptions and metadata).
        # Retries up to 3 times with validation feedback if schema doesn't match.
        response = await call_llm_with_retries(
            system_prompt=rendered.system,
            user_prompt=rendered.user,
            llm_provider=provider,
            llm_model=model,
            response_model=LLMPlanResponse,
        )

        # Convert LLM response (Pydantic models) to internal Plan/PlanStep dataclasses.
        # This decouples LLM response schema from internal planning representation,
        # allowing us to change LLM format without breaking downstream code.
        steps = [
            PlanStep(description=step.description, metadata=dict(step.metadata))
            for step in response.steps
        ]
        return Plan(steps=steps, metadata=dict(response.metadata))


class LLMReflectionEngine(ReflectionEngine):
    """Reflection engine that requests diary entries from an LLM.

    Stanford Generative Agents pattern: Periodically synthesize recent experiences into
    higher-level insights. LLM reviews recent memories and generates reflections like
    "I'm running low on resources" or "My relationship with Agent X is deteriorating".
    Reflections are stored as high-importance memories to influence future decisions.
    """

    def __init__(
        self,
        *,
        template_name: str = "reflect_diary",
        prompt_library: Optional[PromptLibrary] = None,
    ) -> None:
        # Template name identifies reflection style (diary entries, strategic analysis, etc.)
        self.template_name = template_name
        # Custom prompt library for scenario-specific reflection styles
        self.prompt_library = prompt_library

    async def maybe_reflect(
        self,
        agent_id: str,
        scratchpad: Scratchpad,
        recent_memories,
        *,
        trigger_context: dict[str, Any] | None = None,
        context: PromptContext | None = None,
    ) -> List[ReflectionResult]:
        # Context is required for reflection - contains memories, world state, plan.
        # If None, caller didn't build context properly - return empty list.
        if context is None:
            return []

        # Extract LLM credentials from context. If not configured, skip reflection
        # (reflections are optional - agent can function without them).
        provider = context.extra.get("llm_provider")
        model = context.extra.get("llm_model")
        if not provider or not model:
            # No LLM available - skip reflection rather than failing
            return []

        # Resolve prompt library with three-level fallback (same as planner)
        library = self.prompt_library or context.extra.get("prompt_library") or DEFAULT_PROMPTS
        try:
            # Look up reflection template. Reflection prompts typically ask LLM to
            # identify patterns, synthesize insights, or update beliefs based on memories.
            template = library.get(self.template_name)
        except KeyError:
            # Template not found - fall back to default diary-style reflection
            template = DEFAULT_PROMPTS.get("reflect_diary")

        # Render template with context. Reflection prompts receive larger memory window
        # (20 vs 10) to identify longer-term patterns.
        rendered = render_prompt(template, context, include_default=False)

        # Call LLM with retry logic. LLM generates structured response containing list
        # of reflections. Each reflection has content (text), importance (1-10), and
        # optional metadata (tags, categories, etc.).
        response = await call_llm_with_retries(
            system_prompt=rendered.system,
            user_prompt=rendered.user,
            llm_provider=provider,
            llm_model=model,
            response_model=LLMReflectionResponse,
        )

        # Convert LLM response to ReflectionResult objects. Reflections are stored as
        # memories with elevated importance (typically 6-10 vs 5 for actions), making
        # them more likely to surface in future memory retrieval.
        return [
            ReflectionResult(
                content=item.content,
                importance=item.importance,
                metadata=dict(item.metadata),
            )
            for item in response.reflections
        ]


class LLMExecutor(Executor):
    """Executor that calls LLM for action selection on every tick.

    Pure LLM-driven executor with NO deterministic fallback. Unlike SimpleExecutor
    (which returns "rest" action when LLM is unavailable), LLMExecutor REQUIRES
    LLM configuration and raises ValueError if missing.

    Use cases:
    - When you want guaranteed LLM-driven agent behavior
    - Social simulations where emergent decisions are critical
    - Scenarios where agents must adapt to unpredictable conditions

    For deterministic fallback behavior, use SimpleExecutor instead.
    For pure deterministic logic, create custom executor (e.g., RuleBasedExecutor).

    Example:
        # Pure LLM cognition stack
        cognition = AgentCognition(
            planner=LLMPlanner(template_name="warehouse_plan"),
            executor=LLMExecutor(template_name="warehouse_execute"),
            reflection=LLMReflectionEngine(template_name="warehouse_reflect"),
            scratchpad=Scratchpad(),
        )

    Args:
        template_name: Name of execution prompt template in library (default: "execute_tick")
        prompt_library: Custom prompt library (optional, falls back to defaults)
    """

    def __init__(
        self,
        *,
        template_name: str = "execute_tick",
        prompt_library: Optional[PromptLibrary] = None,
    ) -> None:
        # Template name identifies which execution prompt to use. Common patterns:
        # - "execute_tick": General-purpose action selection
        # - "execute_warehouse": Domain-specific (inventory vs fulfillment decisions)
        # - "execute_social": Communication-focused actions
        self.template_name = template_name
        # Custom prompt library for scenario-specific execution styles
        self.prompt_library = prompt_library

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
        # Extract LLM credentials from context. These come from Orchestrator initialization.
        provider = context.extra.get("llm_provider")
        model = context.extra.get("llm_model")

        # LLMExecutor is strict: if no LLM configured, it's a configuration error.
        # This is intentional - forces users to be explicit about LLM vs deterministic modes.
        if not provider or not model:
            raise ValueError(
                f"LLMExecutor requires LLM configuration (agent: {agent_id}). "
                f"Set LLM_PROVIDER and LLM_MODEL environment variables, or use SimpleExecutor "
                f"for automatic fallback to deterministic actions when LLM unavailable."
            )

        # Resolve prompt library with three-level fallback (same as planner/reflection)
        library = self.prompt_library or context.extra.get("prompt_library") or DEFAULT_PROMPTS
        try:
            # Look up execution template. Execution prompts guide LLM to choose actions
            # based on current perception, plan step, and recent memories.
            template = library.get(self.template_name)
        except KeyError:
            # Template not found - fall back to default tick execution template
            template = DEFAULT_PROMPTS.get("execute_tick")

        # Render template with context. Replaces placeholders like {{perception_json}},
        # {{plan_step_description}}, {{memories_text}} with actual agent data.
        rendered = render_prompt(template, context, include_default=False)

        # Call LLM with retry logic. LLM generates structured AgentAction response
        # matching Pydantic schema (action_type, target, parameters, reasoning, communication).
        # Retries up to 3 times if validation fails, providing schema feedback to LLM.
        return await call_llm_with_retries(
            system_prompt=rendered.system,
            user_prompt=rendered.user,
            llm_provider=provider,
            llm_model=model,
            response_model=AgentAction,
        )
