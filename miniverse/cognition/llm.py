"""LLM-backed implementations for planner and reflection engines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from miniverse.llm_calls import call_llm_with_retries

from .context import PromptContext
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
    """Planner that delegates plan generation to an LLM."""

    def __init__(
        self,
        *,
        template_name: str = "plan_daily",
        prompt_library: Optional[PromptLibrary] = None,
    ) -> None:
        self.template_name = template_name
        self.prompt_library = prompt_library

    async def generate_plan(
        self,
        agent_id: str,
        scratchpad: Scratchpad,
        *,
        world_context: Any,
        context: PromptContext,
    ) -> Plan:
        provider = context.extra.get("llm_provider")
        model = context.extra.get("llm_model")
        if not provider or not model:
            return Plan()

        library = self.prompt_library or context.extra.get("prompt_library") or DEFAULT_PROMPTS
        try:
            template = library.get(self.template_name)
        except KeyError:
            template = DEFAULT_PROMPTS.get("plan_daily")

        rendered = render_prompt(template, context, include_default=False)

        response = await call_llm_with_retries(
            system_prompt=rendered.system,
            user_prompt=rendered.user,
            llm_provider=provider,
            llm_model=model,
            response_model=LLMPlanResponse,
        )

        steps = [
            PlanStep(description=step.description, metadata=dict(step.metadata))
            for step in response.steps
        ]
        return Plan(steps=steps, metadata=dict(response.metadata))


class LLMReflectionEngine(ReflectionEngine):
    """Reflection engine that requests diary entries from an LLM."""

    def __init__(
        self,
        *,
        template_name: str = "reflect_diary",
        prompt_library: Optional[PromptLibrary] = None,
    ) -> None:
        self.template_name = template_name
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
        if context is None:
            return []

        provider = context.extra.get("llm_provider")
        model = context.extra.get("llm_model")
        if not provider or not model:
            return []

        library = self.prompt_library or context.extra.get("prompt_library") or DEFAULT_PROMPTS
        try:
            template = library.get(self.template_name)
        except KeyError:
            template = DEFAULT_PROMPTS.get("reflect_diary")

        rendered = render_prompt(template, context, include_default=False)

        response = await call_llm_with_retries(
            system_prompt=rendered.system,
            user_prompt=rendered.user,
            llm_provider=provider,
            llm_model=model,
            response_model=LLMReflectionResponse,
        )

        return [
            ReflectionResult(
                content=item.content,
                importance=item.importance,
                metadata=dict(item.metadata),
            )
            for item in response.reflections
        ]
