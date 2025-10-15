"""Cognition module scaffolding for Miniverse.

This package houses the agent cognition stack (scratchpad, planner,
executor, reflection). Each module currently contains placeholder
implementations so we can wire the orchestrator contract before adding
full LLM-powered behavior.
"""

from .scratchpad import Scratchpad
from .planner import Planner, Plan, PlanStep, SimplePlanner
from .executor import Executor, SimpleExecutor
from .reflection import ReflectionEngine, ReflectionResult, SimpleReflectionEngine
from .runtime import AgentCognition, AgentCognitionMap, build_default_cognition
from .context import PromptContext, build_prompt_context
from .prompts import PromptLibrary, PromptTemplate, DEFAULT_PROMPTS
from .renderers import render_prompt, RenderedPrompt
from .llm import LLMPlanner, LLMReflectionEngine, LLMExecutor
from .cadence import (
    CognitionCadence,
    PlannerCadence,
    ReflectionCadence,
    TickInterval,
    tick_to_time_block,
)

__all__ = [
    "Scratchpad",
    "Planner",
    "Plan",
    "PlanStep",
    "SimplePlanner",
    "Executor",
    "SimpleExecutor",
    "ReflectionEngine",
    "ReflectionResult",
    "SimpleReflectionEngine",
    "AgentCognition",
    "AgentCognitionMap",
    "build_default_cognition",
    "PromptContext",
    "build_prompt_context",
    "PromptLibrary",
    "PromptTemplate",
    "DEFAULT_PROMPTS",
    "render_prompt",
    "RenderedPrompt",
    "LLMPlanner",
    "LLMExecutor",
    "LLMReflectionEngine",
    "CognitionCadence",
    "PlannerCadence",
    "ReflectionCadence",
    "TickInterval",
    "tick_to_time_block",
]
