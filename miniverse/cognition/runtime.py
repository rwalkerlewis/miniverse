"""Agent cognition runtime scaffolding.

Provides a lightweight container that bundles the planner, executor,
reflection engine, and scratchpad for each agent. The goal is to make the
orchestrator agnostic to the specific implementations while giving us a
clear place to hang configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .executor import Executor, SimpleExecutor
from .planner import Planner, SimplePlanner
from .reflection import ReflectionEngine, SimpleReflectionEngine
from .scratchpad import Scratchpad
from .prompts import PromptLibrary, DEFAULT_PROMPTS


@dataclass
class AgentCognition:
    """Collection of cognition modules bound to a single agent."""

    planner: Planner
    executor: Executor
    reflection: ReflectionEngine
    scratchpad: Scratchpad
    prompt_library: PromptLibrary = DEFAULT_PROMPTS


def build_default_cognition() -> AgentCognition:
    """Return a minimal cognition stack for quick-start scenarios.

    Notes
    -----
    The default implementations are intentionally simple:
    - ``SimplePlanner`` produces empty plans
    - ``SimpleExecutor`` emits a rest action
    - ``SimpleReflectionEngine`` returns no reflections
    - ``Scratchpad`` is empty

    These placeholders allow the orchestrator to require cognition modules
    before we wire the full LLM-powered pipeline.
    """

    return AgentCognition(
        planner=SimplePlanner(),
        executor=SimpleExecutor(),
        reflection=SimpleReflectionEngine(),
        scratchpad=Scratchpad(),
        prompt_library=DEFAULT_PROMPTS,
    )


AgentCognitionMap = Dict[str, AgentCognition]
"""Convenience alias for the agent cognition registry."""
