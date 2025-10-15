"""Prompt template scaffolding for cognition stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class PromptTemplate:
    """Represents a templated prompt with placeholders."""

    name: str
    system: str
    user: str
    description: str = ""


class PromptLibrary:
    """Container for named prompt templates per cognition stage."""

    def __init__(self) -> None:
        self.templates: Dict[str, PromptTemplate] = {}

    def register(self, template: PromptTemplate) -> None:
        self.templates[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        return self.templates[name]


# Default placeholders ---------------------------------------------------------

DEFAULT_PROMPTS = PromptLibrary()

DEFAULT_PROMPTS.register(
    PromptTemplate(
        name="plan_daily",
        system=(
            "You are the agent's planning assistant. Review the provided context and produce a JSON schedule "
            "for the next few hours. Always follow the JSON schema shown in the example."
        ),
        user=(
            "Context summary:\n{{context_summary}}\n\n"
            "Full context JSON:\n{{context_json}}\n\n"
            "Example output:\n"
            "{\n"
            "  \"steps\": [\n"
            "    {\"description\": \"coordinate morning stand-up\", \"metadata\": {\"duration_minutes\": 45}},\n"
            "    {\"description\": \"inspect life-support systems\", \"metadata\": {\"priority\": \"high\"}}\n"
            "  ],\n"
            "  \"metadata\": {\"planning_horizon\": \"next 4 hours\"}\n"
            "}\n\n"
            "Respond with JSON only."
        ),
        description="Generates a daily agenda based on goals and memories.",
    )
)

DEFAULT_PROMPTS.register(
    PromptTemplate(
        name="execute_tick",
        system=(
            "You are the agent's execution module. Decide the next action that best follows the current plan and "
            "the situational context. Respond with valid AgentAction JSON."
        ),
        user=(
            "Perception record:\n{{perception_json}}\n\n"
            "Plan state:\n{{plan_json}}\n\n"
            "Recent memories:\n{{memories_text}}\n\n"
            "Additional context:\n{{context_summary}}\n\n"
            "Example output:\n"
            "{\n"
            "  \"agent_id\": \"lead\",\n"
            "  \"tick\": 5,\n"
            "  \"action_type\": \"work\",\n"
            "  \"target\": \"ops\",\n"
            "  \"parameters\": {\"focus\": \"coordinate technicians\"},\n"
            "  \"reasoning\": \"Need to follow up on the backlog and brief the team\",\n"
            "  \"communication\": null\n"
            "}\n\n"
            "Return JSON only."
        ),
        description="Chooses an action for the current tick.",
    )
)

DEFAULT_PROMPTS.register(
    PromptTemplate(
        name="reflect_diary",
        system=(
            "You are the reflection module. Summarize key learnings as a short diary entry. Use the JSON schema in "
            "the example so the system can store your reflections."
        ),
        user=(
            "Context summary:\n{{context_summary}}\n\n"
            "Full context JSON:\n{{context_json}}\n\n"
            "Example output:\n"
            "{\n"
            "  \"reflections\": [\n"
            "    {\"content\": \"Coordinating early keeps the backlog manageable. Need to request more filters.\", \"importance\": 6}\n"
            "  ]\n"
            "}\n\n"
            "Respond with JSON only."
        ),
        description="Produces diary entries when reflection triggers fire.",
    )
)
