"""Prompt rendering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .prompts import PromptTemplate, DEFAULT_PROMPTS
from .context import PromptContext


@dataclass
class RenderedPrompt:
    system: str
    user: str


def render_prompt(
    template: PromptTemplate | None,
    context: PromptContext,
    *,
    include_default: bool = True,
) -> RenderedPrompt:
    """Render a prompt template using the supplied context.

    Parameters
    ----------
    template:
        PromptTemplate to render. When ``None`` and ``include_default`` is True,
        the ``DEFAULT_PROMPTS`` library is consulted.
    context:
        Prompt context assembled by the orchestrator.
    include_default:
        Whether to fall back to ``DEFAULT_PROMPTS`` when template is missing.
    """

    if template is None and include_default:
        template = DEFAULT_PROMPTS.get("execute_tick")
    elif template is None:
        raise ValueError("Prompt template not provided and defaults disabled")

    context_json = context.to_json()
    summary = context.summary()
    perception_json = context.perception_json()
    plan_json = context.plan_json()
    memories_text = context.memories_text()
    scratchpad_json = context.scratchpad_json()

    replacements: Dict[str, str] = {
        "{{context_json}}": context_json,
        "{{context_summary}}": summary,
        "{{perception_json}}": perception_json,
        "{{plan_json}}": plan_json,
        "{{memories_text}}": memories_text,
        "{{scratchpad_json}}": scratchpad_json,
    }

    system = template.system
    user = template.user

    for placeholder, value in replacements.items():
        system = system.replace(placeholder, value)
        user = user.replace(placeholder, value)

    return RenderedPrompt(system=system, user=user)
