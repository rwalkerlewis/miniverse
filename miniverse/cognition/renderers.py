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

    Performs simple placeholder replacement to inject context data into prompt templates.
    Supports multiple format options (JSON, text summary, etc.) to accommodate different
    LLM reasoning styles and token budgets.

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

    # Resolve template reference. If None and defaults enabled, use default execution
    # template. This allows callers to pass None without explicitly specifying template.
    if template is None and include_default:
        template = DEFAULT_PROMPTS.get("execute_tick")
    elif template is None:
        raise ValueError("Prompt template not provided and defaults disabled")

    # Generate all available context formats. Templates choose which placeholders to use
    # based on their needs (compact text vs detailed JSON). Pre-generating all formats
    # simplifies template authoring - no need to call context methods inside templates.
    context_json = context.to_json()  # Full context as JSON (agent profile, perception, world, memories, plan)
    summary = context.summary()  # Human-readable text summary (location, plan steps, recent conversation)
    perception_json = context.perception_json()  # Just perception as JSON (what agent observes)
    plan_json = context.plan_json()  # Just plan as JSON (current step, remaining steps)
    memories_text = context.memories_text()  # Recent memories as bulleted text list
    scratchpad_json = context.scratchpad_json()  # Scratchpad state as JSON (working memory)

    # Build replacement map for placeholder substitution. Placeholders use {{double_brace}}
    # syntax to avoid conflicts with JSON braces. String replacement is simple and fast;
    # no need for complex templating engines (Jinja, Mustache) for this use case.
    # Optionally allow templates to reference base_agent_prompt directly
    base_agent_prompt = context.extra.get("base_agent_prompt", "")

    replacements: Dict[str, str] = {
        "{{context_json}}": context_json,
        "{{context_summary}}": summary,
        "{{perception_json}}": perception_json,
        "{{plan_json}}": plan_json,
        "{{memories_text}}": memories_text,
        "{{scratchpad_json}}": scratchpad_json,
        "{{base_agent_prompt}}": base_agent_prompt,
    }

    # Extract system and user prompt components from template. Templates separate system
    # (role definition, constraints) from user (immediate task) to align with LLM APIs.
    system = template.system
    user = template.user

    # Perform placeholder replacement in both system and user prompts. We iterate through
    # all replacements even if template only uses subset - unused placeholders remain
    # as-is (no harm). Multiple passes not needed since placeholders don't nest.
    for placeholder, value in replacements.items():
        system = system.replace(placeholder, value)
        user = user.replace(placeholder, value)

    # Auto-inject base_agent_prompt at the start of the system prompt if present.
    # This ensures high-priority, per-agent instructions are seen by the LLM even
    # if templates don't explicitly include a placeholder.
    if base_agent_prompt:
        system = base_agent_prompt + "\n\n" + system

    return RenderedPrompt(system=system, user=user)
