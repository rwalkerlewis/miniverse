"""Prompt rendering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import os

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
    # Optionally allow templates to reference initial_state_agent_prompt (first-turn user prompt)
    # Backward-compat alias: base_agent_prompt
    initial_state_agent_prompt = (
        context.extra.get("initial_state_agent_prompt")
        or context.extra.get("base_agent_prompt", "")
    )
    is_first_turn = False
    try:
        is_first_turn = int(getattr(context.perception, "tick", 0)) == 0
    except Exception:
        is_first_turn = False
    # Simulation instructions: allow override via context; provide a concise default
    simulation_instructions = context.extra.get(
        "simulation_instructions",
        "You are an agent in a simulation. Read perception and return an AgentAction JSON.",
    )

    # Build character prompt (identity) from AgentProfile with concise lines.
    # Only include non-empty fields to avoid noise.
    profile = context.agent_profile
    character_lines: List[str] = []
    # Name is always present
    if profile.name:
        character_lines.append(f"I am {profile.name}.")
    # Optional age
    if getattr(profile, "age", None) is not None:
        character_lines.append(f"I am {profile.age} years old.")
    # Role
    if getattr(profile, "role", None):
        character_lines.append(f"I work as a {profile.role}.")
    # Background
    if getattr(profile, "background", None):
        character_lines.append(f"Background: {profile.background}")
    # Personality
    if getattr(profile, "personality", None):
        character_lines.append(f"My personality is {profile.personality}.")
    # Skills
    if getattr(profile, "skills", None):
        if isinstance(profile.skills, dict) and profile.skills:
            skill_parts = [f"{k} ({v})" for k, v in profile.skills.items()]
            character_lines.append("My skills include: " + ", ".join(skill_parts) + ".")
    # Goals
    if getattr(profile, "goals", None):
        if isinstance(profile.goals, list) and profile.goals:
            character_lines.append("My goals are: " + ", ".join(profile.goals) + ".")
    # Relationships
    if getattr(profile, "relationships", None):
        if isinstance(profile.relationships, dict) and profile.relationships:
            character_lines.append("My relationships with others:")
            for other_id, relation in profile.relationships.items():
                character_lines.append(f"- {other_id}: {relation}")
    character_prompt = "\n".join(character_lines).strip()
    # Build action catalog if provided
    action_catalog_items = context.extra.get("available_actions") or []
    if action_catalog_items:
        lines = ["Action Catalog (choose one):"]
        for item in action_catalog_items:
            name = item.get("name", "(unnamed)")
            lines.append(f"- {name}")
            schema = item.get("schema")
            if schema is not None:
                import json as _json
                lines.append("  Schema:")
                lines.append("    " + _json.dumps(schema, indent=2).replace("\n", "\n    "))
            examples = item.get("examples") or []
            if examples:
                lines.append("  Examples:")
                for ex in examples:
                    import json as _json
                    pretty = _json.dumps(ex)
                    lines.append(f"    {pretty}")
        action_catalog_str = "\n".join(lines)
    else:
        action_catalog_str = ""

    replacements: Dict[str, str] = {
        "{{context_json}}": context_json,
        "{{context_summary}}": summary,
        "{{perception_json}}": perception_json,
        "{{plan_json}}": plan_json,
        "{{memories_text}}": memories_text,
        "{{scratchpad_json}}": scratchpad_json,
        "{{action_catalog}}": action_catalog_str,
        # Backward-compat replacement (deprecated):
        "{{base_agent_prompt}}": initial_state_agent_prompt if is_first_turn else "",
        # Preferred placeholder:
        "{{initial_state_agent_prompt}}": initial_state_agent_prompt if is_first_turn else "",
        "{{simulation_instructions}}": simulation_instructions,
        "{{character_prompt}}": character_prompt,
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

    # Fallback auto-injection:
    # - If character_prompt exists but template didn't place it, prepend to SYSTEM
    # - If initial_state_agent_prompt exists on first turn and template didn't place it, prepend to USER
    if character_prompt and "{{character_prompt}}" not in template.system:
        if character_prompt not in system:
            system = character_prompt + "\n\n" + system
    if is_first_turn and initial_state_agent_prompt:
        if ("{{initial_state_agent_prompt}}" not in template.user) and ("{{base_agent_prompt}}" not in template.user):
            if initial_state_agent_prompt not in user:
                user = initial_state_agent_prompt + "\n\n" + user

    # Minimal debug logging for prompt rendering
    # Enable with DEBUG_PROMPT_RENDER=1
    try:
        if os.getenv("DEBUG_PROMPT_RENDER", "").lower() in ("1", "true", "yes"):  # pragma: no cover
            print("[PROMPT_RENDER]")
            print(f"  first_turn={is_first_turn}")
            print(f"  injected: character={'yes' if character_prompt else 'no'}, initial_state={'yes' if (is_first_turn and initial_state_agent_prompt) else 'no'}")
            print(f"  placeholders present: character={{'{{character_prompt}}' in template.system}}, initial_state={{'{{initial_state_agent_prompt}}' in template.user}}")
            print(f"  action_catalog_size={len(action_catalog_items) if action_catalog_items else 0}")
    except Exception:
        pass

    return RenderedPrompt(system=system, user=user)
