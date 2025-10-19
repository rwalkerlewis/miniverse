Miniverse Prompting Guide

Overview
- This guide explains how Miniverse builds LLM prompts for agent cognition.
- It covers: the default template, custom templates, placeholders, the context protocol, and guidance on what to put in each prompt section to get reliable actions.

Key Concepts
- Default template: A minimal scaffold named "default" in `miniverse/cognition/prompts.py`.
- Placeholders: Tokens the renderer replaces from context, e.g. `{{character_prompt}}`, `{{initial_state_agent_prompt}}`, `{{perception_json}}`, `{{action_catalog}}`.
- Context protocol: A `PromptContext` object containing `AgentProfile`, `AgentPerception`, `WorldState`, memories, plan/scratchpad, and `extra`.
- Renderer: `render_prompt()` assembles the final system/user prompts, placing identity and instructions in the correct roles with safe fallbacks.

Default Template (recommended starting point)
System
```text
{{character_prompt}}

{{simulation_instructions}}

Available actions:
{{action_catalog}}
```
User
```text
{{initial_state_agent_prompt}}

Perception:
{{perception_json}}
```

Role Separation (why it matters)
- Identity (character) belongs in the system message: it sets stable persona and long-term constraints.
- Simulation instructions belong in system: global, scenario-level rules and output contract.
- Action catalog belongs in system: invariant action space and schemas/examples.
- Base agent prompt belongs in user (first turn): initial state/task framing for this agent.
- Perception belongs in user (every turn): dynamic, tick-by-tick observations.
- The default template places each in the right role; if a template omits a placeholder, the renderer auto-injects:
  - `character_prompt` at the start of system
  - `base_agent_prompt` at the start of user

Placeholders
- `{{character_prompt}}`: Generated from `AgentProfile` (name, optional age, role, background, personality, skills, goals, relationships). Keep backgrounds first-person, concise, and relevant.
- `{{initial_state_agent_prompt}}`: Per-agent instruction for the first turn only. Use this to set initial priorities, style, or short-term rules (“Focus on safety today.”, “Prefer coordination before solo work.”). Backward-compat: `{{base_agent_prompt}}` is supported but deprecated.
- `{{simulation_instructions}}`: Global simulation rules and output contract. Default: “You are an agent in a simulation. Read perception and return an AgentAction JSON.”
- `{{perception_json}}`: JSON view of what the agent observes (partial observability). Avoid adding redundant narrative text—LLMs do well with clean JSON.
- `{{action_catalog}}`: Renderer formats a provided list of actions with schemas and examples so the LLM returns valid `AgentAction` JSON. Placed in system to be stable across turns.
- Other placeholders: `{{memories_text}}`, `{{plan_json}}`, `{{context_summary}}`, `{{scratchpad_json}}`. Use them in custom templates when needed.

Context Protocol
- Built via `PromptContext` (see `miniverse/cognition/context.py`). It includes:
  - `agent_profile`: identity and traits
  - `perception`: this-tick observables
  - `world_snapshot`: current world state (pruned and structured)
  - `memories`: recent memories
  - `plan_state`/`scratchpad_state`: optional planning/working memory
  - `extra`: free-form bag; commonly includes:
    - `base_agent_prompt`: per-agent instructions
    - `simulation_instructions`: global rules/output contract
    - `available_actions`: action catalog entries (name, schema, examples)
    - `llm_provider`/`llm_model`: for LLM calls
    - `prompt_library`: to override templates

What to Put in Each Section
- System (stable constraints and identity)
  - Use `{{character_prompt}}` (auto-generated).
  - Keep any system-level rules minimal and general (e.g., "Respond with AgentAction JSON only").
  - Avoid task-specific instructions here; those live in user.

- User (task directive and current context)
  - Start with `{{base_agent_prompt}}` for per-agent instructions.
  - Provide `Perception` as JSON (no prose around it).
  - Provide `Available actions` via `{{action_catalog}}` to steer outputs to valid schemas.
  - Optionally include memories or plan state if your scenario relies on it.

Perception and Context Lifecycle
- Perception is the agent’s dynamic view for the current tick; it is built from:
  - Current location and personal attributes
  - Public environment/resource metrics
  - High-severity system alerts (broadcast)
  - Direct messages (from memory stream, role="recipient")
  - Recent observations/memories
- The orchestrator rebuilds perception each tick; messages from other characters arrive via memories and are included in perception.messages.
- The initial_state_agent_prompt is included only on the first user message (tick 0) unless your template explicitly includes it otherwise.
- System (identity, sim instructions, action catalog) remains stable; user (base prompt first turn + perception each turn) updates every tick.

Action Catalog: Best Practices
- Keep action names concise and distinct.
- Include JSON schemas with required fields so responses validate.
- Provide 1–2 realistic examples per action (brief!).
- Avoid mixing examples for incompatible scenarios in the same catalog.

Custom Templates
- You can pass a custom `PromptTemplate` inline to `LLMExecutor(template=...)`, or select by name via `template_name` against a `PromptLibrary`.
- If your custom template omits `{{character_prompt}}` or `{{base_agent_prompt}}`, the renderer will still inject them in the correct roles at render time.

Examples
Inline template with explicit placeholders
```python
from miniverse.cognition.prompts import PromptTemplate
from miniverse.cognition import LLMExecutor

template = PromptTemplate(
    name="inline",
    system=(
        "{{character_prompt}}\n"
        "You are an agent in a simulation. Respond with an AgentAction JSON."
    ),
    user=(
        "{{base_agent_prompt}}\n\n"
        "Perception:\n{{perception_json}}\n\n"
        "Available actions:\n{{action_catalog}}\n"
    ),
)

executor = LLMExecutor(template=template, available_actions=[...])
```

Default template by name
```python
from miniverse.cognition import LLMExecutor

executor = LLMExecutor(template_name="default", available_actions=[...])
```

Troubleshooting
- Getting free-form text instead of JSON?
  - Ensure `action_catalog` includes schemas and you remind “Return JSON only” in user if needed.
- Identity not appearing?
  - The renderer auto-injects `character_prompt` in system; confirm your `AgentProfile` fields are set.
- Per-agent rules not applied?
  - Set `initial_state_agent_prompt` in `context.extra` (or legacy `base_agent_prompt`).

Logging
- To view prompt context logs (what the builder assembled): set `DEBUG_PROMPT_CONTEXT=1`.
- To view renderer placement logs (which placeholders were injected, first-turn status): set `DEBUG_PROMPT_RENDER=1`.
- Too-long prompts?
  - Remove unused placeholders from your custom template; keep JSONs succinct.

References
- Templates: `miniverse/cognition/prompts.py`
- Renderer: `miniverse/cognition/renderers.py`
- Context: `miniverse/cognition/context.py`

