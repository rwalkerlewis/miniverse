# Changelog

## 2025-10-20 (later)

- Fix: Deterministic fallback respects occupancy capacity
  - Only update `status.location` if `occupancy.enter(target, agent_id)` succeeds
  - Leave agent in place on refusal; safe leave/enter order with exception guards
- Fix: No-op moves guard
  - Short-circuit when `target == current location` to avoid enter/leave removing the agent from occupancy lists
- Renderer: Better action catalog names
  - If `name` is missing, fall back to `action_type` (from item or schema) before `(unnamed)`
- Docs: Align examples with renderer expectations
  - `docs/USAGE.md` shows `available_actions` entries with `name/schema/examples`
  - `README.md` clarifies that action catalog entries should include `name`

## 2025-10-20

- Prompt system simplified:
  - Renderer now performs plain placeholder replacement and action catalog formatting only (no auto-injection, no tick logic, no back-compat).
  - Added `character_prompt_text()` to `PromptContext`; default template consumes `{{character_prompt}}`.
  - Orchestrator sets `initial_state_agent_prompt` only on the first tick; `simulation_instructions` comes from `world_prompt`.
  - Auto-select default templates when unspecified:
    - Planner → `plan`
    - Executor → `default`
    - Reflection → `reflect_diary`
  - One-time startup logs per agent indicate when defaults are used.
- Examples: Smallville Valentine’s action catalog made generic to preserve emergence; only Isabella has initial state about the party.
- Docs: `docs/PROMPTS.md` updated to reflect simplified renderer, roles, and defaults.
- Tests: adjusted for simplified renderer and defaults (ensure first-turn-only injection via orchestrator).

