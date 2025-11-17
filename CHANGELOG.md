# Changelog

## 2025-11-17 (local LLM support)

### Added
- **Local LLM support**: Miniverse now supports local LLM servers like Ollama, LM Studio, vLLM, and any OpenAI-compatible API
  - New config options: `LOCAL_LLM_BASE_URL` and `LOCAL_LLM_API_KEY` environment variables
  - All LLM calls (`call_llm_with_retries`, `LLMPlanner`, `LLMExecutor`, `LLMReflectionEngine`) now accept `base_url` and `api_key` parameters
  - Config validation updated to handle local provider configuration
  - **Breaking change**: For local LLMs, set `LLM_PROVIDER=openai` (for compatibility) and `LOCAL_LLM_BASE_URL` to your server endpoint

### Changed
- `miniverse/config.py`: Added `LOCAL_LLM_BASE_URL` and `LOCAL_LLM_API_KEY` configuration
- `miniverse/llm_utils.py`: `call_llm_with_retries()` now accepts optional `base_url` and `api_key` parameters for local servers
- `miniverse/llm_calls.py`: Updated `get_agent_action()` and `process_world_update()` to pass local LLM configuration
- `miniverse/cognition/llm.py`: Updated `LLMPlanner`, `LLMExecutor`, and `LLMReflectionEngine` to support local LLMs

### Documentation
- New guide: `docs/LOCAL_LLMS.md` - comprehensive guide covering:
  - Why use local LLMs (privacy, cost, offline, control)
  - Setup instructions for Ollama, LM Studio, and vLLM
  - Model selection guide with recommendations for different use cases
  - Hardware requirements and performance optimization
  - Troubleshooting common issues
  - Examples for all workshop and Smallville scenarios
- Updated `README.md` with local LLM examples and link to guide

### Example Usage

**Ollama (recommended for beginners):**
```bash
ollama pull llama3.1:8b
export LLM_PROVIDER=openai
export LLM_MODEL=llama3.1:8b
export LOCAL_LLM_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=not-needed
uv run python examples/workshop/run.py --llm --ticks 10
```

**LM Studio:**
```bash
export LLM_PROVIDER=openai
export LLM_MODEL=mistral-7b-instruct
export LOCAL_LLM_BASE_URL=http://localhost:1234/v1
export OPENAI_API_KEY=not-needed
uv run python examples/smallville/valentines_party.py
```

## 2025-10-20 (grid perception)

- Perception: Restored the minimal Tier-2 behavior; `AgentPerception` no longer exposes `grid_ascii`. The builder now only prepends the ASCII view to `recent_observations`, and scenarios inject richer context via `SimulationRules.customize_perception()`.
- Cognition: Prompt renderer gained `{{perception_text}}`, `{{current_tick}}`, and `{{agent_id}}` placeholders so executor templates can work with plain text boards instead of JSON dumps.
- Schemas: `AgentAction.reasoning` defaults to an empty string, letting scenarios skip rationale without breaking validation.
- Example (Snake): Executor uses a custom ASCII-only prompt template and strips structured perception fields, closely mirroring the legacy `snake.py` experience.
- Tests: Updated `tests/test_perception.py` and friends to reflect the simplified perception payload.

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
- Notebook: Smallville Valentine's walkthrough is Jupyter-safe
  - Update `examples/smallville/valentines_party_v2.ipynb` to instantiate `InMemoryPersistence()`
  - Add notebook-safe async runner with `nest_asyncio` and `asyncio.run` fallback
  - Fix orchestrator wiring to avoid class vs instance initialization issues
- Logging: Include action parameters in tick summary
  - Default tick summary now shows `target`, `parameters`, and `comm.to` alongside action type
  - Remove reasoning truncation; full reasoning is now printed
- Repo hygiene: Stop tracking legacy experimental example
  - Add `examples/behavior_is_all_you_need/` to `.gitignore`
  - Remove from Git index while keeping files locally (`git rm --cached`)

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
