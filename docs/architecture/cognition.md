# Cognition Stack Overview

_Last updated: 2025-03-15_

## Purpose

Agents in Miniverse run on a shared cognition contract so simulations can swap planners, executors, reflection engines, and memory strategies without rewriting the orchestrator. This document describes the modules, expected data flow, and upcoming prompt stages.

## Modules

| Component | Location | Responsibility |
|-----------|----------|----------------|
| `Scratchpad` | `miniverse/cognition/scratchpad.py` | Working memory: stores active plans, commitments, temporary notes. |
| `Planner` | `miniverse/cognition/planner.py` | Produces structured `Plan` objects (daily schedules, goal trees, etc.). |
| `Executor` | `miniverse/cognition/executor.py` | Converts the current plan step + perception into an `AgentAction`. |
| `ReflectionEngine` | `miniverse/cognition/reflection.py` | Assesses recent memories and emits diary-style reflections when triggers fire. |
| `AgentCognition` | `miniverse/cognition/runtime.py` | Bundles the above modules per agent and provides defaults. |

All components are dependency-injected: users can provide their own implementations per agent ID, or rely on the default stack (currently no-op placeholders).

## Prompt Stages (planned)

1. **Plan Prompt**
   - Trigger: beginning of simulation, at the start of a new day, or when scratchpad plan is exhausted.
   - Input context: agent profile, goals, retrieved memories (topical), environment summary, prior plan state.
   - Output: structured plan (e.g., list of steps with time windows). Stored in scratchpad.

2. **Execute Prompt**
   - Trigger: every tick (or when agent is “active”).
   - Input context: perception, current plan step, scratchpad, relevant memories.
   - Output: `AgentAction` JSON (work, communicate, move, etc.).

3. **Reflection Prompt**
   - Trigger: when accumulated poignancy/importance crosses a threshold or at scheduled intervals (e.g., end of day).
   - Input context: recent memories, plan outcomes, environment events, scratchpad state.
   - Output: reflection text stored as `AgentMemory` with `memory_type="reflection"`, optionally updating scratchpad/goals.

4. **Conversation Prompt** *(optional)*
   - Trigger: when an action requires messaging or social coordination.
   - Output: structured communication payload appended to action `communication` field.

## Orchestrator Integration

- The orchestrator now stores an `AgentCognition` bundle per agent (`Orchestrator.__init__(..., agent_cognition=...)`).
- Upcoming work will extend the tick loop:
  1. Ensure each agent’s scratchpad has an up-to-date plan (calling planner when needed).
  2. Use executor to produce actions instead of `get_agent_action` directly (LLM prompts will live inside executor).
  3. After processing events, pass recent memories + scratchpad into the reflection engine to emit diary entries.
- Default implementations keep the old behavior (no plans, rest actions, no reflections) so existing simulations continue to run until the new modules are filled in.

### Prompt Rendering

- `PromptContext` collects profile, perception, plan metadata, scratchpad state, and recent memories; helpers serialize them into JSON and text summaries.
- `render_prompt(template, context)` replaces placeholders such as `{{context_json}}`, `{{context_summary}}`, `{{perception_json}}`, `{{plan_json}}`, and `{{memories_text}}` inside a `PromptTemplate`.
- `PromptLibrary` holds named templates. The default templates live in `miniverse/cognition/prompts.py`; provide your own library via `AgentCognition(prompt_library=...)`.
- Executors/planners/reflection engines render templates, combine them with any base agent prompt, and forward the system/user strings to the LLM. The legacy single-call path remains available for deterministic strategies.
- `LLMPlanner` and `LLMReflectionEngine` live in `miniverse/cognition/llm.py`. They use the renderer plus `call_llm_with_retries` to parse JSON outputs into `Plan`/`ReflectionResult` objects. Users can supply their own libraries or swap in deterministic implementations per agent.
- `examples/workshop/run.py --llm` registers `plan_workshop`, `execute_workshop`, and `reflect_workshop` templates to show how domain-specific prompts and the default LLM modules fit together.

## Implementation Notes

- Scratchpad and plan schemas are intentionally flexible so domain-specific planners can attach metadata (time ranges, required participants, spatial targets, etc.).
- Reflection triggers will initially use simple rules (e.g., sum of importance >= threshold) but should support custom logic per agent/cognition strategy.
- Memory metadata (tags, embeddings, branch IDs) is now stored on every `AgentMemory`, ready for richer retrieval algorithms.
- `SimpleMemoryStream.get_relevant_memories` performs a lightweight keyword search over recent memories (content + tags). Advanced retrieval adapters (BM25, embeddings) can be plugged in by implementing `MemoryStrategy`.

Refer to `NEXT_STEPS.md` for the roadmap and `plan.md` for repository workflow guidelines.
