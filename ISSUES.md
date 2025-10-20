# Outstanding Issues

Last Updated: 2025-10-19

This file lists only unresolved items aligned with the current architecture.

## ✅ COMPLETED

### A1: Dual Memory Retrieval in Orchestrator ✅
**Status:** COMPLETED (2025-10-19)
- Orchestrator now fetches memories once via `persistence.get_recent_memories()`
- Derives both `recent_memory_strings` and `recent_messages` from same objects
- Eliminates dual-fetch pattern and reduces DB load

### A2: Perception Should Not Parse Actions for Messages ✅
**Status:** COMPLETED (2025-10-19)
- Removed `recent_actions` parameter from `build_agent_perception()`
- Messages now sourced from memory-derived observations only
- Updated orchestrator call sites and tests accordingly
- Better separation of concerns: perception no longer depends on action parsing

### A4: Communication Data Duplication ✅
**Status:** COMPLETED (2025-10-19)
- Persisted actions are sanitized to keep only minimal communication reference (recipient `to`); message body is not persisted in actions
- Full communication content is stored canonically in memories (sender and recipient entries)
- Prevents duplication and clarifies source of truth for transcripts (read from memories)
- Tests updated to assert persisted actions exclude `communication.message`

### A7: AgentProfile.age Should Be Optional ✅
**Status:** COMPLETED (2025-10-19)
- Changed `age: int` to `age: Optional[int] = Field(None, ...)`
- Non-human agents (snake AI, robots) can now omit age field
- Existing human agents with age continue to work
- All 29 tests pass
- Updated snake.py to remove placeholder age

### A5: World-Update Deterministic Processing Adoption ✅
**Status:** COMPLETED (2025-10-19)
- Deterministic world update hook `SimulationRules.process_actions(...)` is implemented and respected by the orchestrator
- `examples/behavior_is_all_you_need/rules.py` provides a rules-heavy implementation
- Added `tests/test_world_update_modes.py` asserting the orchestrator takes the deterministic branch

### A6: Documentation Sweep (Follow-ups) ✅
**Status:** COMPLETED (2025-10-19)
- Updated `docs/USAGE.md` with explicit template usage (`template_name="default"`, inline `PromptTemplate`), `base_agent_prompt` and `{{action_catalog}}`, communication persistence model, deterministic world updates, and testing guidance (LLM-marked test)
- Updated root `README.md` to reflect new defaults, env vars, and key recent changes
- Updated `docs/README.md` index to highlight `USAGE.md` scope
- Updated example READMEs (smallville, examples index); examples now explicitly use `LLMExecutor(template_name="default")`
- Removed/avoided legacy notebook guidance in examples index

### A9: Agent Prompt Pattern - Identity Buried in JSON ✅
**Status:** COMPLETED (2025-10-19)
- Renderer builds a character prompt from `AgentProfile` and injects it into SYSTEM
- Introduced `initial_state_agent_prompt` (first tick only) in USER; `base_agent_prompt` remains as a deprecated alias
- Added `simulation_instructions` placeholder (SYSTEM) with sensible default
- Action catalog moved to SYSTEM for stability across turns
- Examples updated (Smallville Valentine’s) to use `initial_state_agent_prompt`
- New docs: `docs/PROMPTS.md` with roles, placeholders, perception lifecycle, logging flags
- Tests updated to cover new placeholder behavior

### A10: Prompt Template System - Template Noise Masks Intent ✅
**Status:** COMPLETED (2025-10-19)
- Minimal default template now uses placeholders instead of long generic examples
- SYSTEM: `{{character_prompt}}`, `{{simulation_instructions}}`, `{{action_catalog}}`
- USER: `{{initial_state_agent_prompt}}` (first turn), `{{perception_json}}`
- Renderer and context now support env-gated debug logs: `DEBUG_PROMPT_RENDER`, `DEBUG_PROMPT_CONTEXT`
- `LLMExecutor` continues to accept `template` or `template_name` (default "default")
- Tests pass with explicit template selection and new layout

---

## OUTSTANDING ISSUES

### A3: Memory Retrieval Quality

Problem: `SimpleMemoryStream.get_relevant_memories()` uses naive substring matching.

Impact:
- Poor ranking for semantically similar queries
- Users forced to write custom retrieval for quality scenarios

Plan:
- Short-term: Document `ImportanceWeightedMemory` as recommended default; provide an example adapter for embedding-based retrieval in docs
- Mid-term: Implement a Stanford-style memory adapter (embedding + importance + recency) behind a common `MemoryStrategy` interface
- Future: add fuzzy/BM25 hybrid retrieval as optional package

Effort: 2–4 hours for docs/examples; more for algorithmic upgrade.

---

### A11: Grid Perception Not Exposed to Agents

**Problem:** Tier 2 spatial grid environments (`EnvironmentGridState`) are built and maintained in `WorldState`, but agents cannot perceive grid data. The `AgentPerception` schema and `build_agent_perception()` function do not expose grid tiles, positions, or nearby objects.

**Impact:**
- Snake example LLM is completely blind (can't see food, walls, or own position)
- LLM makes decisions with "No map data available" reasoning
- Crashes into walls within 11 ticks despite grid being correctly maintained
- Grid infrastructure exists (`EnvironmentGridState`, pathfinding helpers) but unusable for LLM cognition
- Any Tier 2 spatial scenario will have same issue

**Current Behavior:**
- `world_state.environment_grid` contains full tile map with collision/objects
- `agent_status.grid_position` tracks agent coordinates
- `build_agent_perception()` ignores both fields entirely
- LLM receives empty perception with no spatial awareness

**Expected Behavior:**
- Agents should perceive visible tiles within some radius (e.g., 5x5 window around position)
- Perception should include: walls, objects, food, nearby agents' positions
- LLM should make informed decisions based on visible grid state

**Technical Details:**
- `AgentPerception` schema (schemas.py:456) has no grid-related fields
- `build_agent_perception()` (perception.py:44) doesn't check `world_state.environment_grid`
- No helper function for "get visible tiles around position(x,y) with radius N"
- Grid visibility likely needs partial observability (fog of war) for believability

**Reproduction:**
```bash
LLM_PROVIDER=openai LLM_MODEL=gpt-5-nano uv run python examples/snake/snake.py --ticks 12
# Snake moves blindly, reasoning shows "No map data available"
# Crashes into wall at tick 11 (score: 0, never saw food)
```

**Proposed Solution:**
1. Add `visible_grid_tiles: Optional[Dict[Tuple[int, int], GridTileState]]` to `AgentPerception`
2. Add `get_visible_tiles(grid, position, radius)` helper to `environment/helpers.py`
3. Update `build_agent_perception()` to populate visible_grid_tiles when `environment_grid` present
4. Update snake example to validate LLM can now see and navigate

**Effort:** 4-6 hours (schema extension, perception builder update, helper function, tests)

**Priority:** Medium (blocks all Tier 2 spatial scenarios with LLM agents)

**Workaround:** Pass grid as string in agent prompt (hacky, doesn't scale to large grids)

---

### A8: Logging UX - Information Scattered Across Phases

**Problem:** Current logging output is difficult to follow because information about a single agent's decision-making is scattered across multiple phases. Users must mentally reconstruct what happened to each agent by reading interleaved logs from all agents.

**Impact:**
- 888KB log files with 5% signal-to-noise ratio
- Agent decisions spread across 450+ lines, interleaved with other agents
- 200+ lines of JSON per agent per tick (8,000+ lines total for 5 agents × 8 ticks)
- Key information (message content) buried in JSON, not shown in summary
- No clear agent boundaries - hard to follow individual agent flow

**Plan:**
1. **Phase 1 (P0 - Immediate):** Add verbosity levels via `MINIVERSE_LOG_LEVEL=0|1|2|3`
   - MINIMAL (0): ~50 lines per tick (97% reduction)
   - SUMMARY (1): ~300 lines per tick (80% reduction) 
   - DETAILED (2): Current output (~1,500 lines per tick)
   - DEBUG (3): Full LLM prompts (~3,000 lines per tick)
2. **Phase 2 (P1 - Short-term):** Agent-centric grouping
   - Visual separators between agents
   - Show full message content (not truncated)
   - Group each agent's full tick together
3. **Phase 3 (P2 - Medium-term):** JSON logging mode for analytics
4. **Phase 4 (P3 - Long-term):** Web dashboard for real-time visualization

**Effort:** Phase 1: 1-2 days; Phase 2: 2-3 days; Phases 3-4: Future

---

---


## Test/Build Status

- Unit/integration tests: 39 passing
- New tests assert truthful logging tags: `[LLM]` only on real LLM branches, `[•]` on deterministic branches

---

Document Owner: Kenneth
Next Review: After A3–A6 changes land

---

## Appendix: Files to Review / Touch per Item

### ✅ COMPLETED
- A1 (Dual memory retrieval): ✅ COMPLETED
  - `miniverse/orchestrator.py` - `_get_single_agent_action` (single fetch implemented)
  - `miniverse/memory.py` (no code change required; reference only)

- A2 (Perception should not parse actions): ✅ COMPLETED
  - `miniverse/perception.py` - Removed `recent_actions` parameter, messages from memory
  - `miniverse/orchestrator.py` - Updated call sites, derives messages from memories
  - `tests/test_perception.py` - Updated to new API

### OUTSTANDING
- A3 (Memory retrieval quality):
  - `miniverse/memory.py` (add or document `ImportanceWeightedMemory` as recommended default)
  - New example adapter in docs (embedding-based strategy)

- A4 (Communication unification):
  - `miniverse/orchestrator.py`
    - `_update_memories` (sender/recipient memories already implemented)
    - `_persist_tick` / call to `save_actions` (adjust to not persist message bodies)
  - `miniverse/persistence.py` (action persistence layout)
  - `miniverse/schemas.py` (reference only; keep `AgentAction.communication` type)
  - Docs: USAGE.md (communication patterns), architecture docs

- A5 (Deterministic world updates adoption):
  - `miniverse/simulation_rules.py` (process_actions hook)
  - `miniverse/orchestrator.py` (world_update_mode selection; preflight summary)
  - Scenario rules where strict mechanics apply (implement `process_actions`)

- A6 (Docs sweep):
  - Replace `SimpleExecutor` with `LLMExecutor`/`RuleBasedExecutor`
  - Ensure `WORLD_UPDATE_MODE` usage is shown in examples

- A8 (Logging UX):
  - `miniverse/logging_utils.py` (new file - verbosity levels)
  - `miniverse/orchestrator.py` (replace direct print with conditional logging)
  - All logging call sites throughout codebase

- A9 (Agent prompt pattern):
  - `miniverse/cognition/renderers.py` (build character prompt from AgentProfile)
  - `miniverse/schemas.py` (update AgentProfile docstrings with voice guidance)
  - `docs/USAGE.md` (migration guide and examples)
  - Examples: update to use new pattern

- A10 (Prompt template system):
  - `miniverse/cognition/prompts.py` (minimal default template + domain variants)
  - `miniverse/cognition/llm.py` (template selection API)
  - Examples: show explicit template usage
  - Docs: explain template system

---

## Acceptance Criteria per Item

### ✅ COMPLETED
- A1: ✅ Orchestrator uses one memory fetch; perception receives strings derived from the same objects.
- A2: ✅ Perception shows messages from memory only; removing `recent_actions` does not regress examples.

### OUTSTANDING
- A3: Docs recommend `ImportanceWeightedMemory`; example adapter for embeddings exists.
- A8: Log output is readable at multiple verbosity levels; agent information is grouped together; message content is visible in summaries.

