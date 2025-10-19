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

---

---

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

### A9: Agent Prompt Pattern - Identity Buried in JSON

**Problem:** AgentProfile (name, background, personality, relationships) is buried in user prompt JSON instead of system prompt, creating suboptimal LLM behavior and confusing separation of concerns.

**Impact:**
- LLM must parse JSON to understand who it is (poor role-play performance)
- Users forced to repeat identity info in `agent_prompts` to compensate
- Unclear mental model: what goes in `AgentProfile` vs `agent_prompts`
- Redundant information and inconsistent voice guidance

**Plan:**
1. **Phase 1:** Update renderer to build character prompt from AgentProfile
   - Move identity (name, background, personality, relationships) to system prompt
   - Keep only situational context in `agent_prompts`
   - Use first-person voice for identity, situational for current state
2. **Phase 2:** Update schema docs with clear guidance
3. **Phase 3:** Migration guide for existing scenarios
4. **Phase 4:** Advanced PromptBuilder class for explicit control

**Effort:** Phase 1: 1 day; Phases 2-3: 1 day each; Phase 4: Future

---

### A10: Prompt Template System - Template Noise Masks Intent

**Problem:** Default templates contain 130+ lines of generic action examples that can conflict with domain-specific agent instructions, creating opaque behavior and debugging difficulty.

**Impact:**
- Domain-specific instructions (e.g., "ONLY move actions" for snake game) get diluted by generic examples
- Template choice is invisible in code (`LLMExecutor()` hides what template is used)
- Users don't understand where tuning is needed due to leaky abstraction

**Plan:**
1. **Short-term:** Create minimal default template (no action examples)
2. **Medium-term:** Add domain-specific templates users can opt into:
   - `execute_ops` - work/rest/monitor examples
   - `execute_social` - communicate/rest examples  
   - `execute_game` - move/action examples
3. **Long-term:** Allow `agent_prompts` to override template when needed

**Effort:** 1-2 days for minimal template + domain variants

---


## Test/Build Status

- Unit/integration tests: 29 passing
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
- A9: AgentProfile identity appears in system prompt; users don't need to repeat identity info in agent_prompts; clear separation between identity and situational context.
- A10: Default template is minimal; domain-specific templates available; template choice is explicit in code; no conflicting action examples.

