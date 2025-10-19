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

### A7: AgentProfile.age Should Be Optional ✅
**Status:** COMPLETED (2025-10-19)
- Changed `age: int` to `age: Optional[int] = Field(None, ...)`
- Non-human agents (snake AI, robots) can now omit age field
- Existing human agents with age continue to work
- All 29 tests pass
- Updated snake.py to remove placeholder age

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

### A4: Communication Data Duplication

Problem: Messages are stored in actions and as sender/recipient memories.

Open Question:
- Do we still need messages in the actions table now that memories are canonical?

 Goal: Single source of truth for communication is the memory stream. Actions remain for action metadata, not message content.

 Unification Plan:
 1) Audit current usage of action-stored communication fields (search analytics and examples).
 2) If safe, stop persisting message content in `save_actions` for the `communication` field (keep the field in the in-memory `AgentAction` for prompts, but do not persist the message body in actions storage).
 3) Ensure orchestrator continues writing sender/recipient memories (already implemented) and update any consumers to read communication from memories only.
 4) Optionally persist a minimal reference in actions (e.g., communication=true, to=..., message_id) to correlate with memories, without duplicating content.
 5) Update docs: "Memories are canonical for communication; actions contain at most references."

 Migration/Compatibility:
 - No schema change required to `AgentAction`; change is in persistence behavior.
 - Any reporting that reads messages from actions must switch to memory queries.

 Validation Checklist:
 - Perception shows multi-tick messages sourced from memory only
 - No double-counting in analytics (actions vs memories)
 - Workshop/Valentines still run; transcripts/awareness validated from memories

Effort: Investigation ~1 hour; fix TBD.

---

### A5: World-Update Deterministic Processing Adoption

Context: Orchestrator supports `world_update_mode` (auto|deterministic|llm) and `SimulationRules.process_actions()` for deterministic world updates.

Plan:
- Migrate scenarios that have strict mechanics (e.g., vote tallying) to implement `process_actions()`
- Keep LLM world-engine mode for narrative/event synthesis scenarios

Effort: Scenario-dependent.

---

### A6: Documentation Sweep (Follow-ups)

Problem: Older docs referenced `SimpleExecutor` and legacy fallback behavior.

Plan:
- Ensure all public docs and notebooks reference `LLMExecutor` for LLM mode and `RuleBasedExecutor` (or custom deterministic executors) for non-LLM mode
- Keep WORLD_UPDATE_MODE guidance and preflight messages up-to-date

Status: In progress; remaining notebooks to verify.

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

---

## Acceptance Criteria per Item

### ✅ COMPLETED
- A1: ✅ Orchestrator uses one memory fetch; perception receives strings derived from the same objects.
- A2: ✅ Perception shows messages from memory only; removing `recent_actions` does not regress examples.

### OUTSTANDING
- A3: Docs recommend `ImportanceWeightedMemory`; example adapter for embeddings exists.
- A4: No communication content is persisted in actions; analytics and perception work from memories only.
- A5: At least one rules-heavy example implements `process_actions`; preflight reports deterministic via rules.
- A6: No public docs/notebooks reference `SimpleExecutor`; examples indicate WORLD_UPDATE_MODE and executor choices clearly.

