# Miniverse Project Plan

_Last updated: 2025-10-16_

## 1. Current Snapshot

- `miniverse/`: Library modules (orchestrator, schemas, cognition, memory, persistence, environment helpers).
- `examples/`: Demonstrations
  - `workshop/`: deterministic + LLM maintenance loop (`--llm`, `--debug`, `--analysis`). ✅ Working
  - `standup/`: conversation-heavy stand-up with structured `communication` payloads. ✅ Working
  - `stanford/`: Tier-2 grid-based Valentine's Day party scenario (Stanford replication). 🚧 In progress
  - `_legacy/`: archived prototypes kept for reference.
- `tests/`: 23-unit pytest suite covering cognition flow, schemas, scenario loading, persistence, environment helpers, etc.
- `docs/`: Usage guides, research notes, example roadmap.
- `scripts/`: Utility CLI helpers (lint/test stubs).
- Root files: `README.md`, `CLAUDE.md`, `pyproject.toml`, `uv.lock`, `.env.example`.

## 2. Simulation Loop & Time Model

- A **tick** is the fundamental time step. The orchestrator increments `WorldState.tick` each loop and applies deterministic physics before cognition.
- Scenarios decide how ticks map to real time (minute, hour, day, etc.). Prompts should describe that mapping explicitly; the library does not hard-code “daily” behaviour.
- Planners run once per tick today. Scheduling (e.g., run planner every N ticks or at simulated timestamps) is deferred to upcoming cognition work.
- Scratchpads track plan progress; executors receive the active `PlanStep`. Using `--debug` on any example prints the computed plan, selected step, and final `AgentAction` JSON so users can audit the flow.

## 3. Implemented Capabilities

- **Cognition modules**: Protocols for Planner / Executor / Reflection, with deterministic defaults and LLM-backed options via Mirascope + Tenacity.
- **Tick listeners**: `Orchestrator` accepts optional callbacks (e.g., `TickAnalyzer`) for post-tick analytics/transcripts.
- **Environment helpers**: Graph/grid state schemas plus occupancy and pathfinding utilities.
- **Persistence & memory**: Injected strategies (in-memory by default); `SimpleMemoryStream` stores ordered memories for perception/prompts.
- **Examples**: Workshop (physics-oriented) and Stand-up (social conversation) cover both KPI updates and communication logging.

## 4. Recent Progress (Session 2025-10-16)

### ✅ Completed
1. **DEBUG_LLM logging feature** – Added comprehensive LLM prompt/response logging for debugging
2. **Prompt template fix** – Added communicate action example to `execute_tick` template (messages now have proper format)
3. **Root cause investigation** – Identified that message format was issue, but deeper memory/perception problem remains
4. **Documentation consolidation** – Created ISSUES.md with all findings and next steps
5. ✨ **CRITICAL BUG FIX: Information diffusion** – Added recipient memory creation (orchestrator.py:542-586)
   - Recipients now receive memories when sent messages
   - Both sender and recipient get appropriate memory entries
   - Added `tests/test_information_diffusion.py` with 2 passing tests
   - All 27 tests passing (no regressions)
6. **Comprehensive code review** – Identified 4 architectural issues (documented in ISSUES.md)

### 🚀 Unblocked
7. **Information diffusion** – ✅ FIXED! Recipients now get memories
8. **Valentine's scenario** – Ready to test with fix

### 📋 Immediate Next Steps
9. **Test Valentine's Scenario** – Run `examples/valentines_party.ipynb` with fix and verify information diffusion works
10. **Eliminate dual memory retrieval** – Single fetch in orchestrator (Issue A1)
11. **Add perception logging** – DEBUG_PERCEPTION mode parallel to DEBUG_LLM
12. **Simplify perception builder** – Remove action-based message filtering (Issue A2)

## 5. Release Readiness (PyPI)

Before publishing `miniverse` to PyPI:

1. **Packaging**: Ensure `pyproject.toml` includes metadata (version, description, classifiers, license) and entry points (if any). Add MANIFEST if extra files needed.
2. **Versioning & changelog**: Adopt semantic versioning (`0.1.0`+) and start `CHANGELOG.md`.
3. **Documentation**: Provide a discoverable “Getting Started” in README plus deep links to `docs/`. Consider ReadTheDocs or mkdocs for hosted docs.
4. **Testing/CI**: Configure GitHub Actions (or similar) to run `uv run pytest` (with LLMs mocked) and packaging checks (`uv build`, `twine check`).
5. **Licensing**: Confirm license headers in source files and include LICENSE in distribution.
6. **Examples**: Tag examples as optional extras (document credentials, deterministic fallback) so the package installs cleanly without API keys.
7. **Final polish**: Review TODOs, remove experimental files, and confirm defaults (e.g., logging verbosity) are user-friendly for first-time adopters.

## 6. Quick Reference

- Run workshop: `UV_CACHE_DIR=.uv-cache uv run python -m examples.workshop.run --ticks 5` ✅
- Run stand-up: `UV_CACHE_DIR=.uv-cache uv run python -m examples.standup.run --ticks 4` ✅
- Run stanford: `UV_CACHE_DIR=.uv-cache uv run python -m examples.stanford.run --ticks 3 --analysis` 🚧 (hangs)
- Enable LLM path: add `--llm` (requires `LLM_PROVIDER`, `LLM_MODEL`, provider API key)
- Debug cognition: add `--debug` to print planner/executor/reflection payloads
- Tests: `UV_CACHE_DIR=.uv-cache uv run pytest` ✅ (23 tests passing)

## 7. Atomic Commits (Session 2025-10-14)

- `e93426b` - Enhance inline documentation across core modules
- `5dfb0aa` - Add structured LLM validation feedback, cognition cadence, and memory enhancements
- `8e02cdd` - Add grid position tracking for Tier-2 spatial environments
- `df781ae` - Add move validation helpers for Tier-1 and Tier-2 environments
- `e50fbed` - Add Stanford Valentine's Day party scenario (Tier-2 grid)

**Uncommitted changes**: Stanford runner script (examples/stanford/run.py, rules.py) and grid_position schema fix (List[int] instead of Tuple[int, int])

## 8. Validated Roadmap (Post-Gap Analysis)

### ✅ Phase 0: Current Architecture (COMPLETE)
- Memory stream with importance/recency scoring
- Reflection engine with periodic synthesis
- Planning system with scratchpad
- Executor for action generation
- Dialogue/communication structured payloads
- Partial observability in perception
- Tier 0/1/2 environment support
- Three persistence backends (InMemory, JSON, Postgres)

### 🎯 Phase 1: Example Validation (NOW - 1-2 days)
**Goal**: Validate current architecture works for both canonical examples

1. **Fix welcome.ipynb (Mars habitat)** – Test incrementally, fix bugs, ensure clean execution
2. **Create/fix Valentine's Day party notebook** – Micro-replication of Stanford scenario showing:
   - Information diffusion (party invitation spreads)
   - Memory retrieval (agents remember who told them)
   - Planning & coordination (agents show up at same time/place)
   - **Documented limitations**: keyword matching vs embeddings, flat locations vs tree
3. **Fix tutorial.ipynb** – Basic library walkthrough

### 🔧 Phase 2: Stanford-Quality Memory Retrieval (2-3 weeks)
**Goal**: Match Stanford's three-factor memory retrieval

1. **Embedding-based relevance scoring**:
   - Add `EmbeddingMemoryStream` strategy class
   - Support sentence-transformers (local) or OpenAI embeddings API
   - Store embeddings in persistence layer (new column/field)
   - Implement cosine similarity calculation
   - Combined score: `α_recency * recency + α_importance * importance + α_relevance * cosine_sim`
2. **Plugin architecture**: Users can swap `SimpleMemoryStream` → `EmbeddingMemoryStream` without code changes
3. **Benchmarking**: Compare retrieval quality (keyword vs embedding) on Stanford scenarios

### 🌳 Phase 3: Hierarchical Environment Model (2-3 weeks)
**Goal**: Enable natural language location specification and action grounding

1. **Extend EnvironmentGraph for semantic containment**:
   - Add parent-child relationships (house → kitchen → stove)
   - Tree traversal utilities (`find_object`, `render_subtree_to_nl`)
   - Each agent maintains partial tree (subgraph of world)
2. **Action grounding utilities** in `miniverse/environment/helpers.py`:
   - LLM-based recursive location selection
   - Fallback to keyword matching
   - Integration with existing Executor (no new module needed)
3. **Backward compatibility**: Tier 2 grid and flat location dicts still work

### 🎭 Phase 4: Enhanced Cognitive Fidelity (2-3 weeks)
**Goal**: Match Stanford's full cognitive loop

1. **Reflection tree structure**:
   - Add `source_memory_ids` to `AgentMemory` schema
   - Update reflection engine to parse and store citations
   - Enable "why did I conclude X?" introspection
2. **Reaction decision loop**:
   - Add "should I react?" check to orchestrator tick flow
   - Plan regeneration on unexpected observations
   - More dynamic, responsive agents
3. **Hierarchical plan decomposition**:
   - Add `substeps` to `PlanStep` schema
   - Three-tier planning: daily → hourly → 5-15min chunks
   - Better interruption and replanning

### 📦 Phase 5: PyPI Release (1 week)
**Goal**: Public release with polished documentation

1. Complete `pyproject.toml` metadata
2. Add CI/CD (GitHub Actions for tests)
3. Host documentation (ReadTheDocs or mkdocs)
4. Publish to PyPI as `miniverse` v0.1.0

## 9. Contacts

Current maintainer: Codex agent (2025-03-16). Reference implementations for research are stored under `reference-work/` (gitignored).
