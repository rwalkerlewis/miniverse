# Claude Code Guide – Miniverse

Authoritative instructions for working on the **Miniverse** codebase.

---

## Quick Orientation

- **README.md** – Installation, examples, quick start, and high-level architecture.
- **ISSUES.md** – Current status, known issues, architectural improvements, and next steps. ⭐ Start here for context!
- **plan.md** – Engineering workflow, repo orientation, and immediate priorities.
- **docs/PROMPTS.md** – Prompt system guide (placeholders, templates, role separation). ⭐ Essential for LLM cognition!
- **docs/architecture/** – Deep dives on cognition stack and environment tiers.
- **docs/USAGE.md** – Step-by-step guide for building new simulations.
- **docs/RESEARCH.md** – Academic foundations (Stanford Generative Agents, AgentTorch, etc.).
- **docs/archive/** – Historical context only; do not resurrect old patterns.

If something isn't covered in these sources, ask before implementing new patterns.

---

## What is Miniverse?

Miniverse is a **generalist agent-based simulation library** designed to replicate and extend Stanford's Generative Agents research. It combines:

- **Deterministic physics** (controllable simulation rules in Python)
- **Emergent behavior** (LLM-driven agent cognition with planning, execution, reflection)
- **Environment tiers** (abstract KPIs → logical graphs → spatial grids)
- **Pluggable persistence** (in-memory, JSON, PostgreSQL)
- **Modular cognition stack** (scratchpad, planner, executor, reflection engine)

**Current Status**: Core architecture is in place. The workshop example demonstrates both deterministic and LLM-driven modes. **Information diffusion fixed (2025-10-16)** – recipients now receive communication memories, enabling Stanford-style information propagation. Valentine's Day party scenario ready for testing. **39 tests passing**.

---

## Core Principles

### 1. Dependency Injection Everywhere
- `Orchestrator` receives all dependencies as constructor arguments: world state, agents, prompts, simulation rules, persistence, memory, cognition modules.
- No hidden globals, no implicit scenario loading.
- Users can swap strategies without modifying library code.

### 2. Deterministic Physics + Emergent Cognition
- **Deterministic layer**: `SimulationRules` subclasses update resources, enforce constraints, generate events (pure Python, predictable).
- **Emergent layer**: Agents use LLM cognition (planner/executor/reflection) for decisions, communication, creativity.
- **Hybrid approach**: Physics is controllable; social dynamics emerge naturally.

### 3. Structured Data (Pydantic Models)
- All state, actions, perceptions, memories use Pydantic schemas (`miniverse/schemas.py`).
- Generic `Stat` / `MetricsBlock` pattern keeps world state flexible across domains.
- When adding fields, update schemas, tests, persistence, and serialization together.

### 4. Three-Tier Environment Model
- **Tier 0** (abstract): Metrics only, no spatial semantics. Current default.
- **Tier 1** (logical graph): Nodes (rooms/teams/channels) + adjacency. Helpers enforce capacity, pathfinding.
- **Tier 2** (spatial grid): Tile-based maps with collision, A* pathfinding. Compatible with Stanford's maze exports.

Scenarios can mix tiers. `WorldState` carries optional `environment_graph` and `environment_grid` fields.

### 5. Cognition Stack (Stanford-Inspired)
Each agent has an `AgentCognition` bundle:
- **Scratchpad**: Working memory (plan state, commitments, flags).
- **Planner**: Produces structured `Plan` objects (daily agenda, multi-step tasks).
- **Executor**: Converts plan step + perception → `AgentAction`.
- **ReflectionEngine**: Assesses recent memories, emits reflections when triggers fire.
- **PromptLibrary**: Named templates (plan, execute, reflect) with context injection.

Implementations can be deterministic (heuristic) or LLM-backed (`LLMPlanner`, `LLMReflectionEngine`). Orchestrator calls these modules during the tick loop.

### 6. Memory as Strategy
- `MemoryStrategy` interface separates storage vs. retrieval.
- `SimpleMemoryStream`: Default recency-based retrieval.
- `AgentMemory` schema includes `importance`, `tags`, `embedding_key`, `branch_id` for richer strategies.
- Reflection entries stored as `memory_type="reflection"`.

### 7. Persistence as Strategy
- Three adapters: `InMemoryPersistence` (fast prototyping), `JsonPersistence` (durable disk storage), `PostgresPersistence` (production analytics).
- Orchestrator accepts any `PersistenceStrategy`; no change to client code.

### 8. Tests First (or Alongside)
- `pytest` + `pytest-asyncio`.
- Mock LLM outputs for unit tests; integration tests cover full orchestrator loops.
- New features need tests matching existing patterns.

---

## Execution Stack & Tooling

- **Python** ≥ 3.10 managed with `uv`.
- **Mirascope** for provider-agnostic LLM calls with structured outputs.
- **Tenacity** for retry logic with schema feedback.
- **pytest** for testing.
- **asyncpg** optional for PostgreSQL persistence.

### Common Commands

```bash
# Install dependencies
uv sync

# Run tests (local cache for speed)
UV_CACHE_DIR=.uv-cache uv run pytest

# Workshop example (deterministic, no LLM)
uv run python examples/workshop/run.py --ticks 6

# Workshop example (LLM cognition)
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4.1
export OPENAI_API_KEY=your_key
uv run python examples/workshop/run.py --llm --ticks 8

# Monte Carlo batch (100 trials with stochastic arrivals)
uv run python examples/workshop/monte_carlo.py --runs 100 --ticks 20 --base-seed 42

# Initialize PostgreSQL (optional)
createdb miniverse
uv run python scripts/init_db.py

# Debugging tools (set environment variables)
DEBUG_LLM=true uv run python examples/workshop/run.py --llm --ticks 3
# → Shows ALL LLM calls: planner prompts/responses, executor prompts/responses, reflection prompts/responses

DEBUG_MEMORY=true uv run python examples/workshop/run.py --llm --ticks 3
# → Shows memory creation and retrieval: who sent/received messages, what memories were stored, what was retrieved

DEBUG_PERCEPTION=true uv run python examples/workshop/run.py --llm --ticks 3
# → Shows what each agent perceives each tick (memories, messages, alerts)

# Combine multiple debug flags for full visibility
DEBUG_LLM=true DEBUG_MEMORY=true MINIVERSE_VERBOSE=true uv run python examples/workshop/run.py --llm --ticks 3
# → Maximum debugging: see everything happening in the simulation
```

Configure LLM providers via environment variables (`LLM_PROVIDER`, `LLM_MODEL`, `OPENAI_API_KEY`, etc.).

**Environment Variables**:
- `DEBUG_LLM=true` – **Comprehensive LLM debugging**. Shows all prompts (system + user) and responses for:
  - Planner (plan generation with all steps)
  - Executor (action selection with reasoning and communication)
  - Reflection (diary entries and insights)
  - World Engine (state updates)
- `DEBUG_MEMORY=true` – **Memory flow debugging**. Shows:
  - Memory creation: sender/recipient memories for communications, action memories, event observations
  - Memory retrieval: what memories each agent retrieved before planning/execution
  - Importance scores and tick timestamps
- `DEBUG_PERCEPTION=true` – **Perception visibility**. Shows what each agent perceives:
  - Recent memories (first 5 shown)
  - Incoming messages (from other agents)
  - System alerts (high-severity events)
- `MINIVERSE_VERBOSE=true` – **Action details**. Show action reasoning and communication content in simulation output (user-friendly demo mode)
- `MINIVERSE_NO_COLOR=true` – Disable color-coded output (for CI/CD or terminals without color support)

**Color-Coded Output** (default enabled):
- `[•]` Blue: Deterministic operations (physics, perception building)
- `[AI]` Yellow: LLM calls (planner, executor, reflection)
- `[✓]` Green: Success/completion
- `[i]` Cyan: Metadata/reasoning (in verbose mode)

**Note**: For comprehensive logging UX improvements (verbosity levels, agent-centric grouping), see ISSUES.md A8.

---

## Current Code Layout

```
miniverse/
  __init__.py                     # Public exports
  config.py                       # LLM provider config (env var defaults)
  orchestrator.py                 # Main simulation loop with tick listeners
  schemas.py                      # Pydantic models (WorldState, AgentProfile, etc.)
  simulation_rules.py             # Deterministic physics interface
  persistence.py                  # Persistence strategies (InMemory/JSON/Postgres)
  memory.py                       # Memory strategies (SimpleMemoryStream, etc.)
  perception.py                   # Builds partial observability views
  llm_calls.py                    # LLM helper functions (process world update)
  llm_utils.py                    # Tenacity-backed retry logic + structured outputs
  scenario.py                     # ScenarioLoader for JSON files
  logging_utils.py                # Color-coded output utilities (DEBUG_* flags)
  cognition/
    __init__.py                   # Cognition module exports
    scratchpad.py                 # Working memory data structure
    planner.py                    # Planner protocol + SimplePlanner
    executor.py                   # Executor protocol + RuleBasedExecutor/LLMExecutor
    reflection.py                 # ReflectionEngine protocol + SimpleReflectionEngine
    runtime.py                    # AgentCognition bundle + defaults
    context.py                    # PromptContext builder
    prompts.py                    # PromptLibrary + PromptTemplate + DEFAULT_PROMPTS
    renderers.py                  # Template rendering ({{context_json}}, etc.)
    llm.py                        # LLMPlanner + LLMReflectionEngine
    cadence.py                    # PlannerCadence + ReflectionCadence scheduling
  environment/
    __init__.py                   # Environment tier exports
    schemas.py                    # EnvironmentGraphState, EnvironmentGridState, etc.
    graph.py                      # EnvironmentGraph + LocationNode
    grid.py                       # EnvironmentGrid + GridTile
    helpers.py                    # GraphOccupancy, shortest_path, grid_shortest_path

examples/
  workshop/
    run.py                        # Deterministic + LLM cognition demo
    llm_demo.py                   # Minimal LLM example
    monte_carlo.py                # Batch runner with stochastic arrivals
    scenario.json                 # Tier 1 scenario (logical graph)
  _legacy/                        # Archived old examples (reference only)

scripts/
  run_simulation.py               # CLI wrapper (if still in use)
  init_db.py                      # PostgreSQL schema setup

tests/
  test_*.py                       # Pytest suite (orchestrator, persistence, etc.)

docs/
  README.md                       # Documentation index
  USAGE.md                        # Building simulations guide
  RESEARCH.md                     # Academic foundations
  architecture/
    cognition.md                  # Cognition stack deep dive
    environment.md                # Environment tiers deep dive
  research/
    agent-simulations/
      stanford-comparison.md      # Gap analysis vs. Reverie
    ...                           # Other research notes
  archive/                        # Historical documents (do not use)
```

---

## How to Work Safely

### 1. Plan
- Check `NEXT_STEPS.md` for strategic priorities (WP1-WP7).
- Check `plan.md` for immediate task breakdown.
- Align new work with roadmap before coding.

### 2. Understand the Modules
- **Orchestrator tick flow**:
  1. Apply `SimulationRules.apply_tick` (deterministic updates).
  2. For each agent: ensure plan is current (call planner if needed), gather perception, run executor, collect action.
  3. Process actions (world update via optional world LLM or deterministic rules).
  4. Persist state/actions/memories.
  5. Run reflection engines (per agent) based on triggers.
  6. Update scratchpads, advance plan pointers.
  7. Invoke tick listeners for analysis/logging.

- **Cognition modules**: See `docs/architecture/cognition.md` for prompt stages, context assembly, template rendering.
- **Environment tiers**: See `docs/architecture/environment.md` for Tier 0/1/2 details and helper utilities.
- **Memory retrieval**: `SimpleMemoryStream.get_relevant_memories` does keyword search over recent memories (content + tags). Advanced retrieval (BM25, embeddings) can be plugged in via `MemoryStrategy`.

### 3. Write Tests
- Mock LLM calls for unit tests (see existing tests for patterns).
- Add integration tests for new cognition modules, persistence adapters, environment helpers.
- Run `UV_CACHE_DIR=.uv-cache uv run pytest` before submitting.

### 4. Update Documentation
- **README.md** – if you change public API or examples.
- **USAGE.md** – if you add new cognition/environment patterns.
- **NEXT_STEPS.md** – check off work packages as they complete.
- **plan.md** – update priorities if roadmap shifts.
- **docs/architecture/** – add deep dives for new modules.
- **CLAUDE.md** (this file) – keep in sync when behavior changes.

### 5. Examples
- Workshop example is the reference implementation (`examples/workshop/run.py`).
- Show both deterministic and LLM modes.
- If you add new features (e.g., Tier 2 grid, advanced memory retrieval), create a new example or extend workshop.

### 6. Persist Your Work
- When introducing new persistence backends, wire them into orchestrator and provide CLI/example integration.
- When adding cognition modules, provide both deterministic and LLM implementations.
- Keep backward compatibility: existing simple simulations must still run with defaults.

---

## Roadmap Summary (from NEXT_STEPS.md)

### Work Packages
1. **WP1 – Memory Schema & Strategy**: ✅ Largely complete. Extended `AgentMemory` with metadata, split storage/retrieval.
2. **WP2 – Cognition Modules**: ✅ Complete. Scratchpad, Planner, Executor, ReflectionEngine interfaces + defaults + LLM implementations.
3. **WP3 – Environment Tiering**: ✅ Complete. Graph/grid schemas, helper utilities (occupancy, pathfinding), scenario loader integration.
4. **WP4 – Prompt Templates & Context Assembly**: ✅ Foundational implementation in place. `render_prompt`, default templates, workshop example.
5. **WP5 – Stanford Scenario Replication**: 🚧 In progress. Blueprint Valentine's Day party scenario.
6. **WP6 – Documentation & Examples**: 🚧 In progress. README updated, workshop example shipped, USAGE.md exists.
7. **WP7 – Testing & Tooling**: 🚧 Ongoing. Add mocks, regression tests, benchmarking.

### Key Milestones
- ✅ Hybrid architecture (deterministic + emergent) is stable.
- ✅ Workshop example demonstrates both modes.
- 🚧 Stanford parity: can we replicate emergent party coordination?
- 🧭 Advanced memory retrieval (BM25, embeddings) – future enhancement.
- 🧭 Branching/Loom – deferred until core loop is validated.

---

## What Makes This Different from Stanford's Reverie

### We Do Better
- **Modular orchestrator**: Clean dependency injection, swap strategies without fork.
- **Deterministic rule hooks**: First-class physics layer for domain-specific constraints.
- **Schema flexibility**: Generic `Stat` model supports KPIs, resources, agent attributes across domains.
- **Pluggable persistence**: In-memory, JSON, PostgreSQL with same interface.

### Gaps Closed (vs. Original Roadmap)
- ✅ Cognition stack (scratchpad, planner, executor, reflection) now implemented.
- ✅ Memory metadata (importance, tags, embedding_key) in place.
- ✅ Environment tiers (Tier 0/1/2) with helpers.
- ✅ Prompt templates + context assembly utilities.

### Remaining Work
- 🚧 Stanford scenario replication (validate emergent behavior).
- 🚧 Advanced memory retrieval algorithms (BM25, semantic embeddings).
- 🧭 Performance tuning (batching, caching, plan scheduling).
- 🧭 Frontend visualization (optional; focus is backend correctness).

---

## Development Guidelines

### Code Style
- Follow existing Pydantic patterns for schemas.
- Use protocols for interfaces (see `Planner`, `Executor`, `ReflectionEngine`).
- Prefer async where persistence/memory/LLM calls are involved.
- Keep deterministic rules pure (optional `random.Random` if stochasticity needed).

### Naming Conventions
- `*Strategy` for pluggable backends (Persistence, Memory).
- `*Engine` for processing modules (ReflectionEngine, maybe WorldEngine).
- `*State` for Pydantic schemas (WorldState, AgentStatus, EnvironmentGraphState).
- `*Rules` for deterministic physics subclasses (WorkshopRules, MarsRules).

### Error Handling
- Use Tenacity for LLM retries (`call_llm_with_retries` in `llm_utils.py`).
- Validation errors should provide schema feedback to LLM for self-correction.
- Deterministic rules should raise clear exceptions for invalid states.

### Performance
- Avoid blocking LLM calls in tight loops; consider batching or async gather.
- Use `InMemoryPersistence` for fast prototyping/testing.
- PostgreSQL for long runs with analytics needs.
- JSON for durable single-session storage.

---

## Key Files to Understand

| File | Purpose |
|------|---------|
| `miniverse/orchestrator.py` | Main simulation loop; dependency injection hub |
| `miniverse/schemas.py` | All Pydantic models (WorldState, AgentProfile, AgentAction, etc.) |
| `miniverse/cognition/runtime.py` | AgentCognition bundle; `build_default_cognition()` |
| `miniverse/cognition/llm.py` | LLM-backed planner + reflection engine |
| `miniverse/cognition/prompts.py` | Default prompt templates; `DEFAULT_PROMPTS` library |
| `miniverse/cognition/renderers.py` | Template rendering logic (`render_prompt`) |
| `miniverse/cognition/cadence.py` | PlannerCadence + ReflectionCadence scheduling |
| `miniverse/memory.py` | MemoryStrategy interface; `SimpleMemoryStream` |
| `miniverse/persistence.py` | PersistenceStrategy interface; three adapters |
| `miniverse/simulation_rules.py` | Deterministic physics interface |
| `miniverse/logging_utils.py` | Color-coded output; DEBUG_* environment variables |
| `miniverse/environment/helpers.py` | Graph/grid utilities (occupancy, pathfinding) |
| `examples/workshop/run.py` | Reference implementation (deterministic + LLM modes) |
| `docs/PROMPTS.md` | Prompt system guide (A9/A10 refactor) |
| `docs/architecture/cognition.md` | Cognition stack deep dive |
| `docs/architecture/environment.md` | Environment tiers deep dive |
| `docs/research/agent-simulations/stanford-comparison.md` | Gap analysis |

---

## Quick Checklist for New Features

- [ ] Does it align with `NEXT_STEPS.md` roadmap?
- [ ] Have you updated relevant schemas in `miniverse/schemas.py`?
- [ ] Have you provided both deterministic and LLM implementations (if applicable)?
- [ ] Have you written tests (unit + integration)?
- [ ] Have you updated `README.md` / `USAGE.md` / architecture docs?
- [ ] Have you verified backward compatibility (existing examples still run)?
- [ ] Have you wired new strategies into orchestrator constructor?
- [ ] Have you run `UV_CACHE_DIR=.uv-cache uv run pytest`?

---

## Getting Help

- **Understanding architecture**: Read `docs/architecture/cognition.md` and `environment.md`.
- **Building a simulation**: Follow `docs/USAGE.md`.
- **Comparing to Stanford**: See `docs/research/agent-simulations/stanford-comparison.md`.
- **Unclear on roadmap**: Check `NEXT_STEPS.md` for work packages.
- **Immediate priorities**: Check `plan.md` for task breakdown.
- **When in doubt**: Ask before implementing new patterns. The codebase is modular by design; new features should fit the existing interfaces.

---

Stay within these guardrails and Miniverse stays maintainable, extensible, and true to its hybrid (deterministic + emergent) design philosophy.

-- Claude | 2025-10-14
- always use gpt-5-nano with openai which just released. you can simply do this by bringing in the env llm model