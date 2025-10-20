# Miniverse

**Build any agent-based simulation with emergent behavior.**

Miniverse is a Python library for creating simulations where LLM-powered agents interact in deterministic environments. Inspired by Stanford's Generative Agents research, it combines:

- **Deterministic physics** (controllable simulation rules)
- **Emergent behavior** (LLM-driven agent decisions)
- **Flexible architecture** (swap cognition modules, persistence strategies)

## Quick Start

Building a simulation takes three steps:

### 1. Define scenario (JSON file)

```json
{
  "name": "Workshop Operations",
  "description": "Three-person maintenance crew coordinating tasks",
  "agents": [
    {
      "profile": {
        "agent_id": "lead",
        "name": "Morgan Reyes",
        "role": "floor_lead",
        "background": "Floor lead ensuring throughput and safety",
        "goals": ["Keep operations smooth", "Reduce task backlog"]
      },
      "status": {
        "location": "ops",
        "attributes": {"energy": 80, "stress": 35}
      }
    }
  ],
  "resources": {"power_kwh": 120.0, "task_backlog": 6},
  "environment": {"temperature_c": 22.0}
}
```

### 2. Define physics (Python)

```python
from miniverse import SimulationRules, WorldState

class WorkshopRules(SimulationRules):
    """Deterministic physics: resources degrade, agents recover energy."""

    def apply_tick(self, state: WorldState, tick: int) -> WorldState:
        updated = state.model_copy(deep=True)

        # Resources degrade over time
        power = updated.resources.get_metric("power_kwh", default=120.0)
        # Reduce power by 2.5 kWh per tick, but never below 0
        # max(0.0, ...) ensures power can't go negative
        power.value = max(0.0, float(power.value) - 2.5)

        # Agents recover energy when resting, lose energy when active
        for agent in updated.agents:
            energy = agent.get_attribute("energy", default=75)
            if agent.activity == "rest":
                # Resting: gain 4 energy points, but cap at 100
                # min(100.0, ...) ensures energy can't exceed maximum
                energy.value = min(100.0, float(energy.value) + 4)
            else:
                # Active: lose 3 energy points, but never below 0
                # max(0.0, ...) ensures energy can't go negative
                energy.value = max(0.0, float(energy.value) - 3)

        updated.tick = tick
        return updated
```

### 3. Run simulation

```python
from miniverse import Orchestrator, ScenarioLoader
from miniverse.cognition import AgentCognition, LLMPlanner, LLMExecutor, LLMReflectionEngine, Scratchpad

# Load scenario
loader = ScenarioLoader()
world_state, agents = loader.load("workshop")

# Configure LLM-driven cognition for each agent
# (Only executor is required - planner/reflection are optional enhancements)
cognition_map = {
    agent.agent_id: AgentCognition(
        executor=LLMExecutor(template_name="default"),  # explicit minimal template
        planner=LLMPlanner(),  # Optional - generates multi-step plans
        reflection=LLMReflectionEngine(),  # Optional - synthesizes insights
        scratchpad=Scratchpad()  # Optional - needed if using planner
    )
    for agent in agents
}

# Run simulation
orchestrator = Orchestrator(
    world_state=world_state,
    agents={a.agent_id: a for a in agents},
    simulation_rules=WorkshopRules(),
    agent_cognition=cognition_map,
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    # Select world update behavior: 'auto' | 'deterministic' | 'llm'
    world_update_mode="auto"
)

result = await orchestrator.run(num_ticks=20)
```

### Workshop example scripts

The repository ships with a concrete workshop scenario (`examples/workshop/`). Use it to explore the deterministic baseline, Monte Carlo batching, and LLM cognition.

```bash
# Deterministic run (5 ticks)
UV_CACHE_DIR=.uv-cache uv run python -m examples.workshop.run --ticks 5

# Monte Carlo batch with stochastic arrivals (100 trials)
UV_CACHE_DIR=.uv-cache uv run python -m examples.workshop.monte_carlo --runs 100 --ticks 20 --base-seed 42

# LLM cognition demo (requires provider/model + API key)
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini
UV_CACHE_DIR=.uv-cache uv run python -m examples.workshop.llm_demo --ticks 8

# Verbose cognition logging + per-tick analysis
UV_CACHE_DIR=.uv-cache uv run python -m examples.workshop.run --llm --ticks 6 --debug --analysis
```

The deterministic scripts run without network access and are ideal for parameter sweeps. Each example includes a Python-based planner so you get meaningful behaviour even without an LLM; add `--debug` to see the generated plan steps, selected actions, and any reflections.

### Stand-up conversation example

To focus on social dynamics and conversation payloads, try the stand-up scenario (`examples/standup/`). Each action emits structured chat messages (`communication.to` / `communication.message`).

```bash
# Deterministic stand-up with transcript
UV_CACHE_DIR=.uv-cache uv run python -m examples.standup.run --ticks 4

# LLM-driven stand-up (requires provider/model)
UV_CACHE_DIR=.uv-cache uv run python -m examples.standup.run --llm --ticks 4

# Debug log of planner/executor payloads + analysis metrics
UV_CACHE_DIR=.uv-cache uv run python -m examples.standup.run --llm --ticks 4 --debug --analysis
```

Key knobs live in `examples/standup/run.py`—tweak `StandupPlanner.ROLE_STEPS`, `StandupExecutor.ROLE_MESSAGES`, or the heuristics inside `ConversationPostTick` to see how the conversation shifts. The debug flag will echo the generated plans, chosen messages, and reflections each tick so you can confirm context is flowing through the pipeline.

## What Makes Miniverse Different

**Hybrid Architecture:**
- **Deterministic layer**: Python code controls physics (resource budgets, machine speeds, scheduling)
- **Emergent layer**: LLMs control agents (decisions, communication, creativity)
- **User control**: Modify simulation rules, branch scenarios, adjust parameters

**Unlike pure LLM simulations**: Physics is predictable and controllable
**Unlike pure game engines**: Social dynamics emerge naturally from agent interactions

## Installation

```bash
# Clone and install
git clone <repository-url>
cd miniverse
uv sync

# Configure LLM provider (examples default to OpenAI gpt-5-nano)
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-5-nano
export OPENAI_API_KEY=your_openai_api_key

# Optional: Initialize PostgreSQL database for persistence
createdb miniverse
uv run python scripts/init_db.py
```

### Persistence Options

Miniverse ships with pluggable persistence strategies:

- **In-memory** (default): fastest for prototyping; state is discarded after each run.
- **JSON files**: durable storage on disk without requiring a database.
- **PostgreSQL**: production-ready history for long simulations and analysis.

Use the CLI to choose the backend:

```bash
# In-memory (default)
uv run python scripts/run_simulation.py --ticks 5 --scenario examples/factory/scenario.json

# JSON storage (writes to ./simulation_runs)
uv run python scripts/run_simulation.py --ticks 5 --scenario examples/retail/scenario.json --persistence json --json-dir runs

# PostgreSQL storage (requires asyncpg and a database URL)
uv run python scripts/run_simulation.py --ticks 5 --scenario examples/habitat/scenario.json --persistence postgres --database-url postgresql://localhost/miniverse
```

Or inject the strategy directly when constructing the orchestrator:

```python
from miniverse import Orchestrator, JsonPersistence

persistence = JsonPersistence("./runs")
orchestrator = Orchestrator(
    world_state=world_state,
    agents=agents,
    world_prompt=world_prompt,
    agent_prompts=agent_prompts,
    llm_provider="openai",
    llm_model="gpt-5-nano",
    persistence=persistence,
)
```

## Core Concepts

### 1. SimulationRules (Deterministic)

Define the physics of your world in Python:

```python
class FactorySimulationRules(SimulationRules):
    def apply_tick(self, state: WorldState, tick: int) -> WorldState:
        # Deterministic physics
        new_state = state.model_copy(deep=True)

        # Assembly line produces 10 units per minute
        new_state.resources.units_produced += 10 * (tick_seconds / 60)

        # Machines degrade 0.1% per tick
        new_state.resources.machine_health *= 0.999

        return new_state
```

### 2. Agent Behavior (Emergent)

Agents decide actions via LLM calls:

```python
perception = {
    "personal_attributes": {
        "energy": {"value": 65, "unit": "%"},
        "focus": {"value": 80, "unit": "%"},
    },
    "visible_resources": {
        "power_kwh": {"value": 118.0, "unit": "kWh", "label": "Battery Reserve"},
    },
    "environment_snapshot": {
        "temperature": {"value": 21.0, "unit": "°C"},
    },
    "system_alerts": ["Line 2 throughput dropped below target"],
    "messages": [{"from": "lead", "message": "Need a technician at station B"}],
    "recent_observations": ["Calibrated station B at tick 4"],
}

action = await llm_call(
    system_prompt=agent_prompt,
    user_prompt=json.dumps(perception),
)
# → AgentAction(action_type="inspect", target="station_b", reasoning="...")
```

### 3. Persistence & Analysis

Choose your storage backend:

```python
from miniverse import JsonPersistence, PostgresPersistence

# JSON files (durable storage)
persistence = JsonPersistence("./simulation_runs")

# PostgreSQL (production analytics)
persistence = PostgresPersistence(
    database_url="postgresql://localhost/miniverse"
)

orchestrator = Orchestrator(
    ...,
    persistence=persistence
)
```

## Project Status

**Current Focus**: Testing information diffusion fix and architectural improvements

✅ **Working Today (v0.1)**
- **Core Architecture**: Memory stream, reflection, planning, execution
- **Communication Actions**: Agents can send messages with proper formatting
- **Information Diffusion**: ✨ **FIXED!** Recipients now receive communication memories (2025-10-16)
- **Persistence**: Three backends (InMemory, JSON, PostgreSQL) with pluggable interface
- **Environment Tiers**: Abstract KPIs (Tier 0), Logical graphs (Tier 1), Spatial grids (Tier 2)
- **Cognition Stack**: Protocols for swappable planner/executor/reflection with deterministic and LLM implementations
- **Examples**: Workshop (operations with deterministic rules) ✅
- **DEBUG_LLM**: Comprehensive logging of all LLM prompts and responses
- **Test Coverage**: 27 passing tests including new information diffusion tests

🔧 **Recent Fixes** (2025-10-16)
- **Critical Bug**: Added recipient memory creation for communication actions
  - Before: Only senders remembered sending messages
  - After: Both senders and recipients get appropriate memories
  - Impact: Enables Stanford-style information diffusion patterns
  - Tests: `tests/test_information_diffusion.py`
  - Details: See `ISSUES.md` for full analysis

⚠️ **Known Limitations**
- **LLM Non-Determinism**: Agent behavior varies between runs (expected with temperature > 0)
- **Memory Retrieval**: SimpleMemoryStream uses basic keyword matching (consider ImportanceWeightedMemory for production)
- See `ISSUES.md` for architectural improvements in progress

🚧 **In Progress**
- Valentine's Day scenario testing with information diffusion fix
- Architectural improvements (dual memory retrieval, perception simplification)
- Stanford Generative Agents comparison study

🔧 **Planned Enhancements (Phase 2-4)**

### Phase 2: Stanford-Quality Memory Retrieval
- **Embedding-based relevance scoring**: Add `EmbeddingMemoryStream` with cosine similarity
- **Three-factor retrieval**: `α_recency * recency + α_importance * importance + α_relevance * embedding_similarity`
- **Pluggable embeddings**: Support sentence-transformers (local) or OpenAI API
- **Why**: Current keyword matching works but misses semantic relationships. Stanford uses embeddings for "relevance" component.

### Phase 3: Hierarchical Environment Model
- **Semantic containment tree**: `house → kitchen → stove` relationships for natural language location queries
- **Action grounding helpers**: LLM-based recursive location resolution (e.g., "get coffee" → find coffee machine)
- **Partial environment trees**: Each agent maintains subgraph of world they've explored
- **Why**: Enables more natural agent reasoning about space. Current flat locations work but require explicit naming.

### Phase 4: Enhanced Cognitive Fidelity
- **Reflection tree structure**: Citations linking reflections to source memories (enable "why did I think X?" introspection)
- **Reaction decision loop**: "Should I react to this observation?" check with plan regeneration on surprises
- **Hierarchical planning**: Three-tier decomposition (daily → hourly → 5-15min chunks) for realistic long-term behavior
- **Why**: Increases agent believability and enables more complex emergent coordination patterns.

### Future (Post-1.0)
- Branching scenarios (explore alternate timelines with Loom-style interface)
- Advanced memory retrieval (BM25, hybrid search, memory decay)
- Additional deterministic helper modules (queuing theory, energy budgeting, resource optimization)
- Large-scale performance testing (100+ agents)

See `docs/README.md` for the documentation index.

### Key recent changes

- Explicit prompt templates: `LLMExecutor(template_name="default")` or inline `PromptTemplate`
- Agent prompts injection: `agent_prompts` are prepended to the system prompt
- Action Catalog: pass `available_actions` to inject `{{action_catalog}}` (each entry should include `name`, `schema`, and optional `examples`)
- Communication persistence: actions store only `{ "to": agent_id }`; full messages live in memories

## Example

- `examples/workshop/run.py` – operations workshop with role-based plans and logical environment graph (`uv run python examples/workshop/run.py`). Pass `--llm` to enable the LLM planner/executor/reflection stack once you have provider credentials configured.

Legacy examples are preserved under `examples/_legacy/` for reference.

## Documentation

- **[USAGE.md](docs/USAGE.md)** - How to build simulations (templates, prompts, comms model) ⭐
- **[ISSUES.md](ISSUES.md)** - Current status, known issues, and next steps
- **[CLAUDE.md](CLAUDE.md)** - Development guide and codebase orientation
- **[Cognition Stack](docs/architecture/cognition.md)** - Planner/executor/reflection flow
- **[Environment Tiers](docs/architecture/environment.md)** - Logical graphs and spatial grids
- **[RESEARCH.md](docs/RESEARCH.md)** - Academic foundations
- **[Debugging Archive](docs/debugging/)** - Investigation notes

## Research Foundation

Based on:
- **Stanford Generative Agents** (Park et al., 2023) - Memory, reflection, partial observability
- **MIT AgentTorch** (2024) - Large-scale agent simulation patterns
- **AgentSociety** (2024) - Multi-agent coordination

See [docs/RESEARCH.md](docs/RESEARCH.md) for detailed analysis.

## Contributing

This project is in active development. See [ROADMAP.md](ROADMAP.md) for current priorities.

Current priorities:
1. Deepening architectural decoupling (no hidden scenario assumptions)
2. Expanding first-party scenario templates across domains
3. Enhancing developer ergonomics (branch explorer, richer persistence tooling)

## License

MIT

## Project Name

The Python package is `miniverse`. The project folder is `varela` (codename inspired by Francisco Varela's work in cybernetics and autopoiesis).
