# Building Simulations with Miniverse

This guide walks through the workflow for authoring a new simulation using the Miniverse library. The goal is to make it easy for developers (or LLM agents) to bootstrap a scenario from scratch.

---

## 1. Define the Scenario

Create a JSON file describing the initial world:

- **Agents** – profiles (identity, skills, goals, relationships) plus status blocks (location, attributes, tags).
- **Resources** – shared metrics such as power, backlog, or budget (via `ResourceState`).
- **Environment** – optional metrics and tier descriptions:
  - Tier 0: metrics only.
  - Tier 1: `environment_graph` (nodes + adjacency) for logical locations.
  - Tier 2: `environment_grid` (width/height + tiles) for spatial worlds.
- **Events** – optional initial events.

Use `ScenarioLoader` to parse the file into `WorldState` and `AgentProfile` objects (`miniverse/scenario.py`). The workshop example (`examples/workshop/scenario.json`) shows a Tier‑1 setup.

```python
loader = ScenarioLoader(scenarios_dir=Path("examples/workshop"))
world_state, profiles = loader.load("scenario")
```

---

## 2. Provide Deterministic Rules

Subclass `SimulationRules` to encode domain physics:

- `apply_tick` – consume resources, adjust agent attributes, generate events.
- `validate_action` – reject impossible actions (room capacity, prerequisites).
- Optional hooks: `on_simulation_start`, `on_simulation_end`, `format_resource_summary`.

These rules run before cognition each tick and ensure predictable system dynamics.
Provide optional stochasticity by passing a `random.Random` instance (or omit it
for strict determinism):

```python
class WorkshopRules(SimulationRules):
    def __init__(
        self,
        occupancy: GraphOccupancy | None = None,
        *,
        rng: random.Random | None = None,
        task_arrival_chance: float = 0.35,
        max_new_tasks: int = 2,
    ) -> None:
        self.occupancy = occupancy
        self.rng = rng
        self.task_arrival_chance = task_arrival_chance
        self.max_new_tasks = max_new_tasks

    def apply_tick(self, state: WorldState, tick: int) -> WorldState:
        ...  # adjust resources, optionally sample new work from rng

    def validate_action(self, action: AgentAction, state: WorldState) -> bool:
        ...  # reject moves into rooms that exceed capacity
```

For environments, use the helper module (`miniverse/environment/helpers.py`) to manage capacities (`GraphOccupancy`) or compute paths (`shortest_path`, `grid_shortest_path`).

### Ticks, Time, and Planning Cadence

- Each iteration of the orchestrator increments `WorldState.tick`. Scenarios decide how that maps to real time (minute, hour, daily loop, etc.)—the library does not assume a day-length tick.
- Planners currently execute once per tick. Use scratchpad flags or custom logic inside your planner to refresh less frequently (e.g., only generate a new plan every 6 ticks) until native scheduling helpers land.
- Reflection engines can trigger on whatever cadence you need (the workshop example reflects every third tick). Future cognition work will expose higher-level scheduling utilities, but today cadence is scenario-defined.

---

## 3. Assemble Cognition Modules

Each agent has an `AgentCognition` bundle containing:

- `Planner` (async) – outputs a `Plan` given `PromptContext`.
- `Executor` – turns plan steps + perception into `AgentAction`.
- `ReflectionEngine` (async) – generates `ReflectionResult` entries for memory.
- `Scratchpad` – shared working memory (plan state, open tasks, custom flags).
- `PromptLibrary` (optional) – named templates for plan/execute/reflect stages.

Miniverse ships two sets of implementations:

1. **Deterministic (example)** – see `examples/workshop/run.py` (`DeterministicPlanner`, `DeterministicExecutor`, `DeterministicReflection`).
2. **LLM-backed** – `LLMPlanner` and `LLMReflectionEngine` in `miniverse/cognition/llm.py`. They render templates, call the LLM via `call_llm_with_retries`, and parse structured JSON responses.

## Choosing Cognition Modules: Deterministic vs LLM

Miniverse supports three patterns for agent cognition. Choose based on your simulation goals:

### Pattern 1: Pure Deterministic (No LLM)

**Use case:** Testing, CI/CD, baseline comparison, Monte Carlo sweeps

**Characteristics:**
- Fast (no network latency)
- Reproducible (same seed → same behavior)
- Zero API costs
- Predictable actions based on hardcoded rules

**Example:**
```python
from miniverse.cognition import AgentCognition, Scratchpad

# Define custom deterministic executor with hardcoded logic
class FixedStrategyExecutor:
    async def choose_action(self, agent_id, perception, scratchpad, *, plan, plan_step, context):
        # Simple rule: if backlog > 50, do fulfillment; else inventory
        backlog = perception.visible_resources.get("backlog", {}).get("value", 0)
        action_type = "fulfillment" if backlog > 50 else "inventory"
        return AgentAction(
            agent_id=agent_id,
            tick=perception.tick,
            action_type=action_type,
            reasoning=f"Backlog at {backlog}"
        )

cognition_map = {
    "worker1": AgentCognition(
        planner=SimplePlanner(),  # Returns empty plan
        executor=FixedStrategyExecutor(),  # Hardcoded rules
        reflection=SimpleReflectionEngine(),  # No-op
        scratchpad=Scratchpad()
    )
}
```

### Pattern 2: Pure LLM (AI-Driven)

**Use case:** Emergent behavior, adaptive strategies, realistic social simulation

**Characteristics:**
- Intelligent decisions based on full context
- Agents adapt to changing conditions
- Creative, unpredictable responses
- Requires API keys and network access

**Example:**
```python
from miniverse.cognition import (
    AgentCognition,
    LLMPlanner,
    LLMExecutor,  # NEW - pure LLM executor
    LLMReflectionEngine,
    Scratchpad
)

cognition_map = {
    "supervisor": AgentCognition(
        planner=LLMPlanner(template_name="warehouse_plan"),
        executor=LLMExecutor(template_name="warehouse_execute"),  # LLM decides every action
        reflection=LLMReflectionEngine(template_name="warehouse_reflect"),
        scratchpad=Scratchpad()
    )
}
```

**Key difference:** `LLMExecutor` REQUIRES LLM configuration (raises error if missing). Use when you want guaranteed AI-driven behavior.

<!--
### Pattern 3: Hybrid (LLM Planning + Deterministic Execution)

**Note:** This pattern uses `SimpleExecutor` which has confusing dual behavior (calls LLM if configured, falls back to "rest" if not). We're deprecating this pattern in favor of clear Pure Deterministic or Pure LLM approaches. For now, use one of the two patterns above.
-->

`PromptLibrary` and `render_prompt` handle the template substitution using data from `PromptContext` (profile, perception, plan, memories, scratchpad state). Default templates live in `miniverse/cognition/prompts.py` and already include JSON examples.

### Controlling Planner/Reflection Cadence

Use `CognitionCadence` to throttle how often planners and reflection engines execute. The orchestrator stores the last run tick in each agent's scratchpad, so you don't have to manage bookkeeping yourself:

```python
from miniverse.cognition import (
    AgentCognition,
    CognitionCadence,
    PlannerCadence,
    ReflectionCadence,
    TickInterval,
)

cadence = CognitionCadence(
    planner=PlannerCadence(interval=TickInterval(every=2, offset=1)),
    reflection=ReflectionCadence(
        interval=TickInterval(every=3, offset=1),
        require_new_memories=True,
    ),
)

cognition = AgentCognition(
    planner=my_planner,
    executor=my_executor,
    reflection=my_reflection,
    scratchpad=Scratchpad(),
    cadence=cadence,
)
```

Need to translate ticks into higher-level units (day/shift/sprint)? Call `tick_to_time_block(tick=tick, ticks_per_block=8, block_label="shift")` inside your prompts or analytics helpers to expose the block index and per-block offset.

---

## 4. Configure the Orchestrator

Instantiate `Orchestrator` with the initial state, agents, rules, and cognition map:

```python
import random

from miniverse.orchestrator import Orchestrator

orchestrator = Orchestrator(
    world_state=world_state,
    agents=profiles_map,
    world_prompt="You oversee operations.",
    agent_prompts={aid: prompt for aid, prompt in base_prompts.items()},
    llm_provider=Config.LLM_PROVIDER,  # or None for deterministic world updates
    llm_model=Config.LLM_MODEL,
    simulation_rules=WorkshopRules(
        occupancy,
        rng=random.Random(42),        # omit rng for a purely deterministic run
        task_arrival_chance=0.4,
        max_new_tasks=2,
    ),
    agent_cognition=cognition_map,
)
```

### Deterministic vs LLM World Updates

- If `llm_provider`/`llm_model` are provided, the orchestrator will call the world-engine LLM via `process_world_update` after each tick.
- If either is missing, the orchestrator logs a warning and applies deterministic updates only (`_apply_deterministic_world_update`)—cloning the state, stamping new `tick`, and updating agent activities/locations.

Deterministic rules always run before cognition; world updates simply decide how the shared state changes afterwards.

---

## 5. Run the Simulation

Use `orchestrator.run(num_ticks=N)` inside `asyncio.run(...)` or your own loop. The method handles persistence strategy initialization, tick iteration, persistence/memory updates, and optional reflections.

```python
result = await orchestrator.run(num_ticks=8)
final_state = result["final_state"]
```

The workshop example shows CI-friendly usage (deterministic by default) and an LLM mode toggled via CLI (`--llm`). Pass `--debug` to log planner/executor/reflection payloads (including provider/model info) and `--analysis` to emit per-tick summaries (backlog deltas, average energy/stress).

Structured schema errors are also surfaced automatically. If an LLM response fails validation, the retry loop prints each offending field (with the received value preview) and appends the same checklist to the next prompt so the model corrects itself without guesswork.

If you need additional diagnostics, pass `tick_listeners` to the orchestrator. Each listener receives `(tick, previous_state, new_state, actions)`—the workshop example wires `TickAnalyzer` (see `examples/workshop/run.py`) to print per-tick backlog deltas and aggregate stats.

For a dialogue-centric walkthrough, see `examples/standup/run.py`. That scenario emits structured `communication` payloads, prints per-tick transcripts, and uses the same `--llm` / `--debug` / `--analysis` switches to compare deterministic and LLM-driven stand-ups.
Key tuning points are documented inline (`StandupPlanner.ROLE_STEPS`, `StandupExecutor.ROLE_MESSAGES`, `ConversationPostTick` heuristics). Debug mode will dump the generated plan/execute payloads so you can verify the prompts are consuming conversation history stored in each agent’s scratchpad.

---

## 6. Expand or Customize

- **Memory retrieval:** `SimpleMemoryStream` offers recency with lightweight keyword boosting; `ImportanceWeightedMemory` blends recency and importance so critical events stay near the top. Both store `tags`/`metadata`, which you can populate when calling `add_memory` (the orchestrator now records action/communication/event tags by default).
- **Branching timelines:** reserved fields (`branch_id`) let you experiment with Loom-style branching later.
- **Testing:** mock `LLMPlanner`/`LLMReflectionEngine` for unit tests (see `tests/test_cognition_flow.py`).
- **Docs to consult:**
  - `docs/architecture/cognition.md` – deep dive into the cognition runtime.
  - `docs/architecture/environment.md` – environment tiers and helpers.
  - `docs/examples/PLAN.md` – upcoming scenarios (Stanford replication, Mars habitat, etc.).

With this workflow you can craft simulations ranging from KPI-only organizational planning (Tier 0) to spatial sandboxes with LLM-driven agents (Tier 2). Start with the workshop example and adapt the templates or rules to your own domain.
