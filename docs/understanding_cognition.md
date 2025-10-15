# Understanding Cognition Modules

_A practical guide to Miniverse's agent cognition system_

---

## Quick Reference

| Module | What It Does | When It Runs | LLM Call? |
|--------|-------------|--------------|-----------|
| **Planner** | Creates multi-step plan | Once at start, when plan exhausted, or per cadence | Only if using `LLMPlanner` |
| **Executor** | Chooses action for this tick | Every single tick | Only if using `LLMExecutor` |
| **Reflection** | Synthesizes recent experiences | Periodically (every N ticks or importance threshold) | Only if using `LLMReflectionEngine` |
| **Scratchpad** | Stores plan state & working memory | N/A (just a dict) | Never |

---

## Two Types of Non-Determinism

Understanding the difference is crucial for designing simulations:

### 1. Stochastic Physics (Random but Controllable)

**What it is:** Random events in the simulation rules (not in agents)

**Examples:**
- Order arrivals follow Poisson distribution (λ=3.5)
- Equipment failures occur with probability 0.05 per tick
- Processing times vary uniformly between 1.0 and 2.0 units

**Defined in:** `SimulationRules.apply_tick()`

**Reproducibility:** Use `random.Random(seed)` for deterministic randomness

```python
class WarehouseRules(SimulationRules):
    def __init__(self, rng_seed=42):
        self.rng = random.Random(rng_seed)

    def apply_tick(self, state, tick):
        # Stochastic: random order arrivals
        new_orders = self.rng.poisson(3.5)
        state.backlog += new_orders

        # Stochastic: variable processing speed
        for agent in state.agents:
            if agent.activity == "fulfillment":
                rate = self.rng.uniform(1.0, 2.0)
                completed = int(rate * agent.attributes["skill"])
                state.backlog = max(0, state.backlog - completed)

        return state
```

**Key insight:** Agent decisions can still be deterministic even with stochastic physics!

### 2. LLM Decisions (Intelligent Adaptation)

**What it is:** Agents analyze perception and make context-aware decisions

**Examples:**
- Agent sees backlog=80 and decides to switch from inventory to fulfillment
- Supervisor analyzes trends and broadcasts strategy change to team
- Agent reflects on recent failures and adjusts risk tolerance

**Defined in:** Executor/Planner/Reflection modules

**Reproducibility:** Non-reproducible (LLM outputs vary)

```python
cognition = AgentCognition(
    planner=LLMPlanner(),       # LLM creates strategy
    executor=LLMExecutor(),      # LLM decides actions
    reflection=LLMReflectionEngine()  # LLM synthesizes insights
)
```

**Key insight:** Creates emergent, creative behavior that can't be scripted!

---

## Decision Matrix: What to Use When

| Simulation Goal | Physics | Agent Cognition | Why |
|----------------|---------|-----------------|-----|
| **Baseline measurement** | Deterministic | Deterministic | Reproducible, fast, no API costs |
| **Monte Carlo analysis** | Stochastic | Deterministic | Test variance in outcomes, compare strategies |
| **Strategy validation** | Stochastic | LLM | See if agents adapt to changing conditions |
| **Social simulation** | Deterministic or Stochastic | LLM | Focus on emergent social behavior |

---

## Cognition Module Deep Dive

### Planner (Strategy Creation)

**Purpose:** Generate multi-step plan for agent to follow

**When it runs:**
- At simulation start (tick 1)
- When current plan is exhausted (all steps completed)
- At configured intervals (e.g., every 8 ticks = "daily" planning)

**Input:**
- Agent profile (personality, goals, skills)
- World state (resources, environment)
- Recent memories (last 10 observations)
- Current scratchpad state

**Output:** Plan object with 3-10 steps

**Example LLM-generated plan:**
```json
{
  "steps": [
    {"description": "Check inventory levels", "metadata": {"priority": "high"}},
    {"description": "Process urgent orders", "metadata": {"target": "fulfillment"}},
    {"description": "Restock low inventory items", "metadata": {"target": "inventory"}},
    {"description": "Review team coordination", "metadata": {"target": "communication"}}
  ]
}
```

**Deterministic alternative:** Hardcoded steps based on agent role
```python
class RoleBasedPlanner:
    ROLE_PLANS = {
        "worker": ["check_backlog", "work", "work", "rest"],
        "supervisor": ["review_metrics", "send_updates", "coordinate"]
    }

    async def generate_plan(self, agent_id, scratchpad, *, world_context, context):
        role = context.agent_profile.role
        steps = [PlanStep(description=s) for s in self.ROLE_PLANS[role]]
        return Plan(steps=steps)
```

---

### Executor (Action Selection)

**Purpose:** Choose specific action for this tick based on current plan step

**When it runs:** **Every single tick** (most frequently called module)

**Input:**
- Current perception (what agent sees now)
- Current plan step (e.g., "Process urgent orders")
- Recent memories (context for decision)
- Scratchpad state

**Output:** AgentAction (action_type, target, parameters, reasoning, communication)

**Example LLM-generated action:**
```json
{
  "agent_id": "worker1",
  "tick": 42,
  "action_type": "fulfillment",
  "target": "warehouse",
  "parameters": {},
  "reasoning": "Backlog is at 75, need to prioritize fulfillment over inventory",
  "communication": {
    "to": "supervisor",
    "message": "Switching to fulfillment - backlog critical"
  }
}
```

**Deterministic alternative:** Rule-based logic
```python
class ThresholdExecutor:
    async def choose_action(self, agent_id, perception, scratchpad, *, plan, plan_step, context):
        backlog = perception.visible_resources.get("backlog", {}).get("value", 0)
        inventory = perception.visible_resources.get("inventory", {}).get("value", 100)

        # Simple threshold rules
        if backlog > 50:
            action_type = "fulfillment"
        elif inventory < 30:
            action_type = "inventory"
        else:
            action_type = "fulfillment"  # Default

        return AgentAction(
            agent_id=agent_id,
            tick=perception.tick,
            action_type=action_type,
            reasoning=f"Backlog={backlog}, Inventory={inventory}"
        )
```

---

### Reflection (Experience Synthesis)

**Purpose:** Periodically look back and extract insights from recent experiences

**When it runs:**
- Every N ticks (e.g., every 3 ticks)
- When importance accumulates above threshold (Stanford pattern: sum >= 150)
- At major events (end of shift, critical incidents)

**Input:**
- Recent memories (last 20-100 memories)
- Current plan state
- World state snapshot

**Output:** ReflectionResult objects stored as high-importance memories

**Example LLM-generated reflections:**
```json
{
  "reflections": [
    {
      "content": "I've been switching between inventory and fulfillment too frequently - need to batch tasks better",
      "importance": 8,
      "metadata": {"category": "efficiency"}
    },
    {
      "content": "Supervisor's recent messages indicate shift in priorities toward fulfillment speed",
      "importance": 7,
      "metadata": {"category": "coordination"}
    }
  ]
}
```

**Why reflections matter:**
- Stored as memories with high importance (6-10 vs 5 for actions)
- Surface in future memory retrieval more often
- Influence future planning and execution decisions
- Create agent "learning" and adaptation over time

**Deterministic alternative:** Custom heuristic reflection or None
```python
class HeuristicReflectionEngine:
    async def maybe_reflect(self, agent_id, scratchpad, recent_memories, *, trigger_context, context):
        if trigger_context.get("tick", 0) % 3 != 0:
            return []  # Only reflect every 3 ticks

        # Simple heuristic reflection
        latest = next(iter(recent_memories), None)
        if latest and "backlog" in latest.content.lower():
            content = "Team is struggling with backlog - should focus on fulfillment"
        else:
            content = "Operations proceeding normally"

        return [ReflectionResult(content=content, importance=6)]

# Or simply: reflection=None (skip reflection phase entirely)
```

---

### Scratchpad (Working Memory)

**Purpose:** Store plan state and temporary data between ticks

**When it runs:** Never (it's just a dictionary)

**What it stores:**
- Current plan object
- Plan index (which step we're on)
- Temporary flags (e.g., "in_crisis_mode": true)
- Custom per-agent data

**Example scratchpad state:**
```python
scratchpad.state = {
    "plan": Plan(steps=[...]),
    "plan_index": 2,  # Currently on step 2
    "last_plan_refresh_tick": 8,
    "execute_prompt_template": "warehouse_execute",  # Custom template
    "recent_transcript": ["message1", "message2"],  # For conversation agents
}
```

**Key insight:** Scratchpad is NOT smart - it's just memory. All logic happens in Planner/Executor/Reflection.

---

## Common Patterns

### Pattern: Deterministic Baseline + LLM Experiment

1. Build simulation with deterministic agents
2. Run 100 trials, measure outcomes
3. Replace one agent type with LLM cognition
4. Re-run 100 trials, compare performance

```python
# Baseline
cognition_baseline = AgentCognition(
    planner=FixedPlanPlanner(),
    executor=ThresholdExecutor(),
    reflection=NoOpReflection()
)

# Experiment
cognition_llm = AgentCognition(
    planner=LLMPlanner(),
    executor=LLMExecutor(),
    reflection=LLMReflectionEngine()
)

# Run both, compare metrics
```

### Pattern: Stochastic Physics + Adaptive Agents

1. Physics generates random events (order spikes, equipment failures)
2. LLM agents observe and adapt strategies
3. Test if agents handle variability better than fixed rules

```python
rules = WarehouseRules(rng_seed=42)  # Stochastic arrivals

cognition = AgentCognition(
    executor=LLMExecutor()  # Adapts to spikes
)
```

### Pattern: Team Coordination

1. One supervisor agent (LLM) analyzes metrics
2. Broadcasts strategy via communication
3. Worker agents (LLM or deterministic) respond to messages

```python
# Supervisor (with planning)
supervisor_cognition = AgentCognition(
    executor=LLMExecutor(template_name="supervisor_broadcast"),
    planner=LLMPlanner(),
    scratchpad=Scratchpad()  # Needed for plan state
)

# Workers (reactive, no planning)
worker_cognition = AgentCognition(
    executor=LLMExecutor(template_name="worker_respond")
    # No planner - workers just react to supervisor messages
)
```

---

## FAQ

**Q: Do I need LLM for realistic simulations?**

A: Depends on your goals:
- **For physics validation, parameter sweeps, reproducibility:** No, use deterministic
- **For emergent behavior, social dynamics, creativity:** Yes, use LLM

**Q: Can I mix LLM and deterministic in same simulation?**

A: Yes! Each agent gets its own cognition configuration. Common pattern: LLM supervisors + deterministic workers.

**Q: How many LLM calls per tick?**

A:
- Planner: 0 (if plan exists), 1 per agent (if plan needs refresh)
- Executor: 1 per agent per tick (if using LLMExecutor)
- Reflection: 0 (most ticks), 1 per agent (when triggered)

Example: 10 agents, 20 ticks, planning every 10 ticks:
- Planning: 10 agents × 2 refresh cycles = 20 calls
- Execution: 10 agents × 20 ticks = 200 calls
- Reflection: 10 agents × ~4 triggers = 40 calls
- **Total: ~260 LLM calls**

**Q: What's the difference between SimpleExecutor and LLMExecutor?**

A:
- `SimpleExecutor`: Calls LLM if configured, returns "rest" if not (dual behavior - confusing)
- `LLMExecutor`: Always calls LLM, raises error if not configured (clear, predictable)

**Recommendation:** Use `LLMExecutor` for pure LLM mode, custom deterministic executors for pure deterministic mode.

**Q: How do I add custom logic without LLM?**

A: Create custom executor:

```python
class MyCustomExecutor:
    async def choose_action(self, agent_id, perception, scratchpad, *, plan, plan_step, context):
        # Your logic here
        return AgentAction(...)

cognition = AgentCognition(
    executor=MyCustomExecutor()
)
```

See `examples/workshop/run.py` for complete deterministic example.

---

## Next Steps

- Read `docs/USAGE.md` for step-by-step simulation building
- See `docs/architecture/cognition.md` for technical implementation details
- Try `examples/workshop/run.py` for working code (both deterministic and LLM modes)
- Explore `miniverse/cognition/llm.py` to understand LLM module implementations
