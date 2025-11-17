# Workshop Examples - Progressive Learning Path

This directory contains a series of examples that progressively introduce Miniverse concepts, from simple deterministic agents to complex LLM-driven simulations.

> **Tip:** Prefer local inference? Set `LLM_PROVIDER=ollama`, `LLM_MODEL=<tag from \`ollama list\`>`, and optionally `OLLAMA_BASE_URL=http://127.0.0.1:11434` to run any `--llm` example without cloud API keys. Full instructions live in `docs/USAGE.md#local-llms-ollama`.

## Learning Progression

### 01_hello_world - The Basics
**Concepts:** Simulation loop, physics, executor, validation

**What you'll learn:**
- How the simulation loop works (physics → action → validation → update)
- Creating simple `SimulationRules` for world physics
- Writing a minimal executor with hardcoded logic
- The absolute minimum needed to run a simulation

**Run:**
```bash
uv run python -m examples.workshop.01_hello_world.run
```

**Key takeaway:** One agent, one action ("always work"), watch power decrease over 5 ticks. No LLM, no complexity - just the core loop.

---

### 02_deterministic - Threshold Logic
**Concepts:** Multiple agents, thresholds, agent attributes, resource tracking

**What you'll learn:**
- Creating agents with attributes (energy, stress)
- Threshold-based decision making (if energy < 30: rest, if backlog > 8: work)
- Balancing multiple factors in deterministic logic
- How agent actions affect world state (working reduces backlog)

**Run:**
```bash
uv run python -m examples.workshop.02_deterministic.run
```

**Key takeaway:** Two workers with deterministic if/then logic adapting to backlog pressure and personal energy. Still no LLM - pure Python logic.

---

### 03_llm_single - Reactive AI
**Concepts:** LLM executor, intelligent decisions, context-aware behavior

**What you'll learn:**
- Using `LLMExecutor` for AI-driven decisions
- How agents receive context (perception, attributes, resources)
- LLM making nuanced decisions beyond simple thresholds
- Minimal LLM configuration (just executor, no planning/reflection)

**Requires:** LLM API key (OpenAI, Anthropic, etc.) or a local Ollama model (`LLM_PROVIDER=ollama`).

**Run:**
```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4
export OPENAI_API_KEY=your_key
uv run python -m examples.workshop.03_llm_single.run
```

Tip: set `WORLD_UPDATE_MODE=deterministic` during development to skip the world engine LLM and keep runs fast. Preflight will print the selected mode at start.

**Key takeaway:** Single LLM agent making intelligent work/rest decisions based on goals (wellbeing + productivity). No hardcoded thresholds - agent reasons about trade-offs.

---

### 04_team_chat - Multi-Agent Coordination
**Concepts:** Communication, emergent coordination, agent relationships

**What you'll learn:**
- Multiple LLM agents coordinating via natural language
- Using the `communication` field in `AgentAction`
- Agents reading and responding to each other's messages
- Emergent team behavior from individual LLM decisions

**Requires:** LLM API key or local Ollama setup

**Run:**
```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4
export OPENAI_API_KEY=your_key
uv run python -m examples.workshop.04_team_chat.run
```

Tip: `WORLD_UPDATE_MODE=deterministic` avoids the world engine LLM; `WORLD_UPDATE_MODE=llm` forces it (fail-fast if misconfigured).

**Key takeaway:** Team leader + 2 workers coordinate via chat. Leader emerges as coordinator without hardcoded control - just through communication and individual decisions.

---

### 05_stochastic - Random Events + Adaptation
**Concepts:** Stochastic physics, LLM adaptation, randomness vs intelligence

**What you'll learn:**
- **Stochastic physics** = random world events (in `SimulationRules`)
- **LLM adaptation** = intelligent response to unpredictability
- Distinction between deterministic/stochastic physics and LLM decisions
- Using `random.Random` with seeds for reproducible stochastic simulations

**Requires:** LLM API key or local Ollama setup

**Run:**
```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4
export OPENAI_API_KEY=your_key
uv run python -m examples.workshop.05_stochastic.run
```

Tip: `WORLD_UPDATE_MODE=llm` showcases non-deterministic world updates; `auto` and `deterministic` remain available.

**Key takeaway:** Agent can't predict random task arrivals or equipment breakdowns, but adapts intelligently. Shows that "non-determinism" has two forms: random physics (world) and intelligent decisions (agents).

---

## Concept Map

| Example | Agents | Decision Logic | Physics | Communication | Complexity |
|---------|--------|---------------|---------|---------------|-----------|
| 01 | 1 | Hardcoded | Deterministic | No | ⭐ |
| 02 | 2 | Threshold (if/then) | Deterministic | No | ⭐⭐ |
| 03 | 1 | LLM (reactive) | Deterministic | No | ⭐⭐⭐ |
| 04 | 3 | LLM (reactive) | Deterministic | Yes | ⭐⭐⭐⭐ |
| 05 | 1 | LLM (adaptive) | Stochastic | No | ⭐⭐⭐⭐ |

## Key Design Patterns Shown

### AgentCognition Patterns

**Minimal (Example 01, 02):**
```python
AgentCognition(executor=MyExecutor())
```

**Simple LLM (Example 03, 04, 05):**
```python
AgentCognition(executor=LLMExecutor())
```

**Full Stanford (not shown, see main workshop example):**
```python
AgentCognition(
    executor=LLMExecutor(),
    planner=LLMPlanner(),
    reflection=LLMReflectionEngine(),
    scratchpad=Scratchpad()
)
```

### SimulationRules Patterns

**Deterministic Physics:**
```python
class MyRules(SimulationRules):
    def apply_tick(self, state, tick):
        # Fixed updates every tick
        power.value -= 5
        return state
```

**Stochastic Physics:**
```python
class MyRules(SimulationRules):
    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def apply_tick(self, state, tick):
        # Random events
        new_tasks = self.rng.randint(0, 5)
        return state
```

## What's Next?

After completing these examples, you're ready for:

1. **Main workshop example** (`examples/workshop/run.py`) - Full-featured simulation with deterministic baseline, LLM mode, and Monte Carlo batching
2. **Stanford replication** - Implement Valentine's Day party scenario with full planning + reflection
3. **Custom domains** - Apply patterns to your own simulation (supply chain, social networks, etc.)

## Tips for Learning

1. **Start with 01** - Even if you want LLM agents, understand the basic loop first
2. **Run each example** - Don't just read code, see the output
3. **Modify examples** - Change thresholds (02), prompts (03-05), add agents (04)
4. **Compare 02 vs 03** - Same scenario, different decision logic (deterministic vs LLM)
5. **Understand 05** - Key distinction between stochastic physics and LLM intelligence

## Common Questions

**Q: Do I need an LLM for realistic simulations?**

A: Depends on your goals:
- **Deterministic (01-02):** Fast, reproducible, good for parameter sweeps
- **LLM (03-05):** Emergent behavior, creativity, realistic social dynamics

**Q: What's the difference between stochastic physics and LLM non-determinism?**

A: **Stochastic physics** = random events in the world (task arrivals, breakdowns) defined in `SimulationRules`. **LLM non-determinism** = intelligent, context-aware decisions by agents. You can have either, both, or neither!

**Q: Can I mix deterministic and LLM agents in the same simulation?**

A: Yes! Each agent gets its own `AgentCognition`. Mix freely:
```python
cognition_map = {
    "manager": AgentCognition(executor=LLMExecutor()),  # LLM
    "worker1": AgentCognition(executor=FixedExecutor()),  # Deterministic
}
```

**Q: Which example is closest to the Stanford Generative Agents paper?**

A: None of these simple examples - they focus on fundamentals. For Stanford parity, you need planning + reflection + rich memory, shown in the main workshop example and upcoming Stanford scenario.
