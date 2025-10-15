# LLM Multi-Agent Simulations

## Overview

Recent research (2024-2025) demonstrates that LLM-powered agents can simulate believable human behavior, social dynamics, and complex interactions at scale.

## Key Research Papers

### AgentSociety (February 2025)

**Paper**: "AgentSociety: Large-Scale Simulation of LLM-Driven Generative Agents Advances Understanding of Human Behaviors and Society"

**Key Findings**:
- Large-scale social simulator integrating LLM-driven agents
- Simulated social lives for over **10,000 agents** with **5 million interactions**
- Successfully reproduced behaviors from 4 real-world social experiments:
  - Polarization dynamics
  - Inflammatory message spread
  - Universal basic income effects
  - Hurricane disaster impacts

**Source**: https://arxiv.org/abs/2502.08691

---

### Stanford Generative Agents (2023)

**Paper**: "Generative Agents: Interactive Simulacra of Human Behavior"

**Authors**: Stanford University & Google Research

**Architecture Components**:
1. **Memory Stream**: Complete record of agent experiences in natural language
2. **Reflection**: Synthesis of memories into higher-level patterns over time
3. **Dynamic Retrieval**: Context-aware memory access for planning behavior

**Demonstration**:
- Populated sandbox environment (inspired by The Sims) with **25 agents**
- Single user prompt: "One agent wants to throw a Valentine's Day party"
- Emergent behaviors:
  - Agents autonomously spread invitations over 2 days
  - Made new acquaintances
  - Asked each other on dates
  - Coordinated to show up together at correct time

**Extension (2024)**:
- Simulated **1,052 real individuals** by applying LLMs to qualitative interviews
- Replicated General Social Survey responses with **85% accuracy**
- Agents matched participants' own answer consistency over 2-week period

**Sources**:
- https://arxiv.org/abs/2304.03442
- https://arxiv.org/abs/2411.10109

---

### Multi-Agent LLM Systems as New Paradigm (2025)

**Paper**: "Beyond Static Responses: Multi-Agent LLM Systems as a New Paradigm for Social Science Research"

**Key Concepts**:
- LLM-based agentic systems incorporate:
  - **Memory**: Persistent experience tracking
  - **Goal-directed behavior**: Purpose-driven action selection
  - **Environmental interaction**: Sensing and responding to context
  - **Adaptive learning**: Behavior modification based on outcomes

**Applications**:
- Synthetic participants in experiments
- Individual differences modeling
- Group dynamics simulation
- Demographic conditioning effects

**Source**: https://arxiv.org/abs/2506.01839

---

## Technical Patterns

### Agent Architecture (Stanford Pattern)

```
Agent
├── Profile
│   ├── Identity (name, age, role)
│   ├── Personality traits
│   └── Goals and motivations
│
├── Memory Stream
│   ├── Observations (timestamped experiences)
│   ├── Conversations
│   └── Actions taken
│
├── Reflection Engine
│   ├── Pattern synthesis
│   ├── Higher-order abstractions
│   └── Self-assessment
│
└── Planning Module
    ├── Memory retrieval (context-aware)
    ├── Action generation
    └── Decision-making
```

### Behavioral Capabilities

**Demonstrated emergent behaviors**:
- Fairness and cooperation
- Social norm adherence
- Collaborative problem-solving
- Deception and cheating (in competitive contexts)
- Relationship formation
- Information spreading

### Memory Systems

**Critical for agent believability**:
- Long-term experience storage
- Context-dependent retrieval
- Abstraction and generalization
- Temporal awareness (recency, importance)

---

## Scaling Insights

| System | Agent Count | Interactions | Key Achievement |
|--------|-------------|--------------|-----------------|
| Stanford Generative Agents | 25 | Hundreds | Emergent social coordination |
| AgentSociety | 10,000+ | 5 million | Large-scale social experiments |
| Personality Simulations | 1,052 | Survey responses | 85% individual accuracy |

---

## Implications for Varela

1. **Architecture Validated**: Memory + Reflection + Planning is proven pattern
2. **Scale Feasible**: 10-50 agents for Mars colony well within demonstrated capabilities
3. **Emergent Behavior**: Complex coordination can arise from simple agent rules
4. **Social Dynamics**: LLMs naturally model human social behavior
5. **Memory Critical**: Persistent experience tracking essential for believability

---

---

## Stanford Implementation Deep Dive (Paper + Code)

### Cognitive Loop (Persona Class)
- `perceive` ➝ spatial memory queries the `Maze` tiles for nearby events filtered by attention bandwidth and recency thresholds.
- `retrieve` ➝ associative memory scores memories using recency, importance (“poignancy”), and semantic relevance via embedding lookups before surfacing context bundles to the LLM.
- `plan` ➝ two-tier planning: (1) daily schedule generation (wake-up hour, long-horizon goals) and (2) minute-level task decomposition; plans persist in scratch memory for execution.
- `execute` ➝ converts the chosen plan node into concrete movement and interaction directives (target tile, action narration) while coordinating with pathfinding (`path_finder.py`).
- `reflect` ➝ when the running sum of poignant memories crosses a threshold the agent triggers reflection prompts that distill new “thought” nodes back into memory.

### Memory Structures
- **Associative memory** stores event/thought/chat nodes with metadata (timestamps, keywords, embeddings, poignancy scores). Retrieval uses keyword maps and embedding similarity to rank relevance.
- **Spatial memory** maintains a world tree (`world ➝ sector ➝ arena ➝ object`) populated from `maze_meta_info.json`, giving agents semantic addresses for navigation.
- **Scratch space** acts as working memory: current time, active plan pointers, recent reflections.

### World Model (Maze)
- Tile-based map generated from Tiled exports (`collision`, `sector`, `arena`, `game_object`, `spawn` layers).
- Tiles cache semantic labels and event sets; reverse indices (`address_tiles`) accelerate pathfinding and event lookup.
- Movement/path planning uses A* variants constrained by collision layers and spawn locations.

### Prompt Patterns
- Prompts live under `persona/prompt_template/` and include few-shot exemplars for planning, reflection, conversation, and location rating.
- Reflection prompts produce diary-style summaries that append back to associative memory as higher-level “thought” entries.

---

## Reference Implementations Reviewed

### mkturkcan/generative-agents
- Simplified Python recreation of the paper’s loop (daily planning, hourly actions, memory rating, location selection) with OpenAI calls.
- Lacks spatial pathfinding or weighted memory retrieval; useful for understanding minimal viable loop and prompt templates.

### nmatter1/smallville
- Java service exposing REST/WebSocket APIs for running generative-agent towns in real time.
- Emphasizes client/server separation, live dashboards, and plug-in clients (Java/JS) that subscribe to simulation updates.
- Demonstrates how to externalize the world state so other applications can render or intervene.

### joonspk-research/generative_agents
- Open-sourced research snapshot of Stanford’s Reverie engine, including the full persona cognitive stack, maze world, and memory persistence.
- Provides concrete implementations of the paper’s memory scoring, reflection triggers, and plan execution pipeline.
- Contains utilities for bootstrap personas and environment authoring (Tiled assets, CSV block definitions).

---

## Integration Opportunities for Miniverse

1. **Memory Architecture Upgrade**
   - Implement an associative memory structure with importance + recency + semantic relevance scoring, using embeddings (e.g., via sentence transformers) and poignancy metadata.
   - Introduce reflection triggers that convert accumulated experiences into higher-level thoughts and inject them back into memory for future retrieval.

2. **Planning & Scratch Layer**
   - Add a scratch workspace to the orchestrator so agents can hold short-term plans, diary entries, and active goals across ticks.
   - Extend agent prompts to request daily/hourly plans and store them as structured tasks executed over multiple ticks rather than one-off actions.

3. **Environment Semantics**
   - Move beyond numeric resource stats by defining semantic locations (`world/sector/arena/object`) and game objects similar to the `Maze` model; scenarios should describe walkable tiles or logical rooms even for abstract org simulations.
   - Add movement/pathfinding helpers so deterministic physics manages spatial consistency while LLMs focus on high-level intent.

4. **Perception Enhancements**
   - Expand `build_agent_perception` to surface nearby events, plan commitments, and reflective insights rather than just metrics and alerts.
   - Feed relevant memories retrieved via combined scoring into the LLM prompt sections (context window framing similar to Reverie’s retrieve module).

5. **Simulation Harness**
   - Consider separating the orchestrator loop from rendering/clients (Smallville pattern) so organizational dashboards or mission control UIs can subscribe to simulation progress.
   - Introduce configurable attention bandwidth, retention windows, and plan refresh cadences to mimic the paper’s tunable cognition parameters.

6. **Diary & Reporting**
   - Log reflective summaries and plan updates as first-class artifacts (diary entries) that users can inspect, mirroring the paper’s qualitative outputs.

These upgrades would shift Miniverse from KPI-focused ticks toward the richer cognitive loop validated by Stanford’s work while preserving our deterministic rule hooks for domain-specific physics.

## References

- AgentSociety: https://arxiv.org/abs/2502.08691
- Stanford Generative Agents: https://arxiv.org/abs/2304.03442
- Personality Simulations: https://arxiv.org/abs/2411.10109
- Multi-Agent Paradigm: https://arxiv.org/abs/2506.01839
- Nature Survey: https://www.nature.com/articles/s41599-024-03611-3
