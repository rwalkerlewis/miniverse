# Research Implementation Patterns for Agent Simulations

## Overview

This document summarizes how leading multi-agent simulation research projects have implemented time management, agent scheduling, and coordination.

---

## Stanford Generative Agents (2023)

**Paper**: "Generative Agents: Interactive Simulacra of Human Behavior"
**Scale**: 25 agents in Smallville sandbox
**GitHub**: https://github.com/joonspk-research/generative_agents
**LLM**: gpt3.5-turbo (ChatGPT)
**Paper Location**: `/Users/ken/Desktop/lab/varela/docs/research/papers/generative-agents-stanford.pdf`

Full implementation details extracted in: `/Users/ken/Desktop/lab/varela/docs/research/game-engines/stanford-implementation-details.md`

### Time System

**Time Step Granularity**:
- 1 game step = **10 seconds** of simulated time
- 1 second real time = 1 minute game time (60x compression)

**Example**:
- `run 100` = simulate 100 steps = 1000 seconds = ~16.7 minutes of sim time

### Simulation Architecture

**Two-Server System** (Section 5):
1. **Environment Server** (Django)
   - Manages world state
   - Handles agent positions on 2D map
   - Serves visual interface (browser-based)
   - Uses Phaser web game development framework

2. **Agent Simulation Server**
   - Runs agent reasoning
   - Processes LLM calls for each agent
   - Manages agent memory and planning
   - Agent state stored as JSON

**Environment Representation**:
- Tree data structure (areas and objects)
- Edge = containment relationship
- Converted to natural language for LLM
- Example: "stove" child of "kitchen" → "there is a stove in the kitchen"

**Agent Environment Trees**:
- Each agent builds individual tree (subgraph of full environment)
- Initialized with: living quarters, workplace, common locations
- Updated as agent navigates
- Can become out-of-date when agent leaves area

### Game Loop Implementation

**Each Time Step** (Section 5 - Server Loop):
```
1. Parse JSON for changes from generative agents

2. Move agents to new positions

3. Update sandbox object states

4. Send observations to each agent's memory
   - Agents/objects within visual range
   - Grounded in natural language

5. Agents process observations
   - Store in memory stream
   - Retrieve relevant memories
   - Generate reflection (if needed)
   - Plan actions
   - Output natural language statement
     (e.g., "Isabella Rodriguez is writing in her journal")

6. Agent outputs update JSON

7. Loop to next time step
```

**Action Grounding** (Section 5.1):
- Recursively traverse environment tree to find appropriate location
- Step 1: Find appropriate area (from root of agent's tree)
- Step 2: Recursively find subarea until reaching leaf node
- Step 3: Animate movement using traditional game path algorithms

**Control Flow**:
- Manual step control via CLI
- Enter `run <n>` to simulate n steps
- Can pause/resume/replay simulations
- Agents visible moving on map in real-time

### Agent Scheduling

**Synchronous Steps**:
- All agents process each time step together
- No async/concurrent agent execution
- Sequential processing within each step

**Planning Granularity**:
- High-level plans decomposed into **hour-long chunks**
- Further decomposed into **5-15 minute chunks**
- Plans updated dynamically at each time step
- Agents can react mid-plan to environmental changes

**Planning Decomposition** (Section 4.3):

**Plan Structure**:
- Location
- Starting time
- Duration
- Natural language description

**Top-Down Recursive Decomposition**:

1. **Daily outline** (5-8 chunks)
   - Prompt with agent summary + previous day summary
   - Example output: "1) wake up and complete the morning routine at 8:00 am, 2) go to Oak Hill College to take classes starting 10:00 am, [...] 7) finish school assignments and go to bed by 11:00 pm"

2. **Hour-long chunks**
   - Example: "1:00 pm to 5:00 pm work on composition" → "1:00 pm: start by brainstorming some ideas [...] 4:00 pm: take a quick break and recharge"

3. **5-15 minute chunks**
   - Example: "4:00 pm: grab a light snack, 4:05 pm: take a short walk [...] 4:50 pm: clean up workspace"

**Reacting and Updating Plans** (Section 4.3.1):
- At each time step, agents perceive world and store observations
- LLM decides if reaction needed based on observation
- If reaction needed: regenerate plan from that point forward
- If interaction indicated: generate dialogue

### Agent Perception

**Observation Mechanism**:
- Sandbox server sends observations to agents
- Limited to **visual range** (spatial locality)
- Natural language descriptions of surroundings
- Agents only know what they can "see"

### Memory Architecture

**Three Components**:
1. **Memory Stream**: Complete record of experiences
2. **Reflection**: Synthesizes memories into higher-level insights
3. **Retrieval**: Dynamically queries relevant memories for planning

**Memory Object Structure**:
- Natural language description
- Creation timestamp
- Most recent access timestamp

**Retrieval Scoring** (Section 4.1 of paper):
```python
score = α_recency * recency + α_importance * importance + α_relevance * relevance
# All α values = 1 in Stanford implementation
# Normalized to [0,1] using min-max scaling
```

**1. Recency**:
- Exponential decay function: `decay^hours_since_last_access`
- Decay factor: 0.995
- Higher score for recently accessed memories

**2. Importance**:
- LLM rates memories 1-10 scale at creation time
- Prompt: "On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following piece of memory."
- Example: "cleaning up the room" = 2, "asking your crush out on a date" = 8

**3. Relevance**:
- LLM embedding vectors of memory descriptions
- Cosine similarity between memory embedding and query embedding
- Conditioned on query context

**Reflection Generation** (Section 4.2):
- **Trigger**: When sum of importance scores for latest events exceeds threshold
- **Threshold**: 150 in Stanford implementation
- **Frequency**: ~2-3 times per day in practice

**Reflection Process**:
1. Query 100 most recent records
2. Generate 3 salient high-level questions
3. Retrieve relevant memories for each question
4. Extract 5 high-level insights with citations
5. Store insights with pointers to cited memory objects

**Memory Updates**:
- Every observation stored in memory stream
- Reflections generated when importance threshold exceeded
- Retrieved context-dependently for decisions

### Coordination

**Emergent Coordination**:
- No explicit coordination mechanism
- Agents coordinate through:
  - Observations of each other's actions
  - Natural language interactions
  - Shared environment state

**Example** (Valentine's Day party from Section 6):
- Isabella planned party at Hobbs Cafe
- Invited 12 agents over 2 days
- Agents autonomously:
  - Spread invitations through conversations
  - Make new acquaintances
  - Ask each other on dates
  - 5 agents showed up
  - 3 cited schedule conflicts
  - 4 expressed interest but didn't attend

**Evaluation Results** (Section 6):
- **Agent Count**: 25 agents in Smallville
- **Duration**: 2 full game days
- **Information Diffusion**: Sam's mayoral candidacy spread from 1 agent (4%) to 8 agents (32%)
- **Relationship Formation**: Network density increased from 0.167 to 0.74
- **Hallucination Rate**: 1.3% (6 out of 453 responses)

**Common Failure Modes** (Section 7.2):
1. **Location Choice Degradation**: As agents learn more locations, choices become less typical
2. **Physical Norm Misunderstanding**: Multiple agents entering "one-person bathroom", entering closed stores
3. **Overly Cooperative Behavior**: Agents rarely say "no", dialogue overly formal (likely from instruction tuning)

---

## AgentSociety (2025)

**Paper**: "AgentSociety: Large-Scale Simulation of LLM-Driven Generative Agents"
**Scale**: 10,000+ agents, 5 million interactions
**GitHub**: https://github.com/tsinghua-fib-lab/AgentSociety
**Paper Location**: `/Users/ken/Desktop/lab/varela/docs/research/papers/agent-society.pdf` (45.4MB - too large to extract via tools)

### Architecture Layers

**Four-Layer System**:

1. **Model Layer**
   - Agent configuration
   - Task definitions
   - Centralized control

2. **Agent Layer**
   - Multi-head workflows
   - Various action types
   - Decision-making logic

3. **Environment Layer**
   - Urban simulation environment
   - Agent-environment interaction
   - Spatial coordination

4. **LLM Layer**
   - LLM integration services
   - Model configuration
   - Request management

### Scalability Strategy

**Asynchronous Architecture**:
- Agents don't all run in lockstep
- Asynchronous simulation model
- Ray distributed computing framework

**Ray Framework Benefits**:
- Managed Ray actors across machines
- Horizontal scaling of compute resources
- Distributed agent processing
- Agent groups share services

**Concurrency Model**:
- Multiple agents work concurrently
- Different experiments use distinct Ray clusters
- Shared services prevent interference

### Time Management

**Discrete Time Steps**:
- System steps through discrete "time steps"
- Specific granularity not specified in abstract
- Likely variable based on simulation needs

### Agent Coordination

**Coordination Mechanisms**:
- Online communication (chat/messaging)
- Offline meetings (scheduled based on online chat)
- Example: Agents discover shared interests → schedule detailed discussion

**Agent Groups**:
- Clients connecting to shared services
- Environment simulator coordination
- Multiple agents work concurrently within groups

---

## Key Patterns from Research

### 1. Fixed Time Steps (Universal)

**All implementations use discrete time steps**:
- Stanford: 10 seconds per step
- AgentSociety: Discrete steps (granularity varies)
- Traditional ABM: Fixed time quanta

**Why Fixed Steps**:
- Deterministic replay
- Easier debugging
- Clear causality
- Manageable state snapshots

### 2. Synchronous vs. Asynchronous

**Stanford (Small Scale)**: Synchronous
- All agents process same time step together
- Sequential within step, parallel across agents
- Simple coordination

**AgentSociety (Large Scale)**: Asynchronous
- Agents run concurrently
- Distributed processing
- More complex but scales better

**Implication**:
- Small sims (< 100 agents): Synchronous works
- Large sims (1000+ agents): Asynchronous necessary

### 3. Perception is Spatial/Local

**Consistent Pattern**:
- Agents perceive within limited range
- No omniscient knowledge
- Forces communication for coordination
- More realistic behavior

### 4. Natural Language Throughout

**Universal Approach**:
- Actions described in natural language
- Observations in natural language
- Plans in natural language
- Grounded to environment via translation layer

### 5. Memory is Critical

**Stanford's Three Components**:
- Stream (raw experiences)
- Reflection (synthesized insights)
- Retrieval (context-aware access)

**This enables**:
- Long-term consistency
- Learning from experience
- Contextual decision-making

### 6. Emergent Coordination

**Not Hardcoded**:
- No explicit coordination protocols
- Emerges from:
  - Observations
  - Communication
  - Shared goals
  - Social dynamics

---

## Recommendations for Varela

Based on research patterns:

### Time System

**Use Fixed Time Steps**:
```python
TIME_STEP = 10  # seconds of sim time per step
# or
TIME_STEP = 60  # 1 minute per step for slower pace
```

**Compression Ratio**:
```python
# Stanford: 1 real second = 1 sim minute (60x)
# For Varela: 1 real second = 1 sim minute (60x)
# Or faster: 1 real second = 10 sim minutes (600x) for data gen
```

### Agent Scheduling

**For Small Scale (3-10 agents)**:
```
Synchronous execution:
├── Step N begins
├── All agents perceive simultaneously
├── All agents decide actions (parallel LLM calls)
├── World engine processes all actions
├── Step N+1 begins
```

**For Large Scale (100+ agents)**:
```
Asynchronous execution:
├── Agents run on distributed workers
├── Actions submitted to queue
├── World engine processes batches
├── State broadcast to agents
```

### Perception Model

**Spatial Locality** (like Stanford):
```python
def get_agent_perception(agent, world_state):
    return {
        "nearby_agents": get_within_range(agent.location, radius=50),
        "visible_resources": get_at_location(agent.location),
        "system_alerts": get_public_alerts(),
        "messages": get_messages_to(agent.id)
    }
```

### Coordination

**Emergent** (like research):
- No hardcoded coordination
- Agents observe each other
- Agents communicate explicitly
- World engine validates/resolves conflicts

### Memory Architecture

**Implement Stanford's Pattern**:
```python
class AgentMemory:
    stream: List[Observation]  # Everything the agent experiences
    reflections: List[Insight]  # Periodic high-level synthesis

    def retrieve(self, query, k=5):
        # Return k most relevant memories for current context
        pass
```

---

## Implementation Template for Varela

### Basic Loop (Stanford Style)

```python
class VarelaSimulation:
    TIME_STEP_SECONDS = 10  # Each step = 10 sim seconds

    async def run(self, num_steps: int):
        for step in range(num_steps):
            print(f"Step {step} (sim time: {step * self.TIME_STEP_SECONDS}s)")

            # 1. Get current world state
            world_state = await self.world.get_state()

            # 2. All agents perceive (parallel)
            perceptions = [
                self.get_perception(agent, world_state)
                for agent in self.agents
            ]

            # 3. All agents decide actions (parallel LLM calls)
            action_tasks = [
                agent.decide(perception)
                for agent, perception in zip(self.agents, perceptions)
            ]
            actions = await asyncio.gather(*action_tasks)

            # 4. World engine processes actions
            new_state = await self.world_engine.process(
                world_state,
                actions,
                time_delta=self.TIME_STEP_SECONDS
            )

            # 5. Update world state
            await self.world.set_state(new_state)

            # 6. Update agent memories
            for agent, action in zip(self.agents, actions):
                agent.memory.add(step, action, new_state)

            # 7. Save to database
            await self.db.save_state(step, new_state)
```

### With Claude Code Integration

```python
# Each agent is Claude Code process with tools

# tool: read_world_state
async def read_world_state():
    """Called by agent to get current state"""
    return await db.get_latest_state()

# tool: submit_action
async def submit_action(action_type: str, **kwargs):
    """Called by agent to submit action for this step"""
    await action_queue.add({
        "agent": AGENT_ID,
        "step": CURRENT_STEP,
        "action": action_type,
        **kwargs
    })

# Agent system prompt
"""
You are {agent_name}.

The simulation runs in 10-second time steps.
Each step:
1. Call read_world_state() to see current situation
2. Decide what to do based on your role and goals
3. Call submit_action() with your decision
4. Your action will be processed with all other agents

You can only act once per step.
Choose wisely based on priorities.
"""
```

---

## Key Takeaway

**Research consensus**:
1. ✅ Fixed time steps (10-60 seconds)
2. ✅ Synchronous within step, parallel LLM calls
3. ✅ Spatial/local perception
4. ✅ Natural language actions and observations
5. ✅ Memory with stream + reflection + retrieval
6. ✅ Emergent coordination (no hardcoded protocols)

**For Varela v0.1**:
- Follow Stanford pattern exactly
- 10-second time steps
- Synchronous step execution
- Parallel agent LLM calls
- Manual step control (run N steps at a time)

**For Varela v0.2+**:
- Consider async for scale (AgentSociety pattern)
- Add distributed processing if needed
- Keep core patterns the same

---

## References

- Stanford Generative Agents: https://arxiv.org/abs/2304.03442
- Stanford GitHub: https://github.com/joonspk-research/generative_agents
- AgentSociety Paper: https://arxiv.org/abs/2502.08691
- AgentSociety GitHub: https://github.com/tsinghua-fib-lab/AgentSociety
