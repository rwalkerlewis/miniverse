# Stanford Generative Agents - Technical Implementation Details

**Source**: Full paper from `/docs/research/papers/generative-agents-stanford.pdf`

---

## Time System (Confirmed from Paper)

### Time Step Granularity

**From Section 5 (Sandbox Environment Implementation)**:
- **1 game step = 10 seconds of simulated time**
- Manual control via CLI: `run <step-count>` command
- Can replay simulations by navigating to specific starting time-step

**Action Loop**:
> "At each time step, the agents output a natural language statement describing their current action, such as 'Isabella Rodriguez is writing in her journal'"

---

## Architecture Components

### Memory Stream (Section 4.1)

**Definition**:
> "A long-term memory module that records, in natural language, a comprehensive list of the agent's experiences"

**Memory Object Structure**:
- Natural language description
- Creation timestamp
- Most recent access timestamp

**Observations**: The most basic element
- Events directly perceived by agent
- Agent's own behaviors
- Behaviors of other agents
- Object state changes

**Example observations**:
1. Isabella Rodriguez is setting out the pastries
2. Maria Lopez is studying for a Chemistry test while drinking coffee
3. Isabella Rodriguez and Maria Lopez are conversing about planning a Valentine's day party
4. The refrigerator is empty

### Retrieval Function (Section 4.1)

**Three-Component Scoring**:

**1. Recency**:
- Exponential decay function over game hours since last retrieval
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

**Final Score**:
```
score = α_recency * recency + α_importance * importance + α_relevance * relevance
```
- All α values set to 1 in their implementation
- Normalize to [0,1] using min-max scaling
- Top-ranked memories that fit in context window are included

### Reflection (Section 4.2)

**When Generated**:
- Periodically when sum of importance scores for latest events exceeds threshold
- Threshold: 150 in their implementation
- In practice: ~2-3 times per day

**Process**:

**Step 1**: Generate questions from recent memories
- Query 100 most recent records
- Prompt: "Given only the information above, what are 3 most salient high-level questions we can answer about the subjects in the statements?"
- Example questions: "What topic is Klaus Mueller passionate about?" "What is the relationship between Klaus Mueller and Maria Lopez?"

**Step 2**: Retrieve relevant memories for each question
- Use questions as retrieval queries
- Gather memories (including other reflections)

**Step 3**: Extract insights
- Prompt: "What 5 high-level insights can you infer from the above statements? (example format: insight (because of 1, 5, 3))"
- Example output: "Klaus Mueller is dedicated to his research on gentrification (because of 1, 2, 8, 15)"
- Parse and store with pointers to cited memory objects

**Reflection Trees**:
- Leaf nodes: Base observations
- Non-leaf nodes: Reflections on observations/other reflections
- Higher nodes = more abstract thoughts

### Planning (Section 4.3)

**Plan Structure**:
- Location
- Starting time
- Duration
- Natural language description

**Example**:
> "for 180 minutes from 9am, February 12th, 2023, at Oak Hill College Dorm: Klaus Mueller's room: desk, read and take notes for research paper"

**Plan Decomposition (Top-Down Recursive)**:

**Step 1**: Create daily outline (5-8 chunks)
- Prompt with agent summary + previous day summary
- Output: "1) wake up and complete the morning routine at 8:00 am, 2) go to Oak Hill College to take classes starting 10:00 am, [...] 5) work on his new music composition from 1:00 pm to 5:00 pm, 6) have dinner at 5:30 pm, 7) finish school assignments and go to bed by 11:00 pm"

**Step 2**: Decompose into hour-long chunks
- Example: "1:00 pm to 5:00 pm work on composition" → "1:00 pm: start by brainstorming some ideas [...] 4:00 pm: take a quick break and recharge"

**Step 3**: Decompose into 5-15 minute chunks
- Example: "4:00 pm: grab a light snack, 4:05 pm: take a short walk [...] 4:50 pm: clean up workspace"

**Granularity**: Adjustable to match desired detail level

### Reacting and Updating Plans (Section 4.3.1)

**Action Loop**:
> "At each time step, they perceive the world around them and those perceived observations are stored in their memory stream"

**Reaction Decision Prompt**:
```
[Agent's Summary Description]
It is February 13, 2023, 4:56 pm.
John Lin's status: John is back home early from work.
Observation: John saw Eddy taking a short walk around his workplace.
Summary of relevant context from John's memory: [retrieved context]
Should John react to the observation, and if so, what would be an appropriate reaction?
```

**If Reaction Needed**:
- Regenerate plan from reaction point forward
- If interaction indicated, generate dialogue

### Dialogue Generation (Section 4.3.2)

**Initiated by Agent A**:
- Use summarized memory about Agent B
- Include intended reaction
- Generate first utterance

**Agent B Response**:
- Retrieves and summarizes memory about A
- Reviews current dialogue history
- Generates response or chooses to end

**Continuation**: Same mechanism until one agent decides to end

---

## Sandbox Environment (Section 5)

### Implementation

**Technology**:
- Phaser web game development framework
- Django server for environment management
- JSON data structure for agent state

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

### Action Grounding (Section 5.1)

**Process**: Recursively traverse environment tree

**Step 1**: Find appropriate area
- Start at root of agent's environment tree
- Flatten portion to natural language
- Prompt LLM to select most suitable area
- Example output: "The Lin family's house"

**Step 2**: Recursively find subarea
- Repeat process within chosen area
- Continue until reaching leaf node
- Example final result: "The Lin family's house: garden: house garden"

**Step 3**: Animate movement
- Use traditional game path algorithms
- Animate agent to leaf node location

### Object State Updates

**When Agent Acts on Object**:
- Query LLM for object state change
- Example: "making espresso for a customer" → coffee machine state: "off" → "brewing coffee"

### Server Loop

**Each Time Step**:
1. Parse JSON for changes from generative agents
2. Move agents to new positions
3. Update sandbox object states
4. Send observations (agents/objects within visual range) to each agent's memory
5. Agent outputs update JSON
6. Loop to next time step

---

## Agent Initialization (Section 5)

**Input**: One paragraph natural language description

**Example** (John Lin):
```
John Lin is a pharmacy shopkeeper at the Willow Market and Pharmacy who loves to help people. He is always looking for ways to make the process of getting medication easier for his customers; John Lin is living with his wife, Mei Lin, who is a college professor, and son, Eddy Lin, who is a student studying music theory; John Lin loves his family very much; John Lin has known the old couple next-door, Sam Moore and Jennifer Moore, for a few years; John Lin thinks Sam Moore is a kind and nice man; [...and so on]
```

**Processing**:
- Split by semicolons into individual statements
- Each statement entered as initial memory
- These seed memories determine initial behavior
- Behavior evolves as agent gains more experiences

---

## Evaluation Results (Section 6)

### Agent Count
- **25 agents** in Smallville
- Ran for **2 full game days**

### Information Diffusion

**Sam's Mayoral Candidacy**:
- Start: 1 agent knew (4%)
- End: 8 agents knew (32%)

**Isabella's Valentine's Day Party**:
- Start: 1 agent knew (4%)
- End: 13 agents knew (52%)

**Verification**: None who claimed knowledge had hallucinated it

### Relationship Formation

**Network Density**:
- Start: 0.167
- End: 0.74
- Hallucination rate: 1.3% (6 out of 453 responses)

### Coordination

**Valentine's Day Party**:
- Isabella planned party
- Invited 12 agents
- 5 agents actually showed up
- 3 cited schedule conflicts
- 4 expressed interest but didn't plan to attend

---

## Model & Costs (Section 4)

**LLM Used**:
- gpt3.5-turbo (ChatGPT)
- Note: GPT-4 API was invitation-only at time of writing

**Cost Estimation** (Not explicitly in paper, but can be inferred):
- Thousands of dollars in token credits
- Multiple days to simulate 25 agents for 2 days

---

## Key Implementation Insights

### 1. Synchronous Time Steps
- All agents process each 10-second step together
- No async/concurrent agent execution within a step
- Sequential processing of agents within each step

### 2. Memory Management
- Everything stored as natural language
- Retrieval is critical bottleneck
- Three-factor scoring essential for relevance

### 3. Planning Granularity
- High-level: hour-long chunks
- Mid-level: 5-15 minute chunks
- Plans can be interrupted and regenerated

### 4. Partial Observability
- Agents only perceive within visual range
- Must be in same area to observe each other
- Forces communication for coordination

### 5. Emergent Behavior
- No hardcoded coordination
- Emerges from: observations, memory retrieval, planning, reactions
- Example: Party coordination across multiple agents with no explicit protocol

---

## Common Failure Modes (Section 7.2)

### 1. Location Choice Degradation
- As agents learn about more locations, choices become less typical
- Example: Choosing bar for lunch instead of cafe
- Cause: Increasing memory makes retrieval harder

### 2. Physical Norm Misunderstanding
- Example: Multiple agents entering "one-person bathroom"
- Example: Entering closed stores after hours
- Cause: Norms hard to convey in natural language

### 3. Overly Cooperative/Formal Behavior
- Likely from instruction tuning in base LLM
- Agents rarely say "no" to suggestions
- Dialogue overly formal
- Interests shaped by others' suggestions

---

## References

Full paper: `/Users/ken/Desktop/lab/varela/docs/research/papers/generative-agents-stanford.pdf`

GitHub: https://github.com/joonspk-research/generative_agents
Demo: https://reverie.herokuapp.com/UIST_Demo/
