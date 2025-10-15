# Structured Outputs & State Management

## Overview

Managing world state through LLM-driven evolution requires structured output formats (primarily JSON) to ensure consistency, validateability, and reliable state transitions.

---

## LLMs as World Simulators

### Core Concept

**Research Finding**: LLMs can serve as "text-based world simulators"

**State Transition Model**:
```
(previous_state + agent_actions + environmental_context) → next_state
```

**Implementation**:
- Previous state encoded as JSON object
- Agent actions provided as structured input
- Context message explains situation
- LLM produces subsequent state as:
  - Complete JSON object (full state replacement), OR
  - Diff/patch (only changes from previous state)

**Source**: "Can Language Models Serve as Text-Based World Simulators?" (2024)
- https://arxiv.org/abs/2406.06485

---

## JSON Schema for Structured Outputs

### What is JSON Schema?

**Definition**: A vocabulary for describing the structure and content of JSON data

**Purpose**: Acts as a blueprint specifying:
- Data types for each field
- Required vs. optional fields
- Format constraints and validation rules
- Nested object structures
- Array element types

### Why JSON Schema Matters

**Benefits**:
1. **Validation**: Ensure LLM output conforms to expected structure
2. **Type Safety**: Prevent malformed data entering system
3. **Documentation**: Schema serves as API contract
4. **Tool Integration**: Enable function calling and structured responses
5. **Diff Computation**: Compare states reliably

**Source**: https://blog.promptlayer.com/how-json-schema-works-for-structured-outputs-and-tool-integration/

---

## Current LLM Capabilities

### Performance on Structured Output

**Historical Challenge**:
- Earlier LLMs struggled to generate valid JSON consistently
- Format errors, missing brackets, incorrect types common

**Current State (2024-2025)**:
- **GPT-4o (2024-08-06)**: Scores **100%** on complex JSON schema following
- **Claude 3.5 Sonnet**: High reliability with structured outputs
- **Gemini 1.5**: Strong JSON generation capabilities

**Key Advancement**: Native structured output modes in latest models
- OpenAI: `response_format: { "type": "json_schema" }`
- Anthropic: Tool use with typed parameters
- Google: Controlled generation with schema constraints

---

## Implementation Strategies

### 1. Full State Representation

**Approach**: LLM generates complete world state each turn

```json
{
  "timestamp": "2157-03-15T14:30:00Z",
  "environment": {
    "location": "Mars Base Alpha",
    "temperature_celsius": -63,
    "oxygen_level_percent": 21,
    "power_available_kwh": 1250
  },
  "agents": [
    {
      "id": "agent_001",
      "name": "Dr. Sarah Chen",
      "role": "Medical Officer",
      "status": "active",
      "location": "medical_bay",
      "health": 85,
      "stress": 42
    }
  ],
  "resources": {
    "water_liters": 5000,
    "food_days": 180,
    "oxygen_hours": 720
  }
}
```

**Pros**:
- Simple to implement
- Complete snapshot each turn
- Easy to serialize/deserialize

**Cons**:
- Verbose for large worlds
- Higher token usage
- May reintroduce errors in unchanged parts

---

### 2. Diff/Patch Representation

**Approach**: LLM generates only changes from previous state

```json
{
  "timestamp": "2157-03-15T14:31:00Z",
  "changes": {
    "agents.agent_001.location": "greenhouse",
    "agents.agent_001.stress": 40,
    "resources.oxygen_hours": 719,
    "events": [
      {
        "type": "agent_moved",
        "agent": "agent_001",
        "from": "medical_bay",
        "to": "greenhouse",
        "reason": "stress_relief"
      }
    ]
  }
}
```

**Pros**:
- Efficient for large states
- Lower token usage
- Highlights what changed

**Cons**:
- More complex merging logic
- Potential inconsistencies
- Harder to validate completeness

---

### 3. Hybrid Approach

**Strategy**: Use full state for critical moments, diffs for routine updates

**When to use full state**:
- Save points / checkpoints
- After major events
- Before branching (Loom entry points)
- Validation checks

**When to use diffs**:
- Turn-by-turn updates
- Minor agent actions
- Resource consumption
- Routine state evolution

---

## Tools & Libraries

### Python Ecosystem

**Pydantic**: Type-safe Python classes that generate JSON schemas
```python
from pydantic import BaseModel

class Agent(BaseModel):
    id: str
    name: str
    role: str
    health: int
    stress: int
```

**jsonschema**: Validation library
```python
import jsonschema

jsonschema.validate(instance=data, schema=schema)
```

**Outlines** (Apache-2.0): Structured text generation
- Supports multiple LLM providers
- Regex patterns, JSON schemas, Pydantic models
- Context-free grammar support
- https://github.com/outlines-dev/outlines

---

### LLM Integration Libraries

**LiteLLM**: Multi-provider structured output support
- Client-side JSON schema validation
- `litellm.enable_json_schema_validation=True`
- Unified API across OpenAI, Anthropic, Google, etc.
- https://docs.litellm.ai/docs/completion/json_mode

**LangChain**: Structured output chains
- Schema-based parsers
- Retry logic on validation failure
- Integration with Pydantic

**Mirascope**: Lightweight structured output library
- Type-safe LLM interactions
- https://mirascope.com/blog/langchain-structured-output

---

## Validation Strategies

### Multi-Layer Validation

**1. Schema Validation**: Does output match expected structure?
**2. Semantic Validation**: Do values make sense? (e.g., health 0-100)
**3. Consistency Validation**: Does new state follow from old state + actions?
**4. Conservation Laws**: Are resources properly accounted for?

### Error Recovery

**When LLM produces invalid output**:
1. **Retry with error message**: "Output failed validation: {error}"
2. **Provide example**: Show correct format
3. **Simplify request**: Break into smaller state updates
4. **Fallback**: Use previous valid state, log error

---

## State Persistence

### Save Point Strategy

**Checkpointing**:
- Save full state at decision points
- Store as JSON file or database record
- Include metadata: timestamp, branch_id, event_trigger

**Branching Support (for Loom)**:
```json
{
  "branch_id": "branch_001",
  "parent_branch": "main",
  "fork_timestamp": "2157-03-15T14:30:00Z",
  "decision_point": "Choose oxygen allocation strategy",
  "state": { ... }
}
```

---

## Implications for Varela

### Recommended Approach

1. **Use Pydantic** for schema definition (type-safe Python)
2. **Full state** for Loom save points
3. **Diffs** for turn-by-turn evolution (optional optimization)
4. **Multi-layer validation** before state acceptance
5. **Git-like branching** model for timeline exploration

### State Schema Design Priorities

**Critical components**:
- Agent states (health, location, inventory, memory)
- Environment conditions (resources, hazards, infrastructure)
- Relationship graph (trust, conflicts, alliances)
- Event log (timestamped actions and outcomes)

**Nice-to-have**:
- Skill progression tracking
- Detailed resource production chains
- Weather/environmental cycles
- Communication message history

---

## References

- LLMs as World Simulators: https://arxiv.org/abs/2406.06485
- JSON Schema Guide: https://blog.promptlayer.com/how-json-schema-works-for-structured-outputs-and-tool-integration/
- Structured Output with LangChain: https://medium.com/@docherty/mastering-structured-output-in-llms-choosing-the-right-model-for-json-output-with-langchain-be29fb6f6675
- Awesome LLM JSON Resources: https://github.com/imaurer/awesome-llm-json
- Schema Reinforcement Learning: https://arxiv.org/abs/2502.18878
