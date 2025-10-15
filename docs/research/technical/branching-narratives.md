# Branching Narratives & The Loom

## Overview

The "Loom" concept in Varela draws from branching narrative systems in interactive storytelling and game design, applied to simulation timelines for scenario planning.

---

## Core Concepts

### Decision Trees

**Definition**: A diagram representing narrative flow where:
- **Nodes** = Story/simulation states
- **Edges** = Decisions or actions
- **Branches** = Different paths through the story/simulation
- **Outcomes** = Multiple possible endings

**Analogy**: Git version control
- Commits = Save points
- Branches = Alternative timelines
- Merges = Insights combined back to main timeline

**Source**: https://fetliu.net/has/2020/05/22/online-branching-scenarios/

---

## Game Design Applications

### RPG Examples

**The Witcher 3**:
- Decision trees track player choices
- Multiple possible endings based on accumulated decisions
- Salient decision points improve gameplay experience

**Research Finding**: "Presence of salient decision points in an RPG leads to improved experiences of game play"

**Key Principle**: Players must be able to predict consequences to make meaningful decisions

**Source**: https://www.researchgate.net/publication/300588610_Narrative_Control_and_Player_Experience_in_Role_Playing_Games_Decision_Points_and_Branching_Narrative_Feedback

---

## Design Principles

### Meaningful Choices

**Requirements for meaningful decisions**:
1. **Predictability**: Player/user has basis to predict consequences
2. **Agency**: Choice actually affects outcomes
3. **Trade-offs**: No obviously "correct" answer
4. **Coherence**: Consequences follow logically from choice

**Anti-pattern**: Random guessing without context

**Source**: https://www.gamedeveloper.com/design/meaningful-decisions-in-branching-narratives

---

### Narrative Coherence

**Best Practice**: Use flowcharts or decision trees to ensure:
- Every choice logically follows from previous events
- Cause-and-effect relationships remain clear
- No matter which path, plot retains consistency

**Implementation**:
- Map out narrative paths visually
- Track dependencies between choices
- Validate that branches don't contradict established facts

**Source**: https://kreonit.com/programming-and-games-development/nonlinear-gameplay/

---

## Technical Tools

### Interactive Fiction Engines

**Twine**:
- Visual node-based editor for branching stories
- Exports to HTML/JavaScript
- Used for both games and educational simulations
- Open source

**Ink** (by Inkle Studios):
- Scripting language for branching narratives
- Integrates with Unity and other engines
- Used in: 80 Days, Heaven's Vault
- JSON export for state management

**Comparison**: https://christytuckerlearning.com/tools-for-building-branching-scenarios/

---

### Standard Patterns in Choice-Based Games

**Common structures**:

1. **Time Cave** (branching then reconvergence)
   - Multiple paths
   - Paths reconverge at key points
   - Memory of choices carried forward

2. **Gauntlet** (sequential challenges)
   - Linear series of choices
   - Each affects character state
   - Final outcome based on accumulated state

3. **Branch and Bottleneck**
   - Major branches with different content
   - Occasional reconvergence points
   - Balance between variety and manageability

4. **Open Map**
   - Non-linear exploration
   - Visit locations in any order
   - State tracks what's been discovered

**Source**: https://heterogenoustasks.wordpress.com/2015/01/26/standard-patterns-in-choice-based-games/

---

## The Loom for Varela

### Core Functionality

**Purpose**: Navigate simulation timelines to explore decision spaces

**Operations**:
1. **Save State**: Create checkpoint at decision point
2. **Branch**: Fork timeline to explore alternative
3. **Navigate**: Move between branches and timepoints
4. **Compare**: Analyze outcomes across branches
5. **Merge**: Bring insights from exploration back to main timeline

---

### Architecture Model

```
Main Timeline
├── T0: Initial state (colony arrival)
├── T1: First crisis (oxygen system failure)
│   ├── Branch A: Prioritize repair (conservative)
│   │   ├── T2a: Repair complete, morale low
│   │   └── T3a: Colony stable but behind schedule
│   │
│   └── Branch B: Improvise workaround (aggressive)
│       ├── T2b: Workaround functional, risk high
│       └── T3b: Innovation success OR catastrophic failure
│
└── T4: [Based on insights from A & B exploration]
```

---

### User Interaction Model

**Workflow**:
1. Simulation runs forward
2. System/user marks **decision point** (critical choice moment)
3. User can:
   - Continue main timeline (commit to choice)
   - Create branch (explore alternative)
   - Review past decision points
   - Compare branch outcomes

**UI Concepts**:
- Timeline visualization (horizontal node graph)
- Branch explorer (tree view)
- State diff viewer (compare two branches)
- Metrics dashboard (outcome comparison)

---

### Decision Point Detection

**Automatic Triggers** (system identifies):
- Resource scarcity reaching threshold
- Agent conflict requiring intervention
- Equipment failure with multiple repair options
- Opportunity for major strategic shift

**Manual Marking** (user decides):
- Interesting simulation moment
- Before testing hypothesis
- After unexpected emergence

**Metadata for Decision Points**:
```json
{
  "decision_id": "dp_001",
  "timestamp": "2157-03-15T14:30:00Z",
  "trigger": "oxygen_critical",
  "description": "Oxygen production down 40%. Repair or improvise?",
  "context": {
    "available_engineers": 2,
    "spare_parts": "limited",
    "time_to_critical": "6 hours"
  },
  "options": [
    {
      "label": "Full repair",
      "estimated_time": "8 hours",
      "risk": "low",
      "resource_cost": "high"
    },
    {
      "label": "Jury-rig workaround",
      "estimated_time": "2 hours",
      "risk": "medium",
      "resource_cost": "low"
    },
    {
      "label": "Reduce consumption",
      "estimated_time": "immediate",
      "risk": "low",
      "impact": "morale decrease"
    }
  ]
}
```

---

### State Comparison

**Metrics to Compare Across Branches**:

**Quantitative**:
- Agent survival count
- Resource levels
- Mission objectives completed
- Time elapsed
- Conflict incidents

**Qualitative**:
- Morale/stress trends
- Relationship network health
- Innovation events
- Crisis response effectiveness

**Visualization**:
- Line charts (metrics over time, overlaid by branch)
- Radar charts (multi-dimensional outcome comparison)
- Network graphs (relationship structure differences)

---

## Implementation Considerations

### Storage Strategy

**Option 1: Full State Snapshots**
- Store complete world state at each branch point
- Fast to load any branch
- High storage cost

**Option 2: Diff Chain**
- Store initial state + action sequences
- Replay to reach any point
- Low storage, slower reconstruction

**Option 3: Hybrid**
- Snapshots at branch points
- Diffs for steps within branch
- Balance performance and storage

**Recommendation**: Option 3 for Varela

---

### Branch Management

**Naming Convention**:
- `main` = Primary timeline
- `explore/oxygen-crisis-v1` = Exploratory branch
- `test/aggressive-expansion` = Hypothesis testing

**Metadata Tracking**:
- Parent branch
- Creation timestamp
- Creator notes/hypothesis
- Outcome summary (when complete)

---

### Pruning Strategy

**When to keep branches**:
- Produced valuable insights
- Interesting unexpected outcomes
- Representative of decision class

**When to delete branches**:
- Exploration complete
- Similar to other branches
- Clear dead-end/failure

**Archive Option**: Save branch summary + decision point, discard full state

---

## Implications for Varela

### MVP Loom Features

**Phase 1**:
1. Save/load full state at any point
2. Create named branches from save points
3. Visual timeline with branch indicators
4. Basic state diff (show what changed)

**Phase 2**:
5. Automatic decision point detection
6. Metrics comparison across branches
7. Branch annotations and notes
8. Timeline visualization UI

**Phase 3**:
9. Merge insights (user records learnings)
10. Recommendation system (suggest interesting branches)
11. Pattern detection across branches

---

### Design Questions

1. **Auto-save frequency**: Every N turns? Only at decision points?
2. **Branch limit**: Max branches per project? Warn when storage grows large?
3. **Simulation speed**: Real-time branches or run exploratory branches fast-forward?
4. **Replay**: Allow editing past decisions and replaying forward?

---

## References

- Decision Trees in Branching Scenarios: https://fetliu.net/has/2020/05/22/online-branching-scenarios/
- Meaningful Decisions: https://www.gamedeveloper.com/design/meaningful-decisions-in-branching-narratives
- Standard Patterns: https://heterogenoustasks.wordpress.com/2015/01/26/standard-patterns-in-choice-based-games/
- RPG Narrative Control: https://www.researchgate.net/publication/300588610
- Tool Comparison: https://christytuckerlearning.com/tools-for-building-branching-scenarios/
