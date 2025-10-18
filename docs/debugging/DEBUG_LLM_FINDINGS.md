# DEBUG_LLM Analysis - Valentine's Day Scenario

**Date**: 2025-10-16
**Tool**: `DEBUG_LLM=true` environment variable
**Scenario**: Valentine's Day party information diffusion test
**Run**: In progress - capturing 8 ticks with full LLM logging

---

## What DEBUG_LLM Shows

Setting `DEBUG_LLM=true` prints complete LLM cognition for each agent:

### For Planner (LLMPlanner)
```
[LLM PLANNER] Agent: {agent_id}
[SYSTEM PROMPT] - Generic planning assistant instructions
[USER PROMPT] - Full JSON context including:
  - profile (goals, personality, skills, relationships)
  - perception (location, resources, observations, messages)
  - world (full state with all agents' locations)
  - scratchpad (current state)
  - plan_state (existing plan if any)
  - memories (recent observations)
  - extra (base_agent_prompt, LLM config)
[LLM RESPONSE] - Generated plan with steps
```

### For Executor (LLMExecutor)
```
[LLM EXECUTOR] Agent: {agent_id}
[SYSTEM PROMPT] - Action selection instructions
[USER PROMPT] - Full context + current plan step
[LLM RESPONSE] - Chosen action with reasoning
```

### For Reflection (LLMReflectionEngine)
```
[LLM REFLECTION] Agent: {agent_id}
[SYSTEM PROMPT] - Reflection synthesis instructions
[USER PROMPT] - Recent memories to reflect on
[LLM RESPONSE] - Generated reflections
```

---

## Initial Observations (from 1-tick test)

### Isabella's Context (Tick 1)

**Base Agent Prompt** (from extra.base_agent_prompt):
```
You are Isabella Rodriguez, owner of Hobbs Cafe.

CRITICAL GOAL: You are planning a Valentine's Day party at Hobbs Cafe on February 14th, 5-7pm.

Your top priority is to INVITE people! When you see someone (at the cafe or elsewhere):
- Use the "communicate" action
- Tell them about the party: date, time, location
- Encourage them to come and spread the word

The party is in 2 days - start inviting NOW!

Available actions: communicate (to invite), work_at_cafe, move_to [location]
```

**✅ GOOD**: The directive is clear and explicit.

**Profile Goals**:
```json
"goals": [
  "Run successful cafe",
  "Build community",
  "Plan Valentine's Day party at Hobbs Cafe on Feb 14, 5-7pm"
]
```

**Visible Agents at Same Location** (hobbs_cafe):
- Ayesha Khan (journalist, regular customer)

**Key Context Available**:
- Current time: 10am, Feb 13
- Party is tomorrow at 5pm (in 2 days minus ~31 hours)
- Someone (Ayesha) is RIGHT THERE to invite

---

## Problem Hypothesis

### Issue: Planner Overrides Executor Directive

The **base_agent_prompt** tells Isabella to "INVITE people NOW!" but:

1. **LLMPlanner** creates a multi-step plan first
2. Plan prioritizes "logistics confirmation" before invitations
3. **LLMExecutor** follows the plan (as designed)
4. Result: Isabella never gets to inviting stage

### Evidence Needed

From full 8-tick DEBUG_LLM log, we need to see:

1. **What plan does LLMPlanner generate?**
   - Does it have "invite" as step 1, or later?
   - What's the actual step sequence?

2. **What does LLMExecutor see?**
   - Does it get the "invite NOW" directive?
   - Does it see Ayesha is present?
   - What's the plan_step it's executing?

3. **Why doesn't it deviate from plan?**
   - Executor is allowed to deviate if circumstances changed
   - Does it recognize Ayesha as an invitation opportunity?

---

## Possible Root Causes

### 1. **Planner Prompt Too Conservative**

The default planning template (`plan_daily`) might encourage:
- Systematic preparation
- Risk-averse sequencing
- "Confirm before acting" patterns

**Check**: What does the `plan_daily` system prompt actually say?

### 2. **Executor Doesn't Override Plan**

Even though executor CAN deviate, it might:
- Prioritize plan adherence too heavily
- Not recognize "person present" as trigger for deviation
- Lack explicit instruction to prioritize immediate opportunities

**Check**: What does the `execute_tick` system prompt say about plan vs immediate action?

### 3. **Context Dilution**

The base_agent_prompt saying "invite NOW!" might get:
- Buried in massive JSON context dump
- Overshadowed by profile goals that mention "planning"
- Treated as optional flavor text vs hard directive

**Check**: How is `base_agent_prompt` positioned in the user prompt?

### 4. **Goal Interpretation**

Profile goal says "Plan Valentine's Day party" - this could be interpreted as:
- ✅ "Execute the party plan" (invite people)
- ❌ "Create a plan for the party" (logistics first)

**Check**: Do other agents with simpler goals behave more directly?

---

## Questions to Answer from Full Log

### Planner Analysis
- [ ] What's the exact system prompt for `plan_daily`?
- [ ] What plan does Isabella's LLMPlanner generate (all steps)?
- [ ] What plan does Maria's LLMPlanner generate (for comparison)?
- [ ] Do any agents generate "communicate" as step 1?

### Executor Analysis
- [ ] What's the exact system prompt for `execute_tick`?
- [ ] What context does Isabella's executor see on tick 1?
- [ ] Does it mention Ayesha being present?
- [ ] What's the plan_step it's told to execute?
- [ ] What reasoning does it give for choosing "work" vs "communicate"?

### Cross-Agent Comparison
- [ ] Do agents without planner (if any) act more directly?
- [ ] Do agents with simpler goals make more immediate actions?
- [ ] Is there correlation between goal complexity and action delay?

---

## Potential Fixes (DO NOT IMPLEMENT YET)

### Option A: Simplify Isabella's Role
Instead of "Plan party" goal, make it "Invite people to party"
- Changes intent from planning → execution
- More aligned with desired behavior

### Option B: Modify Planner Prompt
Add to planning system prompt:
```
If the agent has an urgent time-sensitive goal (party soon, deadline approaching),
prioritize immediate action steps over preparation steps.
```

### Option C: Modify Executor Prompt
Add to execution system prompt:
```
If you see an opportunity related to a high-priority goal (e.g., person to invite for party),
you may deviate from your plan to seize the immediate opportunity.
```

### Option D: Inject First Action
Modify scratchpad initialization to force:
```python
scratchpad.next_action_override = "communicate with Ayesha about party"
```

### Option E: Remove Planner for This Agent
```python
# Make Isabella reactive, not planned
cognition_map['isabella'] = AgentCognition(
    executor=LLMExecutor(),
    # No planner - executor acts directly each tick
)
```

---

## Next Steps

1. **Wait for full DEBUG_LLM log to complete** (~5-10 min)
2. **Extract and analyze**:
   - Isabella's plan (all steps)
   - Isabella's executor reasoning (tick 1)
   - System prompts for planner and executor
3. **Document findings** in this file
4. **Propose specific fix** based on evidence
5. **Test fix** with 1-tick validation
6. **Full 8-tick retest** to verify information diffusion

---

## Status

- ⏳ Running full 8-tick simulation with DEBUG_LLM=true
- ⏳ Log saved to `/tmp/debug_llm_full.log`
- ⏳ Awaiting completion for analysis

---

*Analysis in progress - will update when log is complete*
