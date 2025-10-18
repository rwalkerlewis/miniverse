# Bugs and Issues Found - 2025-10-16

## Critical Issues

### 1. Information Diffusion Failed in Valentine's Party Scenario

**Severity**: HIGH
**Location**: Example prompt design + LLMPlanner behavior
**Description**: Isabella never invited anyone to the party despite explicit prompt instructions.

**Root Cause**:
- LLMPlanner created conservative multi-step plan with gating: "confirm venue logistics" → "draft guest list" → "send invitations"
- Isabella got stuck on step 1 for all 8 ticks
- Never progressed to actually inviting people

**Evidence**:
```
Tick 1: work - confirming logistics before outreach
Tick 2: work - confirm event logistics before outreach
Tick 3: work - confirm logistics and capacity
...
Tick 8: work - confirming venue logistics before invitations
```

**Result**: 0/5 agents learned about party (information did not diffuse)

**Proposed Fixes**:
1. **Short-term**: Change prompt to be more action-oriented:
   - Replace "Plan Valentine's Day party" with "INVITE people to Valentine's Day party"
   - Add explicit first action: "Your FIRST action should be to communicate and invite someone"

2. **Medium-term**: Add planner constraints:
   - Max plan complexity (e.g., 3-5 steps max)
   - Force certain actions in early steps
   - Add "bias toward action" parameter

3. **Long-term**: Implement reaction/replanning:
   - If agent hasn't taken critical action by tick N, trigger replan
   - Add goal progress monitoring

---

### 2. Schema Validation Errors - Communication Field

**Severity**: MEDIUM
**Location**: `miniverse/llm_utils.py` or LLM executor prompts
**Description**: LLM frequently returns `communication` as a string instead of required dict format

**Evidence**:
```
LLM schema validation failed for AgentAction (attempt 1/3).
    - communication: Input should be a valid dictionary [type=dict_type] |
      received='Hello Maria, this is the execution module...'
```

**Frequency**: ~30% of communication actions in both welcome.ipynb and valentines_party.ipynb

**Impact**:
- Retry logic works (usually corrects on attempt 2/3)
- Adds latency (~2-3 extra LLM calls per occurrence)
- Wastes tokens and API costs

**Proposed Fixes**:
1. **Improve schema feedback prompt** in `llm_utils.py`:
   - Currently just shows error
   - Should show EXAMPLE of correct format

2. **Add explicit examples in executor prompt**:
   ```python
   # Good example to add to prompt
   "communication": {"to": "agent_id", "message": "your message here"}
   ```

3. **Consider structured output mode** if provider supports it (OpenAI function calling, etc.)

---

### 3. Schema Validation Errors - Array Parameters

**Severity**: MEDIUM
**Location**: LLM executor / parameter generation
**Description**: LLM returns arrays for parameters expecting single values

**Evidence**:
```
LLM schema validation failed for AgentAction (attempt 1/3).
    - parameters.companions.str: Input should be a valid string | received=['Maria Lopez']
    - parameters.participants.str: Input should be a valid string | received=['Maria Lopez', 'Klaus Mueller']
```

**Impact**: Similar to #2 - retry overhead

**Proposed Fix**:
- Clarify in schema that parameters should be single values (string/int/float/bool)
- OR: Support arrays if that's the intended behavior (update schema)

---

### 4. Memory Retrieval Returns Empty

**Severity**: LOW (expected behavior in some cases)
**Location**: welcome.ipynb Cell 20
**Description**: Engineer's memory retrieval returns 0 memories

**Evidence**:
```
Total memories retrieved: 0
⚠️  No memories found for this agent.
```

**Possible Causes**:
- Memories not being saved properly
- Memory strategy not configured correctly
- OR: Working as intended (short simulation = few memories)

**Action**: Needs investigation. Check if memories ARE being saved during simulation.

---

## Testing Gaps

### 5. Workshop Examples Not Tested After Changes

**Severity**: MEDIUM
**Location**: `examples/workshop/run.py`, `examples/standup/run.py`
**Description**: Haven't verified that existing examples still work after notebook development

**Risk**: Changes to core library (if any) may have broken working examples

**Action Needed**: Run workshop and standup examples to ensure no regressions

---

## Validation Strategy Going Forward

### Phase 1: Fix Critical Prompt Issues (NOW)
1. Update Isabella's prompt to be more directive
2. Test with 8-tick simulation again
3. Verify information diffusion works

### Phase 2: Fix Schema Validation (Next Session)
1. Update `llm_utils.py` schema feedback prompts with examples
2. Add explicit format examples to executor prompts
3. Retest both notebooks

### Phase 3: Comprehensive Example Validation
1. Create test suite that runs ALL examples
2. `test_workshop.py` - deterministic mode
3. `test_standup.py` - deterministic mode
4. `test_welcome_notebook.py` - LLM mode (already have)
5. `test_valentines_notebook.py` - LLM mode (already have)

### Phase 4: Library Improvements
1. Add planner constraints (max steps, action bias)
2. Add goal progress monitoring
3. Add reaction/replanning triggers
4. Consider structured LLM output modes

---

## Success Criteria

Valentine's Day scenario is COMPLETE when:
- ✅ Simulation runs without errors
- ❌ Isabella invites at least 2 people explicitly (via communicate action with party message)
- ❌ At least 2 agents remember the party (embedding retrieval finds it)
- ✅ At least 2 agents show up at Hobbs Cafe (we got 4/5, but not for the right reason!)
- ❌ Information diffusion is observable in logs

**Current Status**: 2/5 criteria met. Core behavior missing.

---

*Generated: 2025-10-16*
*Test run: `fe38aa73-063e-42e2-9b3a-2dd30d7dc916`*
