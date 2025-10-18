# Outstanding Issues & Next Steps

**Last Updated**: 2025-10-16
**Status**: 1 critical bug fixed, verification in progress

---

## 🎯 Current Session Summary

### What We Accomplished

1. ✅ **Added DEBUG_LLM logging** to library (`miniverse/cognition/llm.py`)
   - Shows complete LLM prompts (system + user) for planner, executor, reflection
   - Shows LLM responses with structured data
   - Enable with `DEBUG_LLM=true` environment variable

2. ✅ **Identified root cause of information diffusion failure**
   - Problem: Valentine's Day party scenario showed 0/5 agents learning about party
   - Root cause: `execute_tick` prompt template only showed `work` action example
   - Impact: LLMs generated communicate actions with wrong message format (missing 'message' key)

3. ✅ **Implemented fix**
   - Updated `miniverse/cognition/prompts.py` execute_tick template
   - Added communicate action example alongside work action example
   - Demonstrated fix with `test_simple_communicate.py`

4. ✅ **Fixed EmbeddingMemoryStream bug**
   - Issue: Custom memory adapter didn't handle `importance=None`
   - Location: `test_valentines_notebook.py` line 74
   - Fix: Added None check with default value

5. ✅ **Created comprehensive documentation**
   - `ROOT_CAUSE_ANALYSIS.md` - Complete evidence trail and fix implementation
   - `DEBUG_LLM_FINDINGS.md` - Analysis framework (now superseded by ROOT_CAUSE_ANALYSIS.md)
   - `SESSION_SUMMARY.md` - Previous session overview

---

## 🔄 Currently Running Tests

### Valentine's Day Scenario (test_valentines_notebook.py)
**Status**: 🔄 Running with fixed prompt
**Expected Outcome**: Information diffusion should show >0/5 agents aware of party
**Log**: `/tmp/valentines_fixed.log`

### Workshop Example (examples/workshop/run.py)
**Status**: 🔄 Running with DEBUG_LLM for regression testing
**Expected Outcome**: Should still work (uses SimpleExecutor which calls same code path)
**Log**: `/tmp/workshop_debug.log`

---

## 📋 Immediate Next Steps

### Step 1: Verify Fix ⏳ IN PROGRESS
- [x] Implement fix in prompts.py
- [ ] Wait for Valentine's test completion
- [ ] Check if Isabella's messages now have content
- [ ] Verify knowledge diffusion metric shows >0 agents aware
- [ ] Compare before/after DEBUG_LLM logs

**Success Criteria**:
- Isabella's communicate actions have non-empty `message` field
- Other agents' perceptions include Isabella's messages
- Knowledge diffusion ≥ 2/5 (40%+)

### Step 2: Update Documentation 🔄 IN PROGRESS
- [ ] Update README.md with fix notes
- [ ] Update plan.md to mark WP6 progress
- [ ] Clean up temporary analysis docs (merge into ROOT_CAUSE_ANALYSIS.md)
- [ ] Update CLAUDE.md with new debugging patterns learned

### Step 3: Clean Up Repository 📦 PENDING
- [ ] Remove temporary test files (test_simple_communicate.py, test_valentines_notebook.py, test_welcome_notebook.py)
- [ ] Move analysis docs to docs/debugging/ folder
- [ ] Archive DEBUG_LLM logs or delete if no longer needed
- [ ] Update .gitignore for debug logs

### Step 4: Regression Testing ✅ PENDING
- [ ] Run workshop example without DEBUG_LLM (ensure no performance regression)
- [ ] Run welcome.ipynb notebook (Mars habitat example)
- [ ] Run tutorial.ipynb (if it exists)
- [ ] Verify all examples pass with new prompt template

### Step 5: Write Tests 🧪 PENDING
- [ ] Add unit test for communicate action message format validation
- [ ] Add integration test for information diffusion scenario
- [ ] Add test that verifies both work and communicate examples work
- [ ] Consider adding test for other action types (move, rest, etc.)

---

## 🐛 Known Issues

### ✅ FIXED: Empty Communication Messages
**Status**: FIXED (2025-10-16)
**Severity**: Critical
**Impact**: Information diffusion completely broken in LLM-driven scenarios
**Root Cause**: Missing communicate action example in execute_tick prompt
**Fix**: Added communicate example to DEFAULT_PROMPTS
**Files Modified**: `miniverse/cognition/prompts.py`
**Documentation**: `ROOT_CAUSE_ANALYSIS.md`

### 🟡 Workshop Uses Legacy get_agent_action()
**Status**: Design question, not a bug
**Severity**: Low
**Description**: Workshop example uses `SimpleExecutor` which calls legacy `get_agent_action()` function instead of `LLMExecutor`. Both work, but code paths differ.
**Impact**: Maintenance complexity - two ways to do LLM executor calls
**Recommendation**: Eventually migrate workshop to use `LLMExecutor` directly for consistency
**Priority**: Low - works fine as-is

### 🟡 DEBUG_LLM Not in Legacy get_agent_action()
**Status**: Enhancement opportunity
**Severity**: Low
**Description**: DEBUG_LLM logging added to `LLMExecutor` class but not to legacy `get_agent_action()` function in `llm_calls.py`
**Impact**: Workshop example with --llm doesn't show executor prompts/responses in DEBUG_LLM mode
**Recommendation**: Add DEBUG_LLM to `get_agent_action()` for consistency
**Priority**: Low - can work around by switching to LLMExecutor

---

## 🎯 Strategic Priorities (from NEXT_STEPS.md)

### WP6 - Documentation & Examples 🚧 IN PROGRESS
**Current Status**:
- ✅ README.md updated with workshop example
- ✅ Created tutorial.ipynb, welcome.ipynb
- ✅ Created valentines_party.ipynb (Stanford replication attempt)
- 🔄 Debugging information diffusion issue → FIX IMPLEMENTED
- ⏳ Verification pending

**Next**:
- Verify Valentine's scenario works with fix
- Document best practices for prompt engineering
- Add communication patterns guide
- Update USAGE.md with debugging section

### WP7 - Testing & Tooling 🚧 ONGOING
**Current Status**:
- ✅ DEBUG_LLM feature implemented
- ✅ Root cause analysis methodology established
- 🔄 Fix implementation in progress
- ⏳ Regression tests pending

**Next**:
- Add automated tests for communicate actions
- Add information diffusion test suite
- Consider adding prompt template validation tests
- Benchmark LLM token usage with new longer prompts

### WP5 - Stanford Scenario Replication 🔄 BLOCKED → UNBLOCKING
**Current Status**:
- ✅ Valentine's Day party scenario created
- ❌ Information diffusion failed (0/5 agents)
- ✅ Root cause identified
- ✅ Fix implemented
- ⏳ Verification pending

**Next**:
- Confirm fix works
- Tune scenario parameters if needed (more ticks, better initial conditions)
- Add other Stanford scenarios (cascading failures, emergent coordination)
- Document comparison vs Stanford's Reverie architecture

---

## 📊 Testing Status Matrix

| Example/Test | Status | Last Run | Pass/Fail | Notes |
|-------------|--------|----------|-----------|-------|
| workshop (deterministic) | ✅ Passing | Previous session | ✅ Pass | No LLM calls |
| workshop (--llm) | 🔄 Running | 2025-10-16 | ⏳ Testing | DEBUG_LLM enabled |
| welcome.ipynb | ✅ Passing | Previous session | ✅ Pass | Mars habitat example |
| tutorial.ipynb | ✅ Passing | Previous session | ✅ Pass | Step-by-step guide |
| valentines_party.ipynb | 🔄 Running | 2025-10-16 | ⏳ Testing | With prompt fix |
| test_simple_communicate.py | ✅ Verified | 2025-10-16 | ✅ Pass | Demonstrates fix works |
| Unit tests (pytest) | ⚠️ Unknown | Not run | ⏳ Pending | Need to run full suite |

---

## 🔧 Technical Debt

### 1. Consolidate Executor Implementations
**Priority**: Medium
**Effort**: Medium
**Description**: `SimpleExecutor`, `LLMExecutor`, and `get_agent_action()` provide overlapping functionality. Consider:
- Deprecating `get_agent_action()` in favor of `LLMExecutor`
- Making `SimpleExecutor` a thin wrapper around `LLMExecutor` with deterministic fallback
- Clearer documentation on which to use when

### 2. Prompt Template Validation
**Priority**: Low
**Effort**: Small
**Description**: Add validation that prompt templates include required placeholders and example structures
- Check that execute templates have at least one action example
- Validate communicate examples have correct `{"to": "...", "message": "..."}` format
- Warn if templates missing common placeholders

### 3. Structured Communication Schema
**Priority**: Low
**Effort**: Medium
**Description**: Current communication field is `Optional[Dict[str, str]]` which is very loose
- Consider creating `CommunicationPayload` Pydantic model
- Enforce `to` and `message` fields at schema level
- Better type safety and validation

### 4. Memory Adapter Interface Consistency
**Priority**: Medium
**Effort**: Small
**Description**: Custom memory adapters need to handle `importance=None` gracefully
- Add note to memory adapter documentation
- Consider making `importance` non-optional in orchestrator calls
- Or document that adapters must handle None

---

## 📚 Documentation Improvements Needed

### README.md
- [ ] Add section on DEBUG_LLM feature
- [ ] Update examples section with Valentine's scenario once working
- [ ] Add troubleshooting section for common LLM issues

### USAGE.md
- [ ] Add debugging section with DEBUG_LLM instructions
- [ ] Add communication patterns guide
- [ ] Add prompt engineering best practices
- [ ] Add section on custom memory adapters with gotchas

### CLAUDE.md
- [ ] Document DEBUG_LLM debugging workflow
- [ ] Add communication action debugging patterns
- [ ] Update with lessons learned from this session

### docs/architecture/cognition.md
- [ ] Update executor section with LLMExecutor details
- [ ] Document prompt template requirements
- [ ] Add section on action schema and communication format

---

## 🎓 Lessons Learned

### 1. Always Provide Complete Examples in Few-Shot Prompts
**Problem**: Single work action example led LLM to invent its own communication format
**Solution**: Multiple examples covering different action types
**Takeaway**: Don't assume LLM will generalize from one example to other similar patterns

### 2. DEBUG_LLM is Essential for LLM Debugging
**Problem**: Without visibility into LLM prompts/responses, we were guessing about root cause
**Solution**: Added comprehensive DEBUG_LLM logging showing full context
**Takeaway**: Always instrument LLM calls with detailed logging in development

### 3. Schema Comments ≠ LLM Guidance
**Problem**: Schema had comment saying format should be `{"to": "...", "message": "..."}` but LLM didn't see it
**Solution**: Put examples in prompt, not just in code comments
**Takeaway**: LLMs learn from examples in prompts, not from code comments or schema docstrings

### 4. Test with Multiple Executors
**Problem**: Workshop (SimpleExecutor) worked but Valentine's (LLMExecutor) failed, initially confusing
**Solution**: Realized both use same underlying call_llm_with_retries, issue was prompt template
**Takeaway**: Test with various executor implementations to catch edge cases

### 5. Structured Outputs Need Examples Too
**Problem**: Even with Pydantic schema validation, LLM needs examples to know how to populate fields
**Solution**: Mirascope sends schema, but examples guide LLM on which dict keys to use
**Takeaway**: Structured outputs validate format but don't guide content - examples do

---

## 🚀 Next Session Priorities

When you return to this project, here's the recommended order of operations:

1. **Check Test Results** ⏰ URGENT
   - Review `/tmp/valentines_fixed.log` for Valentine's scenario results
   - Check knowledge diffusion metric
   - Verify workshop regression test passed

2. **Update Documentation** 📝 HIGH
   - Merge findings into main docs
   - Clean up temporary analysis files
   - Update README with new features

3. **Write Regression Tests** 🧪 HIGH
   - Prevent this issue from recurring
   - Test coverage for communicate actions
   - Information diffusion test suite

4. **Clean Up Repository** 🧹 MEDIUM
   - Remove temporary test files
   - Archive or delete debug logs
   - Organize docs folder

5. **Continue Stanford Replication** 🎯 MEDIUM
   - Once Valentine's works, tune parameters
   - Add more complex scenarios
   - Validate emergent behaviors

---

## 📞 Support & Contact

If you encounter issues:
1. Check DEBUG_LLM logs first (`DEBUG_LLM=true` environment variable)
2. Review ROOT_CAUSE_ANALYSIS.md for communication debugging patterns
3. Consult USAGE.md for prompt engineering best practices
4. Check GitHub issues for similar problems

---

**Document Owner**: Claude (Lead Dev)
**Last Updated**: 2025-10-16
**Next Review**: After Valentine's test completion

-- Claude | 2025-10-16
