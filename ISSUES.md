# Outstanding Issues

**Last Updated**: 2025-10-16
**Status**: Critical bug FIXED! Now addressing architectural improvements

---

## ✅ FIXED: Information Diffusion Bug

### Status
**RESOLVED** - Root cause identified and fixed in orchestrator.py:542-586

### Root Cause
Communication memories were only created for SENDERS, never for RECIPIENTS.

**The Bug** (orchestrator.py old code):
```python
if action.communication:
    memory = await self.memory.add_memory(
        agent_id=action.agent_id,  # ← Only sender!
        content=f"I told {recipient}: {message}",
    )
```

**The Fix** (orchestrator.py new code):
- Create sender memory: "I told X: message"
- Create recipient memory: "X told me: message"  ← THIS WAS MISSING!

### Verification
- Added `tests/test_information_diffusion.py` with 2 passing tests
- Added `tests/test_communication_flow.py` for end-to-end verification
- All 26 tests passing (no regressions)
- Fixed event severity validation (default to importance=5)
- Valentine's notebook ready to run

### Why This Broke Information Diffusion
1. Isabella sends: "Hi Ayesha, party on Feb 14 at 5pm..."
2. Memory created for Isabella: "I told Ayesha: ..."
3. ❌ NO memory created for Ayesha
4. When Ayesha's perception built: no party-related memories found
5. Result: 0/5 agents aware despite properly formatted messages

### The Fix in Action
1. Isabella sends message to Ayesha
2. Sender memory: agent_id="isabella", content="I told Ayesha: party..."
3. **Recipient memory**: agent_id="ayesha", content="Isabella told me: party..."
4. Ayesha's perception retrieves: "Isabella told me: party..."
5. ✅ Information successfully diffused!

---

---

## 🟠 NEW: Architectural Issues Discovered During Code Review

### Status
**OPEN** - Non-blocking improvements for DX and performance

These issues don't break functionality but make the library harder to use and maintain.

### Issue A1: Dual Memory Retrieval (orchestrator.py:327-348)

**Problem**: Same memories fetched twice in different formats
```python
# Line 327: Fetch as strings for perception
recent_memory_strings = await self.memory.get_recent_memories(...)

# Line 346: Fetch as objects for prompt context (20 lines later!)
recent_agent_memories = await self.persistence.get_recent_memories(...)
```

**Impact**:
- Confusing for new contributors
- Inefficient (2x database queries)
- Easy to get out of sync
- Unclear which one is "correct"

**Proposed Fix**:
Fetch once, convert once:
```python
# Single fetch
recent_agent_memories = await self.persistence.get_recent_memories(...)
recent_memory_strings = [m.content for m in recent_agent_memories]
```

**Effort**: 30 minutes

---

### Issue A2: Perception Builder Parses Actions for Messages (perception.py:144-154)

**Problem**: Messages extracted from `recent_actions` parameter at perception-building time

**Why This Is Wrong**:
1. Messages are ALREADY in memory stream (after our fix!)
2. Perception builder shouldn't know about actions (separation of concerns)
3. Only shows last-tick messages (older messages lost)
4. Stanford pattern: messages are memories, retrieved via memory strategy

**Current Flow** (bad):
```
Tick N: Isabella → Ayesha message
├─ Stored as sender memory
├─ Stored as recipient memory (after fix)
└─ Stored in actions table

Tick N+1: Build Ayesha's perception
├─ Fetch actions from tick N
├─ Parse actions for messages
└─ Show only tick-N messages (not tick N-1, N-2, etc.)
```

**Better Flow**:
```
Tick N: Isabella → Ayesha message
├─ Stored as sender memory
└─ Stored as recipient memory

Tick N+1: Build Ayesha's perception
└─ Memory strategy retrieves relevant communication memories
    (could be from tick N, N-1, N-2 based on recency/importance)
```

**Proposed Fix**:
1. Remove `recent_actions` parameter from `build_agent_perception()`
2. Messages come from `recent_observations` (already populated from memory)
3. Simplify perception.py significantly

**Effort**: 1-2 hours (includes updating orchestrator call sites)

---

### Issue A3: SimpleMemoryStream Keyword Matching Too Weak (memory.py:318-364)

**Problem**: `get_relevant_memories()` uses crude substring matching

**Example Failure**:
```python
query = "party"
matches = ["third party vendor", "Valentine's party", "party supplies"]
# All match equally despite different relevance!
```

**Impact**:
- Poor retrieval quality for complex queries
- Users forced to write custom strategies (like Valentine's notebook)
- Doesn't match Stanford paper quality

**Current Workaround**: Valentine's notebook implements custom EmbeddingMemoryStream

**Proposed Fixes** (pick one):
A. **Upgrade SimpleMemoryStream** with fuzzy matching / stemming
B. **Promote ImportanceWeightedMemory** as recommended default
C. **Document limitation** and show embedding example

**Recommendation**: Option B + C
- Make ImportanceWeightedMemory the documented default
- Add fuzzy matching in future release
- Keep SimpleMemoryStream for testing only

**Effort**: 2-4 hours

---

### Issue A4: Messages Stored in 3 Places (Conceptual Confusion)

**Problem**: Communication data duplicated across:
1. `actions` table (via save_actions)
2. Sender memory (via add_memory)
3. Recipient memory (via add_memory - after fix)

**Question**: Do we need messages in actions table at all?

**Stanford Pattern**: Messages are just memories. No separate action storage.

**Proposed Investigation**:
- Check if anything uses actions table for message retrieval
- If not, consider dropping communication from action persistence
- Or document the two-track system clearly

**Effort**: Investigation 1 hour, fix TBD

---

## 🟡 OLD: Depr ecated Issue Descriptions (For Reference)

### Status
**ARCHIVED** - Replaced by fix documentation above

### Description (OLD)
Valentine's Day party scenario shows 0/5 agents becoming aware of the party, despite Isabella successfully sending invitation messages.

## 📋 NEW: Next Steps (Prioritized)

### Immediate (This Session - COMPLETED)

1. ✅ **Fix Information Diffusion Bug**
   - Added recipient memory creation
   - Created tests (`test_information_diffusion.py`)
   - All 27 tests passing

### Short-term (Next 1-2 Days)

2. **Test Valentine's Scenario** 🎯
   - Run `examples/valentines_party.ipynb` with fix
   - Verify agents become aware of party
   - Document expected vs actual awareness rates

3. **Eliminate Dual Memory Retrieval** (Issue A1)
   - Single fetch in orchestrator
   - Convert to strings once
   - Update all call sites

4. **Add Perception Logging** (DEBUG_PERCEPTION mode)
   - Show what memories retrieved
   - Show what messages in perception
   - Parallel to DEBUG_LLM for troubleshooting

5. **Enhanced Logging & Observability** (DX Enhancement)
   - **Color-coded output**: Distinguish deterministic vs LLM calls
     - Blue: Deterministic (physics, perception)
     - Yellow: LLM calls (executor, planner, reflection)
     - Red: LLM retries/schema validation
     - Green: Success/completion
   - **Verbose mode**: Show action reasoning, communication content, plan steps
   - **Better demo UX**: Users can verify experiments with human eyes
   - **Effort**: 2-3 hours
   - **Benefit**: Much easier to debug and understand what's happening

### Medium-term (Next Week)

6. **Remove Action-based Message Filtering** (Issue A2)
   - Simplify perception.py
   - Messages from memory only
   - Update orchestrator

7. **Improve Memory Retrieval** (Issue A3)
   - Document ImportanceWeightedMemory as recommended default
   - Add embedding example to docs
   - Consider fuzzy matching for future

8. **Document Communication Architecture** (Issue A4)
   - Clarify why messages in both actions and memories
   - Update USAGE.md with communication patterns
   - Add "Building Social Scenarios" guide

### Long-term (Post-Fix)

9. **Stanford Comparison Study**
   - Run Valentine's with different memory strategies
   - Compare awareness rates with Stanford paper
   - Document gaps and parity

10. **Performance Optimization**
   - Benchmark memory retrieval strategies
   - Profile LLM call patterns
   - Consider caching/batching

---

## 🗃️ ARCHIVED: Old Investigation Notes

These sections kept for historical reference but replaced by fix documentation above.

---

## 🟡 Medium Priority: LLM Non-Determinism

### Status
**DOCUMENTED** - Expected behavior, may need mitigation

### Description
Same prompts produce different agent behaviors across runs due to LLM sampling.

### Impact
- Test results vary between runs
- Isabella sometimes chooses `communicate`, sometimes `work`
- Makes debugging and validation difficult

### Evidence
- Run 1 (no DEBUG_LLM): Isabella chose `work` actions → 0/5 awareness
- Run 2 (with DEBUG_LLM): Isabella chose `communicate` actions → still 0/5 (but different path)

### Proposed Solutions

**Option A: Add temperature=0 for deterministic mode**
- Modify LLM calls to support temperature parameter
- Allow users to set deterministic behavior for testing
- Document trade-off (less creative but reproducible)

**Option B: Multiple test runs with success threshold**
- Run scenario 10 times, check if ≥30% show information diffusion
- Statistical validation instead of single-run expectation
- Better captures emergent behavior

**Option C: Seed-based reproducibility**
- If LLM provider supports seeds, use them
- Document that tests are provider-specific
- Accept some variability as feature, not bug

### Recommendation
Option A for testing, Option C for production. Document non-determinism as expected.

---

## 🟡 Medium Priority: Stanford Comparison Gaps

### Status
**NEEDS INVESTIGATION** - Behavior doesn't match expectations from paper

### Observations

1. **Plan Execution Too Rigid**
   - Agents follow multi-step plans sequentially
   - Don't react dynamically to new information (e.g., receiving invitation)
   - Stanford agents seem more reactive/opportunistic

2. **Memory Retrieval May Be Too Weak**
   - Even with 3-factor scoring (recency + importance + relevance), messages not surfacing
   - Stanford's system clearly got this working
   - May need to tune scoring weights or thresholds

3. **Reflection Not Triggering Information Update**
   - Agents reflect on their actions but don't reflect on received messages
   - Should reflection engine explicitly process communication memories?

### Questions for Further Investigation

1. **How does Stanford handle message delivery?**
   - Do messages create immediate perception updates?
   - Or are they queued for next tick's perception?
   - What memory_type do they use?

2. **How important is the retrieval scoring?**
   - What weights do Stanford use for recency/importance/relevance?
   - Do they have minimum thresholds?
   - How many memories do they retrieve per query?

3. **Should executors see messages directly?**
   - Currently messages go to memory, then retrieved via perception
   - Should there be a "pending messages" queue that's always shown?
   - Stanford paper unclear on this

### Action Items

1. Re-read Stanford paper section on "Information Diffusion"
2. Check if Reverie code is open source for reference
3. Test with different memory retrieval strategies
4. Consider adding "message queue" separate from episodic memory

---

## 🟢 Low Priority: Code Cleanup

### Status
**DEFERRED** - Can wait until core issues resolved

### Items

1. **Temporary Test Files**
   - `test_valentines_notebook.py` - Move to examples/ or delete
   - `test_simple_communicate.py` - Delete (was for debugging)
   - `test_welcome_notebook.py` - Move to examples/ or delete

2. **Debug Logs**
   - `/tmp/debug_llm_full.log` - Archive or delete
   - `/tmp/debug_valentines_v2.log` - Archive or delete
   - `/tmp/valentines_fixed.log` - Delete
   - `/tmp/workshop_debug.log` - Delete

3. **Duplicate Documentation**
   - Multiple session summaries - consolidate
   - Analysis docs - keep ROOT_CAUSE_ANALYSIS.md, archive rest

4. **Example Consolidation**
   - `examples/valentines_party.ipynb` - Needs to incorporate findings
   - `examples/welcome.ipynb` - Verify still works
   - `examples/tutorial.ipynb` - Verify still works

---

## 📋 Next Steps (Prioritized)

### Immediate (This Week)

1. **Investigate Memory/Perception Issue** 🔴
   - Add logging to perception builder
   - Verify message storage format
   - Test with SimpleMemoryStream
   - Fix information diffusion

2. **Update Examples** 📝
   - Fix `valentines_party.ipynb` based on findings
   - Add notes about current limitations
   - Ensure deterministic examples work

3. **Document Current State** 📚
   - Update README with known issues
   - Add troubleshooting section
   - Set expectations for information diffusion

### Short-term (Next 2 Weeks)

4. **Stanford Comparison Study** 🔬
   - Deep dive into Reverie architecture
   - Identify exact gaps in our implementation
   - Prioritize features for parity

5. **Add Deterministic Testing Mode** 🧪
   - Temperature=0 option
   - Reproducible test scenarios
   - Better validation framework

6. **Memory Retrieval Improvements** 🧠
   - Tune scoring weights
   - Add debug logging
   - Consider alternative strategies

### Long-term (Month+)

7. **Advanced Features** 🚀
   - Branching/Loom for scenario exploration
   - Better reflection triggers
   - Hierarchical planning

8. **Performance Optimization** ⚡
   - Batch LLM calls
   - Cache embeddings
   - Parallel agent processing

9. **Production Readiness** 📦
   - Comprehensive test suite
   - Error handling
   - API documentation

---

## 🔍 Open Questions

1. **Why aren't messages appearing in perception?**
   - Storage format issue?
   - Retrieval query issue?
   - Memory type mismatch?

2. **Should we change the message delivery model?**
   - Direct queue vs episodic memory?
   - Immediate vs next-tick delivery?
   - Push vs pull notification?

3. **How much Stanford parity do we need?**
   - Is information diffusion essential for v0.1?
   - Can we ship with known limitations?
   - What's the minimum viable feature set?

4. **Should agents be more reactive?**
   - Current: Plan-driven, sequential execution
   - Stanford: Opportunistic, reactive to environment
   - Is this a planner issue or executor issue?

---

## 📞 Getting Help

If investigating these issues:

1. **Use DEBUG_LLM**: `DEBUG_LLM=true` shows full LLM context
2. **Check logs**: All findings documented in ROOT_CAUSE_ANALYSIS.md
3. **Test with simple scenarios**: Isolate variables
4. **Compare with workshop example**: Known working communication

---

**Document Owner**: Kenneth + Claude
**Next Review**: After memory/perception investigation
**Priority**: Fix information diffusion before shipping

