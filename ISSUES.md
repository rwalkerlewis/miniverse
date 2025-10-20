# Outstanding Issues

Last Updated: 2025-10-20

This file lists only unresolved items aligned with the current architecture.

## (Completed items removed for brevity)

## OUTSTANDING ISSUES

### A3: Memory Retrieval Quality

Problem: `SimpleMemoryStream.get_relevant_memories()` uses naive substring matching.

Impact:
- Poor ranking for semantically similar queries
- Users forced to write custom retrieval for quality scenarios

Plan:
- Short-term: Document `ImportanceWeightedMemory` as recommended default; provide an example adapter for embedding-based retrieval in docs
- Mid-term: Implement a Stanford-style memory adapter (embedding + importance + recency) behind a common `MemoryStrategy` interface
- Future: add fuzzy/BM25 hybrid retrieval as optional package

Effort: 2–4 hours for docs/examples; more for algorithmic upgrade.

---


### A8: Logging UX - Information Scattered Across Phases

**Problem:** Current logging output is difficult to follow because information about a single agent's decision-making is scattered across multiple phases. Users must mentally reconstruct what happened to each agent by reading interleaved logs from all agents.

**Impact:**
- 888KB log files with 5% signal-to-noise ratio
- Agent decisions spread across 450+ lines, interleaved with other agents
- 200+ lines of JSON per agent per tick (8,000+ lines total for 5 agents × 8 ticks)
- Key information (message content) buried in JSON, not shown in summary
- No clear agent boundaries - hard to follow individual agent flow

**Plan:**
1. **Phase 1 (P0 - Immediate):** Add verbosity levels via `MINIVERSE_LOG_LEVEL=0|1|2|3`
   - MINIMAL (0): ~50 lines per tick (97% reduction)
   - SUMMARY (1): ~300 lines per tick (80% reduction) 
   - DETAILED (2): Current output (~1,500 lines per tick)
   - DEBUG (3): Full LLM prompts (~3,000 lines per tick)
2. **Phase 2 (P1 - Short-term):** Agent-centric grouping
   - Visual separators between agents
   - Show full message content (not truncated)
   - Group each agent's full tick together
3. **Phase 3 (P2 - Medium-term):** JSON logging mode for analytics
4. **Phase 4 (P3 - Long-term):** Web dashboard for real-time visualization

**Effort:** Phase 1: 1-2 days; Phase 2: 2-3 days; Phases 3-4: Future

---

---


## Test/Build Status

- Unit/integration tests: 39 passing
- New tests assert truthful logging tags: `[LLM]` only on real LLM branches, `[•]` on deterministic branches

---

Document Owner: Kenneth
Next Review: After A3/A8/A11 changes land

---

## Acceptance Criteria per Item

### ✅ COMPLETED
- A1: ✅ Orchestrator uses one memory fetch; perception receives strings derived from the same objects.
- A2: ✅ Perception shows messages from memory only; removing `recent_actions` does not regress examples.

### OUTSTANDING
- A3: Docs recommend `ImportanceWeightedMemory`; example adapter for embeddings exists.
- A8: Log output is readable at multiple verbosity levels; agent information is grouped together; message content is visible in summaries.

