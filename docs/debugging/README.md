# Debugging Session Archive

This directory contains detailed debugging documentation from the 2025-10-16 session investigating information diffusion failure in the Valentine's Day party scenario.

## Key Files

### ROOT_CAUSE_ANALYSIS.md
Complete forensic analysis of the communication message format issue. Shows how DEBUG_LLM logging revealed that the `execute_tick` prompt template lacked a communicate action example, causing LLMs to generate malformed message objects.

**Resolution**: Added communicate action example to `miniverse/cognition/prompts.py`

### SESSION_2025-10-16_SUMMARY.md
Comprehensive session overview including:
- Investigation process (5 phases)
- Findings and evidence
- Fix implementation details
- Lessons learned

### OUTSTANDING_ISSUES.md (moved to root as ISSUES.md)
Centralized issue tracker with all findings, next steps, and open questions. **This is the current source of truth.**

## Current Status

✅ **Fixed**: Message format issue (communicate actions now have proper `{"to": "...", "message": "..."}` structure)

❌ **Still Broken**: Information diffusion (messages send but don't propagate to other agents)

## Fail-Fast Agent Actions (2025-10-18)

Agent action collection now fails fast instead of silently substituting a default rest action when an agent fails. If any agent errors on a tick, the run aborts with an informative error summarizing failing agents and remediation tips (check LLM configuration, enable DEBUG flags).

Future work (not yet implemented): configurable fallback policy (strict_fail, noop/observe, rest, custom callback). For now, failures are surfaced immediately to avoid misleading results.

🔍 **Under Investigation**: Memory/perception system - why aren't messages appearing in other agents' perceptions?

## How to Debug LLM Issues

1. **Enable DEBUG_LLM**: `DEBUG_LLM=true` when running simulations
2. **Check prompts**: Verify what context agents actually receive
3. **Check responses**: Confirm LLM outputs match expected schema
4. **Compare runs**: LLMs are non-deterministic, run multiple times
5. **Create minimal tests**: Isolate the issue with simple reproduction cases

## Lessons Learned

1. **Always provide complete few-shot examples** - Don't assume LLM generalization
2. **DEBUG_LLM is essential** - Can't debug LLM issues without seeing prompts/responses
3. **Schema comments ≠ LLM guidance** - Examples in prompts, not code comments
4. **Structured outputs need examples too** - Pydantic validates format but doesn't guide content
5. **Test with simple scenarios first** - Complexity hides root causes

## Next Investigation

See `/ISSUES.md` in repository root for current priorities.

