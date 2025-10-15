# Stanford Generative Agents vs. Miniverse

## Purpose of This Brief

We want to deliver a generalist social-simulation library that can replicate the Stanford Generative Agents experiment while remaining modular enough for wildly different scenarios (distributed teams, factory floors, Mars habitats, etc.). This note compares Stanford's open-sourced Reverie stack to our current codebase, highlights what's working, and calls out gaps and risks. The final section proposes high-level next steps before we green-light detailed implementation plans.

---

## Architectural Snapshot

| Concern | Stanford (Reverie) | Miniverse Today | Assessment |
|---------|--------------------|-----------------|------------|
| **Execution Spine** | Persona class runs perception → retrieve → plan → execute → reflect inside a monolithic loop, directly mutating world state. | `Orchestrator` coordinates ticks with clear dependency injection (rules, memory, persistence, prompts). | ✅ Our orchestrator is cleaner/modular; keep it as the backbone. |
| **World State** | Tile-based maze (Tiled exports) with semantic layers (world → sector → arena → object) and reverse indices; movement enforced via pathfinding. | Generic `WorldState` built around `Stat` blocks; deterministic rules optional; optional world LLM updates. No spatial semantics yet. | ⛔ Need semantic environment tiers plus helpers; current KPI-only state can’t replicate spatial behavior. |
| **Memory** | Associative memory graph with recency+poignancy+semantic scoring; scratchpad stores working plan state; reflection updates memory. | `SimpleMemoryStream` (recency-only) + optional persistence. No scoring, no reflection loop, no scratchpad. | ⛔ We must add associative metadata, scratchpad, reflection triggers (even if initial retrieval stays recency-based). |
| **Planning & Reflection** | Long-horizon plan generation, hourly decomposition, reflection thresholds baked into persona modules; diaries added to memory. | Agents generate single-step actions via one prompt; no explicit plans or reflections. | ⛔ Multi-stage cognition is missing; we must introduce plan/execute/reflect prompts and scratchpad state. |
| **Prompts** | Curated templates per cognition phase (plan, schedule, reflection, conversation, location rating). | Single "what do you do" prompt per agent, optional world prompt. | ⛔ Need staged prompts and richer context injection. |
| **Physics / Determinism** | Minimal deterministic rules; mostly LLM-driven but spatial constraints enforced. | `SimulationRules` interface is strong: deterministic hooks for any physics domain. | ✅ Our deterministic layer is more flexible; lean into it. |
| **Persistence / Modularity** | Custom pickled structures tightly coupled to persona code. | Persistence/memory interfaces are pluggable; orchestrator injected dependencies. | ✅ Keep DI approach; expand strategy interfaces. |
| **Front-end / Visualization** | Reverie ships a browser UI; persona loop integrated with environment server. | CLI-based examples; architecture ready for headless mode, but no front-end. | ➖ Visualization optional for now; focus on backend correctness. |
| **Branching / Loom** | Not native; experiments run serially. | Not implemented yet. | ✅ Defer until core cognition pipeline lands. |

---

## What We Do Better (Keep / Extend)

1. **Modular Orchestrator**: The clean injection pattern lets users swap memory, persistence, rules, and prompts—ideal for a generalist toolkit.
2. **Deterministic Rule Hooks**: `SimulationRules` gives us a first-class place for domain physics (queues, resources, equipment). We can add helper utilities without forcing one worldview.
3. **Schema Simplicity**: The `Stat`/`MetricsBlock` approach keeps world state flexible for data-heavy sims (org KPIs, resource dashboards) even when we add spatial tiers.
4. **Retry/Validation**: Unified Tenacity-based retry logic will help once prompts become multi-stage.

---

## Major Gaps to Close

1. **Cognition Stack**: We lack scratchpad, plan decomposition, reflection, and staged prompts. Without these, behavior stays repetitive and can’t reproduce Stanford’s emergent coordination.
2. **Memory Semantics**: Need associative metadata (tick, importance/poignancy, tags, embedding keys), reflection hooks, and retrieval pipelines—even if the first release uses recency scoring. The important bit is the data model and plug-in interface.
3. **Environment Semantics**: Provide optional tiers: abstract (current), logical graph (rooms/teams), spatial grid. Deterministic helpers should enforce constraints when tiers > 0.
4. **Prompt Library**: Build reusable templates (plan, execute, reflect, converse, diary) with config knobs so users or LLMs can script new domains quickly.
5. **Agent Lifecycle Contracts**: Formalize an agent spec that bundles profile, scratchpad, memory strategy, cognition modules. Ensure orchestrator expects these parts when registering agents.

---

## Risks / Unknowns vs. Stanford’s Proven Path

- **World LLM Dependence**: Our design still allows a world-engine LLM; Stanford avoided a second model by letting deterministic/symbolic updates drive the environment. To match them we can keep the world LLM optional and favor deterministic rules (plus reflection) during the replication phase.
- **Testing**: Stanford’s code wasn’t packaged for tests; our suite is focused on existing behavior. Once we add multi-stage cognition we need regression harnesses (mocked LLM outputs) to verify plan/execute/reflect flows.
- **Performance**: Multi-stage prompts increase latency. We should allow batching, caching, or schedule-based triggering (plans once per day, reflections on threshold) similar to Reverie to keep runs practical.
- **User Ergonomics**: We must ensure that “simple sims” don’t drown in config. The cognition modules should have defaults so you can still run a light-weight chatroom sim with minimal setup.

---

## High-Level Next Steps (Proposed)

1. **Agent Cognition Framework**
   - Spec the agent interface: profile + scratchpad + planner + reflection + memory strategy.
   - Implement scratchpad and planner stubs with simple defaults (e.g., daily plan template producing a single task list).
   - Add reflection trigger plumbing (threshold-based, even if initial logic just stores summaries).

2. **Memory Schema Upgrade**
   - Extend `AgentMemory` to include tags, importance, optional embedding keys.
   - Update `MemoryStrategy` interface to separate storage vs. retrieval hooks, enabling custom retrieval strategies without breaking the default recency implementation.
   - Provide a SimpleAssociativeMemory that records metadata and exposes retrieval slots (still recency-driven initially).

3. **Environment Tiering**
   - Define schema additions for logical environment graphs and (optional) spatial grids.
   - Supply deterministic helpers (room occupancy checks, simple path planning) tied to `SimulationRules`.

4. **Prompt Suite & Orchestrator Wiring**
   - Design the staged prompts and orchestrator hooks: plan → execute → reflect per tick, with scheduling knobs.
   - Create example templates mirroring the Stanford prompts (plans, diaries, conversations) but keep them configurable.
   - Update examples to exercise the new pipeline (e.g., retail store with daily schedules, factory shift with spatial tier).

5. **Replication Milestone**
   - Choose a Stanford scenario (Valentine’s Day party or similar) and blueprint the assets we need (personas, environment map, prompt configs).
   - Implement the minimal features to run that scenario end-to-end as validation that our architecture matches the paper’s capabilities.

6. **Documentation for Builders**
   - Produce a “How to Build a Simulation” guide covering cognition modules, environment tiers, memory plug-ins.
   - Ensure the docs are explicit enough for an LLM or human to follow without poking through source.

---

## Summary

Our orchestrator, deterministic rule hooks, and modular interfaces put us ahead for building a generalist simulation platform. To reach Stanford-level emergent behavior (and surpass it), we must add the cognition stack (scratchpad/plan/reflect), enrich memory semantics, and offer optional environment tiers. Once those pieces are in place and validated by replicating a Stanford scenario, we’ll have a foundation that supports both high-control branchable social sims and simpler KPI-driven runs.
