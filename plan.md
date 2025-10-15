# Miniverse Project Plan

_Last updated: 2025-10-14_

## 1. Current Snapshot

- `miniverse/`: Library modules (orchestrator, schemas, cognition, memory, persistence, environment helpers).
- `examples/`: Demonstrations
  - `workshop/`: deterministic + LLM maintenance loop (`--llm`, `--debug`, `--analysis`). ✅ Working
  - `standup/`: conversation-heavy stand-up with structured `communication` payloads. ✅ Working
  - `stanford/`: Tier-2 grid-based Valentine's Day party scenario (Stanford replication). 🚧 In progress
  - `_legacy/`: archived prototypes kept for reference.
- `tests/`: 23-unit pytest suite covering cognition flow, schemas, scenario loading, persistence, environment helpers, etc.
- `docs/`: Usage guides, research notes, example roadmap.
- `scripts/`: Utility CLI helpers (lint/test stubs).
- Root files: `README.md`, `CLAUDE.md`, `pyproject.toml`, `uv.lock`, `.env.example`.

## 2. Simulation Loop & Time Model

- A **tick** is the fundamental time step. The orchestrator increments `WorldState.tick` each loop and applies deterministic physics before cognition.
- Scenarios decide how ticks map to real time (minute, hour, day, etc.). Prompts should describe that mapping explicitly; the library does not hard-code “daily” behaviour.
- Planners run once per tick today. Scheduling (e.g., run planner every N ticks or at simulated timestamps) is deferred to upcoming cognition work.
- Scratchpads track plan progress; executors receive the active `PlanStep`. Using `--debug` on any example prints the computed plan, selected step, and final `AgentAction` JSON so users can audit the flow.

## 3. Implemented Capabilities

- **Cognition modules**: Protocols for Planner / Executor / Reflection, with deterministic defaults and LLM-backed options via Mirascope + Tenacity.
- **Tick listeners**: `Orchestrator` accepts optional callbacks (e.g., `TickAnalyzer`) for post-tick analytics/transcripts.
- **Environment helpers**: Graph/grid state schemas plus occupancy and pathfinding utilities.
- **Persistence & memory**: Injected strategies (in-memory by default); `SimpleMemoryStream` stores ordered memories for perception/prompts.
- **Examples**: Workshop (physics-oriented) and Stand-up (social conversation) cover both KPI updates and communication logging.

## 4. Recent Progress (Session 2025-10-14)

### ✅ Completed
1. **Structured LLM error feedback** – ValidationFeedback dataclass with rich error details, field paths, types, input previews (miniverse/llm_utils.py)
2. **Planner scheduling & time blocks** – CognitionCadence with TickInterval for configurable planner/reflection execution (miniverse/cognition/cadence.py)
3. **Memory metadata & retrieval** – Extended AgentMemory with tags/metadata, ImportanceWeightedMemory with recency+importance scoring (miniverse/memory.py)
4. **Environment tier polish** – ✅ Complete:
   - Grid position tracking in AgentStatus (List[int] for OpenAI schema compatibility)
   - Move validation helpers (validate_grid_move, validate_graph_move)
   - BFS pathfinding with collision detection
   - Scenario loader parses grid positions
5. **Stanford scenario foundation** – Created simplified Valentine's Day party scenario with 3 agents on 10x10 grid (examples/scenarios/stanford.json)

### 🚧 In Progress
6. **Stanford demo execution** – Runner script created but **hanging on initialization**. Issue identified: LLM layer problem during orchestrator setup or first tick. Workshop/standup examples work fine (deterministic mode completes instantly), so issue is specific to stanford runner setup.

### 📋 Outstanding Work
7. **Debug Stanford LLM layer issue** – PRIORITY: Orchestrator hangs even in deterministic mode. Needs investigation:
   - Check if SimplePlanner/SimpleExecutor/SimpleReflectionEngine have bugs
   - Compare stanford runner initialization vs workshop/standup
   - Test minimal orchestrator with grid scenario
8. **Docsite & tutorials** – Expand docs into structured guide ("build your first sim", advanced prompts, custom memory/persistence)
9. **Tooling** – Add CLI/testing conveniences (lint config, benchmarking harness for cognition latency)
10. **Notebooks** – Publish Jupyter notebooks (e.g., stand-up conversation walkthrough)

## 5. Release Readiness (PyPI)

Before publishing `miniverse` to PyPI:

1. **Packaging**: Ensure `pyproject.toml` includes metadata (version, description, classifiers, license) and entry points (if any). Add MANIFEST if extra files needed.
2. **Versioning & changelog**: Adopt semantic versioning (`0.1.0`+) and start `CHANGELOG.md`.
3. **Documentation**: Provide a discoverable “Getting Started” in README plus deep links to `docs/`. Consider ReadTheDocs or mkdocs for hosted docs.
4. **Testing/CI**: Configure GitHub Actions (or similar) to run `uv run pytest` (with LLMs mocked) and packaging checks (`uv build`, `twine check`).
5. **Licensing**: Confirm license headers in source files and include LICENSE in distribution.
6. **Examples**: Tag examples as optional extras (document credentials, deterministic fallback) so the package installs cleanly without API keys.
7. **Final polish**: Review TODOs, remove experimental files, and confirm defaults (e.g., logging verbosity) are user-friendly for first-time adopters.

## 6. Quick Reference

- Run workshop: `UV_CACHE_DIR=.uv-cache uv run python -m examples.workshop.run --ticks 5` ✅
- Run stand-up: `UV_CACHE_DIR=.uv-cache uv run python -m examples.standup.run --ticks 4` ✅
- Run stanford: `UV_CACHE_DIR=.uv-cache uv run python -m examples.stanford.run --ticks 3 --analysis` 🚧 (hangs)
- Enable LLM path: add `--llm` (requires `LLM_PROVIDER`, `LLM_MODEL`, provider API key)
- Debug cognition: add `--debug` to print planner/executor/reflection payloads
- Tests: `UV_CACHE_DIR=.uv-cache uv run pytest` ✅ (23 tests passing)

## 7. Atomic Commits (Session 2025-10-14)

- `e93426b` - Enhance inline documentation across core modules
- `5dfb0aa` - Add structured LLM validation feedback, cognition cadence, and memory enhancements
- `8e02cdd` - Add grid position tracking for Tier-2 spatial environments
- `df781ae` - Add move validation helpers for Tier-1 and Tier-2 environments
- `e50fbed` - Add Stanford Valentine's Day party scenario (Tier-2 grid)

**Uncommitted changes**: Stanford runner script (examples/stanford/run.py, rules.py) and grid_position schema fix (List[int] instead of Tuple[int, int])

## 8. Next Steps (Priority Order)

1. **Debug Stanford runner hang** – Root cause: likely SimplePlanner/SimpleExecutor returning invalid data or infinite loop. Action: Add print statements, compare with workshop deterministic path.
2. **Fix grid_position schema** – Commit List[int] change (OpenAI compatibility fix)
3. **Complete Stanford demo** – Get deterministic mode working first, then test LLM mode
4. **PyPI prep** – Once Stanford validates architecture, proceed with packaging/release

## 9. Contacts

Current maintainer: Codex agent (2025-03-16). Reference implementations for research are stored under `reference-work/` (gitignored).
