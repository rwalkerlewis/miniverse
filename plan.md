# Miniverse Project Plan

_Last updated: 2025-03-16_

## 1. Current Snapshot

- `miniverse/`: Library modules (orchestrator, schemas, cognition, memory, persistence, environment helpers).
- `examples/`: Demonstrations
  - `workshop/`: deterministic + LLM maintenance loop (`--llm`, `--debug`, `--analysis`).
  - `standup/`: conversation-heavy stand-up with structured `communication` payloads.
  - `_legacy/`: archived prototypes kept for reference.
- `tests/`: 19-unit pytest suite covering cognition flow, schemas, scenario loading, persistence, etc.
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

## 4. Outstanding Work

1. **Structured LLM error feedback** – Surface validation failures (missing fields, wrong types) directly in retry prompts and user logs so malformed JSON is easier to correct.
2. **Planner scheduling & time blocks** – Add utilities to run planners/reflections on configurable intervals (e.g., once per “day” or when scratchpad is empty) and expose helpers for scenarios to declare their tick → time mapping.
3. **Memory metadata & retrieval** – ✅ Simple/importance-weighted streams now capture tags/metadata and expose weighted retrieval. Next step: plug in pluggable embedding/BM25 adapters and expose memory clearing on persistence backends.
4. **Environment tier polish** – Finish Tier‑1/ Tier‑2 helpers (capacity enforcement, pathfinding) and ensure scenario loader ergonomics for spatial maps.
5. **Advanced example** – Reproduce the Stanford Valentine’s Day scenario (Tier‑2 grid, richer memory usage) as a proof point for emergent coordination.
6. **Docsite & tutorials** – Expand docs into a structured guide (“build your first sim”, advanced prompts, custom memory/persistence) with cross-links to examples.
7. **Tooling** – Add CLI/testing conveniences (lint config, benchmarking harness for cognition latency).
8. **Notebooks** – Publish Jupyter notebooks (e.g., stand-up conversation walkthrough) to demonstrate interactive usage without reading full scripts.

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

- Run workshop: `UV_CACHE_DIR=.uv-cache uv run python -m examples.workshop.run --ticks 5`
- Run stand-up: `UV_CACHE_DIR=.uv-cache uv run python -m examples.standup.run --ticks 4`
- Enable LLM path: add `--llm` (requires `LLM_PROVIDER`, `LLM_MODEL`, provider API key).
- Debug cognition: add `--debug` to print planner/executor/reflection payloads.
- Tests: `UV_CACHE_DIR=.uv-cache uv run pytest`

## 7. Contacts

Current maintainer: Codex agent (2025-03-16). Reference implementations for research are stored under `reference-work/` (gitignored).
