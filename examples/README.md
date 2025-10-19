# Miniverse Examples

This directory contains examples and tutorials for learning Miniverse.

## 🚀 Start Here

Prefer the script examples under `examples/workshop/` for up-to-date flows. Legacy notebooks may lag behind the latest API.

## Tutorial Notebooks

**`tutorial.ipynb`** - Reference guide for primitives and implementation details (Stat, Plan, Memory, etc.).

### Setup for Notebook

1. **Install notebook dependencies:**
   ```bash
   uv pip install -r examples/requirements-notebook.txt
   ```

2. **Install Miniverse in editable mode:**
   ```bash
   uv pip install -e .
   ```

3. **Register the Jupyter kernel:**
   ```bash
   uv run python -m ipykernel install --user --name=miniverse --display-name="Miniverse (uv)"
   ```

4. **Open in VS Code or Jupyter:**
   - VS Code: Open `tutorial.ipynb` and select "Miniverse (uv)" kernel
   - Jupyter: Run `jupyter notebook` and open `tutorial.ipynb`

### What the Tutorial Covers

- **Part 1:** Core data structures (Stat, AgentProfile, WorldState, Plan, etc.)
- **Part 2:** Cognition modules (Executor, Planner, ReflectionEngine)
- **Part 3:** First simulation (putting it all together)
- **Part 4:** LLM intelligence (optional, requires API keys)
- **Part 5:** Memory and persistence
- **Part 6:** Planning (multi-step reasoning)
- **Part 7:** Reflection (learning from experience)

Each section is executable with detailed explanations and outputs.

## Workshop Examples (Progressive learning)

The `workshop/` directory contains 5 progressive examples showing increasing complexity:

### 01_hello_world - The Basics
```bash
uv run python -m examples.workshop.01_hello_world.run
```
One agent, hardcoded "always work" logic. Demonstrates core simulation loop.

### 02_deterministic - Threshold Logic
```bash
uv run python -m examples.workshop.02_deterministic.run
```
Two workers with if/then decision logic based on energy and backlog thresholds.

### 03_llm_single - Reactive AI (requires LLM)
```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4
export OPENAI_API_KEY=your_key
uv run python -m examples.workshop.03_llm_single.run
```
Single LLM agent making intelligent, context-aware decisions.

### 04_team_chat - Multi-Agent Communication (requires LLM)
```bash
uv run python -m examples.workshop.04_team_chat.run
```
Team of 3 LLM agents coordinating via natural language communication.

### 05_stochastic - Random Events + LLM Adaptation (requires LLM)
```bash
uv run python -m examples.workshop.05_stochastic.run
```
Stochastic physics (random task arrivals, equipment breakdowns) with intelligent LLM adaptation.

See `workshop/README.md` for detailed explanations of each example.

## Which to use?

- **New to Miniverse?** → Start with workshop examples 01-05
- **Want to see complete simulations?** → Run workshop examples
- **Building your own simulation?** → Read `docs/USAGE.md`, then adapt workshop examples

## LLM Configuration (Optional)

For LLM-powered examples (03-05), configure your provider:

```bash
# OpenAI
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4
export OPENAI_API_KEY=your_key

# Anthropic
export LLM_PROVIDER=anthropic
export LLM_MODEL=claude-3-5-sonnet-20241022
export ANTHROPIC_API_KEY=your_key
```

Deterministic examples (01-02) and most of the tutorial work without LLM configuration.
