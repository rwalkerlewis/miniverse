# Valentine's Day Party Test - Quick Guide

This directory contains a working demonstration of Stanford Generative Agents-style information diffusion. Specifically, we construct a micro replication of the experiment focused on the Valentine's Day party scenario where agents spread information (party invites) through their social network.

## 📁 Files

### Main Script
**`valentines_party.py`** - Python script that runs the scenario
- 5 agents (Isabella, Maria, Klaus, Ayesha, Tom)
- 8 ticks (Feb 13 9am → Feb 14 5pm)
- Tests information diffusion through social network

### Notebooks
**`valentines_party.ipynb`** - Original Jupyter notebook (uses embeddings)
- More complex setup with EmbeddingMemoryStream
- Same scenario, more advanced implementation

## 🚀 How to Run

```bash
# Basic run
uv run python examples/valentines_party.py

# With memory debugging (recommended)
DEBUG_MEMORY=true uv run python examples/valentines_party.py

# Full debugging (shows all LLM prompts)

MINIVERSE_VERBOSE=true DEBUG_LLM=true DEBUG_MEMORY=true uv run python examples/valentines_party.py
```

## 🔧 Key Features

- LLM-driven social coordination
- Memory-based information propagation
- Multi-source awareness (redundant social signals)
- Role-consistent emergent behavior (journalist coverage, romantic subplot)
- No hardcoded coordination - all emergent from agent prompts + memories