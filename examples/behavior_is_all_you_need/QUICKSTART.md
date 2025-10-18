# Quick Start Guide

Get up and running with the "Behavior is all you need" example in 5 minutes.

## 1. Install Dependencies

```bash
cd /Users/kenneth/Desktop/lab/miniverse
uv sync
```

## 2. Run Your First Simulation

```bash
cd examples/behavior_is_all_you_need
uv run python run.py --ticks 12 --debug
```

**What you'll see:**
- 6 agents with different personalities
- 12 ticks (~4 game phases: day → voting → night → day...)
- Personality descriptions for each agent
- Game events (accusations, eliminations, emotional reactions)
- Final emotional states and motivations

## 3. Run the Personality Experiment

```bash
cd experiments
uv run python personality_test.py
```

**What this does:**
- Runs 3 trials with different personality types
- Measures behavioral differences (accusations, support, defense, silence)
- Validates personality → behavior correlations
- Saves results to `experiments/results/`

**Expected output:**
```
Running trial: anxious_villager
  Personality: reserved, trusting, emotionally reactive...

Running trial: confident_leader
  Personality: outgoing, confident, assertive...

Running trial: manipulative_deceiver
  Personality: creative, competitive, low empathy...

PERSONALITY CONSISTENCY ANALYSIS
  anxious_villager:    Accuse: 2, Support: 4, Defend: 5, Quiet: 4
  confident_leader:    Accuse: 7, Support: 3, Defend: 2, Quiet: 1
  manipulative_deceiver: Accuse: 8, Support: 1, Defend: 2, Quiet: 2
```

## 4. Analyze Results

```bash
uv run python analysis.py
```

**What you'll see:**
- Correlation coefficients between personality traits and actions
- Emotional trajectory analysis
- Validation of expected patterns (e.g., Extraversion → Accuse)

## 5. Visualize (Optional)

```bash
cd ../notebooks
jupyter notebook personality_analysis.ipynb
```

**Visualizations:**
- Big Five trait profiles (bar charts)
- Action frequency comparison (bar charts)
- Personality-action correlation heatmap
- Emotional states in valence-arousal space

## What Each Module Does

| Module | What It Does |
|--------|--------------|
| `personality.py` | Big Five traits → action weights |
| `emotion.py` | Emotional states → transient behavior modulation |
| `needs.py` | Maslow hierarchy → goal-directed priorities |
| `rules.py` | Social deception game mechanics |
| `cognition.py` | Ties everything together: Action = P × E × N |
| `run.py` | Main simulation loop |

## Understanding the Output

### During Simulation

```
=================================================================
Starting simulation: 12 ticks (~4 game phases)
=================================================================

Tick 1 (Day Phase):
  Alice (anxious): Feeling slightly fearful; staying quiet
  Bob (confident): Feeling neutral; accusing Charlie
  Charlie (analytical): Investigating Bob's behavior
  ...

Tick 4 (Voting Phase):
  Votes tallied: Charlie eliminated (revealed: Seer)
  Emotional reactions: Alice → sadness, Bob → joy, ...

Tick 7 (Night Phase):
  Werewolves choose victim: Alice eliminated
  Diana → intense fear (survival threatened)
  ...

GAME OVER! Winner: werewolves
```

### Personality Consistency Test

```
PERSONALITY CONSISTENCY ANALYSIS

Action Frequency Comparison:
Personality              Accuse  Support  Defend   Quiet
-----------------------------------------------------------
anxious_villager              2        4       5       4
confident_leader              7        3       2       1
manipulative_deceiver         8        1       2       2

✓ Personality-Behavior Validation:
  anxious_villager: ✓ (Expected: High defend/quiet, low accuse)
  confident_leader: ✓ (Expected: High accuse, low quiet/defend)
  manipulative_deceiver: ✓ (Expected: High accuse, low support)
```

## Key Insights to Look For

1. **Personality Consistency**: Same personality type → similar action patterns across runs
2. **Emotional Reactivity**: High neuroticism agents show more emotional volatility
3. **Trait-Action Correlations**:
   - High Extraversion → more accusations
   - High Agreeableness → more support
   - High Neuroticism → more defense
4. **Need-Driven Behavior**: Low safety → defensive actions, low belonging → alliance-seeking

## Troubleshooting

### "No module named 'miniverse'"
```bash
# Make sure you're in the miniverse directory
cd /Users/kenneth/Desktop/lab/miniverse
uv sync
```

### "Results directory not found"
```bash
# Run experiments first
cd examples/behavior_is_all_you_need/experiments
uv run python personality_test.py
```

### Want to use LLM?
```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4
export OPENAI_API_KEY=your_key

uv run python run.py --llm --ticks 12
```

### Seeing warnings about missing data?
```
⚠️  Personality data missing from scratchpad!
⚠️  Emotional state missing from scratchpad!
```

**This means agents aren't properly initialized.** The simulation will still run (using neutral defaults), but behavior won't be personality-driven.

**To test with strict validation:**
```bash
export STRICT_MODE=true
uv run python run.py --ticks 12
# Now raises errors instead of warnings if data is missing
```

## Next Steps

1. **Modify personalities** in `scenario.json` (change Big Five scores 0-100)
2. **Add new emotions** in `emotion.py` (extend `DiscreteEmotion` enum)
3. **Add new needs** in `needs.py` (extend `NeedType` and hierarchy)
4. **Create new scenarios** (copy scenario.json, change roles/agents/personalities)
5. **Run longer simulations** (`--ticks 30` for more game phases)

## Questions?

- **Paper**: See `Whitepaper.pdf` or [online](https://www.egoai.com/research/Whitepaper.pdf)
- **Documentation**: See `README.md` for detailed explanations
- **Implementation**: See `IMPLEMENTATION_SUMMARY.md` for technical details
- **Miniverse**: See main repo docs in `/docs/` directory

---

**Have fun exploring personality-driven behavior!** 🎮🧠
