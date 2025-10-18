# Behavior is all you need

This example implements the framework from the paper ["Behavior is all you need"](https://www.egoai.com/research/Whitepaper.pdf) by Vishnu Hari and Connor Brennan. It demonstrates how personality, emotion, needs, and memory combine to create believable AI agents without requiring true consciousness.

## Paper Summary

The paper argues that game bots fail to feel human-like because they lack **intentional, emotionally grounded, and contextually coherent behavior**. Instead of trying to replicate consciousness, the authors propose a **"play-acting" approach** that creates the *illusion* of internal life through four key components:

1. **Personality** - Stable traits that modulate decision-making (Big Five model)
2. **Emotion** - Transient states that react to events (valence × arousal + discrete emotions)
3. **Needs** - Internal drives that prioritize goals (Maslow's hierarchy)
4. **Memory** - Past experiences that inform current decisions

The key insight: **believability emerges from behavioral consistency**, not cognitive realism. When agents act as if they have intentions, emotions, and goals, players suspend disbelief—just as audiences do with fictional characters.

## Implementation

This example implements all four components using Miniverse's modular architecture:

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Behavioral State                    │
├─────────────────────────────────────────────────────────────┤
│  Personality (Big Five)    Emotion (Hybrid)    Needs (Maslow)│
│  - Openness                - Valence/Arousal   - Safety      │
│  - Conscientiousness       - Discrete emotions - Belonging   │
│  - Extraversion            - Intensity         - Esteem      │
│  - Agreeableness                                - Growth     │
│  - Neuroticism                                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│               Action Selection (Executor)                    │
├─────────────────────────────────────────────────────────────┤
│  1. Get possible actions (game phase dependent)              │
│  2. Weight by personality (stable preferences)               │
│  3. Weight by emotion (transient modulation)                 │
│  4. Weight by needs (goal-directed priorities)               │
│  5. Select action (weighted random or LLM)                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Game Rules (Physics)                       │
├─────────────────────────────────────────────────────────────┤
│  - Process actions (voting, elimination, etc.)               │
│  - Trigger emotional events (betrayal → anger)               │
│  - Update needs (isolation → belonging deprivation)          │
│  - Advance game state (day → night → day)                    │
└─────────────────────────────────────────────────────────────┘
```

### Module Structure

| Module | Purpose | Paper Section |
|--------|---------|---------------|
| `personality.py` | Big Five trait model + action weighting | 4.3 Agentic Framework |
| `emotion.py` | Hybrid emotion system (dimensional + categorical) | 4.3 Agentic Framework |
| `needs.py` | Maslow's hierarchy + need fulfillment | 4.3 Agentic Framework |
| `rules.py` | Social deception game mechanics | 4.1 Architecture (Perception) |
| `cognition.py` | Personality-aware executor & planner | 4.1 Architecture (Behavior Model) |
| `scenario.json` | Agent profiles with Big Five traits | 4.2 Data Collection |

### Social Deception Game

The scenario is a **Werewolf/Mafia-style social deduction game**:

- **Roles**: Villagers, Werewolves, Seer
- **Day Phase**: Discussion, accusations, voting
- **Night Phase**: Werewolves eliminate, Seer investigates
- **Win Conditions**: Eliminate all werewolves OR werewolves outnumber villagers

This scenario tests the paper's claims about emergent social behavior:
- Do agents with different personalities behave differently?
- Do emotions influence decisions believably (fear → defensive behavior)?
- Do needs drive goal-directed behavior (belonging → alliance-seeking)?

## Usage

### Basic Simulation (Rule-based cognition)

```bash
cd examples/behavior_is_all_you_need
uv run python run.py --ticks 12
```

This runs 12 ticks (~4 game phases) with rule-based action selection. Actions are weighted by personality × emotion × needs.

### LLM-Enhanced Simulation

```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4
export OPENAI_API_KEY=your_key

uv run python run.py --llm --ticks 12
```

LLM mode augments prompts with personality/emotion/needs context, allowing the model to generate more nuanced, character-consistent behavior.

### Debug Mode

```bash
uv run python run.py --debug --ticks 12
```

Shows:
- Big Five trait scores for each agent
- Personality descriptions
- Final emotional states
- Final motivational states
- Behavioral consistency analysis

### Strict Mode (for testing)

```bash
export STRICT_MODE=true
uv run python run.py --ticks 12
```

In strict mode:
- **Raises errors** instead of warnings when behavioral data is missing
- Prevents simulation from running with incomplete state
- Useful for testing to ensure proper initialization
- Without strict mode, warnings are shown but simulation continues with defaults

## Experiments

### Personality Consistency Test

**Hypothesis**: Agents with different Big Five profiles exhibit measurably different behaviors.

```bash
cd experiments
uv run python personality_test.py
```

This experiment:
1. Runs the same scenario 3 times with different personality profiles:
   - **Anxious Villager** (high neuroticism, high agreeableness)
   - **Confident Leader** (high extraversion, low neuroticism)
   - **Manipulative Deceiver** (low agreeableness, high openness)
2. Measures behavioral differences:
   - Accusation frequency
   - Defense frequency
   - Silence frequency
   - Emotional volatility
3. Validates personality → behavior mapping

**Expected Results**:
- Anxious agents: More defensive, less accusatory, higher emotional volatility
- Confident agents: More leadership actions, more accusations, stable emotions
- Manipulative agents: More deception, more accusations, creative strategies

### Analysis Tools

```bash
cd experiments
uv run python analysis.py --run-id <simulation_id>
```

Analyzes a completed simulation:
- Action frequency by personality type
- Emotional trajectory over time
- Need satisfaction curves
- Personality-behavior correlations

## Validation Against Paper

| Paper Requirement | Implementation | Status |
|-------------------|----------------|--------|
| **Stable Personality** | Big Five traits (0-100 scores) | ✅ Implemented |
| **Internal/External Motivations** | Maslow's hierarchy (4 need levels) | ✅ Implemented |
| **Emotionally Modulated Behavior** | Valence×Arousal + discrete emotions | ✅ Implemented |
| **Short/Long-term Memory** | Miniverse memory system | ✅ Using existing |
| **Play-Acting (not sentience)** | Weight-based action selection | ✅ Implemented |
| **Embodied Turing Test** | Personality consistency experiment | ✅ Implemented |

### Gaps & Future Work

| Gap | Status | Priority |
|-----|--------|----------|
| Inverse Personality Model (infer traits from behavior) | Not implemented | Medium |
| Foundation model training (end-to-end behavior) | Not implemented | Low (research) |
| Advanced memory retrieval (emotion-tagged) | Partially implemented | High |
| Procedural memory (how-to knowledge) | Not implemented | Medium |
| Personality evolution over time | Not implemented | Low |

## Key Insights Demonstrated

### 1. Personality × Emotion × Needs = Emergent Behavior

See `cognition.py:_rule_based_choose_action()`:

```python
# Start with base weight
weight = 1.0

# Multiply by personality weight (stable trait influence)
weight *= PersonalityWeights.calculate_action_weight(personality, action_type)

# Multiply by emotion weight (transient state modulation)
weight *= EmotionActionInfluence.get_action_modifier(emotional_state, action_type)

# Multiply by needs weight (goal-directed priorities)
weight *= NeedActionInfluence.get_action_modifier(needs, action_type)
```

This simple multiplication creates complex, context-sensitive behavior patterns WITHOUT requiring:
- Explicit state machines for every situation
- Hardcoded personality → action mappings
- LLM calls (though they can enhance it)

### 2. Emotional Events → Behavioral Change

See `rules.py:_trigger_elimination_emotions()`:

```python
# Eliminated agent: sadness + defeat
self._trigger_event_emotion(eliminated_id, "defeat", EmotionalTriggers.DEFEAT)

# Werewolves eliminated - villagers happy
if eliminated_role == Role.WEREWOLF:
    self._trigger_event_emotion(agent_id, "ally_confirmed", EmotionalTriggers.ALLY_CONFIRMED)
```

Agents don't just *say* they're sad—their emotional state shifts, which *changes their action weights*, which *changes their behavior*. This is the "play-acting" approach: behavioral consequences flow naturally from state changes.

### 3. Needs Drive Goal Pursuit

See `needs.py:NeedActionInfluence.get_action_modifier()`:

```python
# Safety need influences
if primary_type == NeedType.SAFETY:
    if action_type in ["defend", "stay_quiet"]:
        return 1.0 + drive_strength * 0.8  # Seek safety through defense
    elif action_type in ["accuse", "lie"]:
        return 1.0 - drive_strength * 0.6  # Avoid risky confrontations
```

When an agent's safety need is low (unsatisfied), they become more defensive and less confrontational. This creates **goal-directed behavior** without explicit goal trees.

## Extending This Example

### Add New Personality Traits

1. Extend `BigFiveTraits` in `personality.py`
2. Add weighting logic in `PersonalityWeights.calculate_action_weight()`
3. Update scenario.json with new trait values

### Add New Emotions

1. Add discrete emotion to `DiscreteEmotion` enum in `emotion.py`
2. Map to circumplex space in `EmotionalState._update_primary_emotion()`
3. Define action influences in `EmotionActionInfluence.get_action_modifier()`

### Add New Needs

1. Extend `NeedType` enum in `needs.py`
2. Add to `NeedsHierarchy` dataclass
3. Define fulfillment events in `NeedFulfillmentEvents`
4. Add action influences in `NeedActionInfluence`

### Different Game Scenarios

The personality/emotion/needs framework is **domain-agnostic**. You could use it for:
- **FPS bots**: Add aggression trait, fear/excitement emotions, survival/dominance needs
- **NPCs in open worlds**: Add curiosity trait, boredom emotion, exploration needs
- **Economic simulations**: Add risk-tolerance trait, greed/satisfaction emotions, security/wealth needs

## Citations

```bibtex
@article{hari2025behavior,
  title={Behavior is all you need},
  author={Hari, Vishnu and Brennan, Connor},
  journal={arXiv preprint},
  year={2025},
  url={https://www.egoai.com/research/Whitepaper.pdf}
}

@article{park2023generative,
  title={Generative Agents: Interactive Simulacra of Human Behavior},
  author={Park, Joon Sung and O'Brien, Joseph C and Cai, Carrie J and Morris, Meredith Ringel and Liang, Percy and Bernstein, Michael S},
  journal={arXiv preprint arXiv:2304.03442},
  year={2023}
}
```

## Credits

- **Paper**: Vishnu Hari & Connor Brennan (Ego AI)
- **Implementation**: Using [Miniverse](https://github.com/anthropics/miniverse) agent simulation library
- **Inspiration**: Stanford Generative Agents, Maslow's Hierarchy, Russell's Circumplex Model, Big Five Personality Theory

---

For questions or contributions, see the main [Miniverse repository](https://github.com/anthropics/miniverse).
