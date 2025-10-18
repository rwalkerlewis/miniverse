# Code Quality Fixes - "Behavior is all you need"

## Issue: Silent Fallbacks for Missing Data

**Problem**: The original implementation had dangerous silent fallbacks where missing behavioral data (personality, emotion, needs) would be replaced with neutral defaults without any warnings. This made it impossible to detect when agents weren't properly initialized.

**Date Fixed**: 2025-10-18

---

## What Was Fixed

### 1. Personality Fallback (cognition.py)

**Before:**
```python
def _get_personality(self, scratchpad: Scratchpad) -> BigFiveTraits:
    personality_data = scratchpad.state.get("personality", {})
    if personality_data:
        return BigFiveTraits.from_dict(personality_data)
    # DANGER: Silent fallback!
    return BigFiveTraits(openness=50, conscientiousness=50, ...)
```

**After:**
```python
def _get_personality(self, scratchpad: Scratchpad) -> BigFiveTraits:
    personality_data = scratchpad.state.get("personality", {})
    if personality_data:
        return BigFiveTraits.from_dict(personality_data)

    # Missing data: fail loudly or warn based on STRICT_MODE
    error_msg = (
        "Personality data missing from scratchpad! "
        "Agent behavior will be bland (neutral defaults used). "
        "Call initialize_agent_state() to properly set up personality."
    )

    if os.getenv("STRICT_MODE", "").lower() in ("true", "1", "yes"):
        raise ValueError(error_msg)
    else:
        warnings.warn(f"⚠️  {error_msg}", UserWarning, stacklevel=2)

    return BigFiveTraits(openness=50, conscientiousness=50, ...)
```

**Impact**: Developers now know immediately when agents aren't properly initialized.

---

### 2. Emotion Fallback (cognition.py)

**Before:**
```python
def _get_emotional_state(self, scratchpad: Scratchpad) -> EmotionalState:
    emotion_data = scratchpad.state.get("emotional_state", {})
    if emotion_data:
        return EmotionalState.from_dict(emotion_data)
    # DANGER: Silent fallback!
    return EmotionalState()
```

**After:**
```python
def _get_emotional_state(self, scratchpad: Scratchpad) -> EmotionalState:
    emotion_data = scratchpad.state.get("emotional_state", {})
    if emotion_data:
        return EmotionalState.from_dict(emotion_data)

    # Missing data: fail loudly or warn
    error_msg = (
        "Emotional state missing from scratchpad! "
        "Agent will not show emotional reactions (neutral state used). "
        "Call initialize_agent_state() to properly set up emotion."
    )

    if os.getenv("STRICT_MODE", "").lower() in ("true", "1", "yes"):
        raise ValueError(error_msg)
    else:
        warnings.warn(f"⚠️  {error_msg}", UserWarning, stacklevel=2)

    return EmotionalState()
```

**Impact**: Agents with missing emotion data are now flagged instead of silently behaving neutrally.

---

### 3. Needs Fallback (cognition.py)

**Before:**
```python
def _get_needs(self, scratchpad: Scratchpad) -> NeedsHierarchy:
    needs_data = scratchpad.state.get("needs", {})
    if needs_data:
        return NeedsHierarchy.from_dict(needs_data)
    # DANGER: Silent fallback!
    return NeedsHierarchy()
```

**After:**
```python
def _get_needs(self, scratchpad: Scratchpad) -> NeedsHierarchy:
    needs_data = scratchpad.state.get("needs", {})
    if needs_data:
        return NeedsHierarchy.from_dict(needs_data)

    # Missing data: fail loudly or warn
    error_msg = (
        "Needs hierarchy missing from scratchpad! "
        "Agent will not show need-driven behavior (default 50% satisfaction used). "
        "Call initialize_agent_state() to properly set up needs."
    )

    if os.getenv("STRICT_MODE", "").lower() in ("true", "1", "yes"):
        raise ValueError(error_msg)
    else:
        warnings.warn(f"⚠️  {error_msg}", UserWarning, stacklevel=2)

    return NeedsHierarchy()
```

**Impact**: Need-driven behavior failures are now detectable.

---

## New Feature: STRICT_MODE

### Environment Variable Control

```bash
# Lenient mode (default): Shows warnings, continues with defaults
uv run python run.py --ticks 12

# Strict mode: Raises errors, prevents execution with bad data
export STRICT_MODE=true
uv run python run.py --ticks 12
```

### When to Use Each Mode

**Lenient Mode (default)**:
- Demos and tutorials
- Experimenting with code
- Quick prototyping
- Want simulation to continue even with missing data

**Strict Mode**:
- Testing and CI/CD
- Debugging initialization issues
- Validating proper setup
- Ensuring all agents have complete behavioral state

---

## Testing the Fix

### Test 1: Warnings in Lenient Mode

```bash
UV_CACHE_DIR=.uv-cache uv run python -c "
import sys
sys.path.insert(0, 'examples/behavior_is_all_you_need')
import warnings
warnings.simplefilter('always')

from cognition import PersonalityAwareExecutor
from miniverse.cognition import Scratchpad

scratchpad = Scratchpad()  # Empty - no personality/emotion/needs
executor = PersonalityAwareExecutor(use_llm=False)

personality = executor._get_personality(scratchpad)
emotion = executor._get_emotional_state(scratchpad)
needs = executor._get_needs(scratchpad)
"
```

**Expected Output:**
```
⚠️  Personality data missing from scratchpad! ...
⚠️  Emotional state missing from scratchpad! ...
⚠️  Needs hierarchy missing from scratchpad! ...
```

### Test 2: Errors in Strict Mode

```bash
STRICT_MODE=true UV_CACHE_DIR=.uv-cache uv run python -c "
import sys
sys.path.insert(0, 'examples/behavior_is_all_you_need')

from cognition import PersonalityAwareExecutor
from miniverse.cognition import Scratchpad

scratchpad = Scratchpad()
executor = PersonalityAwareExecutor(use_llm=False)

try:
    personality = executor._get_personality(scratchpad)
except ValueError as e:
    print('✓ Raised ValueError as expected')
    print(f'Error: {e}')
"
```

**Expected Output:**
```
✓ Raised ValueError as expected
Error: Personality data missing from scratchpad! ...
```

---

## Files Modified

1. `examples/behavior_is_all_you_need/cognition.py`
   - `PersonalityAwareExecutor._get_personality()`
   - `PersonalityAwareExecutor._get_emotional_state()`
   - `PersonalityAwareExecutor._get_needs()`
   - `PersonalityAwarePlanner._get_personality()`
   - `PersonalityAwarePlanner._get_needs()`

2. `examples/behavior_is_all_you_need/README.md`
   - Added "Strict Mode" section

3. `examples/behavior_is_all_you_need/QUICKSTART.md`
   - Added troubleshooting section for warnings
   - Added strict mode example

4. `examples/behavior_is_all_you_need/IMPLEMENTATION_SUMMARY.md`
   - Added strict mode usage example

---

## Why This Matters

### The Paper's Core Claim
"Behavior is all you need" argues that personality-driven behavior creates believable agents. If personality/emotion/needs data is silently missing, agents behave neutrally and the paper's framework can't be validated.

### Before the Fix
- Agents could run with missing data
- Behavior looked correct but wasn't personality-driven
- No way to detect initialization failures
- False validation of the framework

### After the Fix
- Missing data is immediately visible (warnings or errors)
- Developers can choose strictness level
- Testing ensures proper initialization
- True validation of personality → behavior correlation

---

## Future Improvements

### Potential Enhancements
1. **Validation helper**: `validate_agent_state(scratchpad)` that checks all required fields
2. **Initialization checker**: Run at orchestrator start to validate all agents
3. **Logging mode**: `LOG_MODE=true` to log all fallbacks to a file
4. **Per-agent strictness**: Allow some agents to use defaults while others require full state

### Not Implemented (Intentional)
- ❌ Auto-initialization with defaults (defeats the purpose of validation)
- ❌ Silent logging without warnings (makes issues invisible)
- ❌ Optional strict mode flag in code (env var is cleaner for testing)

---

## Summary

**Changes**: Added loud warnings/errors when behavioral data is missing
**Files Modified**: 4 (cognition.py + 3 docs)
**Lines Changed**: ~100 lines
**Impact**: High - catches initialization bugs early
**Breaking Changes**: None (default behavior is lenient)
**Testing**: Verified both modes work correctly

This fix ensures the "Behavior is all you need" example can properly validate the paper's claims by guaranteeing agents have complete behavioral state.
