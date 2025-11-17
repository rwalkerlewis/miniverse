# ICU Critical Care Simulation

A focused simulation of an Intensive Care Unit (ICU) with realistic patient care dynamics.

## Overview

This simulation models a 12-hour ICU shift with:
- **1 Doctor**: Dr. Emily Wilson (ICU Attending Physician)
- **4 Nurses**: Sarah (Charge Nurse), James (Ventilator Specialist), Priya (Junior Nurse), Michael (Cardiac Specialist)
- **8 Patients**: Various critical conditions including post-operative, respiratory failure, sepsis, trauma, stroke, and cardiac

## Patient Conditions

### Critical Patients (Beds 1-4)
- **Robert Harris** (Bed 1): Post-cardiac surgery, on ventilator
- **Linda Chen** (Bed 2): Severe pneumonia, respiratory failure
- **Marcus Johnson** (Bed 3): Septic shock, unstable vitals
- **Elena Rodriguez** (Bed 4): Trauma from MVA, significant pain

### Stable Patients (Beds 5-8)
- **David Kim** (Bed 5): Post-stroke, neurologically stable
- **Susan Williams** (Bed 6): Diabetic ketoacidosis, improving
- **Thomas Anderson** (Bed 7): CHF exacerbation, on diuretics
- **Maria Santos** (Bed 8): Post-MI with stent, monitoring

## Environment

The ICU has:
- 8 individual patient beds with monitoring equipment
- Nurses station (central hub)
- Medication room
- Supply room

Events from outside the ICU (lab results, radiology reports, family inquiries) are referenced but not explicitly modeled as agents.

## Running the Simulation

### Deterministic Mode (No LLM)
Fast, reproducible, rule-based behavior:

```bash
python examples/icu/run.py --ticks 12
```

### LLM Mode
More realistic, emergent behavior using local or cloud LLMs:

```bash
# With Ollama (local)
export LLM_PROVIDER=ollama
export LLM_MODEL=llama3.1:8b
python examples/icu/run.py --llm --ticks 12

# With OpenAI
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini
export OPENAI_API_KEY=your-key
python examples/icu/run.py --llm --ticks 12
```

## Time Scale

Each tick represents **1 hour** of real ICU time:
- Tick 0: 8:00 AM (shift start)
- Tick 6: 2:00 PM (mid-shift)
- Tick 12: 8:00 PM (shift end)

## Key Dynamics

### Patient Health
- Patients on ventilators require close monitoring
- Stable patients gradually improve
- Unstable patients may deteriorate without intervention
- Pain levels affect patient comfort and cooperation

### Staff Workload
- Each nurse typically manages 2 patients
- Doctor rounds on all patients, focusing on critical cases
- Staff energy decreases and stress increases over shift
- Coordination between staff is essential for patient safety

### External Events
Random events from outside the ICU occur periodically:
- Lab results becoming available
- Radiology reports ready for review
- Family requesting updates
- Pharmacy calling about medications
- Bed availability for transfers

## Scenario Design Philosophy

This simulation demonstrates:

1. **Focused Scope**: Models only ICU, references external hospital
2. **Realistic Staffing**: Typical ICU nurse-to-patient ratios (1:2)
3. **Diverse Acuity**: Mix of critical and stable patients
4. **Professional Roles**: Doctor provides medical decisions, nurses execute care
5. **Shift Dynamics**: Energy/stress changes, handoffs, coordination

## Customization

You can modify `scenario.json` to:
- Change patient conditions or acuity levels
- Adjust staffing (add/remove nurses)
- Modify patient assignments
- Add different types of events
- Change initial patient health states

## Educational Value

This example is useful for:
- Understanding healthcare workflow simulation
- Modeling multi-agent coordination in constrained environments
- Demonstrating role-based agent behavior
- Teaching about ICU operations and patient care
- Exploring emergent behavior in critical care settings
