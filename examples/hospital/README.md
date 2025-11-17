# Hospital Simulation

A comprehensive hospital simulation demonstrating complex agent-based modeling with:

- **Emergency Room (ER)**: Triage, emergency treatment, stabilization
- **Intensive Care Unit (ICU)**: Critical care monitoring and treatment
- **Operating Rooms (OR)**: Surgical procedures
- **Imaging Department**: X-ray, CT, MRI diagnostics
- **Multiple Agent Types**: Doctors, nurses, patients, visitors, administrative staff

## Features

- **Patient Flow**: Emergency arrivals, triage assessment, treatment pathways, transfers between departments, discharge
- **Staff Coordination**: Role-based responsibilities, shift management, communication protocols
- **Resource Management**: Bed occupancy, equipment availability, staff workload tracking
- **Emergent Behavior**: LLM-driven decision-making for treatment plans, communication, and care coordination

## Running the Simulation

### Deterministic Mode (No LLM)
```bash
uv run python examples/hospital/run.py --ticks 10
```

### LLM Mode (Emergent Behavior)
```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini
export OPENAI_API_KEY=your_key
uv run python examples/hospital/run.py --llm --ticks 10
```

### Debug Mode
```bash
DEBUG_MEMORY=true DEBUG_LLM=true uv run python examples/hospital/run.py --llm --ticks 5
```

## Scenario Structure

- **Environment**: Logical graph (Tier 1) with departments as nodes
- **Agents**: 15+ agents across different roles
- **Simulation Rules**: Hospital-specific physics (patient vitals, treatment effects, transfers)
- **Metrics**: Patient outcomes, wait times, bed occupancy, staff workload

## Agent Roles

- **ER Doctor**: Emergency assessments and stabilization
- **ER Nurses**: Triage, monitoring, medication administration
- **ICU Doctor**: Critical care management
- **ICU Nurses**: Intensive monitoring and interventions
- **Surgeons**: Perform operations in OR
- **OR Nurses**: Surgical assistance and prep
- **Radiologists**: Imaging interpretation
- **Imaging Technicians**: Operate diagnostic equipment
- **Patients**: Varying severity levels and conditions
- **Visitors**: Family members providing support
- **Administrative Staff**: Admissions, discharge coordination
