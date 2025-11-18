"""ICU Critical Care simulation.

Simplified ICU scenario focusing on:
- 1 attending physician (Dr. Wilson)
- 2 ICU nurses (Sarah, James)
- 4 patients in various conditions
- Events from outside ICU (transfers, lab results, etc.) are referenced but not explicitly modeled

By default runs in deterministic mode (no LLM calls):
    uv run python examples/icu/run.py --ticks 12

To enable LLM-driven cognition:
    export LLM_PROVIDER=ollama
    export LLM_MODEL=qwen2.5:7b-instruct
    uv run python examples/icu/run.py --llm --ticks 12

Each tick represents 30 minutes of ICU time (12 ticks = 6 hours).
"""

from __future__ import annotations

import argparse
import asyncio
import random
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from miniverse import (
    AgentAction,
    AgentCognition,
    GraphOccupancy,
    Orchestrator,
    Plan,
    PlanStep,
    ReflectionResult,
    SimulationRules,
    Stat,
    WorldState,
)
from miniverse.cognition import (
    LLMExecutor,
    LLMPlanner,
    LLMReflectionEngine,
    PromptTemplate,
    Scratchpad,
)
from miniverse.cognition.context import PromptContext
from miniverse.config import Config
from miniverse.environment import shortest_path
from miniverse.logging_utils import log_info
from miniverse.scenario import ScenarioLoader

if TYPE_CHECKING:
    from miniverse import AgentProfile

# ============================================================================
# ICU PROMPTS AND ACTIONS
# ============================================================================

ICU_EXECUTOR_TEMPLATE = PromptTemplate(
    name="icu_executor",
    system=(
        "You are {{agent_name}}, {{agent_role}} in the ICU.\n\n"
        "Your personality: {{agent_personality}}\n"
        "Your background: {{agent_background}}\n\n"
        "As an ICU medical professional, assess each patient's condition, severity, and appropriate treatment.\n"
        "Consider:\n"
        "- Patient vital signs and current health trajectory\n"
        "- Underlying medical conditions and their typical progression\n"
        "- Severity levels (mild, moderate, severe, critical)\n"
        "- Appropriate interventions (monitoring, medications, procedures)\n"
        "- Risk of deterioration or code blue events\n\n"
        "CRITICAL: You must respond with ONLY valid JSON matching this exact schema:\n"
        "{\n"
        '  "agent_id": "{{agent_id}}",\n'
        '  "tick": {{current_tick}},\n'
        '  "action_type": "<one of: work, rest, move_to, communicate>",\n'
        '  "target": "<location_id or agent_id or null>",\n'
        '  "parameters": {<optional key-value pairs including treatment details>},\n'
        '  "reasoning": "<clinical reasoning: patient assessment, treatment plan, and prognosis>"\n'
        "}\n\n"
        "Available action types:\n"
        "- work: Perform medical care (target: patient care location, params: {\"task\": \"monitor_patient\" | \"administer_medication\" | \"intubate\" | \"start_pressors\" | \"assess_patient\" | \"code_response\", \"treatment\": \"<specific intervention>\"})\n"
        "- rest: Take a break to recover energy\n"
        "- move_to: Go to a different location (target: location_id)\n"
        "- communicate: Talk to another staff member (target: agent_id, use communication field)\n\n"
        "DO NOT include any explanatory text before or after the JSON. Return ONLY the JSON object."
    ),
    user=(
        "Current perception:\n{{perception_json}}\n\n"
        "Based on your clinical assessment, choose your next action. Return ONLY valid JSON."
    ),
)

ICU_AVAILABLE_ACTIONS = [
    {
        "action_type": "work",
        "description": "Perform medical care tasks with specific treatments",
        "valid_targets": ["patient_care", "icu_bed_1", "icu_bed_2", "icu_bed_3", "icu_bed_4", "icu_bed_5", "icu_bed_6"],
        "parameters": {
            "task": ["monitor_patient", "administer_medication", "intubate", "start_pressors", "code_response", "assess_patient", "update_chart", "prepare_body", "transfer_to_general"],
            "treatment": ["antibiotics", "fluids", "vasopressors", "oxygen", "ventilation", "cardiac_meds", "pain_management"]
        }
    },
    {
        "action_type": "rest",
        "description": "Take a break to recover energy and reduce stress",
        "valid_targets": [None],
        "parameters": {}
    },
    {
        "action_type": "move_to",
        "description": "Move to a different ICU location",
        "valid_targets": ["nurses_station", "medication_room", "supply_room", "icu_bed_1", "icu_bed_2", "icu_bed_3", "icu_bed_4", "icu_bed_5", "icu_bed_6"],
        "parameters": {}
    },
    {
        "action_type": "communicate",
        "description": "Talk to another staff member",
        "valid_targets": ["dr_wilson", "nurse_sarah", "nurse_james"],
        "parameters": {}
    },
]

# ============================================================================
# ICU SIMULATION RULES
# ============================================================================


class ICURules(SimulationRules):
    """Deterministic rules for ICU patient dynamics and constraints."""

    def __init__(
        self,
        occupancy: GraphOccupancy | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.occupancy = occupancy
        self.rng = rng or random.Random()
        self.time_step_minutes = 30  # Default: 30 minutes per tick
        self.last_event_severity = 0
        self.pending_admissions = []  # Queue of patients waiting for ICU beds
        self.next_patient_id = 1  # Counter for new patient IDs

    def apply_tick(self, state: WorldState, tick: int) -> WorldState:
        """Update patient health, staff energy, and generate events (30 min intervals)."""
        new_state = state.model_copy(deep=True)
        new_state.tick = tick

        # Track patient metrics
        patient_health_sum = 0
        patient_count = 0
        stable_count = 0
        critical_count = 0
        discharged_count = 0
        deceased_count = 0

        # Update patient attributes
        agents_to_remove = []
        for agent_status in new_state.agents:
            if not agent_status.agent_id.startswith("patient_"):
                continue

            # Skip already discharged or deceased patients
            if agent_status.activity in ["discharged", "deceased"]:
                if agent_status.activity == "discharged":
                    discharged_count += 1
                else:
                    deceased_count += 1
                continue

            patient_count += 1
            health = agent_status.attributes.get("health", Stat(value=50)).value

            # Check for death (health <= 0)
            if health <= 0:
                # Check if already in post-mortem care
                time_of_death = agent_status.attributes.get("time_of_death", Stat(value=0)).value
                if time_of_death == 0:
                    # Just died - start post-mortem care
                    agent_status.activity = "deceased_postmortem_care"
                    agent_status.attributes["health"] = Stat(value=0, label="Health")
                    agent_status.attributes["time_of_death"] = Stat(value=tick, label="Time of Death")
                    agent_status.attributes["instantaneous_interventions"] = Stat(
                        value="Time of death recorded; Post-mortem care initiated; Family notification in progress",
                        label="This Tick"
                    )
                    deceased_count += 1
                    from miniverse.schemas import WorldEvent
                    import uuid
                    new_state.recent_events.append(WorldEvent(
                        event_id=f"death_{tick}_{uuid.uuid4().hex[:8]}",
                        tick=tick,
                        category="patient_outcome",
                        description=f"Patient {agent_status.agent_id} has passed away",
                        severity=5,
                        metadata={"type": "death", "patient": agent_status.agent_id}
                    ))
                elif tick - time_of_death >= 2:
                    # After 2 ticks of post-mortem care, transfer to mortuary
                    agent_status.activity = "transferred_to_mortuary"
                    agent_status.attributes["instantaneous_interventions"] = Stat(
                        value="Body prepared; Belongings collected; Transferred to mortuary",
                        label="This Tick"
                    )
                else:
                    # Still in post-mortem care
                    agent_status.attributes["instantaneous_interventions"] = Stat(
                        value="Continued post-mortem care; Documentation completion",
                        label="This Tick"
                    )
                    deceased_count += 1
                continue
            
            # Check for discharge (health >= 85 and stable)
            vital_signs = agent_status.attributes.get("vital_signs", Stat(value="stable")).value
            if health >= 85 and "stable" in str(vital_signs):
                agent_status.activity = "discharged_to_general_ward"
                agent_status.attributes["instantaneous_interventions"] = Stat(
                    value="Final ICU assessment completed; Cleared for transfer to general medical ward",
                    label="This Tick"
                )
                discharged_count += 1
                from miniverse.schemas import WorldEvent
                import uuid
                new_state.recent_events.append(WorldEvent(
                    event_id=f"discharge_{tick}_{uuid.uuid4().hex[:8]}",
                    tick=tick,
                    category="patient_outcome",
                    description=f"Patient {agent_status.agent_id} discharged to general ward - condition stable",
                    severity=2,
                    metadata={"type": "discharge", "patient": agent_status.agent_id}
                ))
                continue

            patient_health_sum += health

            # Count stability
            if "unstable" in str(vital_signs) or health < 50:
                critical_count += 1
            else:
                stable_count += 1

            # Track instantaneous changes (what happened this tick)
            instantaneous_changes = []
            old_health = health
            
            # Natural disease progression based on condition and severity
            condition = agent_status.attributes.get("condition", Stat(value="unknown")).value
            severity = agent_status.attributes.get("severity", Stat(value="moderate")).value
            
            # Condition-specific progression logic
            if "septic shock" in str(condition).lower():
                # Septic shock can deteriorate rapidly without intervention
                if severity == "severe" and health < 60:
                    # Risk of rapid deterioration
                    deterioration = self.rng.randint(5, 12)
                    new_health = max(health - deterioration, 0)
                    agent_status.attributes["health"] = Stat(value=new_health, label="Health")
                    instantaneous_changes.append(f"Health declined {deterioration} points due to septic shock progression")
                    
                    # Trigger code blue if health drops critically low
                    if new_health < 30 and health >= 30:
                        agent_status.attributes["vital_signs"] = Stat(value="critical_unstable", label="Vital Signs")
                        instantaneous_changes.append("CRITICAL: Vital signs deteriorated to critical_unstable")
                        from miniverse.schemas import WorldEvent
                        import uuid
                        new_state.recent_events.append(WorldEvent(
                            event_id=f"code_blue_{tick}_{uuid.uuid4().hex[:8]}",
                            tick=tick,
                            category="medical_emergency",
                            description=f"CODE BLUE - {agent_status.agent_id} experiencing septic shock with cardiovascular collapse!",
                            severity=5,
                            metadata={"type": "code_blue", "patient": agent_status.agent_id, "cause": "septic_shock"}
                        ))
                elif severity == "moderate":
                    # Slower deterioration
                    change = self.rng.randint(-3, 0)
                    new_health = max(health + change, 0)
                    agent_status.attributes["health"] = Stat(value=new_health, label="Health")
                    if change < 0:
                        instantaneous_changes.append(f"Health declined {abs(change)} points")
                    elif change > 0:
                        instantaneous_changes.append(f"Health improved {change} points")
            
            elif "myocardial infarction" in str(condition).lower() or "heart attack" in str(condition).lower():
                # Heart attack patients at risk of arrhythmia/cardiac arrest
                if severity == "severe" and health < 50:
                    deterioration = self.rng.randint(4, 10)
                    new_health = max(health - deterioration, 0)
                    agent_status.attributes["health"] = Stat(value=new_health, label="Health")
                    
                    if new_health < 25 and health >= 25:
                        agent_status.attributes["vital_signs"] = Stat(value="critical_unstable", label="Vital Signs")
                        from miniverse.schemas import WorldEvent
                        import uuid
                        new_state.recent_events.append(WorldEvent(
                            event_id=f"code_blue_{tick}_{uuid.uuid4().hex[:8]}",
                            tick=tick,
                            category="medical_emergency",
                            description=f"CODE BLUE - {agent_status.agent_id} experiencing cardiac arrest!",
                            severity=5,
                            metadata={"type": "code_blue", "patient": agent_status.agent_id, "cause": "cardiac_arrest"}
                        ))
                else:
                    change = self.rng.randint(-2, 1)
                    new_health = max(health + change, 0)
                    agent_status.attributes["health"] = Stat(value=new_health, label="Health")
            
            elif "respiratory failure" in str(condition).lower():
                # Respiratory failure can deteriorate if not properly ventilated
                on_vent = agent_status.attributes.get("on_ventilator", Stat(value=False)).value
                if not on_vent and severity == "severe":
                    deterioration = self.rng.randint(3, 8)
                    new_health = max(health - deterioration, 0)
                    agent_status.attributes["health"] = Stat(value=new_health, label="Health")
                elif on_vent:
                    # Ventilated patients stabilize or improve slowly
                    change = self.rng.randint(0, 2)
                    new_health = min(health + change, 100)
                    agent_status.attributes["health"] = Stat(value=new_health, label="Health")
            
            # Default progression for other conditions
            elif health < 85 and "stable" in str(vital_signs):
                # Stable patients slowly improve (can reach discharge threshold)
                improvement = self.rng.randint(1, 3)
                new_health = min(health + improvement, 100)
                agent_status.attributes["health"] = Stat(value=new_health, label="Health")
                if improvement > 0:
                    instantaneous_changes.append(f"Health improved {improvement} points")
            elif "unstable" in str(vital_signs):
                # Unstable patients may deteriorate (can die)
                change = self.rng.randint(-2, 1)
                new_health = max(health + change, 0)
                agent_status.attributes["health"] = Stat(value=new_health, label="Health")
                if change < 0:
                    instantaneous_changes.append(f"Health declined {abs(change)} points")
                elif change > 0:
                    instantaneous_changes.append(f"Health improved {change} points")
            
            # Save instantaneous changes for this tick
            if instantaneous_changes:
                agent_status.attributes["instantaneous_interventions"] = Stat(
                    value="; ".join(instantaneous_changes),
                    label="This Tick"
                )
            else:
                agent_status.attributes["instantaneous_interventions"] = Stat(
                    value="Continued monitoring; No significant changes",
                    label="This Tick"
                )

        # Update staff energy (decreases over shift, 30-min intervals)
        for agent_status in new_state.agents:
            if agent_status.agent_id.startswith("nurse_") or agent_status.agent_id.startswith("dr_"):
                energy = agent_status.attributes.get("energy", Stat(value=80)).value
                stress = agent_status.attributes.get("stress", Stat(value=20)).value

                # Energy decreases slower (30 min intervals), stress increases during code blue
                energy_loss = self.rng.randint(0, 2)
                stress_gain = self.rng.randint(0, 1)
                
                # Extra stress during code blue
                if any(e.category == "medical_emergency" for e in new_state.recent_events):
                    stress_gain += self.rng.randint(3, 5)
                
                new_energy = max(energy - energy_loss, 20)
                new_stress = min(stress + stress_gain, 100)
                agent_status.attributes["energy"] = Stat(value=new_energy, label="Energy")
                agent_status.attributes["stress"] = Stat(value=new_stress, label="Stress")

        # Update resource metrics
        avg_health = patient_health_sum / patient_count if patient_count > 0 else 0
        new_state.resources.metrics["avg_patient_health"].value = round(avg_health, 1)
        new_state.resources.metrics["patients_stable"].value = stable_count
        new_state.resources.metrics["patients_critical"].value = critical_count
        new_state.resources.metrics["patients_discharged"] = Stat(value=discharged_count, label="Patients Discharged")
        new_state.resources.metrics["patients_deceased"] = Stat(value=deceased_count, label="Patients Deceased")

        # Check for open ICU beds and randomly admit new patients
        occupied_beds = set()
        for agent_status in new_state.agents:
            if agent_status.agent_id.startswith("patient_") and agent_status.activity == "resting":
                if agent_status.location.startswith("icu_bed_"):
                    occupied_beds.add(agent_status.location)
        
        available_beds = [f"icu_bed_{i}" for i in range(1, 7) if f"icu_bed_{i}" not in occupied_beds]
        
        # Random chance of new patient admission if beds available (higher chance at later ticks)
        if available_beds and tick > 3 and self.rng.random() < 0.25:
            # Admit a new patient
            bed = self.rng.choice(available_beds)
            new_patient_id = f"patient_new_{self.next_patient_id}"
            self.next_patient_id += 1
            
            # Random patient conditions
            conditions = [
                ("Acute respiratory distress syndrome (ARDS)", "severe", 40, "critical", False,
                 "Transferred from ED with severe respiratory failure requiring mechanical ventilation"),
                ("Stroke (ischemic)", "severe", 50, "critical", False,
                 "Admitted with acute stroke; currently being assessed for intervention"),
                ("Gastrointestinal bleeding", "moderate", 55, "unstable", False,
                 "Upper GI bleed; hemodynamically unstable; requiring blood transfusions"),
                ("Acute renal failure", "moderate", 60, "unstable", False,
                 "Admitted with acute kidney injury; may require dialysis"),
                ("Severe COVID-19 pneumonia", "severe", 45, "critical", True,
                 "COVID-19 with severe pneumonia; on high-flow oxygen"),
            ]
            
            condition, severity, health, vitals, on_vent, history = self.rng.choice(conditions)
            
            # Create treatments list
            treatments = []
            if on_vent:
                treatments.append("Mechanical ventilation")
            if "respiratory" in condition.lower() or "pneumonia" in condition.lower():
                treatments.extend(["Antibiotics", "Supplemental oxygen"])
            if "bleeding" in condition.lower():
                treatments.extend(["Blood transfusion", "IV fluids", "Hemostatic agents"])
            if "renal" in condition.lower():
                treatments.extend(["IV fluids", "Electrolyte replacement"])
            if "stroke" in condition.lower():
                treatments.extend(["Anticoagulation", "Blood pressure management"])
            
            from miniverse import AgentProfile, AgentStatus
            from datetime import datetime
            
            # Generate random patient name
            first_names = ["Michael", "Jessica", "David", "Sarah", "Christopher", "Amanda", "Daniel", "Jennifer"]
            last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
            name = f"{self.rng.choice(first_names)} {self.rng.choice(last_names)}"
            age = self.rng.randint(35, 85)
            
            # Add event about new admission
            from miniverse.schemas import WorldEvent
            import uuid
            new_state.recent_events.append(WorldEvent(
                event_id=f"admission_{tick}_{uuid.uuid4().hex[:8]}",
                tick=tick,
                category="patient_admission",
                description=f"New patient {name} admitted to {bed} - {condition}",
                severity=3,
                metadata={"type": "admission", "patient": new_patient_id, "bed": bed}
            ))

        # Random events from outside ICU (lower frequency for 30-min ticks)
        if tick > 0 and self.rng.random() < 0.15:
            event_types = [
                "Lab results available for review",
                "Radiology report ready for ICU patient",
                "Family member requesting update at desk",
                "Pharmacy called about medication clarification",
                "Bed available on general floor for transfer",
            ]
            event_desc = self.rng.choice(event_types)
            from miniverse.schemas import WorldEvent
            import uuid
            new_state.recent_events.append(WorldEvent(
                event_id=f"event_{tick}_{uuid.uuid4().hex[:8]}",
                tick=tick,
                category="external_event",
                description=event_desc,
                severity=3,
                metadata={"type": "external_event"}
            ))

        # Adjust time step based on event severity
        max_severity = max([e.severity for e in new_state.recent_events] + [0])
        
        if max_severity >= 5:  # Code blue or death
            self.time_step_minutes = 5  # 5 minutes during critical events
        elif max_severity >= 4:  # High severity events
            self.time_step_minutes = 10  # 10 minutes
        elif max_severity >= 3:  # Moderate events
            self.time_step_minutes = 15  # 15 minutes
        else:  # Routine care
            self.time_step_minutes = 30  # 30 minutes
        
        self.last_event_severity = max_severity

        return new_state

    def validate_action(self, action: AgentAction, state: WorldState) -> bool:
        """Validate ICU actions."""
        # Patients can't move
        if action.agent_id.startswith("patient_") and action.action_type == "move_to":
            return False

        # Check room capacity if moving
        if action.action_type == "move_to" and self.occupancy:
            target_loc = action.target
            if not target_loc:
                return False
            if not self.occupancy.can_enter(target_loc):
                return False

        return True

    def process_actions(
        self, state: WorldState, actions: list[AgentAction], tick: int
    ) -> WorldState:
        """Process ICU-specific actions deterministically."""
        new_state = state.model_copy(deep=True)

        for action in actions:
            # Find the agent in the agents list
            agent_status = next((a for a in new_state.agents if a.agent_id == action.agent_id), None)
            if not agent_status:
                continue

            # Handle move actions
            if action.action_type == "move_to" and action.target:
                agent_status.location = action.target
                agent_status.activity = f"at_{action.target}"

            # Handle medication administration
            elif action.action_type == "administer_medication":
                new_state.resources.metrics["medications_administered"].value += 1
                agent_status.activity = "administering_medication"

            # Handle patient assessment
            elif action.action_type == "assess_patient":
                agent_status.activity = "assessing_patient"

            # Handle work actions
            elif action.action_type == "work":
                task = action.parameters.get("task", "") if action.parameters else ""
                agent_status.activity = f"working_on_{task}"

            # Handle communicate actions
            elif action.action_type == "communicate":
                agent_status.activity = "communicating"

        return new_state


# ============================================================================
# DETERMINISTIC COGNITION (No LLM)
# ============================================================================


class DeterministicICUPlanner:
    """Simple deterministic planner for ICU staff."""

    async def generate_plan(
        self,
        agent_id: str,
        scratchpad: Scratchpad,
        *,
        world_context: Any,
        context: PromptContext,
    ) -> Plan:
        """Generate a simple plan based on role."""
        agent = context.agent_profile
        steps = []

        if agent_id.startswith("dr_"):
            steps = [
                PlanStep(description="Review all patient charts and vital signs"),
                PlanStep(description="Round on critical patients first"),
                PlanStep(description="Update treatment plans as needed"),
                PlanStep(description="Communicate with nursing staff"),
            ]
        elif agent_id.startswith("nurse_"):
            steps = [
                PlanStep(description="Check on assigned patients"),
                PlanStep(description="Administer scheduled medications"),
                PlanStep(description="Monitor vital signs and respond to alarms"),
                PlanStep(description="Document patient care and updates"),
            ]

        return Plan(steps=steps, metadata={"agent": agent.name})


class DeterministicICUExecutor:
    """Simple deterministic executor for ICU staff."""

    async def choose_action(
        self,
        agent_id: str,
        perception,
        scratchpad: Scratchpad,
        *,
        plan: Plan | None,
        plan_step: PlanStep | None,
        context: PromptContext,
    ) -> AgentAction:
        """Choose action based on role and current state."""
        agent = context.agent_profile
        current_location = perception.location

        # Doctor actions
        if agent_id.startswith("dr_"):
            # Round on patients
            if current_location == "nurses_station":
                return AgentAction(
                    agent_id=agent_id,
                    tick=perception.tick,
                    action_type="move_to",
                    target="icu_bed_1",
                    reasoning="Starting patient rounds",
                )
            else:
                return AgentAction(
                    agent_id=agent_id,
                    tick=perception.tick,
                    action_type="work",
                    target="patient_assessment",
                    parameters={"task": "assess_patient"},
                    reasoning="Assessing patient condition",
                )

        # Nurse actions
        elif agent_id.startswith("nurse_"):
            # Go to medication room if at nurses station
            if current_location == "nurses_station":
                return AgentAction(
                    agent_id=agent_id,
                    tick=perception.tick,
                    action_type="move_to",
                    target="medication_room",
                    reasoning="Getting medications for patients",
                )
            elif current_location == "medication_room":
                # Go to first ICU bed
                return AgentAction(
                    agent_id=agent_id,
                    tick=perception.tick,
                    action_type="move_to",
                    target="icu_bed_1",
                    reasoning="Going to check on patient",
                )
            else:
                # At patient bed - assess patient
                return AgentAction(
                    agent_id=agent_id,
                    tick=perception.tick,
                    action_type="work",
                    target="patient_care",
                    parameters={"task": "monitor_patient"},
                    reasoning="Monitoring patient vital signs",
                )

        # Patient actions (minimal)
        return AgentAction(
            agent_id=agent_id,
            tick=perception.tick,
            action_type="rest",
            reasoning="Resting and recovering",
        )


# ============================================================================
# MAIN SIMULATION
# ============================================================================


async def run_simulation(
    world_state: WorldState,
    agents: dict[str, AgentProfile],
    max_ticks: int,
    target_duration_minutes: int,
    use_llm: bool = False,
    report_file: str | None = None,
) -> dict:
    """Run the ICU simulation.
    
    Args:
        world_state: Initial world state
        agents: Agent profiles
        max_ticks: Maximum number of ticks to prevent infinite loops
        target_duration_minutes: Target simulation duration in minutes
        use_llm: Whether to use LLM for agent cognition
        report_file: Path to write detailed report
    """
    
    # Open report file if specified
    report_handle = None
    if report_file:
        report_handle = open(report_file, 'w')
        report_handle.write(f"ICU SIMULATION DETAILED REPORT\n")
        report_handle.write(f"{'='*80}\n")
        report_handle.write(f"Mode: {'LLM-Driven' if use_llm else 'Deterministic'}\n")
        report_handle.write(f"Target Duration: {target_duration_minutes / 60:.1f} hours ({target_duration_minutes} minutes)\n")
        report_handle.write(f"Time steps: Dynamic (5-30 minutes based on event severity)\n")
        report_handle.write(f"Patient conditions will progress naturally based on severity\n")
        report_handle.write(f"{'='*80}\n\n")
    
    # Set up graph occupancy
    if world_state.environment_graph:
        occupancy = GraphOccupancy(world_state.environment_graph)
        for agent in agents.values():
            # Get the location from world_state.agents list
            agent_status = next((a for a in world_state.agents if a.agent_id == agent.agent_id), None)
            if agent_status:
                occupancy.enter(agent_status.location, agent.agent_id)
    else:
        occupancy = None

    # Create simulation rules
    rules = ICURules(
        occupancy=occupancy, 
        rng=random.Random(42),
    )

    # Build cognition map
    cognition_map = {}
    
    for agent_id, profile in agents.items():
        if use_llm:
            # LLM-based cognition for staff only
            if agent_id.startswith("dr_") or agent_id.startswith("nurse_"):
                cognition_map[agent_id] = AgentCognition(
                    executor=LLMExecutor(
                        template=ICU_EXECUTOR_TEMPLATE,
                        available_actions=ICU_AVAILABLE_ACTIONS
                    ),
                    planner=LLMPlanner(template_name="plan"),
                    scratchpad=Scratchpad(),
                )
            else:
                # Patients use simple deterministic
                cognition_map[agent_id] = AgentCognition(
                    executor=DeterministicICUExecutor(),
                )
        else:
            # Fully deterministic
            cognition_map[agent_id] = AgentCognition(
                executor=DeterministicICUExecutor(),
                planner=DeterministicICUPlanner(),
                scratchpad=Scratchpad(),
            )

    # Create orchestrator
    orchestrator = Orchestrator(
        world_state=world_state,
        agents=agents,
        world_prompt="You oversee an Intensive Care Unit with critical patients requiring constant monitoring and care.",
        agent_prompts={},  # No additional per-agent prompts needed
        simulation_rules=rules,
        agent_cognition=cognition_map,
        llm_provider=Config.LLM_PROVIDER if use_llm else None,
        llm_model=Config.LLM_MODEL if use_llm else None,
    )

    # Storage for agent thoughts per tick
    agent_thoughts = {}
    cumulative_time_minutes = 0  # Track total elapsed time with dynamic time steps

    # Wrapper to capture agent actions and reasoning
    original_run_tick = orchestrator._run_tick
    
    async def run_tick_with_reporting(tick: int):
        """Wrapper to capture agent thoughts and generate reports."""
        nonlocal cumulative_time_minutes
        
        agent_thoughts[tick] = {}
        
        # Run the normal tick
        await original_run_tick(tick)
        
        # Update cumulative time based on the time step used for this tick
        cumulative_time_minutes += rules.time_step_minutes
        
        # Get the current state after tick
        current_state = orchestrator.current_state
        
        # Write detailed report for this tick
        if report_handle:
            hours = cumulative_time_minutes // 60
            minutes = cumulative_time_minutes % 60
            report_handle.write(f"\n{'='*80}\n")
            report_handle.write(f"TICK {tick} REPORT (Step: {rules.time_step_minutes} min | Elapsed: {hours}h {minutes}m)\n")
            report_handle.write(f"{'='*80}\n\n")
            
            # Patient reports
            report_handle.write(f"--- PATIENT STATUS ---\n")
            for agent_status in current_state.agents:
                if agent_status.agent_id.startswith("patient_"):
                    profile = agents[agent_status.agent_id]
                    health = agent_status.attributes.get("health", Stat(value=0)).value
                    vitals = agent_status.attributes.get("vital_signs", Stat(value="unknown")).value
                    pain = agent_status.attributes.get("pain_level", Stat(value=0)).value
                    vent = agent_status.attributes.get("on_ventilator", Stat(value=False)).value
                    condition = agent_status.attributes.get("condition", Stat(value="unknown")).value
                    severity = agent_status.attributes.get("severity", Stat(value="unknown")).value
                    treatments = agent_status.attributes.get("current_treatments", Stat(value="None")).value
                    interventions = agent_status.attributes.get("interventions_today", Stat(value="None")).value
                    instantaneous = agent_status.attributes.get("instantaneous_interventions", Stat(value="")).value
                    
                    status_icon = "✓" if agent_status.activity == "discharged_to_general_ward" else ("†" if "deceased" in agent_status.activity or "mortuary" in agent_status.activity else "•")
                    
                    report_handle.write(f"\n{status_icon} {profile.name} ({agent_status.agent_id}):\n")
                    report_handle.write(f"  Condition: {condition} (Severity: {severity})\n")
                    report_handle.write(f"  Location: {agent_status.location}\n")
                    report_handle.write(f"  Health: {health}/100\n")
                    report_handle.write(f"  Vital Signs: {vitals}\n")
                    report_handle.write(f"  Pain Level: {pain}/10\n")
                    report_handle.write(f"  On Ventilator: {vent}\n")
                    report_handle.write(f"  Activity: {agent_status.activity}\n")
                    if agent_status.activity == "discharged_to_general_ward":
                        report_handle.write(f"  ** DISCHARGED TO GENERAL WARD **\n")
                    elif "deceased" in agent_status.activity:
                        report_handle.write(f"  ** {agent_status.activity.upper().replace('_', ' ')} **\n")
                    elif "mortuary" in agent_status.activity:
                        report_handle.write(f"  ** {agent_status.activity.upper().replace('_', ' ')} **\n")
                    report_handle.write(f"\n  TREATMENT DETAILS:\n")
                    if treatments and treatments != "None":
                        for treatment in treatments.split("; "):
                            report_handle.write(f"    - {treatment}\n")
                    else:
                        report_handle.write(f"    - No active treatments\n")
                    
                    # Show what changed this tick (instantaneous)
                    if instantaneous and instantaneous != "":
                        report_handle.write(f"\n  THIS TICK:\n")
                        for change in instantaneous.split("; "):
                            if change.strip():
                                report_handle.write(f"    → {change}\n")
                    
                    if interventions and interventions != "None" and interventions != "":
                        report_handle.write(f"\n  CUMULATIVE INTERVENTIONS:\n")
                        for intervention in interventions.split("; "):
                            if intervention.strip():
                                report_handle.write(f"    - {intervention}\n")
            
            # Staff reports
            report_handle.write(f"\n--- STAFF STATUS ---\n")
            for agent_status in current_state.agents:
                if agent_status.agent_id.startswith("dr_") or agent_status.agent_id.startswith("nurse_"):
                    profile = agents[agent_status.agent_id]
                    energy = agent_status.attributes.get("energy", Stat(value=0)).value
                    stress = agent_status.attributes.get("stress", Stat(value=0)).value
                    assigned = agent_status.attributes.get("assigned_patients", Stat(value="none")).value
                    
                    report_handle.write(f"\n{profile.name} ({agent_status.agent_id}) - {profile.role}:\n")
                    report_handle.write(f"  Location: {agent_status.location}\n")
                    report_handle.write(f"  Energy: {energy}/100\n")
                    report_handle.write(f"  Stress: {stress}/100\n")
                    report_handle.write(f"  Assigned Patients: {assigned}\n")
                    report_handle.write(f"  Activity: {agent_status.activity}\n")
            
            # Events
            if current_state.recent_events:
                report_handle.write(f"\n--- EVENTS ---\n")
                for event in current_state.recent_events:
                    if event.tick == tick:
                        report_handle.write(f"  [{event.category}] Severity {event.severity}: {event.description}\n")
            
            # Agent thoughts/reasoning (captured from recent actions)
            report_handle.write(f"\n--- AGENT CLINICAL DECISIONS ---\n")
            # Get actions from persistence for this tick
            try:
                tick_actions = await orchestrator.persistence.get_actions(orchestrator.run_id, tick)
                for agent_id, profile in agents.items():
                    if agent_id.startswith("dr_") or agent_id.startswith("nurse_"):
                        agent_action = next((a for a in tick_actions if a.agent_id == agent_id), None)
                        if agent_action:
                            report_handle.write(f"\n{profile.name} ({agent_id}):\n")
                            report_handle.write(f"  Action: {agent_action.action_type}\n")
                            if agent_action.target:
                                report_handle.write(f"  Target: {agent_action.target}\n")
                            if agent_action.parameters:
                                report_handle.write(f"  Parameters: {agent_action.parameters}\n")
                                # Highlight treatment details
                                if 'treatment' in agent_action.parameters:
                                    report_handle.write(f"  ** TREATMENT: {agent_action.parameters['treatment']} **\n")
                                if 'task' in agent_action.parameters:
                                    report_handle.write(f"  ** TASK: {agent_action.parameters['task']} **\n")
                            if agent_action.reasoning:
                                report_handle.write(f"  Clinical Reasoning: {agent_action.reasoning}\n")
            except Exception as e:
                report_handle.write(f"  (Unable to retrieve action details: {e})\n")
            
            # Agent personal comments/reflections
            report_handle.write(f"\n--- AGENT PERSONAL COMMENTS ---\n")
            try:
                for agent_id, profile in agents.items():
                    if agent_id.startswith("dr_") or agent_id.startswith("nurse_"):
                        agent_status = next((a for a in current_state.agents if a.agent_id == agent_id), None)
                        if agent_status:
                            stress = agent_status.attributes.get("stress", Stat(value=0)).value
                            energy = agent_status.attributes.get("energy", Stat(value=0)).value
                            
                            # Generate personal comment
                            if use_llm:
                                # Use LLM to generate authentic personal reflection
                                try:
                                    from miniverse.local_llm import call_ollama_chat
                                    
                                    # Build context for the comment
                                    recent_events_desc = ""
                                    if current_state.recent_events:
                                        recent_events_desc = "; ".join([e.description for e in current_state.recent_events if e.tick == tick])
                                    
                                    assigned_patients = agent_status.attributes.get("assigned_patients", Stat(value="none")).value
                                    
                                    # Get patient conditions for context
                                    patient_context = []
                                    for pid in str(assigned_patients).split(","):
                                        pid = pid.strip()
                                        if pid and pid != "none":
                                            patient_status = next((a for a in current_state.agents if a.agent_id == pid), None)
                                            if patient_status:
                                                health = patient_status.attributes.get("health", Stat(value=0)).value
                                                condition = patient_status.attributes.get("condition", Stat(value="")).value
                                                patient_context.append(f"{pid}: {condition} (health {health}/100)")
                                    
                                    patient_info = "; ".join(patient_context) if patient_context else "no specific assignments"
                                    
                                    comment_prompt = f"""You are {profile.name}, a {profile.role} working in the ICU. {profile.background}

Current situation at this moment:
- Your stress level is {stress}/100 (0=relaxed, 100=overwhelmed)
- Your energy level is {energy}/100 (0=exhausted, 100=fully energized)  
- Patients you're responsible for: {patient_info}
- Recent events this period: {recent_events_desc if recent_events_desc else "Routine monitoring and care"}
- Current activity: {agent_status.activity}

Write a brief personal reflection (2-3 sentences) about how you're feeling right now and what's on your mind. Be specific about the clinical situation, mention concerns or observations about patients by condition (not ID), and be authentic about your emotional state. This is your private thought - be honest.

Respond with ONLY the reflection itself, no quotes or labels."""
                                    
                                    # Call LLM using await (we're already in async context)
                                    if Config.LLM_PROVIDER == "ollama":
                                        comment = await call_ollama_chat(
                                            system_prompt="You are a healthcare professional reflecting on your current situation.",
                                            user_prompt=comment_prompt,
                                            llm_model=Config.LLM_MODEL,
                                        )
                                        comment = comment.strip().strip('"').strip("'").strip()
                                    else:
                                        # For other providers, use a simple fallback
                                        comment = "Focused on providing the best care possible to my patients."
                                    
                                except Exception as e:
                                    # Log the error for debugging
                                    import traceback
                                    log_info(f"⚠️  LLM comment generation failed for {profile.name}: {str(e)}")
                                    log_info(f"Traceback: {traceback.format_exc()}")
                                    # Fallback to template comments if LLM fails
                                    if stress > 70:
                                        comment = "This shift is extremely intense. Every alarm feels critical right now."
                                    elif stress > 50:
                                        comment = "Managing multiple critical patients is mentally draining, but I'm staying focused."
                                    elif stress > 30:
                                        comment = "The ICU is busy today, but I feel we're providing good care to everyone."
                                    else:
                                        comment = "Maintaining steady vigilance. All patients are stable for now."
                                    
                                    if energy < 40:
                                        comment += " I'm getting fatigued and need to be extra careful."
                            else:
                                # Deterministic mode - use template comments
                                comment = ""
                                if stress > 70:
                                    comment = "This shift is extremely intense. The code blue really elevated my stress levels."
                                elif stress > 50:
                                    comment = "Managing multiple critical patients is challenging today."
                                elif stress > 30:
                                    comment = "Staying focused on providing quality care despite the busy ICU."
                                else:
                                    comment = "Maintaining steady vigilance with all patients."
                                
                                if energy < 40:
                                    comment += " Feeling fatigued - need to stay alert."
                            
                            report_handle.write(f"\n{profile.name}: \"{comment}\"\n")
            except Exception as e:
                report_handle.write(f"  (Unable to generate comments: {e})\n")
            
            report_handle.write(f"\n")
            report_handle.flush()
    
    # Replace the tick method
    orchestrator._run_tick = run_tick_with_reporting

    # Run simulation with retry logic for Ollama crashes
    log_info(f"\n{'='*80}")
    log_info(f"ICU SIMULATION - {'LLM' if use_llm else 'Deterministic'} Mode")
    log_info(f"{'='*80}")
    log_info(f"Target Duration: {target_duration_minutes / 60:.1f} hours ({target_duration_minutes} minutes)")
    log_info(f"Dynamic time steps: 5-30 minutes based on event severity\n")
    
    max_retries = 3
    retry_count = 0
    result = None
    
    # Run simulation until we reach target duration or max ticks
    while retry_count < max_retries:
        try:
            # We'll run tick by tick and check elapsed time
            current_tick = 0
            while current_tick < max_ticks and cumulative_time_minutes < target_duration_minutes:
                await orchestrator._run_tick(current_tick + 1)
                current_tick += 1
            
            # Package result similar to orchestrator.run()
            result = {
                "final_state": orchestrator.current_state,
                "run_id": orchestrator.run_id,
            }
            break  # Success!
        except Exception as e:
            error_msg = str(e)
            if "ollama" in error_msg.lower() and ("connection refused" in error_msg.lower() or 
                                                   "unexpectedly stopped" in error_msg.lower()):
                retry_count += 1
                if retry_count < max_retries:
                    log_info(f"\n⚠️  Ollama crashed. Retrying ({retry_count}/{max_retries})...")
                    await asyncio.sleep(2)  # Wait before retry
                else:
                    log_info(f"\n❌ Ollama failed after {max_retries} attempts. Try restarting Ollama with: ollama serve")
                    raise
            else:
                raise  # Not an Ollama crash, re-raise
    
    if result is None:
        raise RuntimeError(f"Simulation failed after {max_retries} retries")
    
    # Close report file if open
    if report_handle:
        final_hours = cumulative_time_minutes // 60
        final_minutes = cumulative_time_minutes % 60
        report_handle.write(f"\n{'='*80}\n")
        report_handle.write(f"SIMULATION COMPLETE\n")
        report_handle.write(f"{'='*80}\n")
        report_handle.write(f"\nFINAL STATISTICS:\n")
        final_state = result["final_state"]
        report_handle.write(f"  Final tick: {final_state.tick}\n")
        report_handle.write(f"  Total elapsed time: {final_hours}h {final_minutes}m ({cumulative_time_minutes} minutes)\n")
        report_handle.write(f"  Active patients health: {final_state.resources.metrics.get('avg_patient_health', Stat(value=0)).value}\n")
        report_handle.write(f"  Stable patients: {final_state.resources.metrics.get('patients_stable', Stat(value=0)).value}\n")
        report_handle.write(f"  Critical patients: {final_state.resources.metrics.get('patients_critical', Stat(value=0)).value}\n")
        report_handle.write(f"  Discharged patients: {final_state.resources.metrics.get('patients_discharged', Stat(value=0)).value}\n")
        report_handle.write(f"  Deceased patients: {final_state.resources.metrics.get('patients_deceased', Stat(value=0)).value}\n")
        report_handle.write(f"  Medications administered: {final_state.resources.metrics.get('medications_administered', Stat(value=0)).value}\n")
        report_handle.close()
        log_info(f"\n✓ Detailed report written to: {report_file}")
    
    # Print summary
    final_state = result["final_state"]
    final_hours = cumulative_time_minutes // 60
    final_mins = cumulative_time_minutes % 60
    log_info(f"\n{'='*80}")
    log_info("SIMULATION COMPLETE")
    log_info(f"{'='*80}")
    log_info(f"Total ticks: {final_state.tick}")
    log_info(f"Elapsed time: {final_hours}h {final_mins}m ({cumulative_time_minutes} minutes)")
    log_info(f"Average patient health: {final_state.resources.metrics.get('avg_patient_health', Stat(value=0)).value}")
    log_info(f"Stable patients: {final_state.resources.metrics.get('patients_stable', Stat(value=0)).value}")
    log_info(f"Critical patients: {final_state.resources.metrics.get('patients_critical', Stat(value=0)).value}")
    log_info(f"Discharged patients: {final_state.resources.metrics.get('patients_discharged', Stat(value=0)).value}")
    log_info(f"Deceased patients: {final_state.resources.metrics.get('patients_deceased', Stat(value=0)).value}")
    log_info(f"Medications administered: {final_state.resources.metrics.get('medications_administered', Stat(value=0)).value}")
    
    return result


async def main(args):
    """Main entry point."""
    # Build scenario programmatically
    from datetime import datetime, timezone
    from miniverse import AgentProfile, AgentStatus, WorldState, EnvironmentState, ResourceState, Stat
    from miniverse.environment.graph import EnvironmentGraph, LocationNode
    from miniverse.environment.schemas import EnvironmentGraphState, LocationNodeState
    
    # Create environment graph
    nodes = {
        f"icu_bed_{i}": LocationNodeState(
            name=f"ICU Bed {i}",
            capacity=2,
            metadata={"bed_type": "critical" if i <= 4 else "stable"}
        )
        for i in range(1, 7)  # 6 ICU beds
    }
    nodes["nurses_station"] = LocationNodeState(name="Nurses Station", capacity=8)
    nodes["medication_room"] = LocationNodeState(name="Medication Room", capacity=2)
    nodes["supply_room"] = LocationNodeState(name="Supply Room", capacity=2)
    
    adjacency = {
        "nurses_station": [f"icu_bed_{i}" for i in range(1, 7)] + ["medication_room", "supply_room"],
        **{f"icu_bed_{i}": ["nurses_station"] for i in range(1, 7)},
        "medication_room": ["nurses_station"],
        "supply_room": ["nurses_station"],
    }
    
    env_graph = EnvironmentGraphState(nodes=nodes, adjacency=adjacency)
    
    # Create agent profiles (separate from status)
    agents = {}
    agent_statuses = []
    
    # Doctor
    agents["dr_wilson"] = AgentProfile(
        agent_id="dr_wilson",
        name="Dr. Emily Wilson",
        role="icu_doctor",
        age=42,
        personality="Decisive, calm under pressure, detail-oriented",
        background="I am a board-certified intensivist with 15 years of critical care experience. I take pride in staying calm during emergencies and making evidence-based decisions.",
        skills={"critical_care": "expert", "diagnosis": "expert"},
        goals=["Ensure all ICU patients receive optimal care", "Respond quickly to patient deterioration"],
        relationships={},
    )
    agent_statuses.append(AgentStatus(
        agent_id="dr_wilson",
        location="nurses_station",
        activity="reviewing_charts",
        attributes={
            "energy": Stat(value=85, label="Energy"),
            "stress": Stat(value=20, label="Stress"),
        }
    ))
    
    # Nurses
    nurse_data = [
        ("nurse_sarah", "Sarah Martinez", 34, "icu_bed_1", ["patient_a", "patient_b"]),
        ("nurse_james", "James Chen", 29, "icu_bed_3", ["patient_c", "patient_d"]),
        ("nurse_patricia", "Patricia Anderson", 38, "icu_bed_5", ["patient_e", "patient_f"]),
    ]
    
    for agent_id, name, age, location, assigned in nurse_data:
        agents[agent_id] = AgentProfile(
            agent_id=agent_id,
            name=name,
            role="icu_nurse",
            age=age,
            personality="Skilled and compassionate ICU nurse",
            background=f"I am {name} and I provide critical care to ICU patients with dedication and skill.",
            skills={"patient_monitoring": "expert", "medication_administration": "advanced"},
            goals=["Monitor assigned patients", "Administer medications", "Document care"],
            relationships={},
        )
        agent_statuses.append(AgentStatus(
            agent_id=agent_id,
            location=location,
            activity="patient_care",
            attributes={
                "energy": Stat(value=75, label="Energy"),
                "stress": Stat(value=30, label="Stress"),
                "assigned_patients": Stat(value=",".join(assigned), label="Assigned Patients"),
            }
        ))
    
    # Patients with realistic medical conditions
    patient_data = [
        # Patient A: Pneumonia with respiratory failure - stable, improving
        ("patient_a", "Robert Harris", 68, "icu_bed_1", 65, "stable", True, 
         "Severe pneumonia with respiratory failure", "moderate", 
         "Admitted 2 days ago with bacterial pneumonia. Currently on mechanical ventilation, showing improvement."),
        
        # Patient B: Septic shock - HIGH RISK for deterioration/code blue
        ("patient_b", "Linda Chen", 55, "icu_bed_2", 48, "unstable", False,
         "Septic shock from abdominal infection", "severe",
         "Post-operative day 1 from emergency bowel surgery. Developed septic shock. High risk for cardiovascular collapse despite fluids and pressors."),
        
        # Patient C: Post-cardiac arrest, recovering
        ("patient_c", "Marcus Johnson", 70, "icu_bed_3", 55, "critical", True,
         "Post-cardiac arrest, therapeutic hypothermia", "severe",
         "Survived out-of-hospital cardiac arrest 18 hours ago. Under therapeutic hypothermia protocol. Prognosis uncertain."),
        
        # Patient D: Diabetic ketoacidosis - improving
        ("patient_d", "Elena Rodriguez", 41, "icu_bed_4", 70, "stable",  False,
         "Diabetic ketoacidosis", "moderate",
         "Admitted with severe diabetic ketoacidosis. Responded well to insulin and fluids. Expected to transfer to general ward soon."),
        
        # Patient E: Acute pancreatitis - moderate condition
        ("patient_e", "Thomas Wright", 52, "icu_bed_5", 60, "unstable", False,
         "Acute pancreatitis", "moderate",
         "Admitted with severe acute pancreatitis. Requiring aggressive fluid resuscitation and pain management."),
        
        # Patient F: Hemorrhagic stroke - critical
        ("patient_f", "Maria Gonzalez", 63, "icu_bed_6", 45, "critical", False,
         "Hemorrhagic stroke", "severe",
         "Admitted with intracerebral hemorrhage. Close neuro monitoring required. Risk of increased intracranial pressure."),
    ]
    
    for agent_id, name, age, location, health, vitals, on_vent, condition, severity, history in patient_data:
        agents[agent_id] = AgentProfile(
            agent_id=agent_id,
            name=name,
            role="patient",
            age=age,
            occupation="Patient",
            personality="ICU patient in critical condition",
            background=f"I am {name}. Medical history: {history}",
            skills={},
            goals=["Survive and recover", "Follow medical advice"],
            relationships={},
        )
        
        # Initialize treatment tracking
        treatments = []
        if on_vent:
            treatments.append("Mechanical ventilation")
        if "septic" in condition.lower():
            treatments.extend(["Broad-spectrum antibiotics (Vancomycin, Meropenem)", "IV fluids (crystalloids)", "Vasopressors (Norepinephrine)"])
        if "pneumonia" in condition.lower():
            treatments.extend(["Antibiotics (Ceftriaxone, Azithromycin)", "Supplemental oxygen"])
        if "cardiac" in condition.lower() or "heart" in condition.lower():
            treatments.extend(["Aspirin", "Beta-blockers", "Anticoagulation", "Continuous cardiac monitoring"])
        if "diabetic" in condition.lower() or "ketoacidosis" in condition.lower():
            treatments.extend(["IV insulin infusion", "Potassium replacement", "Fluid resuscitation"])
        if "pancreatitis" in condition.lower():
            treatments.extend(["IV fluids (aggressive resuscitation)", "Pain management (Fentanyl)", "NPO (nothing by mouth)", "Antiemetics"])
        if "stroke" in condition.lower():
            treatments.extend(["Blood pressure management", "Neuro checks q1h", "Head of bed elevated", "Seizure precautions"])
        
        agent_statuses.append(AgentStatus(
            agent_id=agent_id,
            location=location,
            activity="resting",
            attributes={
                "health": Stat(value=health, label="Health"),
                "pain_level": Stat(value=3, label="Pain Level"),
                "on_ventilator": Stat(value=on_vent, label="On Ventilator"),
                "vital_signs": Stat(value=vitals, label="Vital Signs"),
                "condition": Stat(value=condition, label="Medical Condition"),
                "severity": Stat(value=severity, label="Severity"),
                "medical_history": Stat(value=history, label="Medical History"),
                "current_treatments": Stat(value="; ".join(treatments), label="Current Treatments"),
                "interventions_today": Stat(value="", label="Interventions Today"),
            }
        ))
    
    # Create world state
    world_state = WorldState(
        tick=0,
        timestamp=datetime(2025, 11, 17, 8, 0, 0, tzinfo=timezone.utc),
        agents=agent_statuses,
        environment=EnvironmentState(
            metrics={"icu_status": Stat(value="operational", label="ICU Status")},
            environment_graph=env_graph,
            metadata={}
        ),
        resources=ResourceState(
            metrics={
                "patients_stable": Stat(value=0, label="Stable Patients"),
                "patients_critical": Stat(value=0, label="Critical Patients"),
                "avg_patient_health": Stat(value=0, label="Average Patient Health"),
                "medications_administered": Stat(value=0, label="Medications Administered"),
            }
        ),
        recent_events=[],
    )
    
    # Run simulation
    target_duration_minutes = int(args.duration * 60)
    await run_simulation(
        world_state=world_state,
        agents=agents,
        max_ticks=args.max_ticks,
        target_duration_minutes=target_duration_minutes,
        use_llm=args.llm,
        report_file=args.report_file,
    )


if __name__ == "__main__":
    import sys
    import os
    
    parser = argparse.ArgumentParser(description="ICU simulation")
    parser.add_argument("--duration", type=float, default=3.0, help="Simulation duration in hours (e.g., 3.0 for 3 hours)")
    parser.add_argument("--llm", action="store_true", help="Use LLM for cognition (requires LLM_PROVIDER and LLM_MODEL env vars)")
    parser.add_argument("--log-file", type=str, default=None, help="Write debug output to file instead of terminal")
    parser.add_argument("--report-file", type=str, default="icu_report.txt", help="Path to write detailed tick-by-tick report (default: icu_report.txt)")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG_LLM logging")
    
    args = parser.parse_args()
    
    # Redirect output to file if requested
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file_handle = None
    
    if args.log_file:
        log_file_handle = open(args.log_file, 'w')
        sys.stdout = log_file_handle
        sys.stderr = log_file_handle
    
    # Enable debug logging if requested (either via --debug flag or DEBUG_LLM env var)
    if args.debug:
        os.environ.setdefault('DEBUG_LLM', 'true')
    
    # Convert duration to maximum ticks (with generous buffer since dynamic time steps vary)
    # Worst case: all 5-minute intervals = duration_hours * 60 / 5 ticks
    # We'll use 2x buffer to ensure we hit the duration
    args.max_ticks = int(args.duration * 60 / 5 * 2)
    
    try:
        asyncio.run(main(args))
    finally:
        # Restore original stdout/stderr
        if log_file_handle:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file_handle.close()
            print(f"✓ Simulation output written to: {args.log_file}")
