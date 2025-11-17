"""ICU Critical Care simulation.

Simplified ICU scenario focusing on:
- 1 attending physician (Dr. Wilson)
- 4 ICU nurses (Sarah, James, Priya, Michael)
- 8 patients in various conditions
- Events from outside ICU (transfers, lab results, etc.) are referenced but not explicitly modeled

By default runs in deterministic mode (no LLM calls):
    uv run python examples/icu/run.py --ticks 12

To enable LLM-driven cognition:
    export LLM_PROVIDER=ollama
    export LLM_MODEL=llama3.1:8b
    uv run python examples/icu/run.py --llm --ticks 12

Each tick represents 1 hour of ICU time.
"""

from __future__ import annotations

import argparse
import asyncio
import random
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

    def apply_tick(self, state: WorldState, tick: int) -> WorldState:
        """Update patient health, staff energy, and generate events."""
        new_state = state.model_copy(deep=True)
        new_state.tick = tick

        # Track patient metrics
        patient_health_sum = 0
        patient_count = 0
        stable_count = 0
        critical_count = 0

        # Update patient attributes
        for agent_status in new_state.agents:
            if not agent_status.agent_id.startswith("patient_"):
                continue

            patient_count += 1
            health = agent_status.attributes.get("health", Stat(value=50)).value
            patient_health_sum += health

            # Count stability
            vital_signs = agent_status.attributes.get("vital_signs", Stat(value="stable")).value
            if "unstable" in str(vital_signs) or health < 50:
                critical_count += 1
            else:
                stable_count += 1

            # Simulate gradual health changes
            if health < 70 and "stable" in str(vital_signs):
                # Stable patients slowly improve
                new_health = min(health + self.rng.randint(1, 3), 100)
                agent_status.attributes["health"] = Stat(value=new_health, label="Health")
            elif "unstable" in str(vital_signs):
                # Unstable patients may deteriorate
                change = self.rng.randint(-2, 1)
                new_health = max(health + change, 20)
                agent_status.attributes["health"] = Stat(value=new_health, label="Health")

        # Update staff energy (decreases over shift)
        for agent_status in new_state.agents:
            if agent_status.agent_id.startswith("nurse_") or agent_status.agent_id.startswith("dr_"):
                energy = agent_status.attributes.get("energy", Stat(value=80)).value
                stress = agent_status.attributes.get("stress", Stat(value=20)).value

                # Energy decreases, stress increases slightly
                new_energy = max(energy - self.rng.randint(1, 3), 20)
                new_stress = min(stress + self.rng.randint(0, 2), 100)
                agent_status.attributes["energy"] = Stat(value=new_energy, label="Energy")
                agent_status.attributes["stress"] = Stat(value=new_stress, label="Stress")

        # Update resource metrics
        avg_health = patient_health_sum / patient_count if patient_count > 0 else 0
        new_state.resources.metrics["avg_patient_health"].value = round(avg_health, 1)
        new_state.resources.metrics["patients_stable"].value = stable_count
        new_state.resources.metrics["patients_critical"].value = critical_count

        # Random events from outside ICU
        if tick > 0 and self.rng.random() < 0.2:
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
    ticks: int,
    use_llm: bool = False,
) -> dict:
    """Run the ICU simulation."""
    
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
    rules = ICURules(occupancy=occupancy, rng=random.Random(42))

    # Build cognition map
    cognition_map = {}
    
    for agent_id, profile in agents.items():
        if use_llm:
            # LLM-based cognition for staff only
            if agent_id.startswith("dr_") or agent_id.startswith("nurse_"):
                cognition_map[agent_id] = AgentCognition(
                    executor=LLMExecutor(template_name="default"),
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

    # Run simulation
    log_info(f"\n{'='*80}")
    log_info(f"ICU SIMULATION - {'LLM' if use_llm else 'Deterministic'} Mode")
    log_info(f"{'='*80}\n")
    
    result = await orchestrator.run(num_ticks=ticks)
    
    # Print summary
    final_state = result["final_state"]
    log_info(f"\n{'='*80}")
    log_info("SIMULATION COMPLETE")
    log_info(f"{'='*80}")
    log_info(f"Final tick: {final_state.tick}")
    log_info(f"Average patient health: {final_state.resources.metrics['avg_patient_health'].value}")
    log_info(f"Stable patients: {final_state.resources.metrics['patients_stable'].value}")
    log_info(f"Critical patients: {final_state.resources.metrics['patients_critical'].value}")
    log_info(f"Medications administered: {final_state.resources.metrics['medications_administered'].value}")
    
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
        for i in range(1, 9)
    }
    nodes["nurses_station"] = LocationNodeState(name="Nurses Station", capacity=8)
    nodes["medication_room"] = LocationNodeState(name="Medication Room", capacity=2)
    nodes["supply_room"] = LocationNodeState(name="Supply Room", capacity=2)
    
    adjacency = {
        "nurses_station": [f"icu_bed_{i}" for i in range(1, 9)] + ["medication_room", "supply_room"],
        **{f"icu_bed_{i}": ["nurses_station"] for i in range(1, 9)},
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
        occupation="ICU Attending Physician",
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
    ]
    
    for agent_id, name, age, location, assigned in nurse_data:
        agents[agent_id] = AgentProfile(
            agent_id=agent_id,
            name=name,
            role="icu_nurse",
            age=age,
            occupation="ICU Nurse",
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
    
    # Patients
    patient_data = [
        ("patient_a", "Robert Harris", 68, "icu_bed_1", 60, "critical", True),
        ("patient_b", "Linda Chen", 55, "icu_bed_2", 55, "critical", True),
        ("patient_c", "Marcus Johnson", 70, "icu_bed_3", 45, "unstable", False),
        ("patient_d", "Elena Rodriguez", 41, "icu_bed_4", 50, "stable_but_painful", False),
    ]
    
    for agent_id, name, age, location, health, vitals, on_vent in patient_data:
        agents[agent_id] = AgentProfile(
            agent_id=agent_id,
            name=name,
            role="patient",
            age=age,
            occupation="Patient",
            personality="ICU patient recovering from illness or injury",
            background=f"I am {name} and I am currently receiving critical care in the ICU.",
            skills={},
            goals=["Recover from illness", "Follow medical advice"],
            relationships={},
        )
        agent_statuses.append(AgentStatus(
            agent_id=agent_id,
            location=location,
            activity="resting",
            attributes={
                "health": Stat(value=health, label="Health"),
                "pain_level": Stat(value=3, label="Pain Level"),
                "on_ventilator": Stat(value=on_vent, label="On Ventilator"),
                "vital_signs": Stat(value=vitals, label="Vital Signs"),
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
    await run_simulation(
        world_state=world_state,
        agents=agents,
        ticks=args.ticks,
        use_llm=args.llm,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICU simulation")
    parser.add_argument("--ticks", type=int, default=12, help="Number of ticks (hours) to simulate")
    parser.add_argument("--llm", action="store_true", help="Use LLM for cognition (requires LLM_PROVIDER and LLM_MODEL env vars)")
    
    args = parser.parse_args()
    asyncio.run(main(args))
