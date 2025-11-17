"""Hospital simulation showcasing ER, ICU, imaging, and surgical coordination.

Run in deterministic mode (no LLM calls):

    uv run python examples/hospital/run.py --ticks 6

Enable LLM-driven cognition (requires OpenAI `gpt-5-nano` credentials):

    export LLM_PROVIDER=openai
    export LLM_MODEL=gpt-5-nano
    export OPENAI_API_KEY=your_key
    uv run python examples/hospital/run.py --llm --ticks 8
"""

from __future__ import annotations

import argparse
import asyncio
import random
from pathlib import Path
from typing import Dict, List

from miniverse import (
    AgentAction,
    AgentCognition,
    AgentProfile,
    EnvironmentGraph,
    EnvironmentGraphState,
    GraphOccupancy,
    LocationNode,
    Orchestrator,
    Plan,
    PlanStep,
    ReflectionResult,
    SimulationRules,
    WorldState,
)
from miniverse.cognition import (
    LLMPlanner,
    LLMReflectionEngine,
    PromptLibrary,
    PromptTemplate,
    Scratchpad,
)
from miniverse.cognition.context import PromptContext
from miniverse.cognition.executor import Executor
from miniverse.cognition.llm import LLMExecutor
from miniverse.cognition.planner import Planner
from miniverse.cognition.reflection import ReflectionEngine
from miniverse.config import Config
from miniverse.scenario import ScenarioLoader


ROLE_PLAN_TEMPLATES: Dict[str, List[Dict[str, object]]] = {
    "er_doctor": [
        {"description": "triage incoming patients", "metadata": {"location": "er_triage"}},
        {"description": "coordinate imaging", "metadata": {"location": "imaging_ct"}},
        {"description": "transition stabilized patient to icu", "metadata": {"location": "icu"}},
    ],
    "er_nurse": [
        {"description": "move patients into bays", "metadata": {"location": "er_treatment"}},
        {"description": "coach families", "metadata": {"location": "waiting_room"}},
    ],
    "icu_nurse": [
        {"description": "assess vents and drips", "metadata": {"location": "icu"}},
        {"description": "update families", "metadata": {"location": "family_lounge"}},
    ],
    "imaging_tech": [
        {"description": "prep scanner", "metadata": {"location": "imaging_mri"}},
        {"description": "run stat scan", "metadata": {"location": "imaging_ct"}},
    ],
    "surgeon": [
        {"description": "prep operating room", "metadata": {"location": "operating_room_a"}},
        {"description": "coordinate recovery", "metadata": {"location": "recovery"}},
    ],
    "transport_aide": [
        {"description": "transport patient", "metadata": {"location": "er_triage"}},
        {"description": "sanitize stretchers", "metadata": {"location": "pharmacy"}},
    ],
    "hospital_admin": [
        {"description": "review staffing", "metadata": {"location": "administration"}},
        {"description": "visit er triage", "metadata": {"location": "er_triage"}},
    ],
    "patient": [
        {"description": "ask for updates", "metadata": {"location": "waiting_room"}},
        {"description": "focus on breathing", "metadata": {"location": "waiting_room"}},
    ],
    "visitor": [
        {"description": "check on loved one", "metadata": {"location": "family_lounge"}},
        {"description": "coordinate with staff", "metadata": {"location": "er_triage"}},
    ],
    "default": [
        {"description": "observe floor status", "metadata": {}},
    ],
}


def build_environment_graph(state: EnvironmentGraphState | None) -> EnvironmentGraph | None:
    if state is None:
        return None
    nodes = {
        node_id: LocationNode(
            name=node_state.name,
            capacity=node_state.capacity,
            metadata=node_state.metadata,
        )
        for node_id, node_state in state.nodes.items()
    }
    return EnvironmentGraph(nodes=nodes, adjacency=dict(state.adjacency))


class HospitalRules(SimulationRules):
    """Simple deterministic updates reflecting hospital throughput."""

    CAREGIVER_KEYWORDS = ("doctor", "nurse", "surgeon", "tech", "aide")

    def __init__(
        self,
        occupancy: GraphOccupancy | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.occupancy = occupancy
        self.rng = rng

    def apply_tick(self, state: WorldState, tick: int) -> WorldState:
        updated = state.model_copy(deep=True)
        resources = updated.resources

        er_wait = resources.get_metric("er_waiting_patients", default=5, label="ER Waiting Patients")
        icu_beds = resources.get_metric("icu_beds_available", default=1, label="ICU Beds Available")
        imaging_queue = resources.get_metric("imaging_queue", default=2, label="Imaging Queue")
        or_cases = resources.get_metric("or_cases_pending", default=1, label="OR Cases Pending")
        fatigue = resources.get_metric("staff_fatigue_index", default=40, unit="%", label="Staff Fatigue")

        caregivers = 0
        resting_staff = 0
        imaging_staff = 0
        surgeons_active = 0

        for agent in updated.agents:
            role = (agent.role or "").lower()
            if any(keyword in role for keyword in self.CAREGIVER_KEYWORDS):
                caregivers += 1
                energy = agent.attributes.get("energy")
                stress = agent.attributes.get("stress")
                if energy is not None:
                    energy.value = max(20.0, float(energy.value) - 3.0)
                if stress is not None:
                    stress.value = min(100.0, float(stress.value) + 1.5)
            if role.startswith("patient") or role == "patient" or role == "visitor":
                continue
            if agent.location == "staff_break_room":
                resting_staff += 1
            if "imaging" in role:
                imaging_staff += 1
            if "surgeon" in role and agent.location in {"operating_room_a", "operating_room_b"}:
                surgeons_active += 1

        throughput = max(1, caregivers // 2)
        er_wait.value = max(0, int(er_wait.value) - throughput)
        arrivals = self.rng.randint(0, 2) if self.rng else 1
        er_wait.value = max(0, int(er_wait.value) + arrivals)

        if int(er_wait.value) > 6 and int(icu_beds.value) > 0:
            icu_beds.value = int(icu_beds.value) - 1
        elif int(er_wait.value) <= 4 and int(icu_beds.value) < 3:
            icu_beds.value = int(icu_beds.value) + 1

        imaging_clear = max(1, imaging_staff)
        imaging_queue.value = max(0, int(imaging_queue.value) - imaging_clear)
        if self.rng and self.rng.random() < 0.4:
            imaging_queue.value += 1

        if surgeons_active:
            or_cases.value = max(0, int(or_cases.value) - surgeons_active)
        if self.rng and self.rng.random() < 0.3:
            or_cases.value += 1

        fatigue.value = min(100.0, max(0.0, float(fatigue.value) + caregivers * 0.5 - resting_staff * 1.8))

        updated.tick = tick
        return updated

    def validate_action(self, action: AgentAction, state: WorldState) -> bool:
        if action.action_type == "move" and self.occupancy:
            target = action.target
            if not target:
                return False
            return self.occupancy.can_enter(target, action.agent_id)
        return True


class HospitalDeterministicPlanner(Planner):
    def __init__(self, templates: Dict[str, List[Dict[str, object]]]):
        self.templates = templates

    async def generate_plan(
        self,
        agent_id: str,
        scratchpad: Scratchpad,
        *,
        world_context: WorldState,
        context: PromptContext,
    ) -> Plan:
        profile: AgentProfile = context.agent_profile
        role_key = profile.role or "default"
        steps_data = self.templates.get(role_key, self.templates["default"])
        steps = [
            PlanStep(description=item["description"], metadata=item.get("metadata", {}))
            for item in steps_data
        ]
        return Plan(steps=steps, metadata={"role": role_key})


class HospitalDeterministicExecutor(Executor):
    ROLE_ACTIONS: Dict[str, Dict[str, Dict[str, str]]] = {
        "er_doctor": {
            "triage incoming patients": {"action": "triage", "target": "er_triage"},
            "coordinate imaging": {"action": "coordinate", "target": "imaging_ct"},
            "transition stabilized patient to icu": {"action": "handoff", "target": "icu"},
        },
        "er_nurse": {
            "move patients into bays": {"action": "move", "target": "er_treatment"},
            "coach families": {"action": "communicate", "target": "waiting_room"},
        },
        "icu_nurse": {
            "assess vents and drips": {"action": "care", "target": "icu"},
            "update families": {"action": "communicate", "target": "family_lounge"},
        },
        "imaging_tech": {
            "prep scanner": {"action": "prepare", "target": "imaging_mri"},
            "run stat scan": {"action": "scan", "target": "imaging_ct"},
        },
        "surgeon": {
            "prep operating room": {"action": "operate", "target": "operating_room_a"},
            "coordinate recovery": {"action": "coordinate", "target": "recovery"},
        },
        "transport_aide": {
            "transport patient": {"action": "transport", "target": "er_triage"},
            "sanitize stretchers": {"action": "sanitize", "target": "pharmacy"},
        },
        "hospital_admin": {
            "review staffing": {"action": "analyze", "target": "administration"},
            "visit er triage": {"action": "move", "target": "er_triage"},
        },
        "patient": {
            "ask for updates": {"action": "communicate", "target": "waiting_room"},
            "focus on breathing": {"action": "rest", "target": "waiting_room"},
        },
        "visitor": {
            "check on loved one": {"action": "comfort", "target": "family_lounge"},
            "coordinate with staff": {"action": "communicate", "target": "er_triage"},
        },
    }

    async def choose_action(
        self,
        agent_id: str,
        perception,
        scratchpad: Scratchpad,
        *,
        plan: Plan,
        plan_step: PlanStep | None,
        context: PromptContext,
    ) -> AgentAction:
        profile: AgentProfile = context.agent_profile
        role_key = profile.role or "default"
        action_map = self.ROLE_ACTIONS.get(role_key, {})
        if plan_step is None:
            action_type = "observe"
            target = perception.location
            reasoning = "No plan step available"
        else:
            mapped = action_map.get(plan_step.description)
            if mapped:
                action_type = mapped["action"]
                target = mapped["target"]
            else:
                action_type = "observe"
                target = perception.location
            reasoning = f"Following plan step '{plan_step.description}'"
        return AgentAction(
            agent_id=agent_id,
            tick=perception.tick,
            action_type=action_type,
            target=target,
            parameters={"plan_step": plan_step.description if plan_step else "none"},
            reasoning=reasoning,
            communication=None,
        )


class HospitalDeterministicReflection(ReflectionEngine):
    async def maybe_reflect(
        self,
        agent_id: str,
        scratchpad: Scratchpad,
        recent_memories,
        *,
        trigger_context=None,
        context: PromptContext | None = None,
    ) -> List[ReflectionResult]:
        tick = (trigger_context or {}).get("tick", 0)
        if tick % 3 != 0:
            return []
        latest = next(iter(recent_memories), None)
        if latest is None:
            content = "Reviewed shift status and stayed ready."
        else:
            content = f"Noted: {latest.content}"
        return [ReflectionResult(content=content, importance=6)]


class HospitalTickReporter:
    def report(
        self,
        tick: int,
        previous_state: WorldState,
        new_state: WorldState,
        actions: List[AgentAction],
    ) -> None:
        def metric_val(state: WorldState, key: str) -> int | float | None:
            metric = state.resources.metrics.get(key)
            if metric is None:
                return None
            return metric.value

        er_prev = metric_val(previous_state, "er_waiting_patients")
        er_new = metric_val(new_state, "er_waiting_patients")
        icu_new = metric_val(new_state, "icu_beds_available")
        imaging_new = metric_val(new_state, "imaging_queue")

        print(f"  [Tick {tick}] ER waiting: {er_prev} -> {er_new} | ICU beds: {icu_new} | Imaging queue: {imaging_new}")
        if actions:
            summary = ", ".join(f"{a.agent_id}:{a.action_type}" for a in actions)
            print(f"    Actions: {summary}")


def build_prompt_library() -> PromptLibrary:
    library = PromptLibrary()
    library.register(
        PromptTemplate(
            name="plan_hospital",
            system=(
                "You are a hospital professional planning the next few moves. Use the context"
                " to keep ER flow, ICU beds, imaging scanners, and operating rooms synchronized."
            ),
            user=(
                "Context summary:\n{{context_summary}}\n\n"
                "Environment JSON:\n{{context_json}}\n\n"
                "Return JSON matching {\"steps\": [{\"description\": str, \"metadata\": dict}]}."
            ),
        )
    )
    library.register(
        PromptTemplate(
            name="execute_hospital",
            system=(
                "Choose the next action that respects room capacity and available actions."
                " Respond with AgentAction JSON only."
            ),
            user=(
                "Perception:\n{{perception_json}}\n\n"
                "Plan:\n{{plan_json}}\n\n"
                "Recent memories:\n{{memories_text}}\n\n"
                "Action catalog:\n{{action_catalog}}\n\n"
                "Respond with JSON."
            ),
        )
    )
    library.register(
        PromptTemplate(
            name="reflect_hospital",
            system="Write a concise diary entry linking observations to ER/ICU/OR priorities. Return JSON.",
            user=(
                "Summary:\n{{context_summary}}\n\n"
                "Details:\n{{context_json}}\n\n"
                "Use format {\"reflections\": [{\"content\": str, \"importance\": int}]}."
            ),
        )
    )
    return library


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Riverside General Hospital simulation")
    parser.add_argument("--llm", action="store_true", help="Use LLM cognition stack")
    parser.add_argument("--ticks", type=int, default=6, help="Number of ticks to simulate")
    parser.add_argument("--analysis", action="store_true", help="Print per-tick hospital metrics")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for arrivals (use -1 for deterministic)")
    return parser.parse_args()


async def run_simulation(
    ticks: int,
    *,
    use_llm: bool,
    analysis: bool,
    seed: int,
) -> Dict[str, object]:
    loader = ScenarioLoader(scenarios_dir=Path(__file__).parent)
    world_state, profiles = loader.load("scenario")
    profiles_map = {profile.agent_id: profile for profile in profiles}

    env_graph = build_environment_graph(world_state.environment_graph)
    occupancy = GraphOccupancy(env_graph) if env_graph else None
    if occupancy:
        for agent_status in world_state.agents:
            if agent_status.location:
                occupancy.enter(agent_status.location, agent_status.agent_id)

    rng = None if seed == -1 else random.Random(seed)

    if use_llm:
        Config.validate()

    prompt_library = build_prompt_library() if use_llm else None

    available_actions = [
        {
            "name": "triage",
            "action_type": "triage",
            "description": "Assess incoming patients and prioritize care",
            "schema": {"action_type": "triage", "target": "<location>", "parameters": {}, "reasoning": "<string>", "communication": None},
            "examples": [{"agent_id": "er_doctor", "tick": 1, "action_type": "triage", "target": "er_triage", "parameters": {}, "reasoning": "Assign bays", "communication": None}],
        },
        {
            "name": "coordinate",
            "action_type": "coordinate",
            "description": "Coordinate across departments",
            "schema": {"action_type": "coordinate", "target": "<location>", "parameters": {}, "reasoning": "<string>", "communication": {"to": "<agent_id>", "message": "<text>"}},
            "examples": [{"agent_id": "surgeon", "tick": 2, "action_type": "coordinate", "target": "recovery", "parameters": {}, "reasoning": "Confirm bed availability", "communication": {"to": "icu_nurse", "message": "Will need a bed"}}],
        },
        {
            "name": "move",
            "action_type": "move",
            "description": "Move to another hospital zone",
            "schema": {"action_type": "move", "target": "<location>", "parameters": {}, "reasoning": "<string>", "communication": None},
            "examples": [{"agent_id": "transport_aide", "tick": 1, "action_type": "move", "target": "er_treatment", "parameters": {}, "reasoning": "Guide stretcher", "communication": None}],
        },
        {
            "name": "care",
            "action_type": "care",
            "description": "Provide bedside care",
            "schema": {"action_type": "care", "target": "<location>", "parameters": {"detail": "<text>"}, "reasoning": "<string>", "communication": None},
            "examples": [{"agent_id": "icu_nurse", "tick": 2, "action_type": "care", "target": "icu", "parameters": {"detail": "Adjust drip"}, "reasoning": "Stabilize vitals", "communication": None}],
        },
        {
            "name": "scan",
            "action_type": "scan",
            "description": "Run imaging studies",
            "schema": {"action_type": "scan", "target": "<location>", "parameters": {"modality": "<ct|mri>"}, "reasoning": "<string>", "communication": None},
            "examples": [{"agent_id": "imaging_tech", "tick": 3, "action_type": "scan", "target": "imaging_ct", "parameters": {"modality": "ct"}, "reasoning": "STAT trauma scan", "communication": None}],
        },
        {
            "name": "operate",
            "action_type": "operate",
            "description": "Perform or support surgery",
            "schema": {"action_type": "operate", "target": "<operating_room>", "parameters": {}, "reasoning": "<string>", "communication": None},
            "examples": [{"agent_id": "surgeon", "tick": 2, "action_type": "operate", "target": "operating_room_a", "parameters": {}, "reasoning": "Start trauma case", "communication": None}],
        },
        {
            "name": "comfort",
            "action_type": "comfort",
            "description": "Support patients or visitors",
            "schema": {"action_type": "comfort", "target": "<location>", "parameters": {"audience": "<string>"}, "reasoning": "<string>", "communication": None},
            "examples": [{"agent_id": "visitor", "tick": 1, "action_type": "comfort", "target": "family_lounge", "parameters": {"audience": "patient_er"}, "reasoning": "Offer reassurance", "communication": None}],
        },
        {
            "name": "rest",
            "action_type": "rest",
            "description": "Rest or de-escalate",
            "schema": {"action_type": "rest", "target": "<location>", "parameters": {}, "reasoning": "<string>", "communication": None},
            "examples": [{"agent_id": "er_nurse", "tick": 4, "action_type": "rest", "target": "staff_break_room", "parameters": {}, "reasoning": "Short reset", "communication": None}],
        },
    ]

    cognition_map: Dict[str, AgentCognition] = {}
    for agent_id, profile in profiles_map.items():
        if use_llm and prompt_library is not None:
            cognition_map[agent_id] = AgentCognition(
                planner=LLMPlanner(template_name="plan_hospital", prompt_library=prompt_library),
                executor=LLMExecutor(
                    template_name="execute_hospital",
                    prompt_library=prompt_library,
                    available_actions=available_actions,
                ),
                reflection=LLMReflectionEngine(
                    template_name="reflect_hospital",
                    prompt_library=prompt_library,
                ),
                scratchpad=Scratchpad(),
                prompt_library=prompt_library,
            )
        else:
            cognition_map[agent_id] = AgentCognition(
                planner=HospitalDeterministicPlanner(ROLE_PLAN_TEMPLATES),
                executor=HospitalDeterministicExecutor(),
                reflection=HospitalDeterministicReflection(),
                scratchpad=Scratchpad(),
            )

    tick_listeners = []
    if analysis:
        tick_listeners.append(HospitalTickReporter().report)

    provider = Config.LLM_PROVIDER if use_llm else None
    model = Config.LLM_MODEL if use_llm else None

    orchestrator = Orchestrator(
        world_state=world_state,
        agents=profiles_map,
        world_prompt=(
            "You coordinate Riverside General Hospital's state updates. Keep ER waiting times,"
            " ICU beds, imaging load, and OR throughput synchronized while respecting safety."
        ),
        agent_prompts={
            agent_id: (
                f"You are {profile.name}, the {profile.role.replace('_', ' ')} at Riverside General Hospital. "
                "Share critical updates, support patients and visitors, and keep ER, ICU, imaging,"
                " and OR operations flowing."
            )
            for agent_id, profile in profiles_map.items()
        },
        llm_provider=provider,
        llm_model=model,
        simulation_rules=HospitalRules(occupancy, rng=rng),
        agent_cognition=cognition_map,
        tick_listeners=tick_listeners,
    )

    result = await orchestrator.run(num_ticks=ticks)
    return result


async def main(args: argparse.Namespace) -> None:
    try:
        result = await run_simulation(
            args.ticks,
            use_llm=args.llm,
            analysis=args.analysis,
            seed=args.seed,
        )
    except ValueError as exc:
        print(f"[warning] {exc}. Falling back to deterministic modules.")
        result = await run_simulation(
            args.ticks,
            use_llm=False,
            analysis=args.analysis,
            seed=args.seed,
        )

    final_state: WorldState = result["final_state"]
    er_wait = final_state.resources.get_metric("er_waiting_patients").value
    icu_beds = final_state.resources.get_metric("icu_beds_available").value
    imaging_queue = final_state.resources.get_metric("imaging_queue").value
    print("Simulation complete.")
    print(f"  ER waiting patients: {er_wait}")
    print(f"  ICU beds available: {icu_beds}")
    print(f"  Imaging queue length: {imaging_queue}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
