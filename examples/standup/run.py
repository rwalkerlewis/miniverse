"""Daily stand-up simulation focusing on conversational exchanges.

Run deterministically (no LLM):

    UV_CACHE_DIR=.uv-cache uv run python -m examples.standup.run --ticks 4

Enable LLM cognition (requires provider/model + API key):

    UV_CACHE_DIR=.uv-cache uv run python -m examples.standup.run --llm --ticks 4

Add debugging logs and per-tick analysis:

    UV_CACHE_DIR=.uv-cache uv run python -m examples.standup.run --llm --ticks 4 --debug
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from miniverse import (
    AgentAction,
    AgentCognition,
    AgentProfile,
    Orchestrator,
    Plan,
    PlanStep,
    ReflectionResult,
    SimulationRules,
    WorldEvent,
    WorldState,
)
from miniverse.cognition import (
    LLMPlanner,
    LLMReflectionEngine,
    PromptLibrary,
    PromptTemplate,
    Scratchpad,
    SimpleExecutor,
)
from miniverse.cognition.context import PromptContext
from miniverse.cognition.executor import Executor
from miniverse.cognition.planner import Planner
from miniverse.cognition.reflection import ReflectionEngine
from miniverse.config import Config
from miniverse.scenario import ScenarioLoader


class StandupRules(SimulationRules):
    """Lean deterministic rules for a conversational stand-up."""

    def apply_tick(self, state: WorldState, tick: int) -> WorldState:
        updated = state.model_copy(deep=True)
        alignment = updated.resources.get_metric(
            "alignment_score", default=70, unit="%", label="Team Alignment"
        )
        alignment.value = max(0.0, float(alignment.value) - 1.0)

        for agent in updated.agents:
            energy = agent.get_attribute("energy", default=75, unit="%", label="Energy")
            stress = agent.get_attribute("stress", default=30, unit="%", label="Stress")
            energy.value = min(100.0, float(energy.value) + 1.0)
            stress.value = max(0.0, float(stress.value) - 0.5)

        updated.tick = tick
        return updated

    def validate_action(self, action: AgentAction, state: WorldState) -> bool:
        if action.action_type == "communicate":
            comm = action.communication or {}
            return bool(comm.get("to") and comm.get("message"))
        return True


class StandupPlanner(Planner):
    """Deterministic planner producing role-specific talking points."""

    ROLE_STEPS = {
        "product_manager": ["greet_team", "share_objective", "ask_blockers"],
        "engineer": ["share_progress", "mention_blocker", "ask_support"],
        "designer": ["share_progress", "request_feedback", "close_loop"],
    }

    async def generate_plan(
        self,
        agent_id: str,
        scratchpad: Scratchpad,
        *,
        world_context: WorldState,
        context: PromptContext,
    ) -> Plan:
        profile: AgentProfile = context.agent_profile
        steps = [
            PlanStep(description=step, metadata={"role": profile.role})
            for step in self.ROLE_STEPS.get(profile.role, ["share_progress"])
        ]
        return Plan(steps=steps)


class StandupExecutor(Executor):
    """Maps plan steps to conversational messages."""

    ROLE_MESSAGES = {
        "product_manager": {
            "greet_team": "Good morning everyone, let's sync quickly on priorities.",
            "share_objective": "Reminder: today's focus is stabilizing the workflow rollout.",
            "ask_blockers": "Any blockers we should resolve before sprint review?",
        },
        "engineer": {
            "share_progress": "Finished the API instrumentation, deployment ready after lunch.",
            "mention_blocker": "Still blocked on staging credentials for the incident dashboard.",
            "ask_support": "Avery, could you sync with infra so I can ship today?",
        },
        "designer": {
            "share_progress": "Dashboard mockups are ready for review in Figma.",
            "request_feedback": "Jordan, can you confirm the data schema before I finalize?",
            "close_loop": "No blockers on my side once I get that confirmation—thanks both!",
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
        message = None
        if plan_step is not None:
            message = self.ROLE_MESSAGES.get(profile.role, {}).get(plan_step.description)
        if message is None:
            message = "Offering support if anyone needs help today."
        reasoning = (
            f"Following stand-up plan step '{plan_step.description}'"
            if plan_step
            else "No plan step available; sharing a general update"
        )
        return AgentAction(
            agent_id=agent_id,
            tick=perception.tick,
            action_type="communicate",
            target="team_channel",
            parameters={},
            reasoning=reasoning,
            communication={"to": "team", "message": message},
        )


class StandupReflection(ReflectionEngine):
    """Produces short recap entries every few ticks."""

    async def maybe_reflect(
        self,
        agent_id: str,
        scratchpad: Scratchpad,
        recent_memories,
        *,
        trigger_context=None,
        context: PromptContext | None = None,
    ) -> List[ReflectionResult]:
        if not trigger_context or trigger_context.get("tick", 0) % 4 != 0:
            return []
        latest = next(iter(recent_memories), None)
        summary = (
            "Alignment feels good; everyone is clear on priorities."
            if latest is None
            else f"Noted recent highlight: {latest.content}"
        )
        return [ReflectionResult(content=summary, importance=6)]


class DebugPlanner(Planner):
    """Planner wrapper that logs raw inputs/outputs for debugging."""

    def __init__(
        self,
        inner: Planner,
        agent_id: str,
        provider: str | None,
        model: str | None,
    ) -> None:
        self.inner = inner
        self.agent_id = agent_id
        self.provider = provider
        self.model = model

    async def generate_plan(
        self,
        agent_id: str,
        scratchpad: Scratchpad,
        *,
        world_context: WorldState,
        context: PromptContext,
    ) -> Plan:
        inner_name = self.inner.__class__.__name__
        print(
            f"    [Debug][{self.agent_id}] Planner -> {inner_name}"
            + (
                f" (provider={self.provider}, model={self.model})"
                if self.provider and self.model
                else ""
            )
        )
        plan = await self.inner.generate_plan(
            agent_id,
            scratchpad,
            world_context=world_context,
            context=context,
        )
        print("      Plan output:\n" + json.dumps(asdict(plan), indent=2, default=str))
        return plan


class DebugExecutor(Executor):
    """Executor wrapper logging final AgentAction payloads."""

    def __init__(self, inner: Executor, agent_id: str) -> None:
        self.inner = inner
        self.agent_id = agent_id

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
        inner_name = self.inner.__class__.__name__
        print(f"    [Debug][{self.agent_id}] Executor -> {inner_name}")
        if plan_step is not None:
            print("      Plan step:\n" + json.dumps(asdict(plan_step), indent=2, default=str))
        action = await self.inner.choose_action(
            agent_id,
            perception,
            scratchpad,
            plan=plan,
            plan_step=plan_step,
            context=context,
        )
        print("      Action JSON:\n" + json.dumps(action.model_dump(), indent=2, default=str))
        return action


class DebugReflection(ReflectionEngine):
    """Reflection wrapper logging generated summaries."""

    def __init__(
        self,
        inner: ReflectionEngine,
        agent_id: str,
        provider: str | None,
        model: str | None,
    ) -> None:
        self.inner = inner
        self.agent_id = agent_id
        self.provider = provider
        self.model = model

    async def maybe_reflect(
        self,
        agent_id: str,
        scratchpad: Scratchpad,
        recent_memories,
        *,
        trigger_context=None,
        context: PromptContext | None = None,
    ) -> List[ReflectionResult]:
        inner_name = self.inner.__class__.__name__
        print(
            f"    [Debug][{self.agent_id}] Reflection -> {inner_name}"
            + (
                f" (provider={self.provider}, model={self.model})"
                if self.provider and self.model
                else ""
            )
        )
        reflections = await self.inner.maybe_reflect(
            agent_id,
            scratchpad,
            recent_memories,
            trigger_context=trigger_context,
            context=context,
        )
        payload = [asdict(item) for item in reflections]
        print("      Reflections:\n" + json.dumps(payload, indent=2, default=str))
        return reflections


class ConversationPostTick:
    """Adjusts resources based on messages and prints conversation transcript."""

    POSITIVE_KEYWORDS = ("thanks", "thank you", "ready", "aligned", "good to go")
    RESOLUTION_KEYWORDS = ("resolved", "unblocked", "cleared")
    BLOCKER_KEYWORDS = ("blocker", "blocked", "stuck")

    def __init__(
        self,
        agent_names: Dict[str, str],
        scratchpads: Dict[str, Scratchpad],
        show_conversation: bool = False,
        max_history: int = 6,
    ) -> None:
        self.agent_names = agent_names
        self.scratchpads = scratchpads
        self.show_conversation = show_conversation
        self.max_history = max_history

    def __call__(
        self,
        tick: int,
        previous_state: WorldState,
        new_state: WorldState,
        actions: List[AgentAction],
    ) -> None:
        alignment = new_state.resources.get_metric(
            "alignment_score", default=70, unit="%", label="Team Alignment"
        )
        blockers = new_state.resources.get_metric(
            "open_blockers", default=0, label="Open Blockers"
        )

        transcript: List[str] = []
        blocker_delta = 0

        for idx, action in enumerate(actions):
            if action.action_type != "communicate" or not action.communication:
                continue
            message = action.communication.get("message", "")
            target = action.communication.get("to", "team")
            speaker = self.agent_names.get(action.agent_id, action.agent_id)

            event = WorldEvent(
                event_id=f"standup_msg_{tick}_{idx}",
                tick=tick,
                category="communication",
                description=f"{speaker} to {target}: {message}",
                severity=1,
                affected_agents=[action.agent_id],
            )
            new_state.recent_events.append(event)

            transcript.append(event.description)

            lower_msg = message.lower()
            if any(keyword in lower_msg for keyword in self.POSITIVE_KEYWORDS):
                alignment.value = min(100.0, float(alignment.value) + 2.0)
            else:
                alignment.value = min(100.0, float(alignment.value) + 1.0)

            if any(keyword in lower_msg for keyword in self.RESOLUTION_KEYWORDS):
                blocker_delta = min(blocker_delta, -1)
            elif any(keyword in lower_msg for keyword in self.BLOCKER_KEYWORDS):
                blocker_delta = max(blocker_delta, 1)

        if transcript:
            if self.show_conversation:
                print("  [Conversation] " + " \u2022 ".join(transcript))
            for agent_id, scratchpad in self.scratchpads.items():
                history = scratchpad.state.setdefault("recent_transcript", [])
                history.extend(transcript)
                if len(history) > self.max_history:
                    del history[:-self.max_history]

        if blocker_delta != 0:
            blockers.value = max(0.0, float(blockers.value) + blocker_delta)


class TickAnalyzer:
    """General per-tick summary (alignment + blockers)."""

    def __call__(
        self,
        tick: int,
        previous_state: WorldState,
        new_state: WorldState,
        actions: List[AgentAction],
    ) -> None:
        alignment_prev = previous_state.resources.metrics.get("alignment_score")
        alignment_new = new_state.resources.metrics.get("alignment_score")
        if alignment_prev and alignment_new:
            prev_val = float(alignment_prev.value)
            new_val = float(alignment_new.value)
            print(
                f"  [Analysis] Alignment {prev_val:.1f}% → {new_val:.1f}% (Δ {new_val - prev_val:+.1f})"
            )

        blockers_prev = previous_state.resources.metrics.get("open_blockers")
        blockers_new = new_state.resources.metrics.get("open_blockers")
        if blockers_prev and blockers_new:
            prev_blockers = float(blockers_prev.value)
            new_blockers = float(blockers_new.value)
            print(
                f"  [Analysis] Blockers {prev_blockers:.0f} → {new_blockers:.0f}"
            )


def build_prompt_library() -> PromptLibrary:
    library = PromptLibrary()
    library.register(
        PromptTemplate(
            name="standup_plan",
            system=(
                "You are facilitating a daily software stand-up. Generate a plan with the three talking"
                " points this agent should cover today."
            ),
            user=(
                "Team context:\n{{context_summary}}\n\n"
                "Recent transcript JSON:\n{{scratchpad_json}}\n\n"
                "Agent profile JSON:\n{{context_json}}\n\n"
                "Example output:\n"
                "{\n"
                "  \"steps\": [\n"
                "    {\"description\": \"share_progress\", \"metadata\": {\"target\": \"team\"}},\n"
                "    {\"description\": \"mention_blocker\"},\n"
                "    {\"description\": \"ask_support\"}\n"
                "  ]\n"
                "}\n\n"
                "Return JSON only."
            ),
        )
    )
    library.register(
        PromptTemplate(
            name="standup_execute",
            system=(
                "Craft the next stand-up message as structured JSON. If the action is communicate,"
                " you must provide a communication object with 'to' and 'message'."
            ),
            user=(
                "Perception JSON:\n{{perception_json}}\n\n"
                "Current plan JSON:\n{{plan_json}}\n\n"
                "Recent transcript snippets:\n{{scratchpad_json}}\n\n"
                "Stored memories:\n{{memories_text}}\n\n"
                "Example output:\n"
                "{\n"
                "  \"agent_id\": \"eng\",\n"
                "  \"tick\": 7,\n"
                "  \"action_type\": \"communicate\",\n"
                "  \"target\": \"team_channel\",\n"
                "  \"parameters\": {},\n"
                "  \"reasoning\": \"Provide progress and flag blockers\",\n"
                "  \"communication\": {\"to\": \"team\", \"message\": \"Yesterday...\"}\n"
                "}\n\n"
                "Return JSON only."
            ),
        )
    )
    library.register(
        PromptTemplate(
            name="standup_reflect",
            system=(
                "Summarize the stand-up outcomes in one short sentence. Return JSON with a 'reflections'"
                " list."
            ),
            user=(
                "Stand-up summary:\n{{context_summary}}\n\n"
                "Recent transcript JSON:\n{{context_json}}\n\n"
                "Example output:\n"
                "{\n"
                "  \"reflections\": [\n"
                "    {\"content\": \"Team aligned on deployment support; blockers are tracked.\", \"importance\": 6}\n"
                "  ]\n"
                "}\n\n"
                "Return JSON only."
            ),
        )
    )
    return library


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stand-up conversation simulation")
    parser.add_argument("--llm", action="store_true", help="Use LLM-based cognition modules")
    parser.add_argument("--ticks", type=int, default=4, help="Number of ticks to simulate")
    parser.add_argument("--debug", action="store_true", help="Verbose cognition logging")
    parser.add_argument("--analysis", action="store_true", help="Print per-tick metrics summary")
    return parser.parse_args()


async def run_simulation(
    ticks: int,
    *,
    use_llm: bool = False,
    seed: int | None = None,
    verbose: bool = True,
    debug: bool = False,
    analysis: bool = False,
) -> Dict[str, object]:
    loader = ScenarioLoader(scenarios_dir=Path(__file__).parent)
    world_state, profiles = loader.load("scenario")
    profiles_map = {profile.agent_id: profile for profile in profiles}

    rng = random.Random(seed) if seed is not None else None

    if use_llm:
        Config.validate()

    if debug:
        mode = "LLM" if use_llm else "deterministic"
        print(
            f"[debug] Mode={mode}, provider={Config.LLM_PROVIDER}, model={Config.LLM_MODEL}"
        )

    prompt_library = build_prompt_library() if use_llm else None

    cognition_map: Dict[str, AgentCognition] = {}
    for agent_id, profile in profiles_map.items():
        if use_llm and prompt_library is not None:
            scratchpad = Scratchpad(state={"execute_prompt_template": "standup_execute"})
            cognition_map[agent_id] = AgentCognition(
                planner=LLMPlanner(
                    template_name="standup_plan",
                    prompt_library=prompt_library,
                ),
                executor=SimpleExecutor(),
                reflection=LLMReflectionEngine(
                    template_name="standup_reflect",
                    prompt_library=prompt_library,
                ),
                scratchpad=scratchpad,
                prompt_library=prompt_library,
            )
        else:
            cognition_map[agent_id] = AgentCognition(
                planner=StandupPlanner(),
                executor=StandupExecutor(),
                reflection=StandupReflection(),
                scratchpad=Scratchpad(),
            )

    provider = Config.LLM_PROVIDER if use_llm else None
    model = Config.LLM_MODEL if use_llm else None

    if debug:
        for agent_id, cognition in cognition_map.items():
            cognition_map[agent_id] = AgentCognition(
                planner=DebugPlanner(cognition.planner, agent_id, provider, model),
                executor=DebugExecutor(cognition.executor, agent_id),
                reflection=DebugReflection(cognition.reflection, agent_id, provider, model),
                scratchpad=cognition.scratchpad,
                prompt_library=cognition.prompt_library,
            )

    show_conversation = True
    agent_names = {agent_id: profile.name for agent_id, profile in profiles_map.items()}
    scratchpads = {agent_id: cognition_map[agent_id].scratchpad for agent_id in cognition_map}

    tick_listeners: List = []
    tick_listeners.append(ConversationPostTick(agent_names, scratchpads, show_conversation))
    if analysis:
        tick_listeners.append(TickAnalyzer())

    orchestrator = Orchestrator(
        world_state=world_state,
        agents=profiles_map,
        world_prompt="You are the stand-up facilitator updating team state.",
        agent_prompts={
            agent_id: (
                f"You are {profile.name}. Participate in the daily stand-up, share updates,"
                " and keep the team aligned."
            )
            for agent_id, profile in profiles_map.items()
        },
        llm_provider=provider,
        llm_model=model,
        simulation_rules=StandupRules(),
        agent_cognition=cognition_map,
        tick_listeners=tick_listeners,
    )

    context_manager = contextlib.nullcontext()
    if not verbose:
        context_manager = contextlib.redirect_stdout(io.StringIO())

    with context_manager:
        result = await orchestrator.run(num_ticks=ticks)

    return result


async def main(args: argparse.Namespace) -> None:
    try:
        result = await run_simulation(
            args.ticks,
            use_llm=args.llm,
            seed=None,
            verbose=True,
            debug=args.debug,
            analysis=args.analysis,
        )
    except ValueError as exc:
        print(f"[warning] {exc}. Falling back to deterministic cognition.")
        result = await run_simulation(
            args.ticks,
            use_llm=False,
            seed=None,
            verbose=True,
            debug=args.debug,
            analysis=args.analysis,
        )

    final_state: WorldState = result["final_state"]
    final_alignment = final_state.resources.get_metric("alignment_score").value
    print(f"Final alignment score: {final_alignment}")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
