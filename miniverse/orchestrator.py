"""
Main simulation orchestrator.

Fully decoupled from file I/O, database, and config.
All dependencies are injected by the user.

Coordinates the simulation loop:
1. Apply deterministic physics (if simulation_rules provided)
2. Build agent perceptions (partial observability)
3. Gather agent actions in parallel (LLM calls)
4. Process world update (World Engine LLM call)
5. Persist via injected strategy
6. Update agent memories via injected strategy
"""

import asyncio
from typing import Any, Callable, List, Dict, Optional
from uuid import UUID, uuid4

from .perception import build_agent_perception
from .llm_calls import process_world_update
from .schemas import AgentProfile, WorldState, AgentAction, AgentMemory
from .simulation_rules import SimulationRules, format_resources_generic
from .persistence import PersistenceStrategy, InMemoryPersistence
from .memory import MemoryStrategy, SimpleMemoryStream
from .cognition import (
    AgentCognition,
    AgentCognitionMap,
    PromptContext,
    build_default_cognition,
    build_prompt_context,
    DEFAULT_PROMPTS,
)
from .cognition.cadence import PLANNER_LAST_TICK_KEY, REFLECTION_LAST_TICK_KEY
from .cognition.planner import Plan, PlanStep


class Orchestrator:
    """
    Main simulation orchestrator.

    Fully decoupled - accepts all dependencies as parameters.
    No file I/O, no database requirement, no global config.
    """

    def __init__(
        self,
        world_state: WorldState,
        agents: Dict[str, AgentProfile],
        world_prompt: str,
        agent_prompts: Dict[str, str],
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        simulation_rules: Optional[SimulationRules] = None,
        persistence: Optional[PersistenceStrategy] = None,
        memory: Optional[MemoryStrategy] = None,
        agent_cognition: Optional[AgentCognitionMap] = None,
        tick_listeners: Optional[
            List[Callable[[int, WorldState, WorldState, List[AgentAction]], None]]
        ] = None,
    ):
        """Initialize orchestrator with all dependencies injected.

        Args:
            world_state: Initial WorldState
            agents: Dict mapping agent_id to AgentProfile
            world_prompt: System prompt for world engine
            agent_prompts: Dict mapping agent_id to system prompt
            llm_provider: Optional LLM provider name (e.g., "openai", "anthropic")
            llm_model: Optional model identifier (e.g., "gpt-5-nano")
            simulation_rules: Optional SimulationRules for deterministic physics
            persistence: Optional persistence strategy (defaults to InMemory)
            memory: Optional memory strategy (defaults to SimpleMemoryStream)
            agent_cognition: Optional mapping of agent_id to AgentCognition;
                defaults provide empty planner/executor/reflection stacks.
            tick_listeners: Optional callables invoked after each tick for
                additional analysis or logging. Each listener receives
                (tick, previous_state, new_state, actions).
        """
        self.current_state = world_state
        self.agents = agents
        self.world_prompt = world_prompt
        self.agent_prompts = agent_prompts
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.simulation_rules = simulation_rules
        self._world_llm_warning_emitted = False

        # Use defaults if strategies not provided. InMemoryPersistence is fastest for
        # prototyping but data is lost after run. SimpleMemoryStream uses the same
        # persistence backend to store and retrieve memories with recency-based filtering.
        self.persistence = persistence or InMemoryPersistence()
        self.memory = memory or SimpleMemoryStream(self.persistence)

        # Prepare cognition modules per agent. Each agent gets a bundle containing
        # planner (generates multi-step plans), executor (chooses tick actions),
        # reflection engine (periodically summarizes experiences), and scratchpad
        # (working memory for plan state). If user doesn't provide custom cognition,
        # build_default_cognition() returns simple deterministic implementations.
        cognition_map: AgentCognitionMap = {}
        for agent_id in agents:
            if agent_cognition and agent_id in agent_cognition:
                # User provided custom cognition for this agent
                cognition_map[agent_id] = agent_cognition[agent_id]
            else:
                # Fall back to default cognition (deterministic, no-op reflection)
                cognition_map[agent_id] = build_default_cognition()
        self.agent_cognition: AgentCognitionMap = cognition_map

        # Tick listeners are optional callbacks invoked after each tick completes.
        # They receive (tick, previous_state, new_state, actions) for analysis/logging.
        self.tick_listeners = tick_listeners or []

        # Give simulation rules a chance to modify initial state (e.g., compute
        # derived metrics, initialize tracking structures). Not all rules need this.
        if self.simulation_rules:
            self.current_state = self.simulation_rules.on_simulation_start(
                self.current_state
            )

        # Generate unique ID for this simulation run. All persistence/memory operations
        # tag data with this ID so multiple runs can coexist in the same backend.
        self.run_id: UUID = uuid4()

    async def run(self, num_ticks: int) -> Dict:
        """Run simulation for N ticks.

        Args:
            num_ticks: Number of ticks to simulate

        Returns:
            Dict with run_id and final_state

        Raises:
            Exception: If simulation fails
        """
        # Initialize persistence and memory backends. Each backend (InMemory, JSON, Postgres)
        # may need setup (e.g., opening DB connections, creating directories). Initialization
        # is idempotent - safe to call multiple times.
        await self.persistence.initialize()
        await self.memory.initialize()

        try:
            # Save initial state at tick 0. This establishes baseline for debugging and allows
            # users to rewind simulations. All persistence operations tag data with run_id so
            # multiple simulation runs can coexist in the same backend.
            await self.persistence.save_state(self.run_id, 0, self.current_state)

            print(f"Starting simulation run {self.run_id}")
            print(f"Agents: {len(self.agents)}, Ticks: {num_ticks}\n")

            # Main simulation loop. Each tick is independent - if one tick fails, we propagate
            # the exception immediately rather than continuing with corrupted state.
            for tick in range(1, num_ticks + 1):
                print(f"=== Tick {tick}/{num_ticks} ===")

                try:
                    await self._run_tick(tick)
                except Exception as e:
                    print(f"ERROR at tick {tick}: {e}")
                    raise

            # Give simulation rules a chance to compute final statistics, aggregate metrics,
            # or clean up tracking structures. Not all rules need this hook.
            if self.simulation_rules:
                self.current_state = self.simulation_rules.on_simulation_end(
                    self.current_state, num_ticks
                )

            print("\n✅ Simulation complete!")

            return {"run_id": self.run_id, "final_state": self.current_state}

        finally:
            # Always cleanup persistence and memory backends even if simulation fails.
            # Ensures DB connections are closed, file handles released, etc.
            await self.persistence.close()
            await self.memory.close()

    async def _run_tick(self, tick: int) -> None:
        """Execute single tick.

        Args:
            tick: Current tick number
        """
        # Preserve previous state for tick listeners. Listeners compare before/after states
        # to detect emergent patterns, compute metrics, or trigger custom events.
        previous_state = self.current_state

        # 0. Apply deterministic physics FIRST (if simulation_rules provided).
        # Physics runs BEFORE agent actions to ensure agents perceive the consequences of
        # previous tick's physics. Example: resource consumption, environmental decay,
        # scheduled events. Physics is pure Python (fast, testable, deterministic).
        if self.simulation_rules:
            print(f"  [Physics] Applying deterministic rules for tick {tick}...")
            self.current_state = self.simulation_rules.apply_tick(
                self.current_state, tick
            )
            print(f"  [Physics] ✓ Physics applied")

        # 1. Gather agent actions in parallel. Each agent gets partial observability based on
        # their location and access rights. Running in parallel minimizes total LLM latency
        # (critical for large agent populations). Each agent's cognition stack independently
        # builds perceptions, retrieves memories, generates/advances plans, and selects actions.
        actions = await self._gather_agent_actions(tick)

        # 2. World engine processes all actions and returns updated state. The world engine
        # uses an LLM to resolve action conflicts, apply narrative consistency, and generate
        # emergent events. If LLM fails validation after retries, we fall back to deterministic
        # physics to prevent simulation crashes.
        new_state = await self._process_world_update(tick, actions)

        # 3. Persist tick data (state, actions, memories) to backend. Persistence happens
        # AFTER the world engine succeeds so we never save partial/corrupt data. Memory
        # creation includes reflection engine invocations for agents that hit their cadence.
        await self._persist_tick(tick, new_state, actions)

        # 4. Print human-readable summary for monitoring. Shows resource levels, agent actions,
        # and recent events. Helps users understand simulation progress without reading logs.
        self._print_tick_summary(tick, actions, new_state)

        # 5. Update current state to serve as baseline for next tick. This happens AFTER
        # persistence to avoid race conditions between tick listeners and the next tick.
        self.current_state = new_state

        # 6. Invoke tick listeners for optional analysis/debugging. Listeners receive
        # (tick, prev_state, new_state, actions) to compute custom metrics, detect anomalies,
        # or trigger external integrations (e.g., logging to analytics platforms).
        # Listener failures are logged but don't crash the simulation.
        if self.tick_listeners:
            for listener in self.tick_listeners:
                try:
                    listener(tick, previous_state, new_state, actions)
                except Exception as exc:  # pragma: no cover - diagnostic hook
                    print(f"  [Analysis] Listener failed: {exc}")

    async def _gather_agent_actions(self, tick: int) -> List[AgentAction]:
        """Get all agent actions in parallel.

        Args:
            tick: Current tick number

        Returns:
            List of agent actions (or default rest actions for failures)
        """
        # Build perceptions and gather actions in parallel. Each agent's action decision is
        # independent (no coordination) so we run all agents concurrently to minimize total
        # LLM latency. With N agents and ~2s LLM latency, parallel execution takes ~2s total
        # vs ~2N seconds sequential.
        tasks = []
        for agent_id, profile in self.agents.items():
            task = self._get_single_agent_action(agent_id, tick)
            tasks.append((agent_id, task))

        # Execute all agent queries in parallel. return_exceptions=True ensures one agent's
        # failure doesn't crash the entire tick - we handle failures individually below.
        results = await asyncio.gather(
            *[task for _, task in tasks], return_exceptions=True
        )

        # Handle failures by substituting default rest action. This ensures simulation always
        # progresses even if LLM providers have outages. The world engine receives complete
        # action lists, though failed agents effectively skip the tick.
        valid_actions = []
        for i, result in enumerate(results):
            agent_id = tasks[i][0]
            if isinstance(result, Exception):
                print(f"  Agent {agent_id} failed: {result}")
                print(f"  Using default rest action for {agent_id}")
                # Default action prevents cascading failures - world engine expects exactly
                # one action per agent, so we provide a no-op rather than skipping.
                valid_actions.append(self._default_action(agent_id, tick))
            else:
                valid_actions.append(result)

        return valid_actions

    async def _get_single_agent_action(
        self, agent_id: str, tick: int
    ) -> AgentAction:
        """Get single agent's action decision.

        Args:
            agent_id: ID of agent
            tick: Current tick number

        Returns:
            AgentAction from LLM

        Raises:
            Exception: If LLM call fails
        """
        agent_name = self.agents[agent_id].name
        cognition = self.agent_cognition[agent_id]
        print(f"  [{agent_name}] Building perception...")

        # Retrieve previous tick's actions to extract communications. Agents need to see
        # messages directed at them to maintain conversation continuity. Only fetch
        # tick-1 (not entire history) to keep perception focused on immediate context.
        if tick > 1:
            recent_actions_data = await self.persistence.get_actions(self.run_id, tick - 1)
            recent_actions = recent_actions_data if recent_actions_data else []
        else:
            # Tick 1 has no previous actions - agents start with clean slate
            recent_actions = []

        # Pull recent memories formatted as strings for perception context. These are
        # displayed in the agent's "observable world" section to ground their decisions
        # in past experiences. Limit to 10 to avoid token bloat while maintaining recency.
        recent_memory_strings = await self.memory.get_recent_memories(
            self.run_id, agent_id, limit=10
        )

        # Build partial observability view (Stanford Generative Agents pattern).
        # Perception includes: agent's own status, nearby entities, visible locations,
        # messages directed at them, and recent memory context. Agent cannot see
        # entities/locations outside their access range.
        perception = build_agent_perception(
            agent_id,
            self.current_state,
            recent_actions,
            recent_memory_strings,
        )

        # Retrieve structured memory objects (AgentMemory instances) for prompt context.
        # Note: we fetch memories twice - once as strings for perception, once as structured
        # objects for prompt context. This dual retrieval supports both human-readable
        # perception and machine-readable prompt templates.
        recent_agent_memories = await self.persistence.get_recent_memories(
            self.run_id, agent_id, limit=10
        )

        # Use custom prompt library if agent's cognition provides one, otherwise fall back
        # to system defaults. Prompt library contains templates for planning, execution,
        # and reflection - allowing per-agent customization of reasoning styles.
        prompt_library = cognition.prompt_library or DEFAULT_PROMPTS

        # Assemble common metadata for all prompt contexts (planning, execution, reflection).
        # This avoids repeating the same data across multiple context builds.
        extra_common = {
            "base_agent_prompt": self.agent_prompts.get(agent_id, ""),
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "prompt_library": prompt_library,
            # Allow agents to dynamically switch execution prompt templates via scratchpad
            "execute_prompt_template": cognition.scratchpad.state.get(
                "execute_prompt_template", "execute_tick"
            ),
        }

        # Extract existing plan from scratchpad. Plan may be None (agent hasn't planned yet)
        # or Plan object (agent has active multi-step plan). We normalize to Plan() for
        # consistency - empty plans are valid (agent will rest or use heuristics).
        existing_plan = cognition.scratchpad.state.get("plan")
        existing_plan_index = cognition.scratchpad.state.get("plan_index", 0)
        if isinstance(existing_plan, Plan):
            initial_plan = existing_plan
        else:
            # No plan exists - use empty plan as placeholder
            initial_plan = Plan()
            existing_plan_index = 0

        # Summarize current plan state for planning context. Planner needs to see existing
        # plan to decide whether to refresh, extend, or keep current plan.
        planning_plan_state = self._plan_state_summary(initial_plan, existing_plan_index)

        # Build prompt context for planning. Planner receives full world state, perception,
        # memories, and current plan to generate/refresh multi-step plans. Context is
        # JSON-serializable for template rendering.
        planning_context = await build_prompt_context(
            agent_profile=self.agents[agent_id],
            perception=perception,
            world_state=self.current_state,
            scratchpad_state=dict(cognition.scratchpad.state),
            plan_state=planning_plan_state,
            memories=recent_agent_memories,
            extra=extra_common,
        )

        # Ensure agent has valid plan. Planner cadence determines whether to generate fresh
        # plan or reuse existing. Returns (plan, current_step, step_index) tuple. Plan may
        # be empty if planner is deterministic or LLM unavailable.
        plan, plan_step, plan_index = await self._ensure_plan(
            agent_id, cognition, planning_context, tick
        )

        # Summarize updated plan state for execution context. Executor needs to see full
        # plan to coordinate actions with long-term goals.
        plan_state = self._plan_state_summary(plan, plan_index)

        # Build prompt context for execution. Executor receives same data as planner but
        # with updated plan state (in case planner just refreshed the plan).
        context = await build_prompt_context(
            agent_profile=self.agents[agent_id],
            perception=perception,
            world_state=self.current_state,
            scratchpad_state=dict(cognition.scratchpad.state),
            plan_state=plan_state,
            memories=recent_agent_memories,
            extra=extra_common,
        )

        # Delegate action selection to executor. Executor considers current plan step,
        # perception, memories, and world state to choose concrete action (move, interact,
        # communicate, rest). Executor may deviate from plan if circumstances changed.
        print(f"  [{agent_name}] Choosing action via executor...")
        action = await cognition.executor.choose_action(
            agent_id,
            perception,
            cognition.scratchpad,
            plan=plan,
            plan_step=plan_step,
            context=context,
        )
        print(f"  [{agent_name}] ✓ Got action: {action.action_type}")

        # Ensure agent_id matches the requesting agent. LLM may hallucinate wrong agent_id
        # or executor may use templates that don't populate it correctly. Overwriting here
        # prevents actions from being misattributed.
        action.agent_id = agent_id

        # Advance to next plan step after action executes. If plan is exhausted, wrap around
        # to step 0 (plans loop until planner refreshes). This keeps agents active rather
        # than blocking when plan completes.
        self._advance_plan(agent_id, cognition, plan)

        return action

    async def _process_world_update(
        self, tick: int, actions: List[AgentAction]
    ) -> WorldState:
        """World engine processes actions and returns new state.

        Args:
            tick: Current tick number
            actions: List of all agent actions this tick

        Returns:
            Updated WorldState from World Engine LLM

        Raises:
            Exception: If LLM call fails
        """
        print(f"  [World Engine] Processing {len(actions)} actions...")
        # Tell world engine whether physics was already applied
        physics_applied = self.simulation_rules is not None
        if not self.llm_provider or not self.llm_model:
            if not self._world_llm_warning_emitted:
                print("  [World Engine] No LLM provider configured; using deterministic world update only.")
                self._world_llm_warning_emitted = True
            return self._apply_deterministic_world_update(tick, actions)

        try:
            new_state = await process_world_update(
                self.current_state,
                actions,
                tick,
                self.world_prompt,
                self.llm_provider,
                self.llm_model,
                physics_applied=physics_applied,
            )
        except Exception as exc:  # pragma: no cover - exercised during failed LLM responses
            print(f"  [World Engine] Validation failed: {exc}")
            print(
                "  ⚠️ World engine response remained invalid after retries. "
                "Falling back to deterministic physics for this tick."
            )
            print(
                "  ⚠️ Simulation output may be incomplete for this tick; "
                "review logs if this happens frequently."
            )
            return self._apply_deterministic_world_update(tick, actions)

        print(f"  [World Engine] ✓ World state updated")
        return new_state

    async def _persist_tick(
        self, tick: int, state: WorldState, actions: List[AgentAction]
    ) -> None:
        """Save tick data via persistence strategy.

        Args:
            tick: Current tick number
            state: New world state
            actions: Agent actions this tick
        """
        print(f"  [Persistence] Persisting tick {tick}...")
        await self.persistence.save_state(self.run_id, tick, state)
        await self.persistence.save_actions(self.run_id, tick, actions)
        await self._update_memories(tick, actions, state)
        print(f"  [Persistence] ✓ Tick {tick} persisted")

    async def _update_memories(
        self, tick: int, actions: List[AgentAction], state: WorldState
    ) -> None:
        """Create memory records for actions, communications, and events.

        Args:
            tick: Current tick number
            actions: Agent actions this tick
            state: New world state
        """
        new_memories: Dict[str, List[AgentMemory]] = {}

        # Store actions as memories
        for action in actions:
            # Capture the action itself so downstream retrieval engines can filter by type/target.
            memory = await self.memory.add_memory(
                run_id=self.run_id,
                agent_id=action.agent_id,
                tick=tick,
                memory_type="action",
                content=f"I {action.action_type}: {action.reasoning}",
                importance=5,
                tags=[f"action:{action.action_type}"],
                metadata={
                    "target": action.target,
                    "parameters": action.parameters or {},
                    "reasoning": action.reasoning,
                },
            )
            new_memories.setdefault(action.agent_id, []).append(memory)

            # Store communications as memories
            if action.communication:
                recipient = action.communication.get("to", "unknown")
                message = action.communication.get("message") or action.communication.get("content", "")
                memory = await self.memory.add_memory(
                    run_id=self.run_id,
                    agent_id=action.agent_id,
                    tick=tick,
                    memory_type="communication",
                    content=f"I told {recipient}: {message}",
                    importance=6,
                    tags=["communication", f"to:{recipient}"],
                    metadata={
                        "message": message,
                        "recipient": recipient,
                        "action_type": action.action_type,
                    },
                )
                new_memories.setdefault(action.agent_id, []).append(memory)

        # Store events as observations for affected agents
        for event in state.recent_events:
            if event.tick == tick:  # Only new events
                for agent_id in event.affected_agents:
                    memory = await self.memory.add_memory(
                        run_id=self.run_id,
                        agent_id=agent_id,
                        tick=tick,
                        memory_type="observation",
                        content=event.description,
                        importance=event.severity,
                        tags=[
                            "event",
                            f"severity:{event.severity}" if event.severity is not None else "severity:unknown",
                            event.category,
                        ],
                        metadata={
                            "event_id": event.event_id,
                            "category": event.category,
                        },
                    )
                    new_memories.setdefault(agent_id, []).append(memory)

        await self._run_reflections(tick, new_memories)

    def _default_action(self, agent_id: str, tick: int) -> AgentAction:
        """Fallback action when agent fails.

        Args:
            agent_id: ID of agent
            tick: Current tick number

        Returns:
            Default rest action
        """
        return AgentAction(
            agent_id=agent_id,
            tick=tick,
            action_type="rest",
            target=None,
            parameters={},
            reasoning="System default: agent failed to respond",
            communication=None,
        )

    async def _ensure_plan(
        self,
        agent_id: str,
        cognition: AgentCognition,
        planning_context: PromptContext,
        tick: int,
    ) -> tuple[Plan, PlanStep | None, int]:
        """Ensure the agent has an active plan and return the current step.

        This implements the planner cadence pattern - plans are refreshed periodically
        (e.g., once per day) rather than every tick, reducing LLM calls while maintaining
        long-term coherence. Cadence is configurable per agent via AgentCognition.
        """

        # Extract existing plan from scratchpad. Plan may be None if agent never planned,
        # or Plan object if agent has active plan. We need to check type because scratchpad
        # stores arbitrary key-value pairs.
        plan_state = cognition.scratchpad.state.get("plan")
        if isinstance(plan_state, Plan):
            plan = plan_state
        else:
            plan = None

        # Check cadence to determine if plan needs refresh. Cadence considers:
        # 1. Time elapsed since last plan (e.g., 24 ticks = 1 simulated day)
        # 2. Whether current plan is empty/expired
        # 3. Custom trigger conditions (scenario-specific)
        cadence = cognition.cadence.planner
        last_plan_tick = cognition.scratchpad.state.get(PLANNER_LAST_TICK_KEY)

        should_refresh = cadence.should_generate(
            tick=tick,
            last_run_tick=last_plan_tick,
            current_plan=plan,
        )

        if should_refresh:
            # Generate fresh plan via LLM or deterministic planner. Planner considers
            # agent's personality, current world state, recent memories, and existing plan
            # to produce coherent multi-step agenda. New plan replaces old plan entirely
            # (no merging) and resets plan_index to 0.
            plan = await cognition.planner.generate_plan(
                agent_id,
                cognition.scratchpad,
                world_context=self.current_state,
                context=planning_context,
            )
            # Normalize non-Plan return values (shouldn't happen but defensive)
            if not isinstance(plan, Plan):
                plan = Plan()
            # Store new plan and reset index. Recording tick enables cadence tracking.
            cognition.scratchpad.state["plan"] = plan
            cognition.scratchpad.state["plan_index"] = 0
            cognition.scratchpad.state[PLANNER_LAST_TICK_KEY] = tick
        else:
            # Reuse existing plan. If plan somehow missing (shouldn't happen due to cadence
            # logic), create empty plan as safety net.
            if plan is None:  # safety net; shouldn't happen due to cadence
                plan = Plan()
                cognition.scratchpad.state["plan"] = plan
            cognition.scratchpad.state.setdefault("plan_index", 0)

        # Extract current step from plan. If plan_index exceeds plan length (can happen if
        # plan was shortened), wrap to 0 to keep agent active. Empty plans return None step,
        # signaling executor to use fallback behavior (rest, wander, reactive actions).
        plan_index = cognition.scratchpad.state.get("plan_index", 0)
        if plan.steps:
            if plan_index >= len(plan.steps):
                # Plan exhausted - wrap around to beginning to keep agent active
                plan_index = 0
                cognition.scratchpad.state["plan_index"] = plan_index
            plan_step = plan.steps[plan_index]
        else:
            # Empty plan - executor will use fallback logic
            plan_step = None

        return plan, plan_step, plan_index

    def _advance_plan(
        self, agent_id: str, cognition: AgentCognition, plan: Plan
    ) -> None:
        """Advance the plan index after an action executes.

        This implements sequential plan execution - agent moves to next step after each tick.
        When plan exhausts, wraps to step 0 to keep agent active. Alternative strategies
        (conditional branching, parallel steps) could be implemented by subclassing.
        """

        # Empty plans don't advance - executor will keep using fallback behavior
        if not plan.steps:
            return

        # Move to next step, or wrap to beginning if plan exhausted. Wrap-around prevents
        # agents from blocking when plan completes - they loop through plan until planner
        # refreshes with new agenda. This is appropriate for daily routines (wake -> eat ->
        # work -> sleep -> repeat) but not for project plans (may want to stop at end).
        plan_index = cognition.scratchpad.state.get("plan_index", 0)
        if plan_index < len(plan.steps) - 1:
            # More steps remaining - advance to next step
            cognition.scratchpad.state["plan_index"] = plan_index + 1
        else:
            # Plan exhausted - wrap to beginning to maintain activity
            cognition.scratchpad.state["plan_index"] = 0

    @staticmethod
    def _plan_state_summary(plan: Plan, plan_index: int) -> Dict[str, Any]:
        return {
            "current_index": plan_index,
            "steps": [
                {"description": step.description, "metadata": step.metadata}
                for step in plan.steps
            ],
        }

    async def _run_reflections(
        self, tick: int, new_memories: Dict[str, List[AgentMemory]]
    ) -> None:
        """Invoke reflection engines for each agent.

        Stanford Generative Agents pattern: Periodically synthesize recent experiences into
        higher-level insights (e.g., "I'm running low on resources" from multiple actions).
        Reflections are stored as memories with elevated importance, making them more likely
        to influence future decisions.
        """

        for agent_id, cognition in self.agent_cognition.items():
            # Count new memories created this tick. Reflection cadence may trigger based on
            # memory accumulation (e.g., reflect after every 5 new experiences) rather than
            # fixed time intervals.
            new_memory_count = len(new_memories.get(agent_id, []))
            cadence = cognition.cadence.reflection
            last_reflection_tick = cognition.scratchpad.state.get(
                REFLECTION_LAST_TICK_KEY
            )

            # Check if reflection should trigger. Cadence considers both time elapsed and
            # memory count to balance insight generation with LLM cost. Skip if not ready.
            if not cadence.should_reflect(
                tick=tick,
                last_run_tick=last_reflection_tick,
                new_memories=new_memory_count,
            ):
                continue

            # Retrieve larger memory window (20 vs 10) for reflection. Reflections synthesize
            # longer-term patterns, so need broader historical context than tick actions.
            recent_memories = await self.persistence.get_recent_memories(
                self.run_id, agent_id, limit=20
            )

            prompt_library = cognition.prompt_library or DEFAULT_PROMPTS

            # Pull memory strings for perception context (same as action selection)
            recent_memory_strings = await self.memory.get_recent_memories(
                self.run_id, agent_id, limit=10
            )

            # Build minimal perception for reflection. Reflections focus on internal state
            # (memories, plans, goals) rather than environment, so pass empty actions list.
            perception = build_agent_perception(
                agent_id,
                self.current_state,
                [],  # No actions needed for reflection context
                recent_memory_strings,
            )

            # Extract current plan for reflection context. Reflections may comment on plan
            # progress or suggest adjustments based on accumulated experiences.
            plan_obj = cognition.scratchpad.state.get("plan")
            plan_index = cognition.scratchpad.state.get("plan_index", 0)
            if not isinstance(plan_obj, Plan):
                plan_obj = Plan()
                plan_index = 0
            plan_state = self._plan_state_summary(plan_obj, plan_index)

            # Build prompt context for reflection. Reflection engine receives full memories,
            # current plan, and world state to generate synthesized insights.
            reflection_context = await build_prompt_context(
                agent_profile=self.agents[agent_id],
                perception=perception,
                world_state=self.current_state,
                scratchpad_state=dict(cognition.scratchpad.state),
                plan_state=plan_state,
                memories=recent_memories,
                extra={
                    "llm_provider": self.llm_provider,
                    "llm_model": self.llm_model,
                    "prompt_library": prompt_library,
                },
            )

            # Invoke reflection engine. May return empty list if LLM unavailable or no
            # interesting patterns detected. Reflections are free-form text with importance
            # scores (typically higher than raw observations to influence future retrieval).
            reflections = await cognition.reflection.maybe_reflect(
                agent_id,
                cognition.scratchpad,
                recent_memories,
                trigger_context={
                    "tick": tick,
                    "new_memories": new_memory_count,
                },
                context=reflection_context,
            )

            # Record reflection tick to prevent double-reflection same tick
            cognition.scratchpad.state[REFLECTION_LAST_TICK_KEY] = tick

            # Store reflections as memories. Reflections get memory_type="reflection" for
            # retrieval filtering and typically have elevated importance (6-10) vs actions (5).
            # High importance ensures reflections surface in future memory retrieval.
            for reflection in reflections:
                await self.memory.add_memory(
                    run_id=self.run_id,
                    agent_id=agent_id,
                    tick=tick,
                    memory_type="reflection",
                    content=reflection.content,
                    importance=reflection.importance,
                    metadata=reflection.metadata,
                )

    def _apply_deterministic_world_update(
        self, tick: int, actions: List[AgentAction]
    ) -> WorldState:
        """Clone the current state and apply basic deterministic updates.

        This is the fallback world update when LLM world engine fails validation or is
        unavailable. Applies minimal physics: update agent activities, process move actions,
        update occupancy tracking. Does NOT generate events, resolve conflicts, or apply
        narrative consistency - purely mechanical state updates.
        """

        # Deep copy current state to avoid mutating baseline. Pydantic model_copy ensures
        # nested objects (agents, locations, resources) are fully cloned.
        new_state = self.current_state.model_copy(deep=True)
        new_state.tick = tick
        # Clear recent_events since we're not generating new events in deterministic mode
        new_state.recent_events = []

        # Extract occupancy tracker from simulation rules if present. Occupancy prevents
        # multiple agents from occupying same location and enforces capacity limits.
        occupancy = getattr(self.simulation_rules, "occupancy", None)

        for action in actions:
            # Find agent's status object in world state. Status tracks location, activity,
            # resources, and other dynamic properties. Skip if agent not found (shouldn't happen).
            status = next((agent for agent in new_state.agents if agent.agent_id == action.agent_id), None)
            if status is None:
                continue

            # Record previous location before move for occupancy tracking
            previous_location = status.location
            # Update activity regardless of action type (move, interact, rest, etc.)
            status.activity = action.action_type

            # Process move actions: update location and occupancy. Other action types
            # (interact, rest, communicate) don't affect location.
            if action.action_type == "move" and action.target:
                if occupancy is not None and previous_location:
                    # Remove agent from previous location's occupancy list
                    occupancy.leave(previous_location, action.agent_id)
                # Update agent location to target
                status.location = action.target
                if occupancy is not None:
                    # Add agent to new location's occupancy list. May raise exception if
                    # location at capacity, but in deterministic mode we don't validate.
                    occupancy.enter(action.target, action.agent_id)

        return new_state

    def _print_tick_summary(
        self, tick: int, actions: List[AgentAction], state: WorldState
    ) -> None:
        """Print human-readable tick summary.

        Args:
            tick: Current tick number
            actions: Agent actions this tick
            state: New world state
        """
        # Print resource levels
        resource_summary = (
            self.simulation_rules.format_resource_summary(state)
            if self.simulation_rules
            else format_resources_generic(state)
        )
        if resource_summary:
            print(f"  Resources: {resource_summary}")

        # Print agent actions
        for action in actions:
            agent = self.agents[action.agent_id]
            reasoning_preview = (
                action.reasoning[:60] + "..."
                if len(action.reasoning) > 60
                else action.reasoning
            )
            print(f"  {agent.name}: {action.action_type} - {reasoning_preview}")

        # Print events
        for event in state.recent_events:
            if event.tick == tick:
                print(f"  EVENT (severity {event.severity}): {event.description}")

        print()  # Blank line for readability
