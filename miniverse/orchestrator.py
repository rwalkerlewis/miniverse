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

        # Use defaults if strategies not provided
        self.persistence = persistence or InMemoryPersistence()
        self.memory = memory or SimpleMemoryStream(self.persistence)
        
        # Prepare cognition modules per agent. Defaults keep current behavior
        # but make the contract explicit for upcoming plan/execute/reflect work.
        cognition_map: AgentCognitionMap = {}
        for agent_id in agents:
            if agent_cognition and agent_id in agent_cognition:
                cognition_map[agent_id] = agent_cognition[agent_id]
            else:
                cognition_map[agent_id] = build_default_cognition()
        self.agent_cognition: AgentCognitionMap = cognition_map
        self.tick_listeners = tick_listeners or []

        # Initialize simulation rules if provided
        if self.simulation_rules:
            self.current_state = self.simulation_rules.on_simulation_start(
                self.current_state
            )

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
        # Initialize persistence and memory backends
        await self.persistence.initialize()
        await self.memory.initialize()

        try:
            # Save initial state
            await self.persistence.save_state(self.run_id, 0, self.current_state)

            print(f"Starting simulation run {self.run_id}")
            print(f"Agents: {len(self.agents)}, Ticks: {num_ticks}\n")

            # Main simulation loop
            for tick in range(1, num_ticks + 1):
                print(f"=== Tick {tick}/{num_ticks} ===")

                try:
                    await self._run_tick(tick)
                except Exception as e:
                    print(f"ERROR at tick {tick}: {e}")
                    raise

            # Finalize simulation rules if provided
            if self.simulation_rules:
                self.current_state = self.simulation_rules.on_simulation_end(
                    self.current_state, num_ticks
                )

            print("\n✅ Simulation complete!")

            return {"run_id": self.run_id, "final_state": self.current_state}

        finally:
            # Always cleanup persistence and memory backends
            await self.persistence.close()
            await self.memory.close()

    async def _run_tick(self, tick: int) -> None:
        """Execute single tick.

        Args:
            tick: Current tick number
        """
        previous_state = self.current_state

        # 0. Apply deterministic physics FIRST (if simulation_rules provided)
        if self.simulation_rules:
            print(f"  [Physics] Applying deterministic rules for tick {tick}...")
            self.current_state = self.simulation_rules.apply_tick(
                self.current_state, tick
            )
            print(f"  [Physics] ✓ Physics applied")

        # 1. Gather agent actions (parallel)
        actions = await self._gather_agent_actions(tick)

        # 2. World engine processes actions
        new_state = await self._process_world_update(tick, actions)

        # 3. Persist to database
        await self._persist_tick(tick, new_state, actions)

        # 4. Print summary
        self._print_tick_summary(tick, actions, new_state)

        # 5. Update current state
        self.current_state = new_state

        # 6. Invoke tick listeners for optional analysis/debugging
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
        # Build perceptions and gather actions in parallel
        tasks = []
        for agent_id, profile in self.agents.items():
            task = self._get_single_agent_action(agent_id, tick)
            tasks.append((agent_id, task))

        # Execute all agent queries in parallel
        results = await asyncio.gather(
            *[task for _, task in tasks], return_exceptions=True
        )

        # Handle failures (use default rest action)
        valid_actions = []
        for i, result in enumerate(results):
            agent_id = tasks[i][0]
            if isinstance(result, Exception):
                print(f"  Agent {agent_id} failed: {result}")
                print(f"  Using default rest action for {agent_id}")
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

        # Get recent actions from previous tick (for extracting messages)
        if tick > 1:
            recent_actions_data = await self.persistence.get_actions(self.run_id, tick - 1)
            recent_actions = recent_actions_data if recent_actions_data else []
        else:
            recent_actions = []

        recent_memory_strings = await self.memory.get_recent_memories(
            self.run_id, agent_id, limit=10
        )

        # Build perception
        perception = build_agent_perception(
            agent_id,
            self.current_state,
            recent_actions,
            recent_memory_strings,
        )

        # Assemble prompt context (recent memories pulled from persistence for struct)
        recent_agent_memories = await self.persistence.get_recent_memories(
            self.run_id, agent_id, limit=10
        )

        prompt_library = cognition.prompt_library or DEFAULT_PROMPTS

        extra_common = {
            "base_agent_prompt": self.agent_prompts.get(agent_id, ""),
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "prompt_library": prompt_library,
            "execute_prompt_template": cognition.scratchpad.state.get(
                "execute_prompt_template", "execute_tick"
            ),
        }

        existing_plan = cognition.scratchpad.state.get("plan")
        existing_plan_index = cognition.scratchpad.state.get("plan_index", 0)
        if isinstance(existing_plan, Plan):
            initial_plan = existing_plan
        else:
            initial_plan = Plan()
            existing_plan_index = 0

        planning_plan_state = self._plan_state_summary(initial_plan, existing_plan_index)

        planning_context = await build_prompt_context(
            agent_profile=self.agents[agent_id],
            perception=perception,
            world_state=self.current_state,
            scratchpad_state=dict(cognition.scratchpad.state),
            plan_state=planning_plan_state,
            memories=recent_agent_memories,
            extra=extra_common,
        )

        plan, plan_step, plan_index = await self._ensure_plan(
            agent_id, cognition, planning_context
        )

        plan_state = self._plan_state_summary(plan, plan_index)

        context = await build_prompt_context(
            agent_profile=self.agents[agent_id],
            perception=perception,
            world_state=self.current_state,
            scratchpad_state=dict(cognition.scratchpad.state),
            plan_state=plan_state,
            memories=recent_agent_memories,
            extra=extra_common,
        )

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

        # Ensure agent_id matches the requesting agent
        action.agent_id = agent_id

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
            memory = await self.memory.add_memory(
                run_id=self.run_id,
                agent_id=action.agent_id,
                tick=tick,
                memory_type="action",
                content=f"I {action.action_type}: {action.reasoning}",
                importance=5,
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
    ) -> tuple[Plan, PlanStep | None, int]:
        """Ensure the agent has an active plan and return the current step."""

        plan = cognition.scratchpad.state.get("plan")
        if not isinstance(plan, Plan):
            plan = await cognition.planner.generate_plan(
                agent_id,
                cognition.scratchpad,
                world_context=self.current_state,
                context=planning_context,
            )
            if not isinstance(plan, Plan):
                plan = Plan()
            cognition.scratchpad.state["plan"] = plan
            cognition.scratchpad.state["plan_index"] = 0

        plan_index = cognition.scratchpad.state.get("plan_index", 0)
        if plan.steps:
            if plan_index >= len(plan.steps):
                plan_index = 0
                cognition.scratchpad.state["plan_index"] = plan_index
            plan_step = plan.steps[plan_index]
        else:
            plan_step = None

        return plan, plan_step, plan_index

    def _advance_plan(
        self, agent_id: str, cognition: AgentCognition, plan: Plan
    ) -> None:
        """Advance the plan index after an action executes."""

        if not plan.steps:
            return

        plan_index = cognition.scratchpad.state.get("plan_index", 0)
        if plan_index < len(plan.steps) - 1:
            cognition.scratchpad.state["plan_index"] = plan_index + 1
        else:
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
        """Invoke reflection engines for each agent."""

        for agent_id, cognition in self.agent_cognition.items():
            recent_memories = await self.persistence.get_recent_memories(
                self.run_id, agent_id, limit=20
            )

            prompt_library = cognition.prompt_library or DEFAULT_PROMPTS

            recent_memory_strings = await self.memory.get_recent_memories(
                self.run_id, agent_id, limit=10
            )

            perception = build_agent_perception(
                agent_id,
                self.current_state,
                [],
                recent_memory_strings,
            )

            plan_obj = cognition.scratchpad.state.get("plan")
            plan_index = cognition.scratchpad.state.get("plan_index", 0)
            if not isinstance(plan_obj, Plan):
                plan_obj = Plan()
                plan_index = 0
            plan_state = self._plan_state_summary(plan_obj, plan_index)

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

            reflections = await cognition.reflection.maybe_reflect(
                agent_id,
                cognition.scratchpad,
                recent_memories,
                trigger_context={
                    "tick": tick,
                    "new_memories": len(new_memories.get(agent_id, [])),
                },
                context=reflection_context,
            )

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
        """Clone the current state and apply basic deterministic updates."""

        new_state = self.current_state.model_copy(deep=True)
        new_state.tick = tick
        new_state.recent_events = []

        occupancy = getattr(self.simulation_rules, "occupancy", None)

        for action in actions:
            status = next((agent for agent in new_state.agents if agent.agent_id == action.agent_id), None)
            if status is None:
                continue

            previous_location = status.location
            status.activity = action.action_type

            if action.action_type == "move" and action.target:
                if occupancy is not None and previous_location:
                    occupancy.leave(previous_location, action.agent_id)
                status.location = action.target
                if occupancy is not None:
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
