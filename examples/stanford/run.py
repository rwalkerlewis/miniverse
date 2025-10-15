"""Stanford Valentine's Day party simulation runner.

Demonstrates Tier-2 grid-based spatial simulation with LLM-driven agents
navigating a neighborhood, coordinating socially, and planning a party.

Based on Stanford Generative Agents research paper.
"""

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from miniverse import Orchestrator
from miniverse.cognition import (
    AgentCognition,
    Scratchpad,
    SimplePlanner,
    SimpleExecutor,
    SimpleReflectionEngine,
    LLMPlanner,
    LLMReflectionEngine,
)
from miniverse.config import Config
from miniverse.memory import SimpleMemoryStream
from miniverse.persistence import InMemoryPersistence
from miniverse.scenario import ScenarioLoader

from .rules import StanfordRules


def print_grid_state(state, agents_dict):
    """Print ASCII visualization of agent positions on grid."""
    if state.environment_grid is None:
        return

    grid = state.environment_grid
    print(f"\n  Grid ({grid.width}x{grid.height}):")

    # Build position map (handle both list and tuple formats)
    agent_positions = {}
    for agent in state.agents:
        if agent.grid_position:
            pos = tuple(agent.grid_position) if isinstance(agent.grid_position, list) else agent.grid_position
            agent_positions[pos] = agent.display_name[0]  # first letter

    # Print grid with borders
    for row in range(grid.height):
        line = "  "
        for col in range(grid.width):
            pos = (row, col)
            tile = grid.tiles.get(pos)

            if pos in agent_positions:
                line += f" {agent_positions[pos]} "  # agent
            elif tile and tile.collision:
                line += " # "  # wall/obstacle
            elif tile and tile.game_object:
                line += " · "  # object
            else:
                line += " . "  # empty
        print(line)


async def run_simulation(
    *,
    ticks: int,
    llm_mode: bool,
    debug: bool,
    analysis: bool,
):
    """Run the Stanford Valentine's Day party simulation.

    Args:
        ticks: Number of simulation ticks to run
        llm_mode: If True, use LLM-backed cognition; if False, use deterministic defaults
        debug: If True, print detailed planner/executor/reflection outputs
        analysis: If True, print grid visualization after each tick
    """
    # Load scenario
    loader = ScenarioLoader()
    world_state, agents = loader.load("stanford")

    print(f"Starting Stanford simulation: {world_state.metadata.get('name', 'Stanford Party')}")
    print(f"Agents: {len(agents)}, Ticks: {ticks}, Mode: {'LLM' if llm_mode else 'Deterministic'}")

    if debug:
        print(f"[debug] Mode={'LLM' if llm_mode else 'Deterministic'}, provider={Config.LLM_PROVIDER}, model={Config.LLM_MODEL}")

    # Build agent profiles map
    profiles_map = {agent.agent_id: agent for agent in agents}

    # Initialize persistence and memory
    persistence = InMemoryPersistence()
    await persistence.initialize()

    memory = SimpleMemoryStream(persistence)
    await memory.initialize()

    # Build cognition for each agent
    agent_cognition: Dict[str, AgentCognition] = {}

    for agent_id, profile in profiles_map.items():
        if llm_mode:
            # LLM-backed cognition (plan every tick, reflect every 3 ticks)
            agent_cognition[agent_id] = AgentCognition(
                planner=LLMPlanner(
                    llm_provider=Config.LLM_PROVIDER,
                    llm_model=Config.LLM_MODEL,
                    debug=debug,
                ),
                executor=SimpleExecutor(),  # Uses default execute prompt
                reflection=LLMReflectionEngine(
                    llm_provider=Config.LLM_PROVIDER,
                    llm_model=Config.LLM_MODEL,
                    debug=debug,
                ),
                scratchpad=Scratchpad(),
            )
        else:
            # Deterministic cognition (no LLM calls, fast)
            agent_cognition[agent_id] = AgentCognition(
                planner=SimplePlanner(),
                executor=SimpleExecutor(),
                reflection=SimpleReflectionEngine(),
                scratchpad=Scratchpad(),
            )

    # Initialize simulation rules
    rules = StanfordRules(seconds_per_tick=60)  # 1 minute per tick

    # Build agent prompts properly (not placeholder)
    agent_prompts_map = {}
    for agent_id, profile in profiles_map.items():
        agent_prompts_map[agent_id] = f"""You are {profile.name}, a {profile.age}-year-old {profile.role}.
Background: {profile.background}
Personality: {profile.personality}

Your goals:
{chr(10).join(f"- {goal}" for goal in profile.goals)}

Your relationships:
{chr(10).join(f"- {other}: {relationship}" for other, relationship in profile.relationships.items())}"""

    # Create orchestrator
    # Note: world_prompt="" disables world LLM updates (rely on deterministic rules instead)
    orchestrator = Orchestrator(
        world_state=world_state,
        agents=profiles_map,
        world_prompt="",  # Disable world LLM - use deterministic rules only
        agent_prompts=agent_prompts_map,
        llm_provider=Config.LLM_PROVIDER,
        llm_model=Config.LLM_MODEL,
        simulation_rules=rules,
        persistence=persistence,
        memory=memory,
        agent_cognition=agent_cognition,
    )

    # Run simulation
    print()
    await orchestrator.run(num_ticks=ticks)

    # Print final summary
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)

    final_state = orchestrator.current_state
    print(f"\nFinal timestamp: {final_state.timestamp}")
    print(f"Resources: {rules.format_resource_summary(final_state)}")

    print("\nFinal agent states:")
    for agent in final_state.agents:
        energy = agent.attributes.get("energy")
        energy_str = f"{energy.value:.0f}%" if energy else "N/A"
        print(f"  {agent.display_name} @ {agent.grid_position}: {agent.activity or 'idle'} (Energy: {energy_str})")

    if analysis:
        print_grid_state(final_state, profiles_map)

    await persistence.close()
    await memory.close()


def main():
    """CLI entry point for Stanford simulation."""
    parser = argparse.ArgumentParser(
        description="Stanford Valentine's Day party simulation (Tier-2 grid)"
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=10,
        help="Number of simulation ticks to run (default: 10)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use LLM-backed cognition (requires API keys)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed planner/executor/reflection outputs",
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Print grid visualization and detailed analysis",
    )

    args = parser.parse_args()

    asyncio.run(
        run_simulation(
            ticks=args.ticks,
            llm_mode=args.llm,
            debug=args.debug,
            analysis=args.analysis,
        )
    )


if __name__ == "__main__":
    main()
