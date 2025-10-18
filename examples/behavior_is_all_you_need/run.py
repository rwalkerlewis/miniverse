"""Main entry point for "Behavior is all you need" social deception game simulation.

This example demonstrates the paper's 4-component framework:
1. Personality (Big Five traits) - stable behavioral tendencies
2. Emotion (hybrid dimensional + categorical) - transient state modulation
3. Needs (Maslow hierarchy) - motivational drivers
4. Memory - past experiences inform decisions

Usage:
    # Run with rule-based cognition (no LLM)
    python run.py --ticks 12

    # Run with LLM-enhanced cognition
    python run.py --llm --ticks 12

    # Enable debug output
    python run.py --debug --ticks 12
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from miniverse import (
    AgentCognition,
    AgentProfile,
    Orchestrator,
    Scratchpad,
    WorldState,
)
from miniverse.config import Config
from miniverse.scenario import ScenarioLoader

from personality import BigFiveTraits, PersonalityArchetypes
from emotion import EmotionalState
from needs import NeedsHierarchy
from rules import DeceptionGameRules
from cognition import PersonalityAwareExecutor, PersonalityAwarePlanner


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Social deception game with personality-driven behavior"
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use LLM-based cognition (requires LLM_PROVIDER and LLM_MODEL env vars)"
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=12,
        help="Number of ticks to simulate (default: 12, ~4 game phases)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    return parser.parse_args()


def initialize_agent_state(profile: AgentProfile, scratchpad: Scratchpad):
    """Initialize agent scratchpad with personality, emotion, and needs.

    This sets up the behavioral state that will drive agent decisions.
    """
    # Extract Big Five traits from profile metadata
    big_five_data = profile.metadata.get("big_five", {})
    personality = BigFiveTraits.from_dict(big_five_data) if big_five_data else BigFiveTraits(
        openness=50, conscientiousness=50, extraversion=50,
        agreeableness=50, neuroticism=50
    )

    # Initialize emotional state (start neutral)
    emotional_state = EmotionalState()

    # Initialize needs hierarchy (start at moderate satisfaction)
    needs = NeedsHierarchy()

    # Store in scratchpad
    scratchpad.state["personality"] = personality.to_dict()
    scratchpad.state["emotional_state"] = emotional_state.to_dict()
    scratchpad.state["needs"] = needs.to_dict()

    # Store role from tags
    for tag in profile.relationships.keys():  # This is a hack - should be from agent status tags
        pass  # We'll get role from scenario agent status tags instead

    # Extract role from profile.role field
    scratchpad.state["role"] = profile.role


async def run_simulation(
    ticks: int,
    use_llm: bool = False,
    debug: bool = False,
) -> Dict:
    """Run the social deception game simulation.

    Args:
        ticks: Number of simulation ticks to run
        use_llm: Whether to use LLM for cognition
        debug: Whether to print debug information

    Returns:
        Dictionary with final_state and other simulation results
    """
    # Load scenario
    print("Loading scenario...")
    loader = ScenarioLoader(scenarios_dir=Path(__file__).parent)
    world_state, profiles = loader.load("scenario")
    profiles_map = {p.agent_id: p for p in profiles}

    print(f"Loaded {len(profiles)} agents:")
    for profile in profiles:
        role_tag = "unknown"
        for agent_status in world_state.agents:
            if agent_status.agent_id == profile.agent_id:
                for tag in agent_status.tags:
                    if tag.startswith("role:"):
                        role_tag = tag.split(":", 1)[1]
                break
        big_five = profile.metadata.get("big_five", {})
        archetype = profile.metadata.get("archetype", "unknown")
        print(f"  - {profile.name} ({role_tag}): {archetype}")
        if debug:
            print(f"      Big Five: O={big_five.get('openness', 50)}, "
                  f"C={big_five.get('conscientiousness', 50)}, "
                  f"E={big_five.get('extraversion', 50)}, "
                  f"A={big_five.get('agreeableness', 50)}, "
                  f"N={big_five.get('neuroticism', 50)}")

    # Validate LLM config if needed
    if use_llm:
        try:
            Config.validate()
            print(f"Using LLM: {Config.LLM_PROVIDER}/{Config.LLM_MODEL}")
        except ValueError as e:
            print(f"Warning: {e}")
            print("Falling back to rule-based cognition")
            use_llm = False

    # Initialize game rules
    rules = DeceptionGameRules(ticks_per_phase=3)

    # Build cognition map
    print("\nInitializing agent cognition...")
    cognition_map: Dict[str, AgentCognition] = {}
    for agent_id, profile in profiles_map.items():
        scratchpad = Scratchpad()
        initialize_agent_state(profile, scratchpad)

        # Extract role from world state
        for agent_status in world_state.agents:
            if agent_status.agent_id == agent_id:
                for tag in agent_status.tags:
                    if tag.startswith("role:"):
                        scratchpad.state["role"] = tag.split(":", 1)[1]
                        break
                break

        cognition_map[agent_id] = AgentCognition(
            executor=PersonalityAwareExecutor(use_llm=use_llm),
            planner=PersonalityAwarePlanner(use_llm=use_llm),
            scratchpad=scratchpad,
        )

        if debug:
            personality = BigFiveTraits.from_dict(scratchpad.state["personality"])
            print(f"  {profile.name}: {personality.describe()}")

    # Create orchestrator
    provider = Config.LLM_PROVIDER if use_llm else None
    model = Config.LLM_MODEL if use_llm else None

    agent_prompts = {
        agent_id: (
            f"You are {profile.name}. {profile.background}\n\n"
            f"Your personality: {profile.personality}\n"
            f"Your role in the game: {profile.role}\n"
            f"Your goals: {', '.join(profile.goals)}\n\n"
            f"Stay in character. Your personality and emotions influence your decisions."
        )
        for agent_id, profile in profiles_map.items()
    }

    orchestrator = Orchestrator(
        world_state=world_state,
        agents=profiles_map,
        world_prompt="You manage the social deception game state. Process agent actions according to game rules.",
        agent_prompts=agent_prompts,
        llm_provider=provider,
        llm_model=model,
        simulation_rules=rules,
        agent_cognition=cognition_map,
    )

    # Run simulation
    print(f"\n{'='*80}")
    print(f"Starting simulation: {ticks} ticks (~{ticks//3} game phases)")
    print(f"{'='*80}\n")

    result = await orchestrator.run(num_ticks=ticks)

    # Print final results
    final_state: WorldState = result["final_state"]
    print(f"\n{'='*80}")
    print("SIMULATION COMPLETE")
    print(f"{'='*80}")

    if rules.game_over:
        print(f"\n🎮 GAME OVER! Winner: {rules.winner}")
    else:
        print(f"\n⏱️  Simulation ended (time limit reached)")

    print(f"\n📊 Final Statistics:")
    werewolves_alive = sum(
        1 for a in final_state.agents
        if a.agent_id not in rules.eliminated_agents and any("werewolf" in tag for tag in a.tags)
    )
    villagers_alive = sum(
        1 for a in final_state.agents
        if a.agent_id not in rules.eliminated_agents and not any("werewolf" in tag for tag in a.tags)
    )
    print(f"  Werewolves alive: {werewolves_alive}")
    print(f"  Villagers alive: {villagers_alive}")
    print(f"  Total eliminated: {len(rules.eliminated_agents)}")

    if rules.eliminated_agents:
        print(f"\n⚰️  Eliminated players:")
        for agent_id in rules.eliminated_agents:
            profile = profiles_map.get(agent_id)
            if profile:
                print(f"    - {profile.name}")

    print(f"\n📝 Recent events:")
    for event in final_state.recent_events[-5:]:
        print(f"  • {event}")

    # Print personality-behavior correlations if debug
    if debug:
        print(f"\n{'='*80}")
        print("PERSONALITY-BEHAVIOR ANALYSIS")
        print(f"{'='*80}")
        for agent_id, cognition in cognition_map.items():
            profile = profiles_map[agent_id]
            personality = BigFiveTraits.from_dict(cognition.scratchpad.state["personality"])
            emotional_state = EmotionalState.from_dict(cognition.scratchpad.state["emotional_state"])
            needs = NeedsHierarchy.from_dict(cognition.scratchpad.state["needs"])

            print(f"\n{profile.name}:")
            print(f"  Personality: {personality.describe()}")
            print(f"  Final emotion: {emotional_state.get_emotion_description()}")
            print(f"  Final motivation: {needs.get_motivation_summary()}")

            # This is where we'd show behavioral consistency metrics
            # For now, just show the state
            if agent_id in rules.eliminated_agents:
                print(f"  Status: ELIMINATED")
            else:
                print(f"  Status: Survived")

    return result


async def main():
    """Main entry point."""
    args = parse_args()

    try:
        await run_simulation(
            ticks=args.ticks,
            use_llm=args.llm,
            debug=args.debug,
        )
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\n\nError during simulation: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
