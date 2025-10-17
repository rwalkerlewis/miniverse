"""
Simple test to verify information diffusion works end-to-end.

This test simulates what the Valentine's scenario does:
1. One agent sends a message
2. Check that recipient gets the memory
3. Verify recipient can retrieve it

No LLM required - uses deterministic executor.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from uuid import uuid4

from miniverse import (
    Orchestrator,
    AgentProfile,
    AgentStatus,
    WorldState,
    ResourceState,
    EnvironmentState,
    SimulationRules,
    Stat,
)
from miniverse.cognition import AgentCognition, Scratchpad
from miniverse.cognition.executor import Executor
from miniverse.cognition.context import PromptContext
from miniverse.persistence import InMemoryPersistence
from miniverse.memory import SimpleMemoryStream
from miniverse.schemas import AgentAction, AgentPerception
from miniverse.cognition.planner import Plan


class CommunicateExecutor(Executor):
    """Deterministic executor that always sends a party invitation."""

    async def choose_action(
        self,
        agent_id: str,
        perception: AgentPerception,
        scratchpad,
        plan: Plan | None = None,
        plan_step=None,
        context: PromptContext | None = None,
    ) -> AgentAction:
        # Alice always sends message on tick 1
        if agent_id == "alice" and perception.tick == 1:
            return AgentAction(
                agent_id="alice",
                tick=perception.tick,
                action_type="communicate",
                target="bob",
                reasoning="Inviting Bob to party",
                communication={
                    "to": "bob",
                    "message": "Hey Bob! Party at my place on Friday 5pm. You should come!"
                },
            )

        # Bob rests
        return AgentAction(
            agent_id=agent_id,
            tick=perception.tick,
            action_type="rest",
            target=None,
            reasoning="Resting",
            communication=None,
        )


class SimpleRules(SimulationRules):
    """Minimal physics - just increment tick."""

    def apply_tick(self, state, tick):
        updated = state.model_copy(deep=True)
        updated.tick = tick
        return updated

    def validate_action(self, action, state):
        return True


@pytest.mark.asyncio
async def test_information_diffusion():
    """Test that Bob receives Alice's message."""

    # Setup world
    world_state = WorldState(
        tick=0,
        timestamp=datetime.now(timezone.utc),
        environment=EnvironmentState(metrics={}),
        resources=ResourceState(metrics={}),
        agents=[
            AgentStatus(agent_id="alice", location="home", display_name="Alice"),
            AgentStatus(agent_id="bob", location="home", display_name="Bob"),
        ],
    )

    # Setup agents
    agents = {
        "alice": AgentProfile(
            agent_id="alice",
            name="Alice",
            age=25,
            background="Party planner",
            role="host",
            personality="social",
            skills={},
            goals=["Invite friends to party"],
            relationships={},
        ),
        "bob": AgentProfile(
            agent_id="bob",
            name="Bob",
            age=27,
            background="Friend",
            role="guest",
            personality="friendly",
            skills={},
            goals=["Have fun"],
            relationships={},
        ),
    }

    # Setup cognition with deterministic executor
    cognition_map = {
        "alice": AgentCognition(
            executor=CommunicateExecutor(), scratchpad=Scratchpad()
        ),
        "bob": AgentCognition(executor=CommunicateExecutor(), scratchpad=Scratchpad()),
    }

    # Setup persistence and memory
    persistence = InMemoryPersistence()
    memory = SimpleMemoryStream(persistence)

    # Create orchestrator
    orchestrator = Orchestrator(
        world_state=world_state,
        agents=agents,
        world_prompt="",
        agent_prompts={"alice": "You are Alice", "bob": "You are Bob"},
        simulation_rules=SimpleRules(),
        agent_cognition=cognition_map,
        persistence=persistence,
        memory=memory,
    )

    print("🧪 Testing Information Diffusion")
    print("=" * 60)
    print("\n📋 Scenario:")
    print("  - Tick 1: Alice sends party invitation to Bob")
    print("  - Tick 2: Check if Bob has party-related memory")
    print()

    # Run simulation (but don't close yet - we need to query first!)
    # Note: orchestrator.run() calls close() in finally block which CLEARS InMemoryPersistence
    # So we manually run ticks instead
    await orchestrator.persistence.initialize()
    await orchestrator.memory.initialize()

    run_id = orchestrator.run_id

    # Save initial state
    await orchestrator.persistence.save_state(run_id, 0, orchestrator.current_state)

    print("Starting manual simulation")
    for tick in range(1, 3):
        print(f"\n=== Tick {tick}/2 ===")
        await orchestrator._run_tick(tick)

    print("\n✅ Simulation complete!")

    print("\n📊 Results:")
    print("=" * 60)

    # Check Alice's memories (using persistence to get AgentMemory objects)
    alice_memories = await persistence.get_recent_memories(run_id, "alice", limit=10)
    print(f"\n✉️ Alice's memories ({len(alice_memories)}):")
    for mem in alice_memories:
        print(f"   - [{mem.memory_type}] {mem.content}")

    # Check Bob's memories (THIS IS THE CRITICAL TEST!)
    bob_memories = await persistence.get_recent_memories(run_id, "bob", limit=10)
    print(f"\n📬 Bob's memories ({len(bob_memories)}):")
    for mem in bob_memories:
        print(f"   - [{mem.memory_type}] {mem.content}")

    # Verify Bob received the message
    party_keywords = ["party", "Friday", "5pm"]
    bob_knows_party = any(
        any(keyword.lower() in mem.content.lower() for keyword in party_keywords)
        for mem in bob_memories
    )

    print("\n" + "=" * 60)
    if bob_knows_party:
        print("✅ SUCCESS: Bob received Alice's party invitation!")
        print("   Information successfully diffused from Alice to Bob")
        print("\n🎉 The fix works! Recipients get communication memories.")
    else:
        print("❌ FAILURE: Bob did not receive the message")
        print("   Information diffusion broken")
        print("\n⚠️ The bug is still present")

    # NOW we can close
    await persistence.close()
    await memory.close()

    return bob_knows_party


if __name__ == "__main__":
    success = asyncio.run(test_information_diffusion())
    exit(0 if success else 1)
