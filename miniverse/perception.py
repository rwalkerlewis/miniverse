"""
Perception building module for Varela simulation.

Implements partial observability - agents only know what they can actually perceive:
- Their own status (health, stress, energy)
- Their current location
- Base-wide resource levels (displayed on screens)
- System alerts (high-severity events)
- Direct messages sent to them
- Recent memories (last 10 observations)

Based on Stanford Generative Agents research - partial observability is critical
for realistic agent behavior.
"""

from typing import List

from miniverse.schemas import WorldState, AgentPerception, AgentAction


def build_agent_perception(
    agent_id: str,
    world_state: WorldState,
    recent_actions: List[AgentAction],
    recent_memories: List[str],
) -> AgentPerception:
    """
    Build partial observability perception for a specific agent.

    Args:
        agent_id: ID of agent to build perception for
        world_state: Current complete world state
        recent_actions: Recent actions from all agents (for extracting messages)
        recent_memories: Recent memory strings for this agent (last 10)

    Returns:
        AgentPerception with only what this agent can perceive

    Raises:
        ValueError: If agent_id not found in world state
    """
    # Find this agent's status
    agent_status = None
    for agent in world_state.agents:
        if agent.agent_id == agent_id:
            agent_status = agent
            break

    if agent_status is None:
        raise ValueError(f"Agent {agent_id} not found in world state")

    # Clone metric maps so perceptions remain decoupled from world state
    personal_attributes = {
        key: stat.model_copy(deep=True)
        for key, stat in agent_status.attributes.items()
    }

    visible_resources = {
        key: stat.model_copy(deep=True)
        for key, stat in world_state.resources.metrics.items()
    }

    environment_snapshot = {
        key: stat.model_copy(deep=True)
        for key, stat in world_state.environment.metrics.items()
    }

    # Extract system alerts (only high-severity events)
    system_alerts = [
        event.description
        for event in world_state.recent_events
        if event.severity is not None and event.severity >= 5
    ]

    # Extract messages sent to this agent from recent actions
    messages = []
    for action in recent_actions:
        if action.communication and action.communication.get("to") == agent_id:
            messages.append(
                {
                    "from": action.agent_id,
                    "message": action.communication["message"],
                }
            )

    # Use provided recent memories (already filtered to last 10)
    recent_observations = recent_memories

    return AgentPerception(
        tick=world_state.tick,
        personal_attributes=personal_attributes,
        location=agent_status.location,
        visible_resources=visible_resources,
        environment_snapshot=environment_snapshot,
        system_alerts=system_alerts,
        messages=messages,
        recent_observations=recent_observations,
    )
