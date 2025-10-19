"""Tests for partial perception construction."""

from datetime import datetime

from miniverse.perception import build_agent_perception
from miniverse.schemas import (
    AgentAction,
    AgentPerception,
    AgentStatus,
    ResourceState,
    Stat,
    WorldEvent,
    WorldState,
    EnvironmentState,
)


def make_world_state() -> WorldState:
    environment = EnvironmentState(
        metrics={
            "temperature": Stat(value=-12, unit="°C", label="Ambient Temperature"),
            "wind_speed": Stat(value=22, unit="km/h", label="Wind Speed"),
        }
    )
    resources = ResourceState(
        metrics={
            "power": Stat(value=140.0, unit="kWh", label="Stored Power"),
            "water": Stat(value=460.0, unit="L", label="Water Reserve"),
        }
    )
    agents = [
        AgentStatus(
            agent_id="alpha",
            location="operations",
            activity="monitor",
            attributes={
                "health": Stat(value=92, unit="%", label="Health"),
                "stress": Stat(value=28, unit="%", label="Stress"),
                "energy": Stat(value=76, unit="%", label="Energy"),
            },
        ),
        AgentStatus(
            agent_id="beta",
            location="lab",
            activity="research",
            attributes={
                "health": Stat(value=88, unit="%", label="Health"),
                "stress": Stat(value=34, unit="%", label="Stress"),
                "energy": Stat(value=82, unit="%", label="Energy"),
            },
        ),
    ]

    events = [
        WorldEvent(
            event_id="evt-1",
            tick=3,
            category="alert",
            description="External wind gust exceeding threshold",
            severity=6,
        ),
        WorldEvent(
            event_id="evt-2",
            tick=2,
            category="info",
            description="Routine power check complete",
            severity=2,
        ),
    ]

    return WorldState(
        tick=3,
        timestamp=datetime(2160, 3, 21, 10, 30, 0),
        environment=environment,
        resources=resources,
        agents=agents,
        recent_events=events,
    )


def test_build_agent_perception_structure():
    world_state = make_world_state()
    recent_messages = [
        {"from": "beta", "message": "Expect gusts over 25 km/h."}
    ]

    perception = build_agent_perception(
        agent_id="alpha",
        world_state=world_state,
        recent_messages=recent_messages,
        recent_memories=["Calibrated sensors at tick 2"],
    )

    assert isinstance(perception, AgentPerception)
    assert "health" in perception.personal_attributes
    assert "power" in perception.visible_resources
    assert "temperature" in perception.environment_snapshot
    assert perception.system_alerts == [
        "External wind gust exceeding threshold"
    ]
    assert perception.messages == [
        {"from": "beta", "message": "Expect gusts over 25 km/h."}
    ]
