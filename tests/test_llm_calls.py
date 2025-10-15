"""Tests for the high-level LLM call helpers."""

import json
from datetime import datetime

import pytest

from miniverse.llm_calls import get_agent_action, process_world_update
from miniverse.schemas import (
    AgentAction,
    AgentPerception,
    AgentStatus,
    EnvironmentState,
    ResourceState,
    Stat,
    WorldState,
)


@pytest.mark.asyncio
async def test_get_agent_action_delegates_to_retry_helper(monkeypatch):
    perception = AgentPerception(
        tick=7,
        location="operations",
        personal_attributes={"energy": Stat(value=72, unit="%", label="Energy")},
        visible_resources={},
        environment_snapshot={},
        system_alerts=[],
        messages=[],
        recent_observations=["Checked diagnostics"],
    )

    captured = {}

    async def fake_call_llm_with_retries(**kwargs):
        captured.update(kwargs)
        return AgentAction(
            agent_id="alpha",
            tick=perception.tick,
            action_type="inspect",
            target="station_b",
            parameters=None,
            reasoning="Need to confirm throughput",
            communication=None,
        )

    monkeypatch.setattr(
        "miniverse.llm_calls.call_llm_with_retries", fake_call_llm_with_retries
    )

    result = await get_agent_action(
        system_prompt="You are agent alpha",
        perception=perception,
        llm_provider="openai",
        llm_model="gpt-5-nano",
    )

    assert result.action_type == "inspect"
    assert captured["llm_provider"] == "openai"
    assert captured["llm_model"] == "gpt-5-nano"
    assert "Current situation" in captured["user_prompt"]
    assert json.loads(perception.model_dump_json())


@pytest.mark.asyncio
async def test_process_world_update_formats_prompt(monkeypatch):
    state = WorldState(
        tick=4,
        timestamp=datetime(2160, 3, 21, 10, 0, 0),
        environment=EnvironmentState(
            metrics={"temperature": Stat(value=21.0, unit="°C", label="Ambient Temp")}
        ),
        resources=ResourceState(
            metrics={"power": Stat(value=110.0, unit="kWh", label="Battery Reserve")}
        ),
        agents=[
            AgentStatus(
                agent_id="alpha",
                location="operations",
                attributes={"energy": Stat(value=80, unit="%")},
            )
        ],
    )
    actions = [
        AgentAction(
            agent_id="alpha",
            tick=5,
            action_type="inspect",
            target="station_b",
            parameters=None,
            reasoning="Need to check anomalies",
            communication=None,
        )
    ]

    captured = {}

    async def fake_call_llm_with_retries(**kwargs):
        captured.update(kwargs)
        return state.model_copy(update={"tick": 5})

    monkeypatch.setattr(
        "miniverse.llm_calls.call_llm_with_retries", fake_call_llm_with_retries
    )

    result = await process_world_update(
        current_state=state,
        actions=actions,
        tick=5,
        system_prompt="You are the world engine",
        llm_provider="openai",
        llm_model="gpt-5-nano",
        physics_applied=True,
    )

    assert result.tick == 5
    assert captured["llm_provider"] == "openai"
    assert captured["llm_model"] == "gpt-5-nano"
    assert "Physics already applied" in captured["user_prompt"]
    assert "Agent actions this tick" in captured["user_prompt"]
