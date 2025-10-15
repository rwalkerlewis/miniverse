"""Tests for scenario loading via ScenarioLoader."""

from pathlib import Path

from miniverse.scenario import ScenarioLoader


def test_scenario_loader_parses_metrics():
    loader = ScenarioLoader(scenarios_dir=Path("examples/workshop"))
    world_state, profiles = loader.load("scenario")
    agents = {profile.agent_id: profile for profile in profiles}

    assert world_state.environment.metrics
    assert world_state.resources.metrics
    assert world_state.agents

    first_agent = world_state.agents[0]
    assert "energy" in first_agent.attributes
    assert first_agent.energy > 0
    assert isinstance(first_agent.current_activity, (str, type(None)))

    assert agents[first_agent.agent_id].name == first_agent.display_name
