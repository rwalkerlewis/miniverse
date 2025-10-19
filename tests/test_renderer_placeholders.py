from datetime import datetime
from miniverse.cognition.prompts import PromptTemplate
from miniverse.cognition.context import PromptContext
from miniverse.cognition.renderers import render_prompt
from miniverse.schemas import AgentPerception, AgentProfile, WorldState, EnvironmentState, ResourceState


def _minimal_context(extra=None):
    return PromptContext(
        agent_profile=AgentProfile(
            agent_id="a", name="A", age=18, background="", role="", personality="", skills={}, goals=[], relationships={}
        ),
        perception=AgentPerception(tick=1),
        world_snapshot=WorldState(
            tick=1,
            timestamp=datetime.now(),
            environment=EnvironmentState(metrics={}),
            resources=ResourceState(metrics={}),
            agents=[],
            recent_events=[],
        ),
        scratchpad_state={},
        plan_state={},
        memories=[],
        extra=extra or {},
    )


def test_base_agent_prompt_is_prepended():
    tmpl = PromptTemplate(name="t", system="SYSTEM", user="USER")
    ctx = _minimal_context(extra={"base_agent_prompt": "AGENT_RULES"})
    rendered = render_prompt(tmpl, ctx, include_default=False)
    assert rendered.system.startswith("AGENT_RULES")


def test_action_catalog_is_rendered():
    tmpl = PromptTemplate(name="t", system="S", user="{{action_catalog}}")
    actions = [
        {"name": "move", "schema": {"action_type": "move"}, "examples": [{"action_type": "move", "parameters": {"direction": "up"}}]},
    ]
    ctx = _minimal_context(extra={"available_actions": actions})
    rendered = render_prompt(tmpl, ctx, include_default=False)
    assert "Action Catalog" in rendered.user
    assert "move" in rendered.user

