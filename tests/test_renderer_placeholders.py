from datetime import datetime, timezone
from miniverse.cognition.prompts import PromptTemplate, DEFAULT_PROMPTS
from miniverse.cognition.context import PromptContext
from miniverse.cognition.renderers import render_prompt
from miniverse.schemas import AgentPerception, AgentProfile, WorldState, EnvironmentState, ResourceState


def _minimal_context(extra=None):
    return PromptContext(
        agent_profile=AgentProfile(
            agent_id="alice",
            name="Alice Smith",
            age=28,
            background="I am a test agent for this simulation.",
            role="engineer",
            personality="analytical",
            skills={"engineering": "expert"},
            goals=["Test the system"],
            relationships={"bob": "colleague"},
        ),
        perception=AgentPerception(tick=0),
        world_snapshot=WorldState(
            tick=1,
            timestamp=datetime.now(timezone.utc),
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


def test_base_agent_prompt_goes_to_user_with_fallback():
    # Template without placeholder should get base prompt auto-prepended to USER
    tmpl = PromptTemplate(name="t", system="SYSTEM", user="USER")
    ctx = _minimal_context(extra={"base_agent_prompt": "AGENT_RULES"})
    rendered = render_prompt(tmpl, ctx, include_default=False)
    assert rendered.user.startswith("AGENT_RULES")


def test_action_catalog_is_rendered():
    tmpl = PromptTemplate(name="t", system="S", user="{{action_catalog}}")
    actions = [
        {"name": "move", "schema": {"action_type": "move"}, "examples": [{"action_type": "move", "parameters": {"direction": "up"}}]},
    ]
    ctx = _minimal_context(extra={"available_actions": actions})
    rendered = render_prompt(tmpl, ctx, include_default=False)
    assert "Action Catalog" in rendered.user
    assert "move" in rendered.user


def test_character_prompt_in_system_with_fallback():
    # Template lacks character placeholder → should be auto-prepended to SYSTEM
    tmpl = PromptTemplate(name="t", system="SYSTEM", user="USER")
    ctx = _minimal_context()
    rendered = render_prompt(tmpl, ctx, include_default=False)
    assert rendered.system.startswith("I am Alice Smith.")


def test_default_template_places_identity_and_base_correctly():
    tmpl = DEFAULT_PROMPTS.get("default")
    ctx = _minimal_context(extra={"base_agent_prompt": "Focus on safety."})
    rendered = render_prompt(tmpl, ctx, include_default=False)
    # Identity at the start of SYSTEM
    assert rendered.system.startswith("I am Alice Smith.")
    # Base prompt at the start of USER
    assert rendered.user.startswith("Focus on safety.")

