"""Unit tests for the LLM retry helper."""

import pytest
from pydantic import BaseModel, ValidationError

from miniverse.llm_utils import call_llm_with_retries


class DummyModel(BaseModel):
    content: str


@pytest.mark.asyncio
async def test_call_llm_with_retries_success(monkeypatch):
    recorded_prompts: list[str] = []

    async def fake_caller(prompt: str) -> DummyModel:
        recorded_prompts.append(prompt)
        return DummyModel(content="ok")

    def fake_decorator(*, provider, model, response_model):
        assert response_model is DummyModel

        def wrapper(fn):
            async def inner(prompt: str):
                return await fake_caller(prompt)

            return inner

        return wrapper

    monkeypatch.setattr("miniverse.llm_utils.llm.call", fake_decorator)

    result = await call_llm_with_retries(
        system_prompt="System context",
        user_prompt="What now?",
        llm_provider="openai",
        llm_model="gpt-5-nano",
        response_model=DummyModel,
    )

    assert result.content == "ok"
    assert recorded_prompts == ["System context\n\nWhat now?"]


@pytest.mark.asyncio
async def test_call_llm_with_retries_injects_feedback(monkeypatch):
    attempts: list[str] = []

    try:
        DummyModel.model_validate({})
    except ValidationError as exc:
        validation_error = exc

    async def fake_caller(prompt: str) -> DummyModel:
        attempts.append(prompt)
        if len(attempts) == 1:
            raise validation_error
        return DummyModel(content="fixed")

    def fake_decorator(*, provider, model, response_model):
        def wrapper(fn):
            async def inner(prompt: str):
                return await fake_caller(prompt)

            return inner

        return wrapper

    monkeypatch.setattr("miniverse.llm_utils.llm.call", fake_decorator)

    result = await call_llm_with_retries(
        system_prompt="System",
        user_prompt="User",
        llm_provider="openai",
        llm_model="gpt-5-nano",
        response_model=DummyModel,
    )

    assert result.content == "fixed"
    assert len(attempts) == 2
    assert "Your previous JSON response failed to validate against the required schema." in attempts[1]
    assert "- content: Field required" in attempts[1]


@pytest.mark.asyncio
async def test_call_llm_with_retries_local_provider(monkeypatch):
    captured_kwargs: dict[str, str] = {}

    async def fake_local_call(*, system_prompt, user_prompt, llm_model, base_url=None, timeout=120.0):
        captured_kwargs["system_prompt"] = system_prompt
        captured_kwargs["user_prompt"] = user_prompt
        captured_kwargs["llm_model"] = llm_model
        captured_kwargs["base_url"] = base_url
        captured_kwargs["timeout"] = str(timeout)
        return '{"content":"ok"}'

    def fail_decorator(*args, **kwargs):
        raise AssertionError("remote provider path should not be used for ollama")

    monkeypatch.setattr("miniverse.llm_utils.call_ollama_chat", fake_local_call)
    monkeypatch.setattr("miniverse.llm_utils.llm.call", fail_decorator)

    result = await call_llm_with_retries(
        system_prompt="System context",
        user_prompt="User payload",
        llm_provider="ollama",
        llm_model="llama3.1",
        response_model=DummyModel,
    )

    assert result.content == "ok"
    assert captured_kwargs["system_prompt"] == "System context"
    assert captured_kwargs["user_prompt"] == "User payload"
    assert captured_kwargs["llm_model"] == "llama3.1"
