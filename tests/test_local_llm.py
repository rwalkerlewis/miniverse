import pytest

from miniverse.local_llm import call_ollama_chat


@pytest.mark.asyncio
async def test_call_ollama_chat_builds_payload(monkeypatch):
    captured: dict[str, object] = {}

    def fake_request(payload, base_url, timeout):
        captured["payload"] = payload
        captured["base_url"] = base_url
        captured["timeout"] = timeout
        return '{"content":"ok"}'

    monkeypatch.setattr("miniverse.local_llm._perform_ollama_request", fake_request)

    result = await call_ollama_chat(
        system_prompt="System context",
        user_prompt="User payload",
        llm_model="llama3.1",
        base_url="http://localhost:11434",
        timeout=30,
    )

    assert result == '{"content":"ok"}'
    payload = captured["payload"]
    assert payload["model"] == "llama3.1"
    assert payload["messages"][0] == {"role": "system", "content": "System context"}
    assert payload["messages"][1] == {"role": "user", "content": "User payload"}
    assert captured["base_url"] == "http://localhost:11434"
    assert captured["timeout"] == 30
