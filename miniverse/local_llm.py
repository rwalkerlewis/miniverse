"""Utilities for calling locally hosted LLMs (e.g., Ollama)."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from urllib import error, request

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
_CHAT_ENDPOINT = "/api/chat"


class LocalLLMError(RuntimeError):
    """Raised when a local LLM invocation fails."""


def _perform_ollama_request(
    payload: dict[str, Any],
    base_url: str,
    timeout: float,
) -> str:
    """Execute the blocking HTTP request against the Ollama REST API."""

    url = f"{base_url.rstrip('/')}{_CHAT_ENDPOINT}"
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
        message = body or exc.reason
        raise LocalLLMError(
            f"Ollama chat request failed with status {exc.code}: {message}"
        ) from exc
    except error.URLError as exc:
        raise LocalLLMError(
            f"Could not reach Ollama at {url}: {exc.reason}"
        ) from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LocalLLMError("Ollama returned non-JSON response.") from exc

    message = parsed.get("message") or {}
    content = message.get("content")
    if not content:
        raise LocalLLMError("Ollama response did not include assistant content.")

    return content


async def call_ollama_chat(
    *,
    system_prompt: str,
    user_prompt: str,
    llm_model: str,
    base_url: str | None = None,
    timeout: float = 120.0,
) -> str:
    """Invoke a local Ollama model and return the assistant text."""

    resolved_base = (
        base_url or os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL
    ).rstrip("/")

    system_prompt = system_prompt.strip()
    user_prompt = user_prompt.strip()
    if not user_prompt:
        raise LocalLLMError("Cannot call Ollama with an empty user prompt.")

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": llm_model,
        "messages": messages,
        "stream": False,
    }

    return await asyncio.to_thread(
        _perform_ollama_request,
        payload,
        resolved_base,
        timeout,
    )


__all__ = ["LocalLLMError", "call_ollama_chat", "DEFAULT_OLLAMA_BASE_URL"]
