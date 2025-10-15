"""Helper utilities for LLM-related error handling and retries."""

import asyncio
from typing import Callable, TypeVar

from mirascope import llm
from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt


ModelT = TypeVar("ModelT", bound=BaseModel)


def inject_validation_feedback(error: ValidationError) -> str:
    """Produce a short message describing why validation failed."""

    messages = []
    for err in error.errors(include_url=False):  # pragma: no branch - tiny list
        loc = ".".join(str(part) for part in err.get("loc", [])) or "root"
        msg = err.get("msg", "validation error")
        messages.append(f"- {loc}: {msg}")
    return "Correct the schema errors:\n" + "\n".join(messages)


async def call_llm_with_retries(
    *,
    system_prompt: str,
    user_prompt: str,
    llm_provider: str,
    llm_model: str,
    response_model: type[ModelT],
    max_attempts: int = 3,
    feedback_builder: Callable[[ValidationError], str] = inject_validation_feedback,
) -> ModelT:
    """Invoke a structured LLM call with validation-aware retries."""

    prompt_sections = [section.strip() for section in (system_prompt, user_prompt) if section.strip()]
    base_prompt = "\n\n".join(prompt_sections)

    feedback: str | None = None

    @llm.call(provider=llm_provider, model=llm_model, response_model=response_model)
    async def _invoke(prompt: str) -> str:
        return prompt

    attempt_number = 0
    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type(ValidationError),
        stop=stop_after_attempt(max_attempts),
        reraise=True,
    ):
        with attempt:
            attempt_number += 1
            if attempt_number > 1:
                print(
                    f"LLM retry {attempt_number}/{max_attempts} for {response_model.__name__};"
                    " attempting schema correction."
                )
            final_prompt = base_prompt if feedback is None else f"{base_prompt}\n\n{feedback}"
            try:
                return await asyncio.wait_for(_invoke(final_prompt), timeout=120)
            except ValidationError as exc:  # pragma: no cover - retry path
                feedback = feedback_builder(exc)
                raise
            except asyncio.TimeoutError as exc:  # pragma: no cover - timeout path
                print(
                    f"LLM call timed out after 120s for {response_model.__name__}."
                )
                raise exc

    # AsyncRetrying with reraise=True will always exit via return or raise.
    raise RuntimeError("LLM retry mechanism exited unexpectedly")
