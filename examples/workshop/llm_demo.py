"""Helper script that forces the workshop example to use LLM cognition.

Usage (requires provider/model + API key in environment):

    export MINIVERSE_LLM_PROVIDER=openai
    export MINIVERSE_LLM_MODEL=gpt-5-nano
    export OPENAI_API_KEY=...
    UV_CACHE_DIR=.uv-cache uv run python -m examples.workshop.llm_demo --ticks 8

This simply delegates to ``run_simulation`` with ``use_llm=True`` so you can keep
``examples.workshop.run`` focused on the deterministic defaults.
"""

from __future__ import annotations

import argparse
import asyncio

from .run import run_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Workshop LLM cognition demo")
    parser.add_argument("--ticks", type=int, default=8, help="Number of ticks to simulate")
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    result = await run_simulation(args.ticks, use_llm=True, verbose=True)
    final_state = result["final_state"]
    print("Final backlog:", final_state.resources.get_metric("task_backlog").value)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
