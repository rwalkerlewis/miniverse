"""Tests for keyword-based memory retrieval."""

import asyncio
from uuid import uuid4
import pytest

from miniverse.memory import SimpleMemoryStream
from miniverse.persistence import InMemoryPersistence


@pytest.mark.asyncio
async def test_keyword_retrieval():
    persistence = InMemoryPersistence()
    await persistence.initialize()
    memory = SimpleMemoryStream(persistence)

    run_id = uuid4()
    agent_id = "alpha"

    await memory.add_memory(run_id, agent_id, tick=1, memory_type="observation", content="Agent inspected the oxygen recycler", importance=5)
    await memory.add_memory(run_id, agent_id, tick=2, memory_type="observation", content="Team discussed revenue targets", importance=5)
    await memory.add_memory(run_id, agent_id, tick=3, memory_type="observation", content="Agent repaired the recycler filters", importance=7)

    results = await memory.get_relevant_memories(run_id, agent_id, query="recycler", limit=2)
    assert len(results) == 2
    assert "recycler" in results[0]

    await persistence.close()
