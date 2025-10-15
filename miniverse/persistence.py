"""
PersistenceStrategy interface for storing simulation state.

This module provides the abstract base class for implementing storage backends.
Users can choose to run simulations entirely in-memory (no database required)
or implement custom persistence strategies for PostgreSQL, files, or other backends.

Key responsibilities:
- Save world state at each tick
- Retrieve world state by tick number
- Store agent actions
- Store agent memories
- Query simulation history

Design principle: Library should work without any storage backend (in-memory default).
"""

import asyncio
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict
from uuid import UUID
from datetime import datetime

from miniverse.schemas import (
    WorldState,
    AgentAction,
    AgentMemory,
    SimulationRun,
)
from .config import Config

try:  # Optional dependency (only needed for PostgresPersistence)
    import asyncpg
except ImportError:  # pragma: no cover - asyncpg may not be installed for json/memory usage
    asyncpg = None


class PersistenceStrategy(ABC):
    """
    Abstract base class for simulation state persistence.

    This interface allows users to choose how (or if) they want to store
    simulation data. Default implementation (InMemoryPersistence) requires
    no database setup.

    Use cases:
    - InMemoryPersistence: Fast prototyping, testing, no storage needed
    - PostgresPersistence: Full historical data, analysis, replay
    - FilePersistence: Simple JSON/pickle files for small simulations
    - CloudPersistence: S3, Google Cloud Storage, etc.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the persistence backend.

        Called once before simulation starts. Used to set up database
        connections, create tables, open files, etc.

        Raises:
            Exception: If initialization fails
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close the persistence backend.

        Called once after simulation completes. Used to close database
        connections, flush buffers, close files, etc.

        Raises:
            Exception: If cleanup fails
        """
        pass

    @abstractmethod
    async def save_run_metadata(self, run: SimulationRun) -> None:
        """
        Save metadata about a simulation run.

        Args:
            run: SimulationRun object with run metadata

        Raises:
            Exception: If save fails
        """
        pass

    @abstractmethod
    async def update_run_status(
        self, run_id: UUID, status: str, end_time: Optional[datetime] = None
    ) -> None:
        """
        Update the status of a simulation run.

        Args:
            run_id: Unique run identifier
            status: New status (running, completed, failed)
            end_time: Optional end time if run completed

        Raises:
            Exception: If update fails
        """
        pass

    @abstractmethod
    async def save_state(self, run_id: UUID, tick: int, state: WorldState) -> None:
        """
        Save world state for a specific tick.

        Args:
            run_id: Unique run identifier
            tick: Tick number
            state: Complete world state to save

        Raises:
            Exception: If save fails
        """
        pass

    @abstractmethod
    async def get_state(self, run_id: UUID, tick: int) -> Optional[WorldState]:
        """
        Retrieve world state for a specific tick.

        Args:
            run_id: Unique run identifier
            tick: Tick number to retrieve

        Returns:
            WorldState if found, None otherwise

        Raises:
            Exception: If retrieval fails
        """
        pass

    @abstractmethod
    async def save_action(self, run_id: UUID, action: AgentAction) -> None:
        """
        Save an agent action.

        Args:
            run_id: Unique run identifier
            action: Agent action to save

        Raises:
            Exception: If save fails
        """
        pass

    @abstractmethod
    async def get_actions_for_tick(
        self, run_id: UUID, tick: int
    ) -> List[AgentAction]:
        """
        Retrieve all agent actions for a specific tick.

        Args:
            run_id: Unique run identifier
            tick: Tick number

        Returns:
            List of agent actions (empty if none found)

        Raises:
            Exception: If retrieval fails
        """
        pass

    @abstractmethod
    async def save_memory(self, run_id: UUID, memory: AgentMemory) -> None:
        """
        Save an agent memory.

        Args:
            run_id: Unique run identifier
            memory: Agent memory to save

        Raises:
            Exception: If save fails
        """
        pass

    @abstractmethod
    async def get_recent_memories(
        self, run_id: UUID, agent_id: str, limit: int = 10
    ) -> List[AgentMemory]:
        """
        Retrieve recent memories for an agent.

        Args:
            run_id: Unique run identifier
            agent_id: Agent identifier
            limit: Maximum number of memories to return

        Returns:
            List of recent memories (most recent first)

        Raises:
            Exception: If retrieval fails
        """
        pass

    @abstractmethod
    async def save_actions(self, run_id: UUID, tick: int, actions: List[AgentAction]) -> None:
        """
        Save multiple agent actions for a tick.

        Convenience method for bulk saving. Implementations can optimize
        this for batch operations (e.g., single database transaction).

        Args:
            run_id: Unique run identifier
            tick: Tick number
            actions: List of actions to save

        Raises:
            Exception: If save fails
        """
        pass

    @abstractmethod
    async def get_actions(self, run_id: UUID, tick: int) -> List[AgentAction]:
        """
        Get all actions for a tick.

        Alias for get_actions_for_tick() for naming consistency.

        Args:
            run_id: Unique run identifier
            tick: Tick number

        Returns:
            List of actions for the tick

        Raises:
            Exception: If retrieval fails
        """
        pass


class InMemoryPersistence(PersistenceStrategy):
    """
    In-memory persistence implementation (no database required).

    Stores all simulation data in Python dictionaries. Data is lost when
    the process exits. Perfect for:
    - Rapid prototyping
    - Testing
    - Short simulations where history isn't needed
    - Environments without database access

    Not suitable for:
    - Long-running simulations with large history
    - Multi-process simulations
    - Simulations that need to be replayed later
    """

    def __init__(self):
        """Initialize in-memory storage."""
        self.runs: Dict[UUID, SimulationRun] = {}
        self.states: Dict[tuple[UUID, int], WorldState] = {}
        self.actions: Dict[tuple[UUID, int], List[AgentAction]] = {}
        self.memories: Dict[tuple[UUID, str], List[AgentMemory]] = {}

    async def initialize(self) -> None:
        """
        Initialize in-memory storage.

        No-op for in-memory implementation.
        """
        pass

    async def close(self) -> None:
        """
        Close in-memory storage.

        Clears all data. No-op otherwise.
        """
        self.runs.clear()
        self.states.clear()
        self.actions.clear()
        self.memories.clear()

    async def save_run_metadata(self, run: SimulationRun) -> None:
        """
        Save run metadata to memory.

        Args:
            run: SimulationRun metadata
        """
        self.runs[run.id] = run

    async def update_run_status(
        self, run_id: UUID, status: str, end_time: Optional[datetime] = None
    ) -> None:
        """
        Update run status in memory.

        Args:
            run_id: Run identifier
            status: New status
            end_time: Optional end time
        """
        if run_id in self.runs:
            self.runs[run_id].status = status
            if end_time:
                self.runs[run_id].end_time = end_time

    async def save_state(self, run_id: UUID, tick: int, state: WorldState) -> None:
        """
        Save world state to memory.

        Args:
            run_id: Run identifier
            tick: Tick number
            state: World state to save
        """
        self.states[(run_id, tick)] = state

    async def get_state(self, run_id: UUID, tick: int) -> Optional[WorldState]:
        """
        Retrieve world state from memory.

        Args:
            run_id: Run identifier
            tick: Tick number

        Returns:
            WorldState if found, None otherwise
        """
        return self.states.get((run_id, tick))

    async def save_action(self, run_id: UUID, action: AgentAction) -> None:
        """
        Save agent action to memory.

        Args:
            run_id: Run identifier
            action: Agent action to save
        """
        key = (run_id, action.tick)
        if key not in self.actions:
            self.actions[key] = []
        self.actions[key].append(action)

    async def get_actions_for_tick(
        self, run_id: UUID, tick: int
    ) -> List[AgentAction]:
        """
        Retrieve actions for a tick from memory.

        Args:
            run_id: Run identifier
            tick: Tick number

        Returns:
            List of actions (empty if none)
        """
        return self.actions.get((run_id, tick), [])

    async def save_memory(self, run_id: UUID, memory: AgentMemory) -> None:
        """
        Save agent memory to memory storage.

        Args:
            run_id: Run identifier
            memory: Agent memory to save
        """
        key = (run_id, memory.agent_id)
        if key not in self.memories:
            self.memories[key] = []
        self.memories[key].append(memory)

    async def get_recent_memories(
        self, run_id: UUID, agent_id: str, limit: int = 10
    ) -> List[AgentMemory]:
        """
        Retrieve recent memories from memory storage.

        Args:
            run_id: Run identifier
            agent_id: Agent identifier
            limit: Maximum memories to return

        Returns:
            List of recent memories (most recent first)
        """
        key = (run_id, agent_id)
        all_memories = self.memories.get(key, [])
        # Sort by tick descending, take limit
        sorted_memories = sorted(all_memories, key=lambda m: m.tick, reverse=True)
        return sorted_memories[:limit]

    async def save_actions(self, run_id: UUID, tick: int, actions: List[AgentAction]) -> None:
        """
        Save multiple agent actions for a tick.

        Convenience method for saving multiple actions at once.

        Args:
            run_id: Run identifier
            tick: Tick number
            actions: List of agent actions to save
        """
        for action in actions:
            await self.save_action(run_id, action)

    async def get_actions(self, run_id: UUID, tick: int) -> List[AgentAction]:
        """
        Alias for get_actions_for_tick.

        Args:
            run_id: Run identifier
            tick: Tick number

        Returns:
            List of actions for the tick
        """
        return await self.get_actions_for_tick(run_id, tick)


class PostgresPersistence(PersistenceStrategy):
    """PostgreSQL-backed persistence implementation."""

    def __init__(self, database_url: Optional[str] = None):
        if asyncpg is None:  # pragma: no cover - handled during runtime when dependency missing
            raise ImportError(
                "asyncpg is required for PostgresPersistence. Install with `uv add asyncpg`."
            )

        self.database_url = database_url or Config.DATABASE_URL
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self) -> None:
        if self.pool is None:
            self.pool = await asyncpg.create_pool(self.database_url)

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def save_run_metadata(self, run: SimulationRun) -> None:
        assert self.pool is not None, "Persistence not initialized"

        payload = run.model_dump(mode="json")
        config_json = json.dumps(payload["config"])

        query = """
            INSERT INTO simulation_runs (id, start_time, end_time, num_ticks, num_agents, status, config, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
            ON CONFLICT (id) DO UPDATE
            SET start_time=$2, end_time=$3, num_ticks=$4, num_agents=$5, status=$6, config=$7::jsonb
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                run.id,
                run.start_time,
                run.end_time,
                run.num_ticks,
                run.num_agents,
                run.status,
                config_json,
                run.created_at,
            )

    async def update_run_status(
        self, run_id: UUID, status: str, end_time: Optional[datetime] = None
    ) -> None:
        assert self.pool is not None, "Persistence not initialized"

        query = """
            UPDATE simulation_runs
            SET status = $2, end_time = $3
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, run_id, status, end_time)

    async def save_state(self, run_id: UUID, tick: int, state: WorldState) -> None:
        assert self.pool is not None, "Persistence not initialized"

        state_json = state.model_dump_json()
        query = """
            INSERT INTO world_states (run_id, tick, state)
            VALUES ($1, $2, $3::jsonb)
            ON CONFLICT (run_id, tick) DO UPDATE SET state = $3::jsonb
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, run_id, tick, state_json)

    async def get_state(self, run_id: UUID, tick: int) -> Optional[WorldState]:
        assert self.pool is not None, "Persistence not initialized"

        query = """
            SELECT state
            FROM world_states
            WHERE run_id = $1 AND tick = $2
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, run_id, tick)

        if not row:
            return None

        return WorldState.model_validate_json(row["state"])

    async def save_action(self, run_id: UUID, action: AgentAction) -> None:
        await self.save_actions(run_id, action.tick, [action])

    async def save_actions(
        self, run_id: UUID, tick: int, actions: List[AgentAction]
    ) -> None:
        assert self.pool is not None, "Persistence not initialized"

        if not actions:
            return

        query = """
            INSERT INTO agent_actions (run_id, agent_id, tick, action)
            VALUES ($1, $2, $3, $4::jsonb)
        """

        records = [
            (run_id, action.agent_id, tick, action.model_dump_json())
            for action in actions
        ]

        async with self.pool.acquire() as conn:
            await conn.executemany(query, records)

    async def get_actions_for_tick(
        self, run_id: UUID, tick: int
    ) -> List[AgentAction]:
        return await self.get_actions(run_id, tick)

    async def get_actions(self, run_id: UUID, tick: int) -> List[AgentAction]:
        assert self.pool is not None, "Persistence not initialized"

        query = """
            SELECT action
            FROM agent_actions
            WHERE run_id = $1 AND tick = $2
            ORDER BY created_at
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, run_id, tick)

        return [AgentAction.model_validate_json(row["action"]) for row in rows]

    async def save_memory(self, run_id: UUID, memory: AgentMemory) -> None:
        assert self.pool is not None, "Persistence not initialized"

        query = """
            INSERT INTO agent_memories
            (run_id, agent_id, tick, memory_type, content, importance, tags, metadata, embedding_key, branch_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7::text[], $8::jsonb, $9, $10)
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                run_id,
                memory.agent_id,
                memory.tick,
                memory.memory_type,
                memory.content,
                memory.importance,
                memory.tags,
                json.dumps(memory.metadata),
                memory.embedding_key,
                memory.branch_id,
            )

    async def get_recent_memories(
        self, run_id: UUID, agent_id: str, limit: int = 10
    ) -> List[AgentMemory]:
        assert self.pool is not None, "Persistence not initialized"

        query = """
            SELECT id, tick, memory_type, content, importance, tags, metadata, embedding_key, branch_id, created_at
            FROM agent_memories
            WHERE run_id = $1 AND agent_id = $2
            ORDER BY tick DESC
            LIMIT $3
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, run_id, agent_id, limit)

        memories = []
        for row in rows:
            memories.append(
                AgentMemory(
                    id=row["id"],
                    run_id=run_id,
                    agent_id=agent_id,
                    tick=row["tick"],
                    memory_type=row["memory_type"],
                    content=row["content"],
                    importance=row["importance"],
                    tags=row["tags"] or [],
                    metadata=row["metadata"] or {},
                    embedding_key=row["embedding_key"],
                    branch_id=row["branch_id"],
                    created_at=row["created_at"],
                )
            )

        return memories


class JsonPersistence(PersistenceStrategy):
    """Persistence strategy that stores data as JSON files on disk."""

    def __init__(self, base_path: Path | str = "simulation_runs"):
        self.base_path = Path(base_path)

    async def initialize(self) -> None:
        await asyncio.to_thread(self.base_path.mkdir, parents=True, exist_ok=True)

    async def close(self) -> None:
        # Nothing to clean up for JSON persistence
        return None

    async def save_run_metadata(self, run: SimulationRun) -> None:
        run_dir = self._run_dir(run.id)
        await asyncio.to_thread(run_dir.mkdir, parents=True, exist_ok=True)
        path = run_dir / "run.json"
        data = run.model_dump(mode="json")
        await asyncio.to_thread(
            path.write_text, json.dumps(data, indent=2), "utf-8"
        )

    async def update_run_status(
        self, run_id: UUID, status: str, end_time: Optional[datetime] = None
    ) -> None:
        path = self._run_dir(run_id) / "run.json"
        if not path.exists():  # Nothing to update yet
            return

        def _update() -> None:
            payload = json.loads(path.read_text("utf-8"))
            payload["status"] = status
            payload["end_time"] = end_time.isoformat() if end_time else None
            path.write_text(json.dumps(payload, indent=2), "utf-8")

        await asyncio.to_thread(_update)

    async def save_state(self, run_id: UUID, tick: int, state: WorldState) -> None:
        path = self._run_dir(run_id) / "states" / f"{tick:05d}.json"
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        payload = state.model_dump(mode="json")
        await asyncio.to_thread(
            path.write_text, json.dumps(payload, indent=2), "utf-8"
        )

    async def get_state(self, run_id: UUID, tick: int) -> Optional[WorldState]:
        path = self._run_dir(run_id) / "states" / f"{tick:05d}.json"
        if not path.exists():
            return None

        payload = await asyncio.to_thread(json.loads, path.read_text("utf-8"))
        return WorldState.model_validate(payload)

    async def save_action(self, run_id: UUID, action: AgentAction) -> None:
        await self.save_actions(run_id, action.tick, [action])

    async def save_actions(
        self, run_id: UUID, tick: int, actions: List[AgentAction]
    ) -> None:
        path = self._run_dir(run_id) / "actions" / f"{tick:05d}.json"
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        payload = [action.model_dump(mode="json") for action in actions]
        await asyncio.to_thread(
            path.write_text, json.dumps(payload, indent=2), "utf-8"
        )

    async def get_actions_for_tick(
        self, run_id: UUID, tick: int
    ) -> List[AgentAction]:
        return await self.get_actions(run_id, tick)

    async def get_actions(self, run_id: UUID, tick: int) -> List[AgentAction]:
        path = self._run_dir(run_id) / "actions" / f"{tick:05d}.json"
        if not path.exists():
            return []

        payload = await asyncio.to_thread(json.loads, path.read_text("utf-8"))
        return [AgentAction.model_validate(item) for item in payload]

    async def save_memory(self, run_id: UUID, memory: AgentMemory) -> None:
        directory = self._run_dir(run_id) / "memories"
        await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
        path = directory / f"{memory.agent_id}.jsonl"
        payload = memory.model_dump(mode="json")

        def _append() -> None:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload))
                handle.write("\n")

        await asyncio.to_thread(_append)

    async def get_recent_memories(
        self, run_id: UUID, agent_id: str, limit: int = 10
    ) -> List[AgentMemory]:
        path = self._run_dir(run_id) / "memories" / f"{agent_id}.jsonl"
        if not path.exists():
            return []

        def _read() -> List[str]:
            return path.read_text("utf-8").splitlines()

        lines = await asyncio.to_thread(_read)
        memories = [AgentMemory.model_validate_json(line) for line in lines if line]

        memories.sort(key=lambda m: m.tick, reverse=True)
        return memories[:limit]

    def _run_dir(self, run_id: UUID) -> Path:
        return self.base_path / str(run_id)
