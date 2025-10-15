"""
MemoryStrategy interface for agent memory systems.

This module provides the abstract base class for implementing how agents
remember and recall past experiences. Based on Stanford Generative Agents
research on memory streams.

Key responsibilities:
- Store agent observations, actions, and reflections
- Retrieve relevant memories based on recency, importance, relevance
- Manage memory capacity (forgetting old/unimportant memories)
- Support different memory architectures

Design principle: Start simple (FIFO), enable sophisticated (weighted retrieval).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import UUID
from datetime import datetime

from miniverse.schemas import AgentMemory


class MemoryStrategy(ABC):
    """
    Abstract base class for agent memory systems.

    This interface allows different memory architectures:
    - SimpleMemoryStream: FIFO queue (recent memories only)
    - ImportanceWeightedMemory: Weight by recency + importance
    - RelevanceMemory: Semantic search for relevant memories
    - ReflectionMemory: Periodic higher-level summaries

    Based on Stanford Generative Agents (2023) memory architecture:
    - Memory Stream: Sequential record of observations
    - Retrieval: Recency + importance + relevance scoring
    - Reflection: Periodic summarization of memories
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the memory backend.

        Called once before simulation starts. Used to set up connections,
        allocate resources, load data, etc.

        Raises:
            Exception: If initialization fails
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close the memory backend.

        Called once after simulation completes. Used to close connections,
        flush buffers, cleanup resources, etc.

        Raises:
            Exception: If cleanup fails
        """
        pass

    @abstractmethod
    async def add_memory(
        self,
        run_id: UUID,
        agent_id: str,
        tick: int,
        memory_type: str,
        content: str,
        importance: int = 5,
    ) -> AgentMemory:
        """
        Add a new memory for an agent.

        Args:
            run_id: Simulation run identifier
            agent_id: Agent who owns this memory
            tick: Tick when memory was created
            memory_type: Type (observation, action, communication, reflection)
            content: Memory content (natural language)
            importance: Importance score 1-10 (5 = neutral)

        Returns:
            The created AgentMemory object

        Raises:
            Exception: If memory cannot be stored
        """
        pass

    @abstractmethod
    async def get_recent_memories(
        self, run_id: UUID, agent_id: str, limit: int = 10
    ) -> List[str]:
        """
        Retrieve recent memories for an agent as strings.

        Used to build agent perception (recent_observations field).
        Returns natural language strings, not full AgentMemory objects.

        Args:
            run_id: Simulation run identifier
            agent_id: Agent identifier
            limit: Maximum number of memories to return

        Returns:
            List of memory content strings (most recent first)

        Raises:
            Exception: If retrieval fails
        """
        pass

    @abstractmethod
    async def get_relevant_memories(
        self,
        run_id: UUID,
        agent_id: str,
        query: str,
        limit: int = 5,
    ) -> List[str]:
        """
        Retrieve memories relevant to a query.

        Used for context-aware memory retrieval. Advanced implementations
        can use semantic similarity, keyword matching, etc.

        Args:
            run_id: Simulation run identifier
            agent_id: Agent identifier
            query: Query string to find relevant memories
            limit: Maximum number of memories to return

        Returns:
            List of relevant memory content strings

        Raises:
            Exception: If retrieval fails
        """
        pass

    @abstractmethod
    async def clear_agent_memories(self, run_id: UUID, agent_id: str) -> None:
        """
        Clear all memories for an agent.

        Used for testing or resetting agent state.

        Args:
            run_id: Simulation run identifier
            agent_id: Agent identifier

        Raises:
            Exception: If clearing fails
        """
        pass


class SimpleMemoryStream(MemoryStrategy):
    """
    Simple FIFO memory stream implementation.

    Stores all memories and returns the N most recent when queried.
    No importance weighting, no semantic search, no reflection.

    Good for:
    - Initial prototyping
    - Short simulations (<100 ticks)
    - Testing basic agent behavior

    Limitations:
    - No importance-based retrieval
    - No semantic relevance
    - Memory grows unbounded (should add capacity limit)
    - Delegates storage to persistence layer
    """

    def __init__(self, persistence):
        """
        Initialize memory stream with persistence backend.

        Args:
            persistence: PersistenceStrategy instance for storing memories
        """
        self.persistence = persistence

    async def initialize(self) -> None:
        """
        Initialize memory backend.

        For SimpleMemoryStream, this is a no-op since we delegate
        to the persistence layer which handles its own initialization.
        """
        pass

    async def close(self) -> None:
        """
        Close memory backend.

        For SimpleMemoryStream, this is a no-op since we delegate
        to the persistence layer which handles its own cleanup.
        """
        pass

    async def add_memory(
        self,
        run_id: UUID,
        agent_id: str,
        tick: int,
        memory_type: str,
        content: str,
        importance: int = 5,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding_key: Optional[str] = None,
        branch_id: Optional[str] = None,
    ) -> AgentMemory:
        """
        Add a memory to the stream.

        Stores via persistence layer. Importance is recorded but not
        used for retrieval in this simple implementation.

        Args:
            run_id: Simulation run identifier
            agent_id: Agent who owns this memory
            tick: Tick when memory was created
            memory_type: Type of memory
            content: Memory content
            importance: Importance score (recorded but not used)
            tags: Optional labels for future retrieval engines
            metadata: Arbitrary key/value payload for retrievers
            embedding_key: Pointer into external embedding store (optional)
            branch_id: Timeline identifier for branching simulations (optional)

        Returns:
            The created AgentMemory object
        """
        import uuid

        memory = AgentMemory(
            id=uuid.uuid4(),
            run_id=run_id,
            agent_id=agent_id,
            tick=tick,
            memory_type=memory_type,
            content=content,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
            embedding_key=embedding_key,
            branch_id=branch_id,
            created_at=datetime.now(),
        )

        await self.persistence.save_memory(run_id, memory)
        return memory

    async def get_recent_memories(
        self, run_id: UUID, agent_id: str, limit: int = 10
    ) -> List[str]:
        """
        Get N most recent memories as strings.

        Simple FIFO retrieval: just get the most recent N memories
        by tick number, regardless of importance or relevance.

        Args:
            run_id: Simulation run identifier
            agent_id: Agent identifier
            limit: Maximum memories to return

        Returns:
            List of memory content strings (most recent first)
        """
        memories = await self.persistence.get_recent_memories(run_id, agent_id, limit)
        return [m.content for m in memories]

    async def get_relevant_memories(
        self,
        run_id: UUID,
        agent_id: str,
        query: str,
        limit: int = 5,
    ) -> List[str]:
        """
        Get relevant memories (simple implementation: just recent).

        This simple implementation doesn't do semantic search,
        just returns recent memories. Advanced implementations
        would use embeddings, keyword matching, etc.

        Args:
            run_id: Simulation run identifier
            agent_id: Agent identifier
            query: Query string (unused in simple implementation)
            limit: Maximum memories to return

        Returns:
            List of recent memory content strings
        """
        query = query.lower().strip()
        if not query:
            return await self.get_recent_memories(run_id, agent_id, limit)

        terms = [term for term in query.replace(",", " ").split() if term]
        if not terms:
            return await self.get_recent_memories(run_id, agent_id, limit)

        # Fetch a larger window to score
        candidate_memories = await self.persistence.get_recent_memories(run_id, agent_id, limit * 5)
        scores: List[tuple[float, str]] = []

        for mem in candidate_memories:
            text = mem.content.lower()
            tag_text = " ".join(mem.tags).lower()
            score = 0.0
            for term in terms:
                if term in text:
                    score += 2.0
                if term in tag_text:
                    score += 1.0
            if score > 0.0:
                score += mem.importance * 0.1
                scores.append((score, mem.content))

        if not scores:
            return await self.get_recent_memories(run_id, agent_id, limit)

        scores.sort(key=lambda item: item[0], reverse=True)
        return [content for _, content in scores[:limit]]

    async def clear_agent_memories(self, run_id: UUID, agent_id: str) -> None:
        """
        Clear all memories for an agent.

        Note: This requires the persistence layer to support clearing.
        For InMemoryPersistence, this is straightforward. For database
        persistence, this would need to delete records.

        Args:
            run_id: Simulation run identifier
            agent_id: Agent identifier
        """
        # This would need support in PersistenceStrategy interface
        # For now, this is a placeholder - would need to add
        # clear_memories() method to PersistenceStrategy
        pass


class ImportanceWeightedMemory(MemoryStrategy):
    """
    Memory retrieval weighted by recency + importance.

    Based on Stanford Generative Agents paper:
    - Score = w_recency * recency(m) + w_importance * importance(m)
    - Return top-k memories by score

    To be implemented in future phase (Phase 7+).
    """

    def __init__(self, persistence, recency_weight: float = 0.7, importance_weight: float = 0.3):
        """
        Initialize importance-weighted memory.

        Args:
            persistence: PersistenceStrategy instance
            recency_weight: Weight for recency (0-1)
            importance_weight: Weight for importance (0-1)
        """
        self.persistence = persistence
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight

    async def add_memory(
        self,
        run_id: UUID,
        agent_id: str,
        tick: int,
        memory_type: str,
        content: str,
        importance: int = 5,
    ) -> AgentMemory:
        """Add memory (implementation pending)."""
        raise NotImplementedError("ImportanceWeightedMemory not yet implemented")

    async def get_recent_memories(
        self, run_id: UUID, agent_id: str, limit: int = 10
    ) -> List[str]:
        """Get memories (implementation pending)."""
        raise NotImplementedError("ImportanceWeightedMemory not yet implemented")

    async def get_relevant_memories(
        self,
        run_id: UUID,
        agent_id: str,
        query: str,
        limit: int = 5,
    ) -> List[str]:
        """Get relevant memories (implementation pending)."""
        raise NotImplementedError("ImportanceWeightedMemory not yet implemented")

    async def clear_agent_memories(self, run_id: UUID, agent_id: str) -> None:
        """Clear memories (implementation pending)."""
        raise NotImplementedError("ImportanceWeightedMemory not yet implemented")
