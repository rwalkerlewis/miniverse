"""
Pydantic schemas for Varela simulation system.

All data structures used in the simulation are defined here.
"""

from pydantic import BaseModel, Field

from miniverse.environment import (
    EnvironmentGraphState,
    EnvironmentGridState,
)
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from uuid import UUID


# ============================================================================
# World State Schemas
# ============================================================================


StatValue = Union[int, float, str, bool]


class Stat(BaseModel):
    """Generic metric used across environments, resources, and agents."""

    value: StatValue = Field(..., description="Current value of the metric")
    unit: Optional[str] = Field(None, description="Optional unit label (%, kWh, °C, etc.)")
    label: Optional[str] = Field(None, description="Human-friendly name for UI or prompts")
    description: Optional[str] = Field(None, description="Optional explanation of the metric")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Free-form scenario metadata")


class MetricsBlock(BaseModel):
    """Container for a set of metrics keyed by arbitrary identifiers."""

    metrics: Dict[str, Stat] = Field(default_factory=dict, description="Keyed metrics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Scenario-defined metadata")

    def get_metric(
        self,
        key: str,
        *,
        default: Optional[StatValue] = None,
        unit: Optional[str] = None,
        label: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Stat:
        """Return the Stat for the requested metric, optionally creating it."""

        if key not in self.metrics:
            if default is None:
                raise KeyError(f"Metric '{key}' not found")
            self.metrics[key] = Stat(
                value=default,
                unit=unit,
                label=label or key.replace("_", " ").title(),
                description=description,
            )
        return self.metrics[key]


class EnvironmentState(MetricsBlock):
    """Environmental conditions exposed by the simulation world."""


class ResourceState(MetricsBlock):
    """Shared resources for the scenario (inventory, budgets, etc.)."""


class AgentStatus(BaseModel):
    """Current status of an agent in the simulation."""

    agent_id: str = Field(..., description="Unique agent identifier")
    display_name: Optional[str] = Field(None, description="Optional override for printing")
    role: Optional[str] = Field(None, description="Short summary of agent role")
    location: Optional[str] = Field(None, description="Current location or zone")
    activity: Optional[str] = Field(None, description="Current activity or task")
    attributes: Dict[str, Stat] = Field(
        default_factory=dict, description="Per-agent metrics (health, stress, quota, etc.)"
    )
    tags: List[str] = Field(default_factory=list, description="Extra labels (shift, faction, etc.)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Scenario-defined metadata for this agent"
    )

    def get_attribute(
        self,
        key: str,
        *,
        default: Optional[StatValue] = None,
        unit: Optional[str] = None,
        label: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Stat:
        """Return a Stat for the requested attribute, optionally creating it."""

        if key not in self.attributes:
            if default is None:
                raise KeyError(f"Attribute '{key}' not found for agent {self.agent_id}")
            self.attributes[key] = Stat(
                value=default,
                unit=unit,
                label=label or key.replace("_", " ").title(),
                description=description,
            )
        return self.attributes[key]

    # Convenience accessors for common metrics ---------------------------------

    @property
    def health(self) -> float:
        return float(self.get_attribute("health", default=0, unit="%", label="Health").value)

    @health.setter
    def health(self, value: float) -> None:
        self.get_attribute("health", default=value, unit="%", label="Health").value = value

    @property
    def stress(self) -> float:
        return float(self.get_attribute("stress", default=0, unit="%", label="Stress").value)

    @stress.setter
    def stress(self, value: float) -> None:
        self.get_attribute("stress", default=value, unit="%", label="Stress").value = value

    @property
    def energy(self) -> float:
        return float(self.get_attribute("energy", default=0, unit="%", label="Energy").value)

    @energy.setter
    def energy(self, value: float) -> None:
        self.get_attribute("energy", default=value, unit="%", label="Energy").value = value

    @property
    def current_activity(self) -> Optional[str]:  # pragma: no cover - simple alias
        return self.activity

    @current_activity.setter
    def current_activity(self, value: Optional[str]) -> None:  # pragma: no cover
        self.activity = value


class WorldEvent(BaseModel):
    """An event emitted by the simulation world."""

    event_id: str = Field(..., description="Unique event identifier")
    tick: int = Field(..., ge=0, description="Tick when event occurred")
    category: str = Field(..., description="Domain-specific category (outage, morale, etc.)")
    description: str = Field(..., description="Human-readable description")
    severity: Optional[int] = Field(
        None,
        description="Optional severity ranking (1-10 or scenario-defined scale)",
    )
    affected_agents: List[str] = Field(
        default_factory=list, description="Agent IDs directly affected"
    )
    metrics: Dict[str, Stat] = Field(
        default_factory=dict, description="Metrics captured or impacted by this event"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional contextual data")


class WorldState(BaseModel):
    """Complete state of the simulation world at a specific tick."""

    tick: int = Field(..., ge=0, description="Current simulation tick")
    timestamp: datetime = Field(..., description="Simulated timestamp")
    environment: EnvironmentState = Field(..., description="Environmental conditions")
    resources: ResourceState = Field(..., description="Shared resource pools")
    agents: List[AgentStatus] = Field(..., description="Status of all agents")
    recent_events: List[WorldEvent] = Field(
        default_factory=list, description="Recent events (typically last few ticks)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Scenario-defined world metadata"
    )
    environment_graph: Optional[EnvironmentGraphState] = Field(
        None,
        description="Tier 1 logical environment graph (rooms/teams/etc.)",
    )
    environment_grid: Optional[EnvironmentGridState] = Field(
        None,
        description="Tier 2 spatial grid description",
    )


# ============================================================================
# Agent Schemas
# ============================================================================


class AgentProfile(BaseModel):
    """Complete profile of an agent (colonist)."""

    agent_id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent's full name")
    age: int = Field(..., ge=18, le=70, description="Age in years")
    background: str = Field(
        ..., description="Detailed backstory in interview style"
    )
    role: str = Field(
        ...,
        description="Role (medical_officer, engineer, commander_botanist, geologist, etc.)",
    )
    personality: str = Field(
        ..., description="Personality type (agreeable, neurotic, reactive, social)"
    )
    skills: Dict[str, str] = Field(
        ...,
        description="Skills and proficiency levels (expert, advanced, intermediate, basic)",
    )
    goals: List[str] = Field(..., description="Personal and role-based goals")
    relationships: Dict[str, str] = Field(
        ..., description="Relationships with other agents {agent_id: description}"
    )


class AgentAction(BaseModel):
    """An action taken by an agent in a single tick."""

    agent_id: str = Field(..., description="Agent performing the action")
    tick: int = Field(..., ge=0, description="Tick when action was taken")
    action_type: str = Field(
        ...,
        description="Type of action (work, rest, communicate, repair, monitor, investigate)",
    )
    target: Optional[str] = Field(
        None, description="Target of action (system, agent, location, equipment)"
    )
    parameters: Optional[Dict[str, Union[str, int, float, bool]]] = Field(
        None, description="Action-specific parameters"
    )
    reasoning: str = Field(..., description="Why this action was chosen")
    communication: Optional[Dict[str, str]] = Field(
        None, description="Communication content if action includes messaging"
    )


class AgentPerception(BaseModel):
    """What an agent can perceive in a single tick (partial observability)."""

    tick: int = Field(..., ge=0, description="Current tick")
    location: Optional[str] = Field(None, description="Agent's current location")
    personal_attributes: Dict[str, Stat] = Field(
        default_factory=dict, description="Self-reported attributes (health, morale, etc.)"
    )
    visible_resources: Dict[str, Stat] = Field(
        default_factory=dict, description="Shared resources visible to this agent"
    )
    environment_snapshot: Dict[str, Stat] = Field(
        default_factory=dict, description="Relevant environment metrics"
    )
    system_alerts: List[str] = Field(
        default_factory=list, description="Current system warnings and alerts"
    )
    messages: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Messages sent to this agent {from: agent_id, message: str}",
    )
    recent_observations: List[str] = Field(
        default_factory=list,
        description="Recent memories (last 10 observations from memory)",
    )


# ============================================================================
# Database Schemas
# ============================================================================


class SimulationRun(BaseModel):
    """Metadata for a simulation run."""

    id: UUID = Field(..., description="Unique run identifier")
    start_time: datetime = Field(..., description="When simulation started")
    end_time: Optional[datetime] = Field(None, description="When simulation ended")
    num_ticks: int = Field(..., ge=0, description="Number of ticks to run")
    num_agents: int = Field(..., ge=1, description="Number of agents in simulation")
    status: str = Field(
        ..., description="Run status (running, completed, failed)"
    )
    config: Dict = Field(..., description="Simulation configuration")
    created_at: datetime = Field(..., description="When record was created")


class AgentMemory(BaseModel):
    """A memory stored for an agent."""

    id: UUID = Field(..., description="Unique memory identifier")
    run_id: UUID = Field(..., description="Simulation run this memory belongs to")
    agent_id: str = Field(..., description="Agent who owns this memory")
    tick: int = Field(..., ge=0, description="Tick when memory was created")
    memory_type: str = Field(
        ...,
        description="Type of memory (observation, action, communication, reflection)",
    )
    content: str = Field(..., description="Memory content (natural language)")
    importance: int = Field(
        ..., ge=1, le=10, description="Importance score (1-10)"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Optional labels for retrieval (topics, entities, etc.)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary structured metadata for custom retrievers",
    )
    embedding_key: Optional[str] = Field(
        None,
        description="Reference to an external embedding/vector store entry",
    )
    branch_id: Optional[str] = Field(
        None,
        description="Timeline identifier for branching/loom scenarios",
    )
    created_at: datetime = Field(..., description="When record was created")


class SimulationError(BaseModel):
    """An error that occurred during simulation."""

    id: UUID = Field(..., description="Unique error identifier")
    run_id: UUID = Field(..., description="Simulation run where error occurred")
    tick: int = Field(..., ge=0, description="Tick when error occurred")
    error_type: str = Field(..., description="Type of error")
    error_message: str = Field(..., description="Error message")
    stack_trace: Optional[str] = Field(None, description="Full stack trace if available")
    created_at: datetime = Field(..., description="When record was created")
