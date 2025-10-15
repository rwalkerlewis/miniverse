"""
Scenario loading and management.

Scenarios define the initial state of a simulation including agents,
environment, resources, and optional starting events.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

from .config import Config
from .schemas import (
    WorldState,
    AgentProfile,
    EnvironmentState,
    ResourceState,
    AgentStatus,
    WorldEvent,
    Stat,
)
from .environment import (
    EnvironmentGraphState,
    EnvironmentGridState,
    LocationNodeState,
    GridTileState,
)


class ScenarioLoader:
    """Load and validate simulation scenarios from JSON files."""

    def __init__(self, scenarios_dir: Optional[Path] = None):
        """Initialize scenario loader.

        Args:
            scenarios_dir: Directory containing scenario files.
                          Defaults to project root /scenarios
        """
        self.scenarios_dir = scenarios_dir or (Config.PROJECT_ROOT / "examples" / "scenarios")

    def load(self, scenario_name: str) -> Tuple[WorldState, List[AgentProfile]]:
        """Load a scenario by name.

        Args:
            scenario_name: Name of scenario (without .json extension)

        Returns:
            Tuple of (initial WorldState, list of AgentProfiles)

        Raises:
            FileNotFoundError: If scenario file doesn't exist
            ValueError: If scenario JSON is invalid
        """
        scenario_path = self.scenarios_dir / f"{scenario_name}.json"

        if not scenario_path.exists():
            raise FileNotFoundError(
                f"Scenario '{scenario_name}' not found at {scenario_path}"
            )

        data = json.loads(scenario_path.read_text())

        # Validate required fields
        self._validate_scenario(data)

        # Parse agents
        agents: List[AgentProfile] = []
        agent_statuses: List[AgentStatus] = []

        for agent_entry in data["agents"]:
            profile_data = agent_entry["profile"]
            status_data = agent_entry["status"]

            profile = AgentProfile(**profile_data)
            agents.append(profile)

            status = self._parse_agent_status(status_data, profile.agent_id)
            agent_statuses.append(status)

        # Parse initial world state
        world_state = self._build_world_state(data, agent_statuses)

        return world_state, agents

    def _validate_scenario(self, data: Dict) -> None:
        """Validate scenario data has required fields.

        Args:
            data: Scenario JSON data

        Raises:
            ValueError: If required fields are missing
        """
        required = ["name", "description", "agents", "environment", "resources"]
        missing = [field for field in required if field not in data]

        if missing:
            raise ValueError(f"Scenario missing required fields: {missing}")

        if not data["agents"]:
            raise ValueError("Scenario must have at least one agent")

        for agent in data["agents"]:
            if "profile" not in agent or "status" not in agent:
                raise ValueError(
                    "Each agent entry must include 'profile' and 'status' blocks"
                )

    def _build_world_state(self, data: Dict, agent_statuses: List[AgentStatus]) -> WorldState:
        """Build initial WorldState from scenario data.

        Args:
            data: Scenario JSON data

        Returns:
            Initial WorldState
        """
        # Parse environment
        environment = self._parse_environment(data.get("environment", {}))

        # Parse resources
        resources = self._parse_resources(data.get("resources", {}))

        environment_graph = None
        if "environment_graph" in data:
            environment_graph = self._parse_environment_graph(data["environment_graph"])

        environment_grid = None
        if "environment_grid" in data:
            environment_grid = self._parse_environment_grid(data["environment_grid"])

        # Build agent statuses from agent data
        # Parse initial events if provided
        initial_events = []
        if "initial_events" in data:
            for event_data in data["initial_events"]:
                event = self._parse_event(event_data)
                initial_events.append(event)

        # Get starting timestamp
        timestamp_str = data.get("initial_timestamp", "2157-03-15T08:00:00")
        timestamp = datetime.fromisoformat(timestamp_str)

        return WorldState(
            tick=0,
            timestamp=timestamp,
            environment=environment,
            resources=resources,
            agents=agent_statuses,
            recent_events=initial_events,
            metadata=data.get("metadata", {}),
            environment_graph=environment_graph,
            environment_grid=environment_grid,
        )

    def _parse_metrics(self, raw: Dict[str, Any]) -> Dict[str, Stat]:
        """Convert raw metric dict into Stat objects."""

        metrics: Dict[str, Stat] = {}
        for key, value in raw.items():
            if isinstance(value, dict) and "value" in value:
                metrics[key] = Stat(**value)
            else:
                metrics[key] = Stat(value=value)
        return metrics

    def _parse_environment(self, data: Dict[str, Any]) -> EnvironmentState:
        metrics = self._parse_metrics(data.get("metrics", {}))
        return EnvironmentState(
            metrics=metrics,
            metadata=data.get("metadata", {}),
        )

    def _parse_resources(self, data: Dict[str, Any]) -> ResourceState:
        metrics = self._parse_metrics(data.get("metrics", {}))
        return ResourceState(
            metrics=metrics,
            metadata=data.get("metadata", {}),
        )

    def _parse_agent_status(
        self, data: Dict[str, Any], fallback_agent_id: str
    ) -> AgentStatus:
        attributes = self._parse_metrics(data.get("attributes", {}))

        return AgentStatus(
            agent_id=data.get("agent_id", fallback_agent_id),
            display_name=data.get("display_name"),
            role=data.get("role"),
            location=data.get("location"),
            activity=data.get("activity"),
            attributes=attributes,
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

    def _parse_event(self, data: Dict[str, Any]) -> WorldEvent:
        metrics = self._parse_metrics(data.get("metrics", {}))
        return WorldEvent(
            event_id=data["event_id"],
            tick=data["tick"],
            category=data.get("category", "event"),
            description=data["description"],
            severity=data.get("severity"),
            affected_agents=data.get("affected_agents", []),
            metrics=metrics,
            metadata=data.get("metadata", {}),
        )

    def _parse_environment_graph(self, data: Dict[str, Any]) -> EnvironmentGraphState:
        nodes: Dict[str, LocationNodeState] = {}
        raw_nodes = data.get("nodes", {})
        for key, value in raw_nodes.items():
            if isinstance(value, dict):
                nodes[key] = LocationNodeState(
                    name=value.get("name", key),
                    capacity=value.get("capacity"),
                    metadata=value.get("metadata", {}),
                )
            else:
                nodes[key] = LocationNodeState(name=key)

        adjacency: Dict[str, List[str]] = {}
        raw_adj = data.get("adjacency", {})
        for key, neighbors in raw_adj.items():
            if isinstance(neighbors, (list, tuple)):
                adjacency[key] = list(neighbors)
            else:
                adjacency[key] = []

        return EnvironmentGraphState(nodes=nodes, adjacency=adjacency)

    def _parse_environment_grid(self, data: Dict[str, Any]) -> EnvironmentGridState:
        width = int(data.get("width", 0))
        height = int(data.get("height", 0))
        tiles_data = data.get("tiles", {})
        tiles: Dict[Tuple[int, int], GridTileState] = {}

        if isinstance(tiles_data, dict):
            for key, value in tiles_data.items():
                if isinstance(key, str) and "," in key:
                    row_str, col_str = key.split(",", 1)
                    row, col = int(row_str), int(col_str)
                else:
                    continue
                tile_kwargs = value if isinstance(value, dict) else {}
                tiles[(row, col)] = GridTileState(**tile_kwargs)
        elif isinstance(tiles_data, list):
            for item in tiles_data:
                if not isinstance(item, dict):
                    continue
                if "coordinate" in item and isinstance(item["coordinate"], (list, tuple)):
                    row, col = item["coordinate"]
                else:
                    row = item.get("row")
                    col = item.get("col")
                if row is None or col is None:
                    continue
                tile_kwargs = {
                    "world": item.get("world"),
                    "sector": item.get("sector"),
                    "arena": item.get("arena"),
                    "game_object": item.get("game_object"),
                    "collision": item.get("collision", False),
                    "metadata": item.get("metadata", {}),
                }
                tiles[(int(row), int(col))] = GridTileState(**tile_kwargs)

        return EnvironmentGridState(width=width, height=height, tiles=tiles)

    def list_scenarios(self) -> List[str]:
        """List all available scenario files.

        Returns:
            List of scenario names (without .json extension)
        """
        if not self.scenarios_dir.exists():
            return []

        return [
            f.stem for f in self.scenarios_dir.glob("*.json")
            if not f.name.startswith("_")
        ]

    def get_scenario_info(self, scenario_name: str) -> Dict[str, str]:
        """Get scenario metadata without loading full scenario.

        Args:
            scenario_name: Name of scenario

        Returns:
            Dict with name, description, num_agents, etc.
        """
        scenario_path = self.scenarios_dir / f"{scenario_name}.json"
        data = json.loads(scenario_path.read_text())

        return {
            "name": data.get("name", scenario_name),
            "description": data.get("description", "No description"),
            "num_agents": len(data.get("agents", [])),
            "recommended_ticks": data.get("recommended_ticks", 50),
        }


def load_scenario(scenario_name: str) -> Tuple[WorldState, List[AgentProfile]]:
    """Convenience function to load a scenario.

    Args:
        scenario_name: Name of scenario to load

    Returns:
        Tuple of (initial WorldState, list of AgentProfiles)
    """
    loader = ScenarioLoader()
    return loader.load(scenario_name)
