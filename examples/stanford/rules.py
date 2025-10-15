"""Simulation rules for Stanford Valentine's Day party scenario."""

from miniverse.simulation_rules import SimulationRules
from miniverse.schemas import WorldState, AgentAction
from miniverse.environment import validate_grid_move


class StanfordRules(SimulationRules):
    """Deterministic physics for grid-based social simulation.

    Handles:
    - Time progression (tick → simulated time mapping)
    - Grid movement validation (collision detection, pathfinding)
    - Energy/social comfort updates based on activities
    - Optional resource tracking (party supplies, etc.)
    """

    def __init__(self, *, seconds_per_tick: int = 60):
        """Initialize Stanford rules.

        Args:
            seconds_per_tick: How many simulated seconds pass per tick.
                             60 = 1 minute per tick (default)
                             300 = 5 minutes per tick
        """
        self.seconds_per_tick = seconds_per_tick

    def get_tick_duration_seconds(self) -> int:
        """Return simulated seconds per tick for timestamp advancement."""
        return self.seconds_per_tick

    def apply_tick(self, state: WorldState, tick: int) -> WorldState:
        """Apply passive world evolution for one tick.

        Updates:
        - Agent energy/social_comfort based on activities
        - Time of day based on simulation time
        - Any environmental changes

        Does NOT handle agent actions - those are processed separately by orchestrator.
        """
        # Deep copy to avoid mutating input state
        new_state = state.model_copy(deep=True)

        # Update time of day based on timestamp (for prompts/perception)
        hour = new_state.timestamp.hour
        if 6 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"

        # Update environment time_of_day metric if it exists
        if "time_of_day" in new_state.environment.metrics:
            new_state.environment.metrics["time_of_day"].value = time_of_day

        # Apply passive energy/social comfort decay (agents get tired, socially drained)
        for agent in new_state.agents:
            # Small energy decay per tick (agents need rest)
            if "energy" in agent.attributes:
                current_energy = float(agent.attributes["energy"].value)
                # Decay 0.5% per tick (1 minute = 0.5% energy loss)
                agent.attributes["energy"].value = max(0.0, current_energy - 0.5)

            # Social comfort changes based on activity
            if "social_comfort" in agent.attributes:
                current_comfort = float(agent.attributes["social_comfort"].value)
                activity = agent.activity or ""

                # Social activities increase comfort for extroverts, decrease for introverts
                if "talking" in activity.lower() or "party" in activity.lower():
                    # Klaus (low initial comfort) drains, Isabella/Maria gain
                    if current_comfort < 60:
                        agent.attributes["social_comfort"].value = max(0.0, current_comfort - 1.0)
                    else:
                        agent.attributes["social_comfort"].value = min(100.0, current_comfort + 0.5)

        return new_state

    def validate_action(self, action: AgentAction, state: WorldState) -> bool:
        """Validate whether an action is physically possible.

        Checks:
        - Grid movement: path exists, no collisions, reasonable distance
        - Action feasibility based on agent state

        Returns True if action should be processed, False to reject it.
        """
        # Find agent attempting action
        agent = next((a for a in state.agents if a.agent_id == action.agent_id), None)
        if agent is None:
            return False  # unknown agent

        # Validate move actions if agent has grid position
        if action.action_type == "move" and agent.grid_position is not None:
            # Extract target position from parameters
            # Expected format: {"target_position": [row, col]} or {"target_position": (row, col)}
            if action.parameters and "target_position" in action.parameters:
                target_pos_raw = action.parameters["target_position"]

                # Normalize to list (our schema uses List[int] for OpenAI compatibility)
                if isinstance(target_pos_raw, (list, tuple)) and len(target_pos_raw) == 2:
                    target_pos = [int(target_pos_raw[0]), int(target_pos_raw[1])]
                else:
                    return False  # malformed target position

                # Validate move using grid helpers
                if state.environment_grid is not None:
                    # Allow moves up to 3 tiles per tick (reasonable walking distance)
                    return validate_grid_move(
                        state.environment_grid,
                        agent.grid_position,
                        target_pos,
                        max_distance=3
                    )

        # Default: allow action (non-movement actions don't need spatial validation)
        return True

    def format_resource_summary(self, state: WorldState) -> str:
        """Format resource metrics for console output."""
        if not state.resources.metrics:
            return ""

        parts = []
        for key, stat in state.resources.metrics.items():
            label = stat.label or key.replace("_", " ").title()
            value = stat.value
            unit = stat.unit or ""
            suffix = f" {unit}" if unit else ""
            parts.append(f"{label}={value}{suffix}")

        return ", ".join(parts)
