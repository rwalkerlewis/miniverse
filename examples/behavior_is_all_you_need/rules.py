"""Social deception game rules (Werewolf/Mafia style).

This module implements the game mechanics for a social deduction game where
agents must identify hidden werewolves through discussion, voting, and deduction.

Game flow:
1. Day Phase: All players discuss, accuse, and vote to eliminate someone
2. Night Phase: Werewolves choose a victim; special roles take actions
3. Repeat until: Werewolves eliminated OR werewolves equal/outnumber villagers

Roles:
- Villagers: Find and eliminate werewolves
- Werewolves: Eliminate villagers without being caught
- Seer: Can investigate one player per night
- Doctor: Can protect one player per night (optional)
"""

from typing import Dict, List, Optional, Set
from miniverse import SimulationRules, WorldState, AgentAction, AgentStatus
from emotion import EmotionalState, EmotionalTriggers
from needs import NeedsHierarchy, NeedFulfillmentEvents


class GamePhase:
    """Game phase constants."""
    DAY = "day"
    NIGHT = "night"
    VOTING = "voting"
    GAME_OVER = "game_over"


class Role:
    """Player role constants."""
    VILLAGER = "villager"
    WEREWOLF = "werewolf"
    SEER = "seer"
    DOCTOR = "doctor"


class DeceptionGameRules(SimulationRules):
    """Simulation rules for social deception game.

    Manages game state, phases, voting, eliminations, and win conditions.
    Triggers emotional and need responses based on game events.
    """

    def __init__(self, ticks_per_phase: int = 3):
        """Initialize game rules.

        Args:
            ticks_per_phase: How many ticks per day/night phase
        """
        self.ticks_per_phase = ticks_per_phase
        self.current_phase = GamePhase.DAY
        self.phase_tick_counter = 0
        self.votes: Dict[str, str] = {}  # voter_id -> target_id
        self.night_actions: Dict[str, Dict] = {}  # agent_id -> {action_type, target}
        self.eliminated_agents: Set[str] = set()
        self.revealed_info: Dict[str, List[str]] = {}  # agent_id -> list of revelations
        self.game_over = False
        self.winner: Optional[str] = None

    def apply_tick(self, state: WorldState, tick: int) -> WorldState:
        """Apply game rules for the current tick.

        Manages phase transitions, vote counting, eliminations, and win condition checks.
        """
        updated = state.model_copy(deep=True)
        updated.tick = tick

        # Check if game is over
        if self.game_over:
            self.current_phase = GamePhase.GAME_OVER
            return updated

        # Increment phase counter
        self.phase_tick_counter += 1

        # Phase transition logic
        if self.phase_tick_counter >= self.ticks_per_phase:
            self._transition_phase(updated)
            self.phase_tick_counter = 0

        # Update phase in world state
        phase_stat = updated.resources.get_metric("game_phase", default=self.current_phase, label="Current Phase")
        phase_stat.value = self.current_phase

        # Update alive counts
        alive_agents = [a for a in updated.agents if a.agent_id not in self.eliminated_agents]
        werewolves_alive = sum(1 for a in alive_agents if self._get_role(a) == Role.WEREWOLF)
        villagers_alive = len(alive_agents) - werewolves_alive

        werewolf_stat = updated.resources.get_metric("werewolves_alive", default=werewolves_alive, label="Werewolves Alive")
        werewolf_stat.value = werewolves_alive

        villager_stat = updated.resources.get_metric("villagers_alive", default=villagers_alive, label="Villagers Alive")
        villager_stat.value = villagers_alive

        # Check win conditions
        if werewolves_alive == 0:
            self.game_over = True
            self.winner = "villagers"
            self._trigger_game_end_emotions(updated, "villagers")
        elif werewolves_alive >= villagers_alive:
            self.game_over = True
            self.winner = "werewolves"
            self._trigger_game_end_emotions(updated, "werewolves")

        # Decay emotions and needs for all alive agents
        for agent in updated.agents:
            if agent.agent_id not in self.eliminated_agents:
                self._decay_agent_state(agent)

        return updated

    def _transition_phase(self, state: WorldState):
        """Handle phase transitions and resolve phase-end actions."""
        if self.current_phase == GamePhase.DAY:
            # Transition to voting
            self.current_phase = GamePhase.VOTING
            self._trigger_phase_emotions(state, "voting_start")

        elif self.current_phase == GamePhase.VOTING:
            # Count votes and eliminate
            self._resolve_day_vote(state)
            self.current_phase = GamePhase.NIGHT
            self._trigger_phase_emotions(state, "night_start")

        elif self.current_phase == GamePhase.NIGHT:
            # Resolve night actions
            self._resolve_night_actions(state)
            self.current_phase = GamePhase.DAY
            self.votes = {}  # Reset votes
            self.night_actions = {}  # Reset night actions
            self._trigger_phase_emotions(state, "day_start")

    def _resolve_day_vote(self, state: WorldState):
        """Count votes and eliminate the player with most votes."""
        if not self.votes:
            return

        # Count votes
        vote_counts: Dict[str, int] = {}
        for target in self.votes.values():
            vote_counts[target] = vote_counts.get(target, 0) + 1

        # Find player with most votes
        max_votes = max(vote_counts.values())
        candidates = [target for target, count in vote_counts.items() if count == max_votes]

        if len(candidates) == 1:
            eliminated = candidates[0]
        else:
            # Tie - pick first alphabetically (could be random instead)
            eliminated = sorted(candidates)[0]

        # Eliminate player
        self.eliminated_agents.add(eliminated)
        self._trigger_elimination_emotions(state, eliminated, "day_vote")

        # Add event to world state
        role = self._get_role_by_id(state, eliminated)
        event = f"{eliminated} was eliminated by vote and revealed to be a {role}!"
        state.recent_events.append(event)

    def _resolve_night_actions(self, state: WorldState):
        """Resolve night phase actions (werewolf kill, seer investigate, doctor protect)."""
        # Werewolf actions
        werewolf_votes: Dict[str, int] = {}  # target -> count
        for agent_id, action in self.night_actions.items():
            agent = self._get_agent_by_id(state, agent_id)
            if agent and self._get_role(agent) == Role.WEREWOLF:
                target = action.get("target")
                if target:
                    werewolf_votes[target] = werewolf_votes.get(target, 0) + 1

        # Select werewolf victim (most votes)
        victim = None
        if werewolf_votes:
            max_votes = max(werewolf_votes.values())
            candidates = [t for t, c in werewolf_votes.items() if c == max_votes]
            victim = sorted(candidates)[0]  # Tie-break alphabetically

        # Doctor protection
        protected = None
        for agent_id, action in self.night_actions.items():
            agent = self._get_agent_by_id(state, agent_id)
            if agent and self._get_role(agent) == Role.DOCTOR:
                protected = action.get("target")
                break

        # Apply kill if not protected
        if victim and victim != protected:
            self.eliminated_agents.add(victim)
            self._trigger_elimination_emotions(state, victim, "werewolf_kill")
            state.recent_events.append(f"{victim} was killed by werewolves during the night!")
        elif victim and victim == protected:
            state.recent_events.append(f"Someone was attacked but saved by the doctor!")
            self._trigger_event_emotion(state, victim, "protected", EmotionalTriggers.CLEARED_OF_SUSPICION)
            self._trigger_event_need(state, victim, "protected", NeedFulfillmentEvents.PROTECTED_BY_ROLE)

        # Seer investigation
        for agent_id, action in self.night_actions.items():
            agent = self._get_agent_by_id(state, agent_id)
            if agent and self._get_role(agent) == Role.SEER:
                target = action.get("target")
                if target:
                    target_agent = self._get_agent_by_id(state, target)
                    if target_agent:
                        role = self._get_role(target_agent)
                        alignment = "werewolf" if role == Role.WEREWOLF else "not a werewolf"
                        revelation = f"{target} is {alignment}"
                        if agent_id not in self.revealed_info:
                            self.revealed_info[agent_id] = []
                        self.revealed_info[agent_id].append(revelation)
                        # Trigger esteem need (successful investigation)
                        self._trigger_event_need(state, agent_id, "successful_deduction",
                                                NeedFulfillmentEvents.SUCCESSFUL_DEDUCTION)

    def _trigger_elimination_emotions(self, state: WorldState, eliminated_id: str, elimination_type: str):
        """Trigger emotional responses to elimination."""
        # Eliminated agent: sadness + defeat
        self._trigger_event_emotion(state, eliminated_id, "defeat", EmotionalTriggers.DEFEAT)
        self._trigger_event_need(state, eliminated_id, "isolated", NeedFulfillmentEvents.ISOLATED)

        # Other agents react based on role alignment
        eliminated_agent = self._get_agent_by_id(state, eliminated_id)
        if not eliminated_agent:
            return

        eliminated_role = self._get_role(eliminated_agent)

        for agent in state.agents:
            if agent.agent_id == eliminated_id or agent.agent_id in self.eliminated_agents:
                continue

            agent_role = self._get_role(agent)

            # Werewolves eliminated - villagers happy
            if eliminated_role == Role.WEREWOLF and agent_role != Role.WEREWOLF:
                self._trigger_event_emotion(state, agent.agent_id, "ally_confirmed", EmotionalTriggers.ALLY_CONFIRMED)
                self._trigger_event_need(state, agent.agent_id, "helped_win", NeedFulfillmentEvents.HELPED_TEAM_WIN)

            # Villagers eliminated - villagers sad, werewolves happy
            elif eliminated_role != Role.WEREWOLF:
                if agent_role == Role.WEREWOLF:
                    self._trigger_event_emotion(state, agent.agent_id, "successful_deception",
                                              EmotionalTriggers.SUCCESSFUL_DECEPTION)
                else:
                    self._trigger_event_emotion(state, agent.agent_id, "ally_died", EmotionalTriggers.ALLY_DIED)
                    self._trigger_event_need(state, agent.agent_id, "ally_died", NeedFulfillmentEvents.ALLY_DIED)

    def _trigger_phase_emotions(self, state: WorldState, phase_event: str):
        """Trigger emotions for phase transitions."""
        trigger_map = {
            "day_start": EmotionalTriggers.DAY_PHASE_START,
            "night_start": EmotionalTriggers.NIGHT_PHASE_START,
            "voting_start": EmotionalTriggers.SUSPICION_RAISED,
        }

        trigger = trigger_map.get(phase_event)
        if not trigger:
            return

        for agent in state.agents:
            if agent.agent_id not in self.eliminated_agents:
                self._trigger_event_emotion(state, agent.agent_id, phase_event, trigger)

    def _trigger_game_end_emotions(self, state: WorldState, winners: str):
        """Trigger emotions when game ends."""
        for agent in state.agents:
            agent_role = self._get_role(agent)
            is_werewolf = agent_role == Role.WEREWOLF
            won = (winners == "werewolves" and is_werewolf) or (winners == "villagers" and not is_werewolf)

            if won:
                self._trigger_event_emotion(state, agent.agent_id, "victory", EmotionalTriggers.VICTORY)
                self._trigger_event_need(state, agent.agent_id, "helped_win", NeedFulfillmentEvents.HELPED_TEAM_WIN)
            else:
                self._trigger_event_emotion(state, agent.agent_id, "defeat", EmotionalTriggers.DEFEAT)

    def _trigger_event_emotion(self, state: WorldState, agent_id: str, event_type: str, trigger: tuple):
        """Apply emotional trigger to specific agent."""
        agent = self._get_agent_by_id(state, agent_id)
        if not agent:
            return

        emotional_state = self._get_emotional_state(agent)
        valence_delta, arousal_delta = trigger
        emotional_state.apply_emotional_event(valence_delta, arousal_delta, event_type)
        self._set_emotional_state(agent, emotional_state)

    def _trigger_event_need(self, state: WorldState, agent_id: str, event_name: str, need_event: tuple):
        """Apply need fulfillment event to specific agent."""
        agent = self._get_agent_by_id(state, agent_id)
        if not agent:
            return

        needs = self._get_needs(agent)
        need_type, satisfaction_delta = need_event
        need = needs.get_all_needs()[need_type]
        need.apply_satisfaction_change(satisfaction_delta)
        self._set_needs(agent, needs)

    def _decay_agent_state(self, agent: AgentStatus):
        """Decay emotion and needs for an agent."""
        emotional_state = self._get_emotional_state(agent)
        emotional_state.decay_emotion()
        self._set_emotional_state(agent, emotional_state)

        needs = self._get_needs(agent)
        needs.decay_all()
        self._set_needs(agent, needs)

    def validate_action(self, action: AgentAction, state: WorldState) -> bool:
        """Validate if an action is legal in current game state."""
        agent = self._get_agent_by_id(state, action.agent_id)
        if not agent:
            return False

        # Dead agents can't act
        if action.agent_id in self.eliminated_agents:
            return False

        action_type = action.action_type

        # Day phase actions
        if self.current_phase == GamePhase.DAY:
            # Allow discussion, accusations, investigations
            return action_type in ["speak", "accuse", "defend", "support", "investigate", "stay_quiet"]

        # Voting phase
        elif self.current_phase == GamePhase.VOTING:
            # Only allow voting
            return action_type == "vote"

        # Night phase
        elif self.current_phase == GamePhase.NIGHT:
            role = self._get_role(agent)
            if role == Role.WEREWOLF:
                return action_type in ["kill", "discuss"]  # Werewolves choose victim
            elif role == Role.SEER:
                return action_type in ["investigate"]  # Seer investigates
            elif role == Role.DOCTOR:
                return action_type in ["protect"]  # Doctor protects
            else:
                return action_type == "sleep"  # Regular villagers sleep

        return True

    def record_vote(self, voter_id: str, target_id: str):
        """Record a vote during voting phase."""
        self.votes[voter_id] = target_id

    def record_night_action(self, agent_id: str, action_type: str, target: Optional[str] = None):
        """Record a night action."""
        self.night_actions[agent_id] = {"action_type": action_type, "target": target}

    def _get_role(self, agent: AgentStatus) -> str:
        """Get agent's role from tags."""
        for tag in agent.tags:
            if tag.startswith("role:"):
                return tag.split(":", 1)[1]
        return Role.VILLAGER

    def _get_role_by_id(self, state: WorldState, agent_id: str) -> str:
        """Get role for agent by ID."""
        agent = self._get_agent_by_id(state, agent_id)
        return self._get_role(agent) if agent else Role.VILLAGER

    def _get_agent_by_id(self, state: WorldState, agent_id: str) -> Optional[AgentStatus]:
        """Get agent status by ID."""
        for agent in state.agents:
            if agent.agent_id == agent_id:
                return agent
        return None

    def _get_emotional_state(self, agent: AgentStatus) -> EmotionalState:
        """Get agent's emotional state from attributes."""
        if "emotional_state" in agent.attributes:
            return EmotionalState.from_dict(agent.attributes["emotional_state"].metadata)
        return EmotionalState()

    def _set_emotional_state(self, agent: AgentStatus, emotional_state: EmotionalState):
        """Set agent's emotional state in attributes."""
        from miniverse.schemas import Stat
        agent.attributes["emotional_state"] = Stat(
            value=emotional_state.get_intensity(),
            unit="intensity",
            label="Emotional State",
            metadata=emotional_state.to_dict()
        )

    def _get_needs(self, agent: AgentStatus) -> NeedsHierarchy:
        """Get agent's needs from attributes."""
        if "needs" in agent.attributes:
            return NeedsHierarchy.from_dict(agent.attributes["needs"].metadata)
        return NeedsHierarchy()

    def _set_needs(self, agent: AgentStatus, needs: NeedsHierarchy):
        """Set agent's needs in attributes."""
        from miniverse.schemas import Stat
        primary_type, primary_need = needs.get_primary_need()
        agent.attributes["needs"] = Stat(
            value=primary_need.satisfaction,
            unit="satisfaction",
            label=f"Primary Need: {primary_type.value}",
            metadata=needs.to_dict()
        )
