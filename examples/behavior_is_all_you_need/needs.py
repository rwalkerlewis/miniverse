"""Maslow's hierarchy of needs for agent motivation.

This module implements the needs component from "Behavior is all you need" paper.
It models internal drives that motivate agent behavior, based on Maslow's
hierarchy of needs pyramid.

The hierarchy (bottom to top):
1. Safety: Physical safety, freedom from threats
2. Belonging: Social bonds, acceptance by group
3. Esteem: Recognition, status, achievement
4. Self-Actualization: Purpose, creativity, growth

Lower needs take priority when unsatisfied. As needs are met, agents pursue
higher-level goals. This creates dynamic, context-sensitive motivation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple


class NeedType(str, Enum):
    """Types of needs in Maslow's hierarchy."""
    SAFETY = "safety"
    BELONGING = "belonging"
    ESTEEM = "esteem"
    SELF_ACTUALIZATION = "self_actualization"


@dataclass
class Need:
    """Individual need with satisfaction level and priority."""
    need_type: NeedType
    satisfaction: float = 0.5  # 0.0 (completely unsatisfied) to 1.0 (fully satisfied)
    importance: float = 1.0  # Weight for this specific agent (some care more about certain needs)
    decay_rate: float = 0.05  # How fast satisfaction decreases per tick
    fulfillment_threshold: float = 0.7  # Above this, need is considered "satisfied"
    critical_threshold: float = 0.2  # Below this, need becomes urgent/desperate

    def is_satisfied(self) -> bool:
        """Check if need is currently satisfied."""
        return self.satisfaction >= self.fulfillment_threshold

    def is_critical(self) -> bool:
        """Check if need is at critical level (urgent)."""
        return self.satisfaction <= self.critical_threshold

    def get_drive_strength(self) -> float:
        """Calculate motivation strength from this need (0.0-1.0).

        Higher when unsatisfied, lower when satisfied.
        Critical needs create very strong drive.
        """
        if self.is_satisfied():
            return 0.1  # Minimal drive when satisfied

        # Inverse of satisfaction, scaled by importance
        unsatisfied_amount = 1.0 - self.satisfaction
        drive = unsatisfied_amount * self.importance

        # Critical needs create even stronger drive
        if self.is_critical():
            drive *= 1.5

        return min(1.0, drive)

    def apply_satisfaction_change(self, delta: float):
        """Change satisfaction level.

        Args:
            delta: Amount to change (-1.0 to 1.0). Positive increases satisfaction.
        """
        self.satisfaction = max(0.0, min(1.0, self.satisfaction + delta))

    def decay(self):
        """Decay satisfaction over time (needs gradually become unsatisfied)."""
        self.satisfaction = max(0.0, self.satisfaction - self.decay_rate)


@dataclass
class NeedsHierarchy:
    """Maslow's hierarchy of needs for an agent.

    Tracks satisfaction of each need level and determines which need
    is currently driving behavior. Lower needs take priority.
    """
    safety: Need = field(default_factory=lambda: Need(NeedType.SAFETY, importance=1.0, decay_rate=0.03))
    belonging: Need = field(default_factory=lambda: Need(NeedType.BELONGING, importance=0.8, decay_rate=0.04))
    esteem: Need = field(default_factory=lambda: Need(NeedType.ESTEEM, importance=0.7, decay_rate=0.05))
    self_actualization: Need = field(default_factory=lambda: Need(NeedType.SELF_ACTUALIZATION, importance=0.5, decay_rate=0.02))

    def get_all_needs(self) -> Dict[NeedType, Need]:
        """Get dictionary of all needs."""
        return {
            NeedType.SAFETY: self.safety,
            NeedType.BELONGING: self.belonging,
            NeedType.ESTEEM: self.esteem,
            NeedType.SELF_ACTUALIZATION: self.self_actualization,
        }

    def get_primary_need(self) -> Tuple[NeedType, Need]:
        """Get the most pressing need according to hierarchy.

        Lower needs take priority. Returns the lowest unsatisfied need,
        or highest need if all are satisfied.
        """
        # Check hierarchy from bottom to top
        for need_type in [NeedType.SAFETY, NeedType.BELONGING, NeedType.ESTEEM, NeedType.SELF_ACTUALIZATION]:
            need = self.get_all_needs()[need_type]
            if not need.is_satisfied():
                return (need_type, need)

        # All needs satisfied - pursue self-actualization
        return (NeedType.SELF_ACTUALIZATION, self.self_actualization)

    def get_critical_needs(self) -> list[Tuple[NeedType, Need]]:
        """Get all needs currently at critical level."""
        critical = []
        for need_type, need in self.get_all_needs().items():
            if need.is_critical():
                critical.append((need_type, need))
        return critical

    def decay_all(self):
        """Decay all needs over time."""
        for need in self.get_all_needs().values():
            need.decay()

    def get_motivation_summary(self) -> str:
        """Get human-readable summary of current motivational state."""
        primary_type, primary_need = self.get_primary_need()
        critical = self.get_critical_needs()

        if critical:
            critical_names = [need_type.value for need_type, _ in critical]
            return f"URGENT: {', '.join(critical_names)} need(s) critical; primarily motivated by {primary_type.value}"
        elif primary_need.is_satisfied():
            return f"All basic needs satisfied; pursuing {primary_type.value}"
        else:
            return f"Primarily motivated by {primary_type.value} (satisfaction: {primary_need.satisfaction:.1%})"

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "safety": {
                "satisfaction": self.safety.satisfaction,
                "is_critical": self.safety.is_critical(),
                "drive_strength": self.safety.get_drive_strength()
            },
            "belonging": {
                "satisfaction": self.belonging.satisfaction,
                "is_critical": self.belonging.is_critical(),
                "drive_strength": self.belonging.get_drive_strength()
            },
            "esteem": {
                "satisfaction": self.esteem.satisfaction,
                "is_critical": self.esteem.is_critical(),
                "drive_strength": self.esteem.get_drive_strength()
            },
            "self_actualization": {
                "satisfaction": self.self_actualization.satisfaction,
                "is_critical": self.self_actualization.is_critical(),
                "drive_strength": self.self_actualization.get_drive_strength()
            },
            "summary": self.get_motivation_summary()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "NeedsHierarchy":
        """Deserialize from dictionary."""
        return cls(
            safety=Need(NeedType.SAFETY, satisfaction=data["safety"]["satisfaction"]),
            belonging=Need(NeedType.BELONGING, satisfaction=data["belonging"]["satisfaction"]),
            esteem=Need(NeedType.ESTEEM, satisfaction=data["esteem"]["satisfaction"]),
            self_actualization=Need(NeedType.SELF_ACTUALIZATION, satisfaction=data["self_actualization"]["satisfaction"])
        )


class NeedFulfillmentEvents:
    """Maps game events to need satisfaction changes.

    Defines how social deception game events affect various needs.
    """

    # Safety need events
    SURVIVED_ELIMINATION = (NeedType.SAFETY, 0.3)
    NEARLY_ELIMINATED = (NeedType.SAFETY, -0.4)
    PROTECTED_BY_ROLE = (NeedType.SAFETY, 0.2)
    THREATENED = (NeedType.SAFETY, -0.3)
    NIGHT_SURVIVED = (NeedType.SAFETY, 0.1)

    # Belonging need events
    SUPPORTED_BY_GROUP = (NeedType.BELONGING, 0.3)
    ACCUSED_BY_GROUP = (NeedType.BELONGING, -0.4)
    ALLY_FORMED = (NeedType.BELONGING, 0.4)
    ALLY_DIED = (NeedType.BELONGING, -0.3)
    ISOLATED = (NeedType.BELONGING, -0.5)
    INCLUDED_IN_DISCUSSION = (NeedType.BELONGING, 0.2)

    # Esteem need events
    PRAISED_FOR_INSIGHT = (NeedType.ESTEEM, 0.4)
    MOCKED_OR_DISMISSED = (NeedType.ESTEEM, -0.3)
    SUCCESSFUL_DEDUCTION = (NeedType.ESTEEM, 0.3)
    FAILED_DEDUCTION = (NeedType.ESTEEM, -0.2)
    LEADERSHIP_ACKNOWLEDGED = (NeedType.ESTEEM, 0.5)
    IGNORED_BY_GROUP = (NeedType.ESTEEM, -0.2)

    # Self-actualization need events
    SOLVED_MYSTERY = (NeedType.SELF_ACTUALIZATION, 0.4)
    HELPED_TEAM_WIN = (NeedType.SELF_ACTUALIZATION, 0.5)
    CREATIVE_STRATEGY_WORKED = (NeedType.SELF_ACTUALIZATION, 0.3)
    PLAYED_ROLE_PERFECTLY = (NeedType.SELF_ACTUALIZATION, 0.3)

    @staticmethod
    def get_event(event_name: str) -> Optional[Tuple[NeedType, float]]:
        """Get need fulfillment change for an event.

        Returns:
            (need_type, satisfaction_delta) or None if event not found
        """
        event_map = {
            # Safety
            "survived_elimination": NeedFulfillmentEvents.SURVIVED_ELIMINATION,
            "nearly_eliminated": NeedFulfillmentEvents.NEARLY_ELIMINATED,
            "protected": NeedFulfillmentEvents.PROTECTED_BY_ROLE,
            "threatened": NeedFulfillmentEvents.THREATENED,
            "night_survived": NeedFulfillmentEvents.NIGHT_SURVIVED,
            # Belonging
            "supported": NeedFulfillmentEvents.SUPPORTED_BY_GROUP,
            "accused": NeedFulfillmentEvents.ACCUSED_BY_GROUP,
            "ally_formed": NeedFulfillmentEvents.ALLY_FORMED,
            "ally_died": NeedFulfillmentEvents.ALLY_DIED,
            "isolated": NeedFulfillmentEvents.ISOLATED,
            "included": NeedFulfillmentEvents.INCLUDED_IN_DISCUSSION,
            # Esteem
            "praised": NeedFulfillmentEvents.PRAISED_FOR_INSIGHT,
            "mocked": NeedFulfillmentEvents.MOCKED_OR_DISMISSED,
            "successful_deduction": NeedFulfillmentEvents.SUCCESSFUL_DEDUCTION,
            "failed_deduction": NeedFulfillmentEvents.FAILED_DEDUCTION,
            "leadership": NeedFulfillmentEvents.LEADERSHIP_ACKNOWLEDGED,
            "ignored": NeedFulfillmentEvents.IGNORED_BY_GROUP,
            # Self-actualization
            "solved_mystery": NeedFulfillmentEvents.SOLVED_MYSTERY,
            "helped_win": NeedFulfillmentEvents.HELPED_TEAM_WIN,
            "creative_success": NeedFulfillmentEvents.CREATIVE_STRATEGY_WORKED,
            "role_perfection": NeedFulfillmentEvents.PLAYED_ROLE_PERFECTLY,
        }
        return event_map.get(event_name)


class NeedActionInfluence:
    """How needs influence action selection in social deception game.

    Maps unsatisfied needs to action preferences. This creates goal-directed
    behavior that shifts based on what the agent currently needs.
    """

    @staticmethod
    def get_action_modifier(needs: NeedsHierarchy, action_type: str) -> float:
        """Get need-based weight modifier for an action.

        Args:
            needs: Agent's current needs hierarchy
            action_type: Action being considered

        Returns:
            Weight multiplier (0.5-2.0). Multiplied with personality and emotion weights.
        """
        primary_type, primary_need = needs.get_primary_need()
        drive_strength = primary_need.get_drive_strength()

        # Safety need influences
        if primary_type == NeedType.SAFETY:
            if action_type in ["defend", "stay_quiet", "trust"]:
                return 1.0 + drive_strength * 0.8  # Seek safety through defense/alliances
            elif action_type in ["accuse", "lie"]:
                return 1.0 - drive_strength * 0.6  # Avoid risky confrontations

        # Belonging need influences
        elif primary_type == NeedType.BELONGING:
            if action_type in ["support", "trust", "ally"]:
                return 1.0 + drive_strength * 1.0  # Strongly seek social bonds
            elif action_type in ["accuse", "suspect"]:
                return 1.0 - drive_strength * 0.5  # Avoid damaging relationships

        # Esteem need influences
        elif primary_type == NeedType.ESTEEM:
            if action_type in ["investigate", "accuse", "lead"]:
                return 1.0 + drive_strength * 0.9  # Seek recognition through active participation
            elif action_type == "stay_quiet":
                return 1.0 - drive_strength * 0.7  # Avoid being invisible

        # Self-actualization need influences
        elif primary_type == NeedType.SELF_ACTUALIZATION:
            if action_type in ["investigate", "creative_strategy"]:
                return 1.0 + drive_strength * 0.7  # Seek growth and mastery
            # Self-actualization doesn't inhibit other actions much

        # Critical need modifiers (override normal hierarchy)
        critical_needs = needs.get_critical_needs()
        for need_type, need in critical_needs:
            if need_type == NeedType.SAFETY:
                if action_type in ["defend", "retreat"]:
                    return 2.0  # Desperate safety-seeking
                elif action_type in ["accuse", "confront"]:
                    return 0.5  # Avoid conflict at all costs
            elif need_type == NeedType.BELONGING:
                if action_type in ["support", "trust"]:
                    return 1.8  # Desperate for connection
                elif action_type in ["accuse", "betray"]:
                    return 0.6  # Cling to any relationships

        return 1.0  # Default: no need influence

    @staticmethod
    def get_goal_description(needs: NeedsHierarchy) -> str:
        """Get description of agent's current goal based on needs.

        Used in prompts to guide LLM toward need-appropriate behavior.
        """
        primary_type, primary_need = needs.get_primary_need()
        critical_needs = needs.get_critical_needs()

        if critical_needs:
            # Urgent/desperate mode
            critical_type = critical_needs[0][0]
            if critical_type == NeedType.SAFETY:
                return "survive at all costs; avoid elimination; seek protection"
            elif critical_type == NeedType.BELONGING:
                return "desperately seek allies; avoid isolation; repair damaged relationships"
            elif critical_type == NeedType.ESTEEM:
                return "prove your worth; gain recognition; restore reputation"
            else:
                return "demonstrate competence; achieve mastery; fulfill purpose"

        # Normal priority mode
        if primary_type == NeedType.SAFETY:
            if primary_need.satisfaction < 0.4:
                return "prioritize safety; identify threats; build defensive alliances"
            else:
                return "maintain security while pursuing other goals"
        elif primary_type == NeedType.BELONGING:
            if primary_need.satisfaction < 0.4:
                return "build social connections; gain group acceptance; form alliances"
            else:
                return "maintain relationships while pursuing recognition"
        elif primary_type == NeedType.ESTEEM:
            if primary_need.satisfaction < 0.4:
                return "seek recognition; demonstrate competence; lead discussions"
            else:
                return "maintain status while pursuing personal growth"
        else:  # SELF_ACTUALIZATION
            return "pursue mastery; solve mysteries; play your role perfectly; help team win"
