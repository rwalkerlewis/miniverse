"""Big Five (OCEAN) personality model for behavior modulation.

This module implements the personality component from "Behavior is all you need" paper.
It uses the scientifically-validated Big Five personality traits to create consistent,
believable agent behavior patterns.

The Big Five traits:
- Openness: Curiosity, creativity, openness to new experiences
- Conscientiousness: Organization, responsibility, goal-directed behavior
- Extraversion: Sociability, assertiveness, energy level
- Agreeableness: Cooperation, trust, empathy
- Neuroticism: Emotional instability, anxiety, vulnerability to stress

Each trait score ranges from 0-100 (normalized percentile scores).
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class Trait(str, Enum):
    """Big Five personality trait dimensions."""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


@dataclass
class BigFiveTraits:
    """Big Five personality trait scores (0-100 scale).

    Higher scores indicate stronger trait expression:
    - High Openness (70-100): Creative, curious, adventurous, unconventional
    - Low Openness (0-30): Traditional, practical, conventional, resistant to change

    - High Conscientiousness (70-100): Organized, disciplined, reliable, goal-focused
    - Low Conscientiousness (0-30): Spontaneous, flexible, disorganized, impulsive

    - High Extraversion (70-100): Outgoing, energetic, talkative, assertive
    - Low Extraversion (0-30): Reserved, quiet, solitary, introspective

    - High Agreeableness (70-100): Cooperative, trusting, empathetic, kind
    - Low Agreeableness (0-30): Competitive, skeptical, challenging, self-focused

    - High Neuroticism (70-100): Anxious, emotionally reactive, stressed, worried
    - Low Neuroticism (0-30): Calm, emotionally stable, resilient, secure
    """
    openness: float  # 0-100
    conscientiousness: float  # 0-100
    extraversion: float  # 0-100
    agreeableness: float  # 0-100
    neuroticism: float  # 0-100

    def __post_init__(self):
        """Validate trait scores are in valid range."""
        for trait_name in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            value = getattr(self, trait_name)
            if not 0 <= value <= 100:
                raise ValueError(f"{trait_name} must be between 0 and 100, got {value}")

    def get(self, trait: Trait) -> float:
        """Get trait score by enum."""
        return getattr(self, trait.value)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "BigFiveTraits":
        """Create from dictionary."""
        return cls(
            openness=data["openness"],
            conscientiousness=data["conscientiousness"],
            extraversion=data["extraversion"],
            agreeableness=data["agreeableness"],
            neuroticism=data["neuroticism"],
        )

    def describe(self) -> str:
        """Generate human-readable personality description."""
        traits = []

        # Openness
        if self.openness >= 70:
            traits.append("highly creative and open to new ideas")
        elif self.openness >= 40:
            traits.append("moderately open to experiences")
        else:
            traits.append("traditional and prefers familiar approaches")

        # Conscientiousness
        if self.conscientiousness >= 70:
            traits.append("highly organized and disciplined")
        elif self.conscientiousness >= 40:
            traits.append("moderately organized")
        else:
            traits.append("spontaneous and flexible")

        # Extraversion
        if self.extraversion >= 70:
            traits.append("very outgoing and energetic")
        elif self.extraversion >= 40:
            traits.append("balanced between introversion and extraversion")
        else:
            traits.append("reserved and introspective")

        # Agreeableness
        if self.agreeableness >= 70:
            traits.append("highly cooperative and empathetic")
        elif self.agreeableness >= 40:
            traits.append("balanced in cooperation")
        else:
            traits.append("competitive and assertive")

        # Neuroticism
        if self.neuroticism >= 70:
            traits.append("emotionally reactive and anxious")
        elif self.neuroticism >= 40:
            traits.append("moderately emotionally stable")
        else:
            traits.append("very emotionally stable and calm")

        return "; ".join(traits)


class PersonalityWeights:
    """Maps personality traits to behavior weights for action selection.

    This class implements the core insight from "Behavior is all you need":
    personality traits directly modulate action probabilities. Instead of trying
    to model internal consciousness, we create behavioral patterns that FEEL
    human-like through consistent trait expression.

    Example usage:
        traits = BigFiveTraits(openness=80, conscientiousness=60, ...)
        weights = PersonalityWeights.calculate(traits, action_type="accuse", context={"stress_level": 0.8})
        # Returns weight multiplier that adjusts action probability
    """

    @staticmethod
    def calculate_action_weight(
        traits: BigFiveTraits,
        action_type: str,
        context: Optional[Dict] = None
    ) -> float:
        """Calculate personality weight for an action.

        Args:
            traits: Agent's Big Five personality
            action_type: Action being considered (e.g., "accuse", "defend", "investigate")
            context: Additional context (stress_level, relationships, etc.)

        Returns:
            Weight multiplier (0.0-2.0). Values >1.0 increase action probability,
            <1.0 decrease it. 1.0 = neutral (no personality influence).
        """
        context = context or {}
        base_weight = 1.0

        # Social deception game actions
        if action_type == "accuse":
            # Accusing others requires low agreeableness and moderate extraversion
            # High neuroticism increases accusatory behavior under stress
            weight = base_weight
            weight *= 1.0 + (100 - traits.agreeableness) / 200  # Low agreeableness boosts
            weight *= 1.0 + traits.extraversion / 200  # Extraversion boosts
            stress = context.get("stress_level", 0.0)
            if stress > 0.5:
                weight *= 1.0 + (traits.neuroticism / 100) * stress  # Neurotic agents accuse more under stress
            return min(2.0, weight)

        elif action_type == "defend":
            # Defending self requires extraversion and low neuroticism (confidence)
            weight = base_weight
            weight *= 1.0 + traits.extraversion / 150
            weight *= 1.0 + (100 - traits.neuroticism) / 200  # Low neuroticism boosts confidence
            return min(2.0, weight)

        elif action_type == "support":
            # Supporting others requires high agreeableness
            weight = base_weight
            weight *= 1.0 + traits.agreeableness / 100  # High agreeableness strongly boosts
            return min(2.0, weight)

        elif action_type == "investigate":
            # Investigation requires openness and conscientiousness
            weight = base_weight
            weight *= 1.0 + traits.openness / 150
            weight *= 1.0 + traits.conscientiousness / 150
            return min(2.0, weight)

        elif action_type == "stay_quiet":
            # Staying quiet favored by introverts, high agreeableness, high neuroticism
            weight = base_weight
            weight *= 1.0 + (100 - traits.extraversion) / 150  # Introversion boosts
            weight *= 1.0 + traits.agreeableness / 200  # Agreeable people avoid conflict
            weight *= 1.0 + traits.neuroticism / 200  # Anxious people stay quiet
            return min(2.0, weight)

        elif action_type == "lie" or action_type == "deceive":
            # Lying requires low agreeableness, high openness (creative lying), low neuroticism (confidence)
            weight = base_weight
            weight *= 1.0 + (100 - traits.agreeableness) / 100  # Low agreeableness enables lying
            weight *= 1.0 + traits.openness / 200  # Openness enables creative deception
            weight *= 1.0 + (100 - traits.neuroticism) / 200  # Low anxiety helps lie convincingly
            return min(2.0, weight)

        elif action_type == "confess" or action_type == "reveal":
            # Confessing requires high conscientiousness (honesty) or high neuroticism (guilt)
            weight = base_weight
            weight *= 1.0 + traits.conscientiousness / 100
            stress = context.get("stress_level", 0.0)
            if stress > 0.7:  # High stress makes neurotic agents confess
                weight *= 1.0 + (traits.neuroticism / 100) * stress
            return min(2.0, weight)

        elif action_type == "trust":
            # Trusting others requires high agreeableness, low neuroticism
            weight = base_weight
            weight *= 1.0 + traits.agreeableness / 100
            weight *= 1.0 + (100 - traits.neuroticism) / 200
            return min(2.0, weight)

        elif action_type == "suspect":
            # Being suspicious favored by low agreeableness, high openness (consider alternatives)
            weight = base_weight
            weight *= 1.0 + (100 - traits.agreeableness) / 150
            weight *= 1.0 + traits.openness / 200
            return min(2.0, weight)

        # Default: no personality influence
        return 1.0

    @staticmethod
    def get_communication_style(traits: BigFiveTraits) -> Dict[str, str]:
        """Get communication style modifiers based on personality.

        Returns a dict with style hints that can be used in prompts:
        - tone: formal/casual/friendly/direct
        - verbosity: verbose/concise/minimal
        - assertiveness: passive/balanced/assertive/aggressive
        """
        style = {}

        # Tone (influenced by agreeableness and extraversion)
        if traits.agreeableness >= 70:
            style["tone"] = "friendly and warm"
        elif traits.agreeableness <= 30:
            style["tone"] = "direct and challenging"
        elif traits.extraversion >= 60:
            style["tone"] = "casual and energetic"
        else:
            style["tone"] = "formal and measured"

        # Verbosity (influenced by extraversion and openness)
        if traits.extraversion >= 70 and traits.openness >= 60:
            style["verbosity"] = "verbose and expressive"
        elif traits.extraversion <= 30:
            style["verbosity"] = "minimal and reserved"
        else:
            style["verbosity"] = "concise and clear"

        # Assertiveness (influenced by extraversion, agreeableness, neuroticism)
        assertiveness_score = (traits.extraversion + (100 - traits.agreeableness) + (100 - traits.neuroticism)) / 3
        if assertiveness_score >= 70:
            style["assertiveness"] = "aggressive and confrontational"
        elif assertiveness_score >= 50:
            style["assertiveness"] = "assertive and confident"
        elif assertiveness_score >= 30:
            style["assertiveness"] = "balanced and diplomatic"
        else:
            style["assertiveness"] = "passive and cautious"

        return style


# Predefined personality archetypes for testing
class PersonalityArchetypes:
    """Common personality archetypes for social deception games."""

    @staticmethod
    def anxious_villager() -> BigFiveTraits:
        """High neuroticism, high agreeableness - panics easily, trusts others."""
        return BigFiveTraits(
            openness=45,
            conscientiousness=60,
            extraversion=30,
            agreeableness=75,
            neuroticism=85
        )

    @staticmethod
    def confident_leader() -> BigFiveTraits:
        """High extraversion, low neuroticism, moderate agreeableness."""
        return BigFiveTraits(
            openness=60,
            conscientiousness=70,
            extraversion=85,
            agreeableness=55,
            neuroticism=25
        )

    @staticmethod
    def manipulative_deceiver() -> BigFiveTraits:
        """Low agreeableness, high openness, low neuroticism - skilled liar."""
        return BigFiveTraits(
            openness=75,
            conscientiousness=40,
            extraversion=65,
            agreeableness=20,
            neuroticism=30
        )

    @staticmethod
    def quiet_observer() -> BigFiveTraits:
        """Low extraversion, high openness, moderate everything else."""
        return BigFiveTraits(
            openness=70,
            conscientiousness=55,
            extraversion=20,
            agreeableness=60,
            neuroticism=45
        )

    @staticmethod
    def emotional_supporter() -> BigFiveTraits:
        """High agreeableness, high extraversion, moderate neuroticism."""
        return BigFiveTraits(
            openness=55,
            conscientiousness=50,
            extraversion=75,
            agreeableness=85,
            neuroticism=55
        )

    @staticmethod
    def skeptical_analyst() -> BigFiveTraits:
        """High openness, high conscientiousness, low agreeableness."""
        return BigFiveTraits(
            openness=80,
            conscientiousness=75,
            extraversion=45,
            agreeableness=35,
            neuroticism=40
        )
