"""Hybrid emotion model combining dimensional and categorical approaches.

This module implements the emotion component from "Behavior is all you need" paper.
It uses a hybrid system:
1. Dimensional: Valence (pleasant/unpleasant) × Arousal (calm/excited) space
2. Categorical: Discrete emotions (joy, anger, fear, sadness, etc.)

The dimensional system provides smooth transitions and mathematical operations,
while discrete emotions provide intuitive labels and action mappings.

References:
- Russell's Circumplex Model of Affect (valence × arousal)
- Ekman's Basic Emotions (categorical system)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple
import math


class DiscreteEmotion(str, Enum):
    """Basic discrete emotions following Ekman's model plus neutral state."""
    NEUTRAL = "neutral"
    JOY = "joy"
    ANGER = "anger"
    FEAR = "fear"
    SADNESS = "sadness"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    TRUST = "trust"
    ANTICIPATION = "anticipation"


@dataclass
class EmotionalState:
    """Hybrid emotional state tracking both dimensional and categorical emotions.

    Dimensional component:
    - valence: Pleasant (1.0) to Unpleasant (-1.0)
    - arousal: Excited (1.0) to Calm (-1.0)

    Categorical component:
    - primary_emotion: Dominant discrete emotion
    - emotion_intensities: Intensity of each discrete emotion (0.0-1.0)

    The discrete emotion is derived from dimensional state using Russell's circumplex:
    - High valence + high arousal = Joy
    - Low valence + high arousal = Anger/Fear
    - Low valence + low arousal = Sadness
    - High valence + low arousal = Trust/Contentment
    """
    valence: float = 0.0  # -1.0 (unpleasant) to 1.0 (pleasant)
    arousal: float = 0.0  # -1.0 (calm) to 1.0 (excited)
    primary_emotion: DiscreteEmotion = DiscreteEmotion.NEUTRAL
    emotion_intensities: Dict[DiscreteEmotion, float] = field(default_factory=dict)

    # Decay parameters (emotions fade over time)
    valence_decay: float = 0.1  # How much valence decays toward neutral per tick
    arousal_decay: float = 0.15  # How much arousal decays toward neutral per tick

    def __post_init__(self):
        """Validate and initialize emotional state."""
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(-1.0, min(1.0, self.arousal))

        # Initialize emotion intensities if empty
        if not self.emotion_intensities:
            self.emotion_intensities = {emotion: 0.0 for emotion in DiscreteEmotion}

        # Update primary emotion based on dimensional state
        self._update_primary_emotion()

    def _update_primary_emotion(self):
        """Derive primary discrete emotion from valence/arousal coordinates.

        Uses Russell's Circumplex Model:
        - Quadrant I (high valence, high arousal): Joy, Surprise
        - Quadrant II (low valence, high arousal): Anger, Fear
        - Quadrant III (low valence, low arousal): Sadness
        - Quadrant IV (high valence, low arousal): Trust, Anticipation
        """
        # Neutral zone in center
        if abs(self.valence) < 0.15 and abs(self.arousal) < 0.15:
            self.primary_emotion = DiscreteEmotion.NEUTRAL
            self.emotion_intensities[DiscreteEmotion.NEUTRAL] = 1.0
            return

        # Calculate angle and magnitude in circumplex space
        angle = math.atan2(self.arousal, self.valence)
        magnitude = math.sqrt(self.valence**2 + self.arousal**2)

        # Map angle to discrete emotion (in radians)
        # 0° = right (high valence, neutral arousal) = Trust
        # 45° = upper right (high valence, high arousal) = Joy
        # 90° = top (neutral valence, high arousal) = Surprise
        # 135° = upper left (low valence, high arousal) = Fear/Anger
        # 180° = left (low valence, neutral arousal) = Disgust
        # -135° = lower left (low valence, low arousal) = Sadness
        # -90° = bottom (neutral valence, low arousal) = Anticipation

        angle_deg = math.degrees(angle)

        if -22.5 <= angle_deg < 22.5:  # Right
            emotion = DiscreteEmotion.TRUST
        elif 22.5 <= angle_deg < 67.5:  # Upper right
            emotion = DiscreteEmotion.JOY
        elif 67.5 <= angle_deg < 112.5:  # Top
            emotion = DiscreteEmotion.SURPRISE
        elif 112.5 <= angle_deg <= 180 or -180 <= angle_deg < -157.5:  # Left
            # Distinguish fear vs anger vs disgust based on arousal
            if self.arousal > 0.3:  # High arousal = fear or anger
                emotion = DiscreteEmotion.FEAR if self.arousal > 0.6 else DiscreteEmotion.ANGER
            else:
                emotion = DiscreteEmotion.DISGUST
        elif -157.5 <= angle_deg < -112.5:  # Lower left
            emotion = DiscreteEmotion.SADNESS
        elif -112.5 <= angle_deg < -67.5:  # Bottom
            emotion = DiscreteEmotion.SADNESS if self.valence < -0.3 else DiscreteEmotion.NEUTRAL
        else:  # -67.5 to -22.5 (lower right)
            emotion = DiscreteEmotion.ANTICIPATION

        self.primary_emotion = emotion

        # Set intensity based on magnitude (distance from center)
        intensity = min(1.0, magnitude)
        self.emotion_intensities[emotion] = intensity

        # Reduce other emotion intensities
        for other_emotion in DiscreteEmotion:
            if other_emotion != emotion:
                self.emotion_intensities[other_emotion] *= 0.5

    def apply_emotional_event(
        self,
        valence_delta: float,
        arousal_delta: float,
        description: Optional[str] = None
    ):
        """Apply an emotional event that shifts dimensional state.

        Args:
            valence_delta: Change in valence (-1.0 to 1.0)
            arousal_delta: Change in arousal (-1.0 to 1.0)
            description: Optional description of what caused this emotion
        """
        self.valence = max(-1.0, min(1.0, self.valence + valence_delta))
        self.arousal = max(-1.0, min(1.0, self.arousal + arousal_delta))
        self._update_primary_emotion()

    def decay_emotion(self):
        """Decay emotional state toward neutral over time.

        Called each simulation tick to simulate emotional regulation.
        """
        # Decay valence toward 0
        if self.valence > 0:
            self.valence = max(0.0, self.valence - self.valence_decay)
        elif self.valence < 0:
            self.valence = min(0.0, self.valence + self.valence_decay)

        # Decay arousal toward 0
        if self.arousal > 0:
            self.arousal = max(0.0, self.arousal - self.arousal_decay)
        elif self.arousal < 0:
            self.arousal = min(0.0, self.arousal + self.arousal_decay)

        self._update_primary_emotion()

    def get_intensity(self) -> float:
        """Get overall emotional intensity (magnitude in circumplex space)."""
        return math.sqrt(self.valence**2 + self.arousal**2)

    def get_emotion_description(self) -> str:
        """Get human-readable description of current emotional state."""
        intensity = self.get_intensity()

        if intensity < 0.15:
            return "neutral and calm"

        intensity_label = "slightly" if intensity < 0.4 else ("moderately" if intensity < 0.7 else "intensely")
        emotion_name = self.primary_emotion.value

        return f"{intensity_label} {emotion_name}"

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "primary_emotion": self.primary_emotion.value,
            "emotion_intensities": {k.value: v for k, v in self.emotion_intensities.items()},
            "intensity": self.get_intensity(),
            "description": self.get_emotion_description()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EmotionalState":
        """Deserialize from dictionary."""
        emotion_intensities = {
            DiscreteEmotion(k): v for k, v in data.get("emotion_intensities", {}).items()
        }
        return cls(
            valence=data["valence"],
            arousal=data["arousal"],
            primary_emotion=DiscreteEmotion(data["primary_emotion"]),
            emotion_intensities=emotion_intensities
        )


class EmotionalTriggers:
    """Common emotional triggers for social deception games.

    Maps game events to (valence_delta, arousal_delta) tuples.
    These values are calibrated for believable emotional responses.
    """

    # Positive events
    CLEARED_OF_SUSPICION = (0.4, -0.2)  # Relief (positive, calming)
    ALLY_CONFIRMED = (0.3, 0.1)  # Joy (positive, mild excitement)
    SUCCESSFUL_DECEPTION = (0.5, 0.3)  # Excitement (positive, high arousal)
    VICTORY = (0.8, 0.6)  # Intense joy
    SUPPORTED_BY_OTHERS = (0.3, 0.0)  # Trust (positive, neutral arousal)

    # Negative events
    ACCUSED_FALSELY = (-0.5, 0.6)  # Anger/fear (negative, high arousal)
    ALLY_DIED = (-0.6, 0.2)  # Sadness/anger (negative, moderate arousal)
    BETRAYED = (-0.7, 0.5)  # Intense anger
    NEAR_ELIMINATION = (-0.4, 0.7)  # Fear (negative, very high arousal)
    ISOLATED = (-0.4, -0.2)  # Sadness (negative, low arousal)
    DEFEAT = (-0.7, 0.3)  # Sadness/frustration

    # Neutral but arousing events
    SUSPICION_RAISED = (-0.2, 0.4)  # Anxiety (slightly negative, arousal)
    NEW_INFORMATION = (0.1, 0.5)  # Surprise (slightly positive, high arousal)
    DAY_PHASE_START = (0.0, 0.3)  # Anticipation (neutral, mild arousal)
    NIGHT_PHASE_START = (-0.1, 0.2)  # Tension (slightly negative, mild arousal)

    @staticmethod
    def get_trigger(event_type: str) -> Optional[Tuple[float, float]]:
        """Get emotional impact for an event type.

        Returns:
            (valence_delta, arousal_delta) or None if event not found
        """
        trigger_map = {
            "cleared": EmotionalTriggers.CLEARED_OF_SUSPICION,
            "ally_confirmed": EmotionalTriggers.ALLY_CONFIRMED,
            "successful_deception": EmotionalTriggers.SUCCESSFUL_DECEPTION,
            "victory": EmotionalTriggers.VICTORY,
            "supported": EmotionalTriggers.SUPPORTED_BY_OTHERS,
            "accused": EmotionalTriggers.ACCUSED_FALSELY,
            "ally_died": EmotionalTriggers.ALLY_DIED,
            "betrayed": EmotionalTriggers.BETRAYED,
            "near_elimination": EmotionalTriggers.NEAR_ELIMINATION,
            "isolated": EmotionalTriggers.ISOLATED,
            "defeat": EmotionalTriggers.DEFEAT,
            "suspicion": EmotionalTriggers.SUSPICION_RAISED,
            "new_info": EmotionalTriggers.NEW_INFORMATION,
            "day_start": EmotionalTriggers.DAY_PHASE_START,
            "night_start": EmotionalTriggers.NIGHT_PHASE_START,
        }
        return trigger_map.get(event_type)


class EmotionActionInfluence:
    """How emotions influence action selection in social deception game.

    Maps emotional states to action weight modifiers. This creates the core
    "play-acting" behavior from the paper - agents don't need to actually FEEL
    emotions, they just need to ACT like they do.
    """

    @staticmethod
    def get_action_modifier(emotional_state: EmotionalState, action_type: str) -> float:
        """Get emotion-based weight modifier for an action.

        Args:
            emotional_state: Current emotional state
            action_type: Action being considered

        Returns:
            Weight multiplier (0.0-2.0). Multiplied with personality weight.
        """
        emotion = emotional_state.primary_emotion
        intensity = emotional_state.get_intensity()

        # Fear influences
        if emotion == DiscreteEmotion.FEAR:
            if action_type in ["stay_quiet", "defend"]:
                return 1.0 + intensity * 0.8  # Fear increases defensive behavior
            elif action_type in ["accuse", "investigate"]:
                return 1.0 - intensity * 0.5  # Fear decreases proactive behavior

        # Anger influences
        elif emotion == DiscreteEmotion.ANGER:
            if action_type in ["accuse", "suspect"]:
                return 1.0 + intensity * 1.0  # Anger increases accusatory behavior
            elif action_type in ["support", "trust"]:
                return 1.0 - intensity * 0.6  # Anger decreases cooperative behavior

        # Joy influences
        elif emotion == DiscreteEmotion.JOY:
            if action_type in ["support", "trust"]:
                return 1.0 + intensity * 0.5  # Joy increases cooperative behavior
            elif action_type in ["accuse", "suspect"]:
                return 1.0 - intensity * 0.4  # Joy decreases suspicion

        # Sadness influences
        elif emotion == DiscreteEmotion.SADNESS:
            if action_type == "stay_quiet":
                return 1.0 + intensity * 0.7  # Sadness increases withdrawal
            elif action_type in ["investigate", "accuse"]:
                return 1.0 - intensity * 0.5  # Sadness decreases engagement

        # Trust influences
        elif emotion == DiscreteEmotion.TRUST:
            if action_type in ["support", "trust"]:
                return 1.0 + intensity * 0.6
            elif action_type in ["suspect", "accuse"]:
                return 1.0 - intensity * 0.7

        # Surprise influences
        elif emotion == DiscreteEmotion.SURPRISE:
            if action_type == "investigate":
                return 1.0 + intensity * 0.5  # Surprise increases curiosity
            elif action_type == "stay_quiet":
                return 1.0 - intensity * 0.3  # Surprise decreases passivity

        # Default: no emotion influence
        return 1.0

    @staticmethod
    def get_communication_modifier(emotional_state: EmotionalState) -> Dict[str, str]:
        """Get emotion-based communication style modifiers.

        Returns dict with:
        - tone: emotional tone for messages
        - urgency: how urgent/impulsive communication should be
        """
        emotion = emotional_state.primary_emotion
        intensity = emotional_state.get_intensity()

        modifiers = {}

        # Tone
        if emotion == DiscreteEmotion.FEAR:
            modifiers["tone"] = "anxious and defensive" if intensity > 0.5 else "cautious"
        elif emotion == DiscreteEmotion.ANGER:
            modifiers["tone"] = "hostile and accusatory" if intensity > 0.5 else "irritated"
        elif emotion == DiscreteEmotion.JOY:
            modifiers["tone"] = "cheerful and supportive" if intensity > 0.5 else "friendly"
        elif emotion == DiscreteEmotion.SADNESS:
            modifiers["tone"] = "withdrawn and somber" if intensity > 0.5 else "subdued"
        elif emotion == DiscreteEmotion.TRUST:
            modifiers["tone"] = "warm and cooperative"
        elif emotion == DiscreteEmotion.SURPRISE:
            modifiers["tone"] = "curious and questioning"
        else:
            modifiers["tone"] = "neutral and measured"

        # Urgency (based on arousal)
        if emotional_state.arousal > 0.6:
            modifiers["urgency"] = "speak immediately and impulsively"
        elif emotional_state.arousal > 0.3:
            modifiers["urgency"] = "speak promptly"
        elif emotional_state.arousal > -0.3:
            modifiers["urgency"] = "take time to consider words"
        else:
            modifiers["urgency"] = "speak slowly and carefully"

        return modifiers
