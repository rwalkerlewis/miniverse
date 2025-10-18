"""Personality-aware cognition modules for "Behavior is all you need" implementation.

This module implements the core insight from the paper: personality, emotion, and needs
directly modulate behavior. It extends Miniverse's executor and planner to incorporate
the paper's 4-component framework.

Key components integrated:
1. Personality (Big Five traits) - stable behavioral tendencies
2. Emotion (dimensional + categorical) - transient state modulation
3. Needs (Maslow hierarchy) - motivational drivers
4. Memory - past experiences inform current decisions

These components create emergent, believable behavior WITHOUT requiring true consciousness.
"""

from typing import Dict, List, Optional
import random

from miniverse import AgentAction, AgentProfile, Plan, PlanStep
from miniverse.cognition import Executor, Planner, Scratchpad
from miniverse.cognition.context import PromptContext
from miniverse.cognition.prompts import PromptLibrary, PromptTemplate
from miniverse.cognition.renderers import render_prompt
from miniverse.llm_calls import call_llm_with_retries
from miniverse.schemas import AgentPerception

from personality import BigFiveTraits, PersonalityWeights
from emotion import EmotionalState, EmotionActionInfluence
from needs import NeedsHierarchy, NeedActionInfluence


class PersonalityAwareExecutor(Executor):
    """Executor that weights actions by personality, emotion, and needs.

    This implements the paper's core behavioral model:
    - Personality provides stable action preferences
    - Emotion provides transient modulation
    - Needs provide goal-directed priorities
    - All three combine to create believable, consistent behavior
    """

    def __init__(self, use_llm: bool = False, template_name: str = "execute_deception"):
        """Initialize personality-aware executor.

        Args:
            use_llm: If True, use LLM for action generation. If False, use rule-based selection.
            template_name: Prompt template name for LLM mode
        """
        self.use_llm = use_llm
        self.template_name = template_name

    async def choose_action(
        self,
        agent_id: str,
        perception: AgentPerception,
        scratchpad: Scratchpad,
        *,
        plan: Optional[Plan] = None,
        plan_step: Optional[PlanStep] = None,
        context: PromptContext,
    ) -> AgentAction:
        """Choose action based on personality, emotion, and needs.

        This method demonstrates the paper's key insight: behavior = personality × emotion × needs.
        """
        # Extract behavioral state from scratchpad
        personality = self._get_personality(scratchpad)
        emotional_state = self._get_emotional_state(scratchpad)
        needs = self._get_needs(scratchpad)

        if self.use_llm:
            return await self._llm_choose_action(
                agent_id, perception, scratchpad, plan, plan_step, context,
                personality, emotional_state, needs
            )
        else:
            return self._rule_based_choose_action(
                agent_id, perception, scratchpad, plan, plan_step, context,
                personality, emotional_state, needs
            )

    async def _llm_choose_action(
        self,
        agent_id: str,
        perception: AgentPerception,
        scratchpad: Scratchpad,
        plan: Optional[Plan],
        plan_step: Optional[PlanStep],
        context: PromptContext,
        personality: BigFiveTraits,
        emotional_state: EmotionalState,
        needs: NeedsHierarchy,
    ) -> AgentAction:
        """Use LLM to generate action, augmented with personality/emotion/needs context."""
        # Build enhanced prompt with behavioral state
        enhanced_context = self._build_enhanced_context(
            context, personality, emotional_state, needs
        )

        provider = context.extra.get("llm_provider")
        model = context.extra.get("llm_model")
        if not provider or not model:
            raise ValueError(f"LLM executor requires provider and model configuration")

        # Get prompt library and render template
        library = context.extra.get("prompt_library") or PromptLibrary()
        try:
            template = library.get(self.template_name)
        except KeyError:
            # Fall back to default template
            template = self._get_default_template()

        rendered = render_prompt(template, enhanced_context, include_default=False)

        # Call LLM
        response = await call_llm_with_retries(
            system_prompt=rendered.system,
            user_prompt=rendered.user,
            llm_provider=provider,
            llm_model=model,
            response_model=AgentAction,
        )

        return response

    def _rule_based_choose_action(
        self,
        agent_id: str,
        perception: AgentPerception,
        scratchpad: Scratchpad,
        plan: Optional[Plan],
        plan_step: Optional[PlanStep],
        context: PromptContext,
        personality: BigFiveTraits,
        emotional_state: EmotionalState,
        needs: NeedsHierarchy,
    ) -> AgentAction:
        """Rule-based action selection weighted by personality/emotion/needs.

        This demonstrates how behavioral components combine WITHOUT requiring LLM:
        1. Define possible actions based on game phase
        2. Weight each action by personality, emotion, and needs
        3. Select action with highest combined weight (with some randomness)
        """
        # Get game phase from perception
        game_phase = self._get_game_phase(perception)

        # Define possible actions based on phase
        possible_actions = self._get_possible_actions(game_phase, scratchpad)

        # Weight each action
        action_weights: Dict[str, float] = {}
        for action_type in possible_actions:
            # Start with base weight of 1.0
            weight = 1.0

            # Apply personality weight
            stress_level = self._get_stress_level(perception, emotional_state)
            personality_weight = PersonalityWeights.calculate_action_weight(
                personality, action_type, {"stress_level": stress_level}
            )
            weight *= personality_weight

            # Apply emotion weight
            emotion_weight = EmotionActionInfluence.get_action_modifier(emotional_state, action_type)
            weight *= emotion_weight

            # Apply needs weight
            needs_weight = NeedActionInfluence.get_action_modifier(needs, action_type)
            weight *= needs_weight

            action_weights[action_type] = weight

        # Select action (weighted random selection for variety)
        action_type = self._weighted_random_choice(action_weights)

        # Determine target based on action type
        target = self._select_target(action_type, perception, scratchpad, personality, emotional_state)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            action_type, target, personality, emotional_state, needs
        )

        return AgentAction(
            agent_id=agent_id,
            tick=perception.tick,
            action_type=action_type,
            target=target,
            parameters={},
            reasoning=reasoning,
            communication=None,
        )

    def _get_possible_actions(self, game_phase: str, scratchpad: Scratchpad) -> List[str]:
        """Get legal actions for current game phase."""
        if game_phase == "day":
            return ["speak", "accuse", "defend", "support", "investigate", "stay_quiet"]
        elif game_phase == "voting":
            return ["vote"]
        elif game_phase == "night":
            role = scratchpad.state.get("role", "villager")
            if role == "werewolf":
                return ["kill", "discuss"]
            elif role == "seer":
                return ["investigate"]
            elif role == "doctor":
                return ["protect"]
            else:
                return ["sleep"]
        else:
            return ["wait"]

    def _weighted_random_choice(self, weights: Dict[str, float]) -> str:
        """Select action using weighted random choice."""
        if not weights:
            return "wait"

        # Normalize weights
        total = sum(weights.values())
        if total == 0:
            return random.choice(list(weights.keys()))

        # Weighted random selection
        r = random.random() * total
        cumulative = 0.0
        for action, weight in weights.items():
            cumulative += weight
            if r <= cumulative:
                return action

        return list(weights.keys())[-1]

    def _select_target(
        self,
        action_type: str,
        perception: AgentPerception,
        scratchpad: Scratchpad,
        personality: BigFiveTraits,
        emotional_state: EmotionalState,
    ) -> Optional[str]:
        """Select target for action based on context."""
        # Get list of other agents from perception
        # This is a simplified version - real implementation would parse perception
        other_agents = []  # TODO: extract from perception.visible_resources or messages

        if not other_agents:
            return None

        if action_type in ["accuse", "suspect", "vote", "kill"]:
            # Select target based on suspicion or emotion
            # For now, random selection - could be enhanced with memory/reasoning
            return random.choice(other_agents)
        elif action_type in ["support", "trust", "protect"]:
            # Select ally
            return random.choice(other_agents)
        elif action_type == "investigate":
            # Seer investigates most suspicious
            return random.choice(other_agents)

        return None

    def _generate_reasoning(
        self,
        action_type: str,
        target: Optional[str],
        personality: BigFiveTraits,
        emotional_state: EmotionalState,
        needs: NeedsHierarchy,
    ) -> str:
        """Generate human-readable reasoning for action."""
        parts = []

        # Describe emotional state
        emotion_desc = emotional_state.get_emotion_description()
        if emotion_desc != "neutral and calm":
            parts.append(f"Feeling {emotion_desc}")

        # Describe primary need
        primary_type, primary_need = needs.get_primary_need()
        if not primary_need.is_satisfied():
            parts.append(f"seeking {primary_type.value}")

        # Describe action
        if target:
            parts.append(f"choosing to {action_type} {target}")
        else:
            parts.append(f"choosing to {action_type}")

        return "; ".join(parts) if parts else f"Performing {action_type}"

    def _get_game_phase(self, perception: AgentPerception) -> str:
        """Extract game phase from perception."""
        # Look for game_phase in visible_resources
        if "game_phase" in perception.visible_resources:
            return perception.visible_resources["game_phase"].get("value", "day")
        return "day"

    def _get_stress_level(self, perception: AgentPerception, emotional_state: EmotionalState) -> float:
        """Calculate stress level from perception and emotion."""
        # High arousal + negative valence = high stress
        stress = 0.0
        if emotional_state.arousal > 0.3:
            stress += emotional_state.arousal * 0.5
        if emotional_state.valence < 0:
            stress += abs(emotional_state.valence) * 0.5
        return min(1.0, stress)

    def _get_personality(self, scratchpad: Scratchpad) -> BigFiveTraits:
        """Extract personality from scratchpad."""
        import os
        import warnings

        personality_data = scratchpad.state.get("personality", {})
        if personality_data:
            return BigFiveTraits.from_dict(personality_data)

        # Missing personality data - fail or warn based on strict mode
        error_msg = (
            "Personality data missing from scratchpad! "
            "Agent behavior will be bland (neutral defaults used). "
            "Call initialize_agent_state() to properly set up personality."
        )

        if os.getenv("STRICT_MODE", "").lower() in ("true", "1", "yes"):
            raise ValueError(error_msg)
        else:
            warnings.warn(f"⚠️  {error_msg}", UserWarning, stacklevel=2)

        # Fallback to neutral personality
        return BigFiveTraits(
            openness=50, conscientiousness=50, extraversion=50,
            agreeableness=50, neuroticism=50
        )

    def _get_emotional_state(self, scratchpad: Scratchpad) -> EmotionalState:
        """Extract emotional state from scratchpad."""
        import os
        import warnings

        emotion_data = scratchpad.state.get("emotional_state", {})
        if emotion_data:
            return EmotionalState.from_dict(emotion_data)

        # Missing emotion data - fail or warn based on strict mode
        error_msg = (
            "Emotional state missing from scratchpad! "
            "Agent will not show emotional reactions (neutral state used). "
            "Call initialize_agent_state() to properly set up emotion."
        )

        if os.getenv("STRICT_MODE", "").lower() in ("true", "1", "yes"):
            raise ValueError(error_msg)
        else:
            warnings.warn(f"⚠️  {error_msg}", UserWarning, stacklevel=2)

        return EmotionalState()

    def _get_needs(self, scratchpad: Scratchpad) -> NeedsHierarchy:
        """Extract needs from scratchpad."""
        import os
        import warnings

        needs_data = scratchpad.state.get("needs", {})
        if needs_data:
            return NeedsHierarchy.from_dict(needs_data)

        # Missing needs data - fail or warn based on strict mode
        error_msg = (
            "Needs hierarchy missing from scratchpad! "
            "Agent will not show need-driven behavior (default 50% satisfaction used). "
            "Call initialize_agent_state() to properly set up needs."
        )

        if os.getenv("STRICT_MODE", "").lower() in ("true", "1", "yes"):
            raise ValueError(error_msg)
        else:
            warnings.warn(f"⚠️  {error_msg}", UserWarning, stacklevel=2)

        return NeedsHierarchy()

    def _build_enhanced_context(
        self,
        base_context: PromptContext,
        personality: BigFiveTraits,
        emotional_state: EmotionalState,
        needs: NeedsHierarchy,
    ) -> PromptContext:
        """Build enhanced context with personality/emotion/needs."""
        # Add behavioral state to extra dict
        enhanced_extra = base_context.extra.copy()
        enhanced_extra["personality"] = personality.to_dict()
        enhanced_extra["personality_description"] = personality.describe()
        enhanced_extra["emotional_state"] = emotional_state.to_dict()
        enhanced_extra["emotion_description"] = emotional_state.get_emotion_description()
        enhanced_extra["needs"] = needs.to_dict()
        enhanced_extra["needs_summary"] = needs.get_motivation_summary()
        enhanced_extra["primary_goal"] = NeedActionInfluence.get_goal_description(needs)
        enhanced_extra["communication_style"] = PersonalityWeights.get_communication_style(personality)
        enhanced_extra["emotional_tone"] = EmotionActionInfluence.get_communication_modifier(emotional_state)

        return PromptContext(
            agent_profile=base_context.agent_profile,
            perception=base_context.perception,
            plan=base_context.plan,
            plan_step=base_context.plan_step,
            memories=base_context.memories,
            scratchpad_state=base_context.scratchpad_state,
            extra=enhanced_extra,
        )

    def _get_default_template(self) -> PromptTemplate:
        """Get default template for action selection."""
        return PromptTemplate(
            name="execute_deception",
            system=(
                "You are playing a social deception game. Choose your next action based on:\n"
                "- Your personality traits (stay in character)\n"
                "- Your current emotional state (emotions influence decisions)\n"
                "- Your primary need/motivation (what you're trying to achieve)\n"
                "- Game state and recent events\n\n"
                "Return a valid AgentAction JSON."
            ),
            user=(
                "Personality: {{personality_description}}\n"
                "Current emotion: {{emotion_description}}\n"
                "Primary goal: {{primary_goal}}\n"
                "Needs: {{needs_summary}}\n\n"
                "Game state:\n{{perception_json}}\n\n"
                "Recent memories:\n{{memories_text}}\n\n"
                "Plan:\n{{plan_json}}\n\n"
                "Choose your action as JSON following the AgentAction schema."
            ),
        )


class PersonalityAwarePlanner(Planner):
    """Planner that generates personality-consistent plans.

    Plans reflect the agent's personality, current emotional state, and primary needs.
    """

    def __init__(self, use_llm: bool = False):
        """Initialize personality-aware planner."""
        self.use_llm = use_llm

    async def generate_plan(
        self,
        agent_id: str,
        scratchpad: Scratchpad,
        *,
        world_context,
        context: PromptContext,
    ) -> Plan:
        """Generate plan based on personality and needs."""
        personality = self._get_personality(scratchpad)
        needs = self._get_needs(scratchpad)

        if self.use_llm:
            # Use LLM with personality context
            return await self._llm_generate_plan(agent_id, scratchpad, world_context, context, personality, needs)
        else:
            # Simple rule-based plan
            return self._rule_based_plan(agent_id, personality, needs, scratchpad)

    def _rule_based_plan(
        self,
        agent_id: str,
        personality: BigFiveTraits,
        needs: NeedsHierarchy,
        scratchpad: Scratchpad,
    ) -> Plan:
        """Generate simple plan based on role and needs."""
        role = scratchpad.state.get("role", "villager")
        primary_type, _ = needs.get_primary_need()

        steps = []

        # Plan based on role
        if role == "werewolf":
            steps.append(PlanStep(description="Blend in with villagers"))
            steps.append(PlanStep(description="Coordinate with other werewolves at night"))
            if personality.agreeableness < 40:
                steps.append(PlanStep(description="Deflect suspicion onto others"))
        elif role == "seer":
            steps.append(PlanStep(description="Investigate suspicious players"))
            if personality.conscientiousness > 60:
                steps.append(PlanStep(description="Share findings carefully with trusted allies"))
        else:  # villager
            steps.append(PlanStep(description="Observe player behavior for inconsistencies"))
            if personality.extraversion > 60:
                steps.append(PlanStep(description="Lead group discussions"))
            else:
                steps.append(PlanStep(description="Listen and analyze quietly"))

        # Add need-based goal
        if primary_type.value == "safety":
            steps.append(PlanStep(description="Prioritize survival and avoid risks"))
        elif primary_type.value == "belonging":
            steps.append(PlanStep(description="Build alliances and social bonds"))
        elif primary_type.value == "esteem":
            steps.append(PlanStep(description="Demonstrate competence and leadership"))

        return Plan(steps=steps)

    async def _llm_generate_plan(
        self,
        agent_id: str,
        scratchpad: Scratchpad,
        world_context,
        context: PromptContext,
        personality: BigFiveTraits,
        needs: NeedsHierarchy,
    ) -> Plan:
        """Use LLM to generate personality-consistent plan."""
        # Enhance context with personality/needs
        enhanced_context = context
        enhanced_context.extra["personality_description"] = personality.describe()
        enhanced_context.extra["primary_goal"] = NeedActionInfluence.get_goal_description(needs)

        # Use default LLM planner logic
        # (This would call call_llm_with_retries with enhanced context)
        provider = context.extra.get("llm_provider")
        model = context.extra.get("llm_model")

        if not provider or not model:
            # Fall back to rule-based
            return self._rule_based_plan(agent_id, personality, needs, scratchpad)

        # Build prompt
        prompt = (
            f"You are {context.agent_profile.name}, a {scratchpad.state.get('role', 'villager')} "
            f"in a social deception game.\n\n"
            f"Personality: {personality.describe()}\n"
            f"Current goal: {NeedActionInfluence.get_goal_description(needs)}\n\n"
            f"Generate a plan that reflects your personality and goals. "
            f"Return as JSON with 'steps' list."
        )

        # This is simplified - real implementation would use proper prompt templates
        return self._rule_based_plan(agent_id, personality, needs, scratchpad)

    def _get_personality(self, scratchpad: Scratchpad) -> BigFiveTraits:
        """Extract personality from scratchpad."""
        import os
        import warnings

        personality_data = scratchpad.state.get("personality", {})
        if personality_data:
            return BigFiveTraits.from_dict(personality_data)

        # Missing personality data - fail or warn based on strict mode
        error_msg = (
            "Personality data missing from scratchpad (Planner)! "
            "Plans will not reflect personality traits. "
            "Call initialize_agent_state() to properly set up personality."
        )

        if os.getenv("STRICT_MODE", "").lower() in ("true", "1", "yes"):
            raise ValueError(error_msg)
        else:
            warnings.warn(f"⚠️  {error_msg}", UserWarning, stacklevel=2)

        return BigFiveTraits(
            openness=50, conscientiousness=50, extraversion=50,
            agreeableness=50, neuroticism=50
        )

    def _get_needs(self, scratchpad: Scratchpad) -> NeedsHierarchy:
        """Extract needs from scratchpad."""
        import os
        import warnings

        needs_data = scratchpad.state.get("needs", {})
        if needs_data:
            return NeedsHierarchy.from_dict(needs_data)

        # Missing needs data - fail or warn based on strict mode
        error_msg = (
            "Needs hierarchy missing from scratchpad (Planner)! "
            "Plans will not be goal-directed. "
            "Call initialize_agent_state() to properly set up needs."
        )

        if os.getenv("STRICT_MODE", "").lower() in ("true", "1", "yes"):
            raise ValueError(error_msg)
        else:
            warnings.warn(f"⚠️  {error_msg}", UserWarning, stacklevel=2)

        return NeedsHierarchy()
