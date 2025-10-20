"""
Snake Game with LLM Agent

A simple snake game where an LLM-powered agent learns to play by observing
the grid and making movement decisions.

Features:
- 2D grid environment with collision detection
- Snake grows when eating food
- LLM agent sees grid state and decides direction
- Game ends on wall collision or self-collision

Run: uv run python examples/snake.py
"""

import asyncio
import os
import random
from datetime import datetime, timezone
from typing import List, Tuple

from miniverse import (
    Orchestrator, AgentProfile, AgentStatus, WorldState,
    ResourceState, EnvironmentState, SimulationRules, Stat
)
from miniverse.cognition import AgentCognition, LLMExecutor, Scratchpad
from miniverse.cognition.planner import Plan, Planner
from miniverse.environment import EnvironmentGridState, GridTileState
from miniverse.persistence import InMemoryPersistence
from miniverse.memory import SimpleMemoryStream
from miniverse.schemas import AgentAction, AgentPerception

# Grid dimensions
GRID_WIDTH = 20
GRID_HEIGHT = 20

# Directions
DIRECTIONS = {
    'up': (0, 1),
    'down': (0, -1),
    'left': (-1, 0),
    'right': (1, 0)
}


class SnakeRules(SimulationRules):
    """Snake game physics - movement, collision, food spawning."""

    def __init__(self):
        self.snake_body: List[Tuple[int, int]] = [(10, 10)]  # Start in middle
        self.direction = 'right'
        self.food_pos: Tuple[int, int] = self._spawn_food()
        self.score = 0
        self.game_over = False

    def _spawn_food(self) -> Tuple[int, int]:
        """Spawn food at random empty location."""
        while True:
            x = random.randint(1, GRID_WIDTH - 2)
            y = random.randint(1, GRID_HEIGHT - 2)
            if (x, y) not in self.snake_body:
                return (x, y)

    def apply_tick(self, state, tick):
        """Update game state - move snake, check collisions, handle food."""
        if self.game_over:
            return state

        updated = state.model_copy(deep=True)

        # Get head position
        head_x, head_y = self.snake_body[0]

        # Calculate new head position
        dx, dy = DIRECTIONS[self.direction]
        new_head = (head_x + dx, head_y + dy)

        # Check wall collision
        if (new_head[0] <= 0 or new_head[0] >= GRID_WIDTH - 1 or
            new_head[1] <= 0 or new_head[1] >= GRID_HEIGHT - 1):
            self.game_over = True
            updated.resources.get_metric('game_status').value = 'game_over'
            return updated

        # Check self collision
        if new_head in self.snake_body:
            self.game_over = True
            updated.resources.get_metric('game_status').value = 'game_over'
            return updated

        # Move snake
        self.snake_body.insert(0, new_head)

        # Check food collision
        if new_head == self.food_pos:
            self.score += 1
            self.food_pos = self._spawn_food()
            updated.resources.get_metric('score').value = self.score
        else:
            # Remove tail (snake doesn't grow)
            self.snake_body.pop()

        # Update grid in state
        grid = self._build_grid()
        updated.environment_grid = grid

        # Update snake position in agent status
        snake_agent = next(a for a in updated.agents if a.agent_id == 'snake')
        snake_agent.grid_position = list(new_head)

        updated.tick = tick
        return updated

    def _build_grid(self) -> EnvironmentGridState:
        """Build grid with current snake and food positions."""
        grid = EnvironmentGridState(width=GRID_WIDTH, height=GRID_HEIGHT)

        # Add walls
        for x in range(GRID_WIDTH):
            grid.tiles[(x, 0)] = GridTileState(game_object='wall', collision=True)
            grid.tiles[(x, GRID_HEIGHT-1)] = GridTileState(game_object='wall', collision=True)
        for y in range(GRID_HEIGHT):
            grid.tiles[(0, y)] = GridTileState(game_object='wall', collision=True)
            grid.tiles[(GRID_WIDTH-1, y)] = GridTileState(game_object='wall', collision=True)

        # Add snake body
        for i, (x, y) in enumerate(self.snake_body):
            obj = 'snake_head' if i == 0 else 'snake_body'
            grid.tiles[(x, y)] = GridTileState(game_object=obj, collision=True)

        # Add food
        grid.tiles[self.food_pos] = GridTileState(game_object='food', collision=False)

        return grid

    def validate_action(self, action, state):
        """Validate movement action."""
        if action.action_type == 'move':
            direction = action.parameters.get('direction')
            return direction in DIRECTIONS
        return True

    def process_action(self, action):
        """Process movement action - change direction."""
        if action.action_type == 'move':
            new_dir = action.parameters.get('direction')

            # Can't reverse direction
            opposite = {
                'up': 'down', 'down': 'up',
                'left': 'right', 'right': 'left'
            }
            if new_dir != opposite.get(self.direction):
                self.direction = new_dir


class NoOpPlanner(Planner):
    """Snake doesn't need planning - just reacts."""
    async def generate_plan(self, agent_id, scratchpad, *, world_context, context):
        return Plan(steps=[], metadata={})


def visualize_grid(rules: SnakeRules):
    """Print ASCII visualization of game state."""
    import sys
    print('\n' + '=' * (GRID_WIDTH * 2 + 4), flush=True)
    for y in range(GRID_HEIGHT - 1, -1, -1):
        row = ''
        for x in range(GRID_WIDTH):
            if (x, y) == rules.snake_body[0]:
                row += 'O '  # Head
            elif (x, y) in rules.snake_body:
                row += 'o '  # Body
            elif (x, y) == rules.food_pos:
                row += '* '  # Food
            elif x == 0 or x == GRID_WIDTH-1 or y == 0 or y == GRID_HEIGHT-1:
                row += '# '  # Wall
            else:
                row += '. '  # Empty
        print(row, flush=True)
    print('=' * (GRID_WIDTH * 2 + 4), flush=True)
    print(f'Score: {rules.score} | Direction: {rules.direction} | Game Over: {rules.game_over}\n', flush=True)


async def main():
    """Run snake game with LLM agent."""

    # Initialize
    persistence = InMemoryPersistence()
    await persistence.initialize()

    memory = SimpleMemoryStream(persistence)

    rules = SnakeRules()

    # Agent profile
    agents = {
        'snake': AgentProfile(
            agent_id='snake',
            name='Snake AI',
            background='LLM-powered snake that learns to survive',
            role='player',
            personality='cautious, strategic',
            skills={'pattern_recognition': 'learning'},
            goals=['Eat food', 'Avoid walls', 'Avoid self-collision', 'Maximize score'],
            relationships={}
        )
    }

    # World state
    initial_grid = rules._build_grid()
    world_state = WorldState(
        tick=0,
        timestamp=datetime.now(timezone.utc),
        environment=EnvironmentState(metrics={}),
        resources=ResourceState(metrics={
            'score': Stat(value=0, unit='points', label='Score'),
            'game_status': Stat(value='playing', unit='', label='Status')
        }),
        agents=[
            AgentStatus(
                agent_id='snake',
                display_name='Snake AI',
                grid_position=list(rules.snake_body[0])
            )
        ],
        environment_grid=initial_grid
    )

    # Agent prompt - directive instructions for snake AI
    agent_prompts = {
        'snake': '''[SNAKE_AI_MARKER_XYZ123] You are a snake. Your ONLY action is "move" with direction parameter.

RULES:
- O = your head, o = body, * = food, # = wall
- Eat food to grow and score
- Hit wall or body = game over

REQUIRED ACTION FORMAT:
{
  "action_type": "move",
  "parameters": {"direction": "up|down|left|right"}
}

DO NOT use any other action type. ONLY move actions.'''
    }

    # Cognition - use standard LLMExecutor (agent_prompts now properly injected!)
    cognition_map = {
        'snake': AgentCognition(
            executor=LLMExecutor(template_name='default'),
            planner=NoOpPlanner(),
            scratchpad=Scratchpad()
        )
    }

    # LLM config
    provider = os.getenv('LLM_PROVIDER', 'openai')
    model = os.getenv('LLM_MODEL', 'gpt-5-nano')

    orchestrator = Orchestrator(
        world_state=world_state,
        agents=agents,
        world_prompt='',
        agent_prompts=agent_prompts,
        simulation_rules=rules,
        agent_cognition=cognition_map,
        llm_provider=provider,
        llm_model=model,
        persistence=persistence,
        memory=memory,
        world_update_mode='deterministic'  # Physics only, no LLM for world updates
    )

    print('🐍 SNAKE GAME - LLM Edition', flush=True)
    print(f'Using: {provider}/{model}', flush=True)
    print('\nPress Ctrl+C to stop\n', flush=True)

    # Game loop
    tick = 0
    max_ticks = 100

    while tick < max_ticks and not rules.game_over:
        tick += 1

        # Visualize current state
        visualize_grid(rules)

        # Get LLM decision
        result = await orchestrator.run(num_ticks=1)

        # Process any actions
        actions = result.get('actions', [])
        if actions and actions[0]:
            action = actions[0][0]  # First action from snake
            rules.process_action(action)
            print(f'🎮 Action: {action.action_type} {action.parameters}', flush=True)

        # Small delay for readability
        await asyncio.sleep(0.5)

    # Final state
    visualize_grid(rules)

    if rules.game_over:
        print('💀 GAME OVER!')
    else:
        print('⏱️  TIME UP!')

    print(f'\n📊 Final Score: {rules.score}')

    await persistence.close()


if __name__ == '__main__':
    asyncio.run(main())
