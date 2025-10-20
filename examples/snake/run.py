"""
Snake Game with LLM Agent

Simple snake game where an LLM decides which direction to move.
The snake sees an ASCII grid and chooses: up, down, left, or right.

Run: uv run python examples/snake/run.py
"""

import asyncio
import os
import random
from datetime import datetime, timezone
from typing import List, Tuple

from miniverse import (
    Orchestrator,
    AgentProfile,
    AgentStatus,
    WorldState,
    ResourceState,
    EnvironmentState,
    SimulationRules,
    Stat,
    EnvironmentGridState,
    GridTileState,
)
from miniverse.cognition import AgentCognition, LLMExecutor
from miniverse.persistence import InMemoryPersistence
from miniverse.memory import SimpleMemoryStream
from miniverse.schemas import AgentAction

# Smaller grid for better LLM comprehension
GRID_WIDTH = 12
GRID_HEIGHT = 12
VISIBILITY_RADIUS = 3

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
        self.snake_body: List[Tuple[int, int]] = [(6, 6)]  # Start in middle
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

    def _build_grid(self) -> EnvironmentGridState:
        """Build sparse grid representation with walls, snake, and food."""
        grid = EnvironmentGridState(width=GRID_WIDTH, height=GRID_HEIGHT)

        # Border walls
        for x in range(GRID_WIDTH):
            grid.tiles[(x, 0)] = GridTileState(game_object='wall', collision=True)
            grid.tiles[(x, GRID_HEIGHT - 1)] = GridTileState(game_object='wall', collision=True)
        for y in range(GRID_HEIGHT):
            grid.tiles[(0, y)] = GridTileState(game_object='wall', collision=True)
            grid.tiles[(GRID_WIDTH - 1, y)] = GridTileState(game_object='wall', collision=True)

        # Snake body (head first)
        for idx, (x, y) in enumerate(self.snake_body):
            obj = 'snake_head' if idx == 0 else 'snake_body'
            grid.tiles[(x, y)] = GridTileState(game_object=obj, collision=True)

        # Food
        fx, fy = self.food_pos
        grid.tiles[(fx, fy)] = GridTileState(game_object='food', collision=False)

        return grid

    def apply_tick(self, state, tick):
        """Update world state (snake movement happens in process_action)."""
        if self.game_over:
            return state

        updated = state.model_copy(deep=True)
        updated.tick = tick
        return updated

    def validate_action(self, action, state):
        """Validate movement action."""
        if action.action_type == 'move':
            params = action.parameters or {}
            direction = params.get('direction')
            return direction in DIRECTIONS
        return True

    def process_actions(self, state, actions, tick):
        """Process agent actions deterministically - move snake based on LLM decision."""
        visibility_radius = VISIBILITY_RADIUS

        if self.game_over or not actions:
            updated = state.model_copy(deep=True)
            updated.environment_grid = self._build_grid()
            updated.metadata['grid_visibility_radius'] = visibility_radius
            if updated.agents:
                updated.agents[0].grid_position = list(self.snake_body[0])
                updated.agents[0].metadata['grid_visibility_radius'] = visibility_radius
            return updated

        # Process first action (only one snake)
        action = actions[0]
        if action.action_type != 'move':
            updated = state.model_copy(deep=True)
            updated.environment_grid = self._build_grid()
            updated.metadata['grid_visibility_radius'] = visibility_radius
            if updated.agents:
                updated.agents[0].grid_position = list(self.snake_body[0])
                updated.agents[0].metadata['grid_visibility_radius'] = visibility_radius
            return updated

        params = action.parameters or {}
        new_dir = params.get('direction')
        if new_dir not in DIRECTIONS:
            updated = state.model_copy(deep=True)
            updated.environment_grid = self._build_grid()
            updated.metadata['grid_visibility_radius'] = visibility_radius
            if updated.agents:
                updated.agents[0].grid_position = list(self.snake_body[0])
                updated.agents[0].metadata['grid_visibility_radius'] = visibility_radius
            return updated

        self.direction = new_dir

        # Calculate new head position
        head_x, head_y = self.snake_body[0]
        dx, dy = DIRECTIONS[self.direction]
        new_head = (head_x + dx, head_y + dy)

        # Check wall collision
        if (new_head[0] <= 0 or new_head[0] >= GRID_WIDTH - 1 or
            new_head[1] <= 0 or new_head[1] >= GRID_HEIGHT - 1):
            self.game_over = True
            updated = state.model_copy(deep=True)
            updated.resources.get_metric('game_status').value = 'game_over'
            updated.environment_grid = self._build_grid()
            updated.metadata['grid_visibility_radius'] = visibility_radius
            if updated.agents:
                updated.agents[0].grid_position = list(self.snake_body[0])
                updated.agents[0].metadata['grid_visibility_radius'] = visibility_radius
            return updated

        # Check self collision
        if new_head in self.snake_body:
            self.game_over = True
            updated = state.model_copy(deep=True)
            updated.resources.get_metric('game_status').value = 'game_over'
            updated.environment_grid = self._build_grid()
            updated.metadata['grid_visibility_radius'] = visibility_radius
            if updated.agents:
                updated.agents[0].grid_position = list(self.snake_body[0])
                updated.agents[0].metadata['grid_visibility_radius'] = visibility_radius
            return updated

        # Move snake
        self.snake_body.insert(0, new_head)

        # Check food collision
        if new_head == self.food_pos:
            self.score += 1
            self.food_pos = self._spawn_food()
        else:
            # Remove tail (snake doesn't grow unless eating)
            self.snake_body.pop()

        # Update world state with new snake position and grid snapshot
        updated = state.model_copy(deep=True)
        updated.resources.get_metric('score').value = self.score
        updated.environment_grid = self._build_grid()
        updated.metadata['grid_visibility_radius'] = visibility_radius

        if updated.agents:
            updated.agents[0].grid_position = list(new_head)
            updated.agents[0].metadata['grid_visibility_radius'] = visibility_radius

        return updated


def render_grid(rules: SnakeRules) -> str:
    """Render ASCII grid with clearer symbols (2x width for readability)."""
    lines = []

    # Top border (double width)
    lines.append('██' * GRID_WIDTH)

    # Grid rows (reversed so (0,0) is bottom-left)
    for y in range(GRID_HEIGHT - 1, -1, -1):
        row = ''
        for x in range(GRID_WIDTH):
            pos = (x, y)

            # Border walls (double width)
            if x == 0 or x == GRID_WIDTH - 1 or y == 0 or y == GRID_HEIGHT - 1:
                row += '██'
            # Snake head (with space)
            elif pos == rules.snake_body[0]:
                row += '● '
            # Snake body (with space)
            elif pos in rules.snake_body:
                row += 'o '
            # Food (with space)
            elif pos == rules.food_pos:
                row += '★ '
            # Empty (double space)
            else:
                row += '  '

        lines.append(row)

    # Bottom border (double width)
    lines.append('██' * GRID_WIDTH)

    return '\n'.join(lines)


async def main():
    # Initialize simulation
    rules = SnakeRules()

    # World state
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
                location=None,
                grid_position=list(rules.snake_body[0]),
                metadata={'grid_visibility_radius': VISIBILITY_RADIUS}
            )
        ],
        environment_grid=rules._build_grid(),
        metadata={'grid_visibility_radius': VISIBILITY_RADIUS}
    )

    # Agent profile
    agents = {
        'snake': AgentProfile(
            agent_id='snake',
            name='Snake AI',
            role='player',
            background='I am a snake learning to play the game.',
            personality='Strategic, cautious, food-seeking',
            skills={'navigation': 'Expert at spatial reasoning and pathfinding'},
            goals=['Eat food (★)', 'Avoid walls (█)', 'Avoid my body (o)', 'Get high score'],
            relationships={}
        )
    }

    # Simple prompt - just show the grid and let LLM decide
    agent_prompts = {
        'snake': (
            "You are a snake in a grid world. Each perception includes your `grid_position`, a structured "
            "`grid_visibility.tiles` window, and a quick `grid_ascii` rendering of nearby tiles. Use those to "
            "choose a safe direction (`up`, `down`, `left`, or `right`) that moves toward the food while avoiding "
            "walls (`game_object='wall'`) and your own body (`snake_body`). Always return an action with "
            "`action_type='move'` and `parameters={\"direction\": <direction>}` and explain your reasoning."
        )
    }

    # Simple action catalog - just move with direction
    available_actions = [
        {
            "name": "move",
            "action_type": "move",
            "description": "Move snake in a direction",
            "schema": {
                "action_type": "move",
                "parameters": {"direction": "up|down|left|right"}
            }
        }
    ]

    cognition_map = {
        'snake': AgentCognition(
            executor=LLMExecutor(template_name='default', available_actions=available_actions)
        )
    }

    # LLM config
    provider = os.getenv('LLM_PROVIDER', 'openai')
    model = os.getenv('LLM_MODEL', 'gpt-5-nano')

    # Memory and persistence
    persistence = InMemoryPersistence()
    await persistence.initialize()
    memory = SimpleMemoryStream(persistence)

    # Orchestrator
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
        world_update_mode='deterministic'
    )

    print('SNAKE GAME - LLM Edition')
    print(f'Using: {provider}/{model}')
    print(f'Grid: {GRID_WIDTH}x{GRID_HEIGHT}')
    print('\nPress Ctrl+C to stop\n')

    max_ticks = 50

    # Show initial state
    print(f'\n{"="*40}')
    print(f'Tick 0 | Score: {rules.score}')
    print(render_grid(rules))

    game_over_announced = False

    def print_tick(tick: int, prev_state, new_state, actions):
        nonlocal game_over_announced
        if game_over_announced:
            return

        print(f'\n{"="*40}')
        print(f'Tick {tick} | Score: {rules.score}')
        print(render_grid(rules))

        if actions:
            action = actions[0]
            direction = (action.parameters or {}).get('direction')
            if direction:
                print(f'Action: move {direction}  |  Reason: {action.reasoning}')

        if rules.game_over:
            print('GAME OVER!')
            game_over_announced = True

    orchestrator.tick_listeners.append(print_tick)

    await orchestrator.run(num_ticks=max_ticks)

    print(f'\n{"="*40}')
    if rules.game_over:
        print('GAME OVER!')
    else:
        print('Time limit reached!')

    print(f'\nFinal Score: {rules.score}')
    print(f'Snake Length: {len(rules.snake_body)}')


if __name__ == '__main__':
    asyncio.run(main())
