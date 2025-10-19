"""Valentine's Day Party - Test Script

Run the Stanford Valentine's party scenario to verify information diffusion works.
This tests the recipient memory fix.

Usage:
    DEBUG_MEMORY=true python valentines_party_test.py
"""

import os
import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from uuid import UUID

# Enable debugging
os.environ['DEBUG_MEMORY'] = 'true'
os.environ['DEBUG_PERCEPTION'] = 'true'
os.environ['DEBUG_LLM'] = 'true'  # Show full LLM prompts/responses
os.environ['MINIVERSE_VERBOSE'] = 'true'

# Miniverse core
from miniverse import (
    Orchestrator, AgentProfile, AgentStatus, WorldState,
    ResourceState, EnvironmentState, SimulationRules,
    Stat, AgentAction
)
from miniverse.cognition import AgentCognition, LLMExecutor, LLMPlanner, Scratchpad
from miniverse.memory import SimpleMemoryStream
from miniverse.schemas import AgentMemory
from miniverse.persistence import InMemoryPersistence


class TownSimulationRules(SimulationRules):
    """Simple town: agents move between locations, time passes."""

    def apply_tick(self, state, tick):
        updated = state.model_copy(deep=True)

        # Update time resource
        hour = updated.resources.get_metric('hour', default=9, unit='am')
        hour.value = 9 + tick  # Each tick is 1 hour

        updated.tick = tick
        return updated

    def validate_action(self, action, state):
        return True


async def main():
    print('🎭 Valentine\'s Day Party - Information Diffusion Test')
    print('=' * 60)

    # LLM config
    provider = os.getenv('LLM_PROVIDER', 'openai')
    model = os.getenv('LLM_MODEL', 'gpt-4o-mini')
    print(f'\n🤖 LLM: {provider}/{model}')

    # Initialize world state
    world_state = WorldState(
        tick=0,
        timestamp=datetime.now(timezone.utc),
        environment=EnvironmentState(metrics={}),
        resources=ResourceState(metrics={
            'hour': Stat(value=9, unit='am', label='Current Time'),
            'day': Stat(value=13, unit='Feb', label='Date')
        }),
        agents=[
            AgentStatus(agent_id='isabella', location='hobbs_cafe', display_name='Isabella Rodriguez'),
            AgentStatus(agent_id='maria', location='park', display_name='Maria Lopez'),
            AgentStatus(agent_id='klaus', location='park', display_name='Klaus Mueller'),
            AgentStatus(agent_id='ayesha', location='hobbs_cafe', display_name='Ayesha Khan'),
            AgentStatus(agent_id='tom', location='park', display_name='Tom Moreno'),
        ]
    )

    print('\n📍 Initial locations:')
    for agent in world_state.agents:
        print(f'   • {agent.display_name}: {agent.location}')

    # Create agent profiles (Stanford-style: rich personalities, minimal task instructions)
    agents = {
        'isabella': AgentProfile(
            agent_id='isabella',
            name='Isabella Rodriguez',
            age=28,
            background='I grew up in a small town where everyone knew each other. When I moved here, I wanted to recreate that sense of community, so I opened Hobbs Cafe. I love hosting events and seeing people connect. My cafe is like my living room - I want everyone to feel welcome.',
            role='cafe_owner',
            personality='Warm, outgoing, and naturally social. I thrive on bringing people together and love organizing gatherings. I pay attention to what people enjoy and try to create experiences they\'ll remember. I\'m a natural host and take pride in making others feel included.',
            skills={'hospitality': 'expert', 'event_planning': 'expert', 'cooking': 'advanced'},
            goals=['Build a thriving community hub', 'Make the cafe financially sustainable', 'Host a Valentine\'s Day party at the cafe on Feb 14, 5-7pm'],
            relationships={
                'maria': 'Maria is one of my closest friends. We met when she started coming to the cafe to study. She\'s sweet and a bit shy, but I can tell she has a crush on Klaus.',
                'ayesha': 'Ayesha comes in regularly for her morning coffee. She\'s a local journalist and we chat about community events. I appreciate how she spotlights local businesses.',
                'klaus': 'Klaus comes in occasionally. He\'s talented but introverted. I think he and Maria would be good together.',
                'tom': 'Tom runs a shop nearby. We support each other\'s businesses and chat when he stops by.'
            }
        ),
        'maria': AgentProfile(
            agent_id='maria',
            name='Maria Lopez',
            age=26,
            background='I\'m a graduate student working on my thesis about social networks in urban communities. I spend a lot of time at Hobbs Cafe because it\'s quiet and Isabella is so welcoming. I\'m shy around people I\'m attracted to, which makes it hard to talk to Klaus even though we were friends in college.',
            role='student',
            personality='Thoughtful and academically focused, but also romantic and a bit dreamy. I overthink social situations and worry about saying the wrong thing. I\'m loyal to my friends and genuinely interested in people\'s lives. I get excited about social events but nervous about actually attending.',
            skills={'research': 'expert', 'writing': 'advanced', 'active_listening': 'advanced'},
            goals=['Finish thesis by spring', 'Maintain close friendships', 'Work up courage to spend more time with Klaus'],
            relationships={
                'isabella': 'Isabella is like a big sister. She gives great advice and I trust her completely. She knows I like Klaus.',
                'klaus': 'We were friends in college but lost touch. I\'ve always been attracted to him but never said anything. Now that we\'re in the same town again, I don\'t know how to reconnect without being awkward.',
                'ayesha': 'We\'ve chatted a few times at the cafe. She seems interesting.',
                'tom': 'I\'ve seen him around but we haven\'t really talked.'
            }
        ),
        'klaus': AgentProfile(
            agent_id='klaus',
            name='Klaus Mueller',
            age=27,
            background='I moved here after music school to focus on composing. I make money playing local gigs but my real passion is original composition. I spend a lot of time alone working on music, which means I don\'t socialize as much as I probably should. I\'m not great at picking up on social cues.',
            role='musician',
            personality='Creative and introspective. I get lost in my work and lose track of time. I\'m comfortable in small groups but feel awkward at large gatherings. I appreciate when people are direct with me because I\'m not great at reading between the lines. Music is how I express myself best.',
            skills={'music_composition': 'expert', 'piano': 'expert', 'guitar': 'advanced'},
            goals=['Finish composing a full album by summer', 'Build a local following', 'Make enough from music to live on'],
            relationships={
                'maria': 'We were friends in college. She was always easy to talk to and genuinely interested in my music. I haven\'t seen her much since we both moved here.',
                'isabella': 'She runs the cafe I sometimes visit. She\'s friendly but very extroverted, which can be overwhelming.',
                'ayesha': 'I don\'t know her well.',
                'tom': 'I don\'t know him.'
            }
        ),
        'ayesha': AgentProfile(
            agent_id='ayesha',
            name='Ayesha Khan',
            age=30,
            background='I\'m a journalist covering local community stories. I moved here three years ago and fell in love with the neighborhood\'s character. I believe local journalism matters - it\'s how communities stay connected and informed. I\'m always looking for stories about people making a difference.',
            role='journalist',
            personality='Curious and observant. I notice details others miss and I\'m good at asking questions that get people talking. I\'m professional but genuinely care about the communities I cover. I have a knack for being in the right place at the right time. I naturally spread information - it\'s my job and my nature.',
            skills={'journalism': 'expert', 'writing': 'expert', 'interviewing': 'expert', 'networking': 'advanced'},
            goals=['Cover meaningful local stories', 'Build trust in the community', 'Eventually start my own local news publication'],
            relationships={
                'isabella': 'Isabella is a great source for community stories. Her cafe is a hub and she knows everyone. We have a good rapport.',
                'maria': 'I\'ve seen her at the cafe. She seems nice but quiet.',
                'klaus': 'I know he\'s a local musician but haven\'t interviewed him yet.',
                'tom': 'Another local business owner. I should do a piece on small businesses in the neighborhood.'
            }
        ),
        'tom': AgentProfile(
            agent_id='tom',
            name='Tom Moreno',
            age=32,
            background='I run a hardware store my grandfather started. Business is steady but not exciting - mostly regulars and contractors. I spend most of my time working and don\'t have much of a social life. I appreciate the other local business owners because we look out for each other.',
            role='shopkeeper',
            personality='Practical and straightforward. I\'m not big on small talk but I\'m reliable and helpful when people need something. I tend to observe rather than insert myself into situations. I respect hard work and value consistency over excitement.',
            skills={'business': 'advanced', 'hardware_expertise': 'expert', 'practical_problem_solving': 'expert'},
            goals=['Keep the family business running', 'Support other local businesses', 'Maybe eventually expand the shop'],
            relationships={
                'isabella': 'We support each other\'s businesses. She sends people my way when they need tools or supplies.',
                'maria': 'I\'ve seen her at the cafe but don\'t really know her.',
                'klaus': 'Don\'t know him.',
                'ayesha': 'She interviewed me once about local businesses. Nice enough.'
            }
        )
    }

    print('\n🎯 Isabella\'s key goal: "Plan Valentine\'s Day party at Hobbs Cafe on Feb 14, 5-7pm"')

    # Create shared persistence and memory
    persistence = InMemoryPersistence()
    await persistence.initialize()

    memory = SimpleMemoryStream(persistence)

    # Configure cognition for each agent
    cognition_map = {
        agent_id: AgentCognition(
            executor=LLMExecutor(template_name="default"),
            planner=LLMPlanner(),
            scratchpad=Scratchpad()
        )
        for agent_id in agents.keys()
    }

    # Agent prompts - backstory and current situation (not explicit instructions)
    agent_prompts = {
        'isabella': '''It's February 13th. You've been excited about hosting a Valentine's Day party at the cafe tomorrow evening (Feb 14, 5-7pm). You want it to feel warm and inclusive - not just for couples, but for anyone who wants to celebrate community and connection. You've been thinking about who might enjoy coming.''',
        'maria': '''You've been spending long days at the cafe working on your thesis. It's nice to have a regular spot where you feel comfortable. Isabella has become a good friend - she has a way of making you feel welcome and noticed.''',
        'klaus': '''You've been in a creative flow lately, working on new compositions. Sometimes you forget to eat or see people for days. You do miss having friends around occasionally, especially people who understand what you're trying to do with your music.''',
        'ayesha': '''You're always on the lookout for community stories - the kinds of things that bring neighborhoods together. Local events, new businesses, people connecting - that's what makes good local journalism. You've found that being genuinely curious and sharing what you learn helps you build trust.''',
        'tom': '''Business has been steady. You keep regular hours and take pride in being reliable. You don't go out of your way to socialize, but you appreciate the other local business owners and try to support them when you can.'''
    }

    # Create orchestrator
    orchestrator = Orchestrator(
        world_state=world_state,
        agents=agents,
        world_prompt='This is smallville, a quiet town with a few local businesses and a community cafe called Hobbs Cafe.',
        agent_prompts=agent_prompts,
        simulation_rules=TownSimulationRules(),
        agent_cognition=cognition_map,
        llm_provider=provider,
        llm_model=model,
        persistence=persistence,
        memory=memory
    )

    # Run simulation
    result = await orchestrator.run(num_ticks=8)

    print('=' * 60)
    print('\n✅ Simulation complete!')
    print(f'   Run ID: {result["run_id"]}')

    # Analyze memories
    print('\n=== MEMORY ANALYSIS ===')
    run_id = result['run_id']

    for agent_id, profile in agents.items():
        memories = await persistence.get_recent_memories(run_id, agent_id, limit=20)
        print(f'\n{profile.name} ({len(memories)} memories):')

        party_aware = False
        for mem in memories:
            print(f'  [Tick {mem.tick}] {mem.content}')
            if 'party' in mem.content.lower() or 'valentine' in mem.content.lower():
                party_aware = True

        if party_aware:
            print(f'  ✅ Knows about party!')
        else:
            print(f'  ❌ No party awareness')

    # Check attendance
    print('\n=== PARTY ATTENDANCE ===')
    final_state = result['final_state']

    attendees = []
    for agent in final_state.agents:
        profile = agents[agent.agent_id]
        at_party = agent.location == 'hobbs_cafe'
        status = '🎉' if at_party else '  '

        print(f'{status} {profile.name}: {agent.location}')
        if at_party:
            attendees.append(profile.name)

    print(f'\n📊 Attendance: {len(attendees)}/{len(agents)} agents at Hobbs Cafe')
    print(f'   Stanford paper: 5/12 invited showed up (42%)')
    print(f'   Our result: {len(attendees)}/{len(agents)} showed up ({len(attendees)/len(agents)*100:.0f}%)')

    if attendees:
        print(f'\n   Party guests: {", ".join(attendees)}')


if __name__ == '__main__':
    asyncio.run(main())
