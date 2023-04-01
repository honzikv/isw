import pickle
from collections import defaultdict

import numpy as np

from pathlib import Path

from agent.qlearning_agent.qlearning_agent import QLearningAgent


def save(file: str, qlearning_agent: QLearningAgent):
    q_table = qlearning_agent._q_table
    timesteps = qlearning_agent._timesteps
    episode_timesteps = qlearning_agent._episode_timesteps
    successful_episodes = qlearning_agent._successful_episodes
    eps = qlearning_agent._eps
    config = qlearning_agent.config
    gamma = config.gamma
    k = qlearning_agent._k

    with open(file, 'wb') as file:
        pickle.dump(file=file, obj={
            'q_table': dict(q_table),
            'timesteps': timesteps,
            'episode_timesteps': episode_timesteps,
            'successful_episodes': successful_episodes,
            'eps': eps,
            'gamma': gamma,
            'k': k,
        })


def load(file: Path, agent: QLearningAgent):
    with open(file, 'rb') as file:
        state_dict = pickle.load(file=file)

    agent._q_table = defaultdict(lambda: np.zeros(agent.env.num_actions), state_dict['q_table'])
    agent._timesteps = state_dict['timesteps']
    agent._episode_timesteps = state_dict['episode_timesteps']
    agent._successful_episodes = state_dict['successful_episodes']
    agent._eps = state_dict['eps']
    agent._gamma = state_dict['gamma']
    agent._k = state_dict['k']

    return agent
