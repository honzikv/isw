from dataclasses import dataclass
from pathlib import Path

import pickle

from agent.qlearning_agent import QLearningAgent


# This is very stupid but I cba and copilot generated this fast

class QLearningSerializer:
    def __init__(self, agent: QLearningAgent):
        self._q_table = agent._q_table
        self._timesteps = agent._timesteps
        self._episode_timesteps = agent._episode_timesteps
        self._config = agent._config
        self._successful_episodes = agent._successful_episodes
        self._eval = agent._eval
        self._k = agent._k
        self._eps = agent._eps

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, environment, path: str) -> QLearningAgent:
        with open(path, 'rb') as f:
            agent_pickle = pickle.load(f)

        agent = QLearningAgent(environment, agent_pickle._config)
        agent._q_table = agent_pickle._q_table
        agent._timesteps = agent_pickle._timesteps
        agent._episode_timesteps = agent_pickle._episode_timesteps
        agent._successful_episodes = agent_pickle._successful_episodes
        agent._eval = agent_pickle._eval
        agent._k = agent_pickle._k
        agent._eps = agent_pickle._eps

        return agent
