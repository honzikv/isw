from src.agent import Agent
from torch import nn

from collections import deque, namedtuple

import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *transition):
        self.memory.append(Transition(*transition))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DeepQLearningNet(nn.Module):

    def __init__(self, state_shape, num_actions):
        super().__init__()
        self.state_shape = state_shape
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(state_shape, 256),
            nn.Mish(),
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Linear(128, num_actions),
        )

    def forward(self, state):
        return self.net(state)


class DeepQLearningAgent(Agent):

    def best_action(self, state, valid_actions) -> int:
        pass
