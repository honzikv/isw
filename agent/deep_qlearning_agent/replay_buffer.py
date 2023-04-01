import torch
import numpy as np

from collections import deque
from typing import NamedTuple

from torch.utils.data import IterableDataset

Experience = NamedTuple('Experience', (
    ('state', torch.tensor),
    ('action', int),
    ('next_state', torch.tensor),
    ('reward', float),
    ('next_valid_actions', list),
    ('is_terminal', bool),
))


class ReplayBuffer:

    def __init__(self, capacity=1000):
        self._buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self._buffer)

    def push(self, experience: Experience):
        self._buffer.append(experience)

    def sample_batch(self, batch_size: int):
        indices = np.random.choice(len(self._buffer), batch_size, replace=False)
        states, actions, next_states, rewards, next_valid_actions, are_terminal = zip(
            *(self._buffer[idx] for idx in indices))

        return (
            np.array(states),
            np.array(actions),
            np.array(next_states),
            np.array(rewards),
            np.array(next_valid_actions),
            np.array(are_terminal),
        )


class ExperienceDataset(IterableDataset):

    def __init__(self, replay_buffer: ReplayBuffer, sample_size: int = 200):
        self.buffer = replay_buffer
        self.sample_size = sample_size

    def __iter__(self):
        states, actions, next_states, rewards, next_valid_actions, are_terminal = self.buffer.sample_batch(
            self.sample_size)
        for idx in range(len(states)):
            yield states[idx], actions[idx], next_states[idx], rewards[idx], next_valid_actions[idx], are_terminal[idx]
