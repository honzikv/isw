from dataclasses import dataclass
from functools import reduce
from itertools import count

import tqdm
import wandb
from tqdm import trange

from src.agent import Agent
from torch import nn, optim

from collections import deque, namedtuple

import random
import torch
import numpy as np

from src.environment import Environment
from matplotlib import pyplot as plt

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'next_valid_actions'))


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
        super(DeepQLearningNet, self).__init__()
        self.state_shape = state_shape
        self.num_actions = num_actions

        self.input_layer = nn.Linear(state_shape, 256)
        self.hidden1 = nn.Linear(256, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, num_actions)

    def forward(self, state: torch.Tensor):
        x = torch.relu(self.input_layer(state))
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output_layer(x)
        return x


@dataclass(frozen=True)
class DeepQLearningConfig:
    """
    A Q-Learning agent configuration
    """
    eps_init: float
    eps_final: float
    eps_decay_timesteps: int
    beta: float
    gamma: float
    lr: float
    replay_memory_size: int = 10_000  # 10k
    batch_size: int = 256
    n_episodes: int = 1000
    wandb_logging: bool = False
    wandb_logging_interval: int = 100


class DeepQLearningAgent(Agent):

    def __init__(self, env: Environment, config: DeepQLearningConfig, device: torch.device = None):
        super().__init__(env)
        self._steps_done = 0
        self.config = config
        self._eval = False
        self.input_size = self._get_input_size()
        self._policy_net = DeepQLearningNet(self.input_size, env.num_actions)
        self._target_net = DeepQLearningNet(self.input_size, env.num_actions)
        self._memory = ReplayMemory(config.replay_memory_size)
        self._timesteps = 0
        self._k = (config.eps_init - config.eps_final) / config.eps_decay_timesteps

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def _get_input_size(self) -> np.shape:
        shape = self._env.observation_shape
        if isinstance(shape, tuple):
            return reduce(lambda x, y: x * y, shape)
        else:
            return shape

    @property
    def policy_net(self) -> DeepQLearningNet:
        return self._policy_net

    @property
    def target_net(self) -> DeepQLearningNet:
        return self._target_net

    @property
    def steps_done(self):
        return self._steps_done

    @property
    def eps(self):
        if self._timesteps < self.config.eps_decay_timesteps:
            return self.config.eps_init - self._k * self._timesteps

        return self.config.eps_final

    def best_action(self, state, valid_actions):
        state = torch.tensor(list(state))
        random_action = random.random()
        valid_actions = list(valid_actions)
        valid_actions.sort()

        eps = 0 if self._eval else self.eps
        if random_action > eps:
            with torch.no_grad():
                valid_actions.sort()
                out = self._policy_net(state).detach().numpy()
                out = np.take(out, valid_actions)
                return torch.tensor(valid_actions[np.argmax(out)], dtype=torch.int64).view(1, 1)
        else:
            return torch.tensor(random.choice(valid_actions)).view(1, 1)

    def _optimize(self, loss_fn, optimizer):
        if len(self._memory) < self.config.batch_size:
            return

        # Sample batch
        transitions = self._memory.sample(self.config.batch_size)
        batch = Transition(
            *zip(*transitions))  # convert list[Transition] to Transition(list of states, list of actions, etc)

        # Mask out None values
        non_final_mask = torch.tensor(
            tuple(map(lambda state: state is not None, batch.next_state)),
            dtype=torch.bool,
        )

        non_final_next_states = torch.stack([state for state in batch.next_state if state is not None])

        # Extract batch of states, actions, and rewards
        state_batch = torch.cat(batch.state).view(self.config.batch_size, self.input_size).float()
        action_batch = torch.cat(batch.action)
        reward_batch = torch.tensor(np.asarray(batch.reward, dtype=np.float32))

        # Get Q(s_t, a) from the batch
        state_action_values = self._policy_net(state_batch).gather(1, action_batch)

        # Compute values of Q(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.config.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self._target_net(non_final_next_states).detach().max(1)[0]

            # # TODO this is such a garbage code
            # for i in range(len(batch_out)):
            #     valid_actions = batch.next_valid_actions[i]
            #     valid_actions = list(valid_actions)
            #     valid_actions.sort()
            #     out = np.max(np.take(batch_out[i], valid_actions))
            #     next_state_values[i] = out.item()

        expected_state_action_values = (next_state_values * self.config.gamma) + reward_batch
        loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

        if self.config.wandb_logging and self._timesteps % self.config.wandb_logging_interval \
                == self.config.wandb_logging_interval - 1:
            wandb.log({'loss': loss.item()})

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self._policy_net.parameters(), 100)
        optimizer.step()

    def train(self, loss_fn=None, optimizer=None, n_episodes: int = 10_000):
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        if optimizer is None:
            optimizer = optim.AdamW(self._policy_net.parameters(), lr=self.config.lr)

        self._eval = False
        for episode in trange(n_episodes):
            cumu_reward, ep_len = 0, 0
            state, reward, valid_actions, is_terminal = self._env.set_to_initial_state()
            state = torch.tensor(state, dtype=torch.float32)

            for t in count():
                action = self.best_action(state, valid_actions)
                next_state, reward, next_valid_actions, is_terminal = self._env.act(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)

                if is_terminal:
                    next_state = None

                cumu_reward += reward
                ep_len += 1

                self._memory.push(state, action, next_state, reward, next_valid_actions)
                self._optimize(loss_fn, optimizer)

                state = next_state
                valid_actions = next_valid_actions

                target_net_state_dict = self._target_net.state_dict()
                policy_net_state_dict = self._policy_net.state_dict()

                # Update target network
                for k in policy_net_state_dict:
                    target_net_state_dict[k] = self.config.beta * policy_net_state_dict[k] + (1 - self.config.beta) * \
                                               target_net_state_dict[k]

                self._target_net.load_state_dict(target_net_state_dict)
                self._timesteps += 1

                if self.config.wandb_logging and self._timesteps % self.config.wandb_logging_interval \
                        == self.config.wandb_logging_interval - 1:
                    wandb.log({
                        'episode_reward': cumu_reward,
                        'episode_length': ep_len,
                        'episode': episode,
                        'eps': self.eps
                    })

                if is_terminal:
                    break
