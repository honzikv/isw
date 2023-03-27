import random
import torch
import pytorch_lightning as pl
import numpy as np

from collections import deque, namedtuple

from pytorch_lightning.loggers import WandbLogger
from torch.optim import Optimizer
from torch.utils.data import IterableDataset
from functools import reduce
from dataclasses import dataclass

from src.environment import CartPole
from src.environment import Environment
from src.agent import Agent

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'next_valid_actions', 'is_terminal'))


class ReplayBuffer:

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, experience: Experience):
        self.buffer.append(experience)

    def sample_batch(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, next_state, reward, next_valid_actions, is_terminal = zip(
            *(self.buffer[idx] for idx in indices))

        return (
            np.array(states),
            np.array(actions),
            np.array(next_state),
            np.array(reward),
            np.array(next_valid_actions),
            np.array(is_terminal),
        )


class ExperienceDataset(IterableDataset):

    def __init__(self, replay_buffer: ReplayBuffer, sample_size: int = 200):
        self.buffer = replay_buffer
        self.sample_size = sample_size

    def __iter__(self):
        states, actions, next_state, reward, next_valid_actions, are_terminal = self.buffer.sample_batch(
            self.sample_size)
        for idx in range(len(states)):
            yield states[idx], actions[idx], next_state[idx], reward[idx], next_valid_actions[idx], are_terminal[idx]


class DeepQLearningAgent(Agent):

    def __init__(self, env: Environment, replay_buffer: ReplayBuffer):
        super().__init__(env)
        self._replay_buffer = replay_buffer
        self.state, _, self.valid_actions, self.is_terminal = self._env.set_to_initial_state()
        self.valid_actions = list(self.valid_actions)

    @property
    def observation_space(self):
        shape = self._env.observation_shape
        if isinstance(shape, tuple):
            return reduce(lambda x, y: x * y, shape)
        else:
            return shape

    @property
    def n_actions(self):
        return self._env.num_actions

    @torch.no_grad()
    def best_action(self, state: torch.tensor, valid_actions: list[int], net: torch.nn.Module, eps: float, device):
        if random.uniform(0, 1) < eps:
            valid_actions.sort()

            state = torch.tensor(state, dtype=torch.float32, device=device)

            q_vec = net(state)
            action = torch.argmax(q_vec, dim=0)  # TODO should only pick from valid actions
            action = int(action.item())
        else:
            action = random.choice(valid_actions)

        return action

    @torch.no_grad()
    def perform_step(self, net: torch.nn.Module, eps: float, device):
        # Get next action
        action = self.best_action(self.state, self.valid_actions, net, eps, device)

        # Step
        new_state, reward, new_valid_actions, is_terminal = self._env.act(action)
        experience = Experience(
            state=self.state,
            action=action,
            next_state=new_state,
            reward=reward,
            next_valid_actions=list(new_valid_actions),
            is_terminal=is_terminal,
        )

        self._replay_buffer.push(experience)

        self.state = new_state
        if is_terminal:
            self.state, _, self.valid_actions, self.is_terminal = self._env.set_to_initial_state()
            self.valid_actions = list(self.valid_actions)
        else:
            self.valid_actions = list(new_valid_actions)

        return reward, is_terminal


class DQN(torch.nn.Module):

    def __init__(self, input_size: int, n_actions: int, hidden_size: int = 256, activation_fn=torch.nn.ReLU):
        super().__init__()
        self.input_layer = torch.nn.Linear(in_features=input_size, out_features=hidden_size)
        self.output_layer = torch.nn.Linear(in_features=hidden_size, out_features=n_actions)
        self.activation = activation_fn()

    def forward(self, x):
        out = self.input_layer(x)
        out = self.activation(out)
        out = self.output_layer(out)
        return out


@dataclass(frozen=True)
class DQNConfig:
    batch_size: int = 16
    lr: float = 1e-2
    replay_size: int = 1000
    init_buffer_steps: int = 1000
    eps_init: float = 1.0
    eps_final: float = .05
    eps_decay_timesteps: int = 5_000
    beta: float = .5
    gamma: float = .99
    sync_rate: int = 10


class DQNLightning(pl.LightningModule):

    def __init__(self, env: Environment, config: DQNConfig, device, optimizer_fn: Optimizer = torch.optim.AdamW):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self._replay_buffer = ReplayBuffer(config.replay_size)
        self._device = device

        self._agent = DeepQLearningAgent(
            env=env,
            replay_buffer=self._replay_buffer
        )

        # Candidate and target nets
        self.net = DQN(input_size=self._agent.observation_space, n_actions=self._agent.n_actions)
        self.target_net = DQN(input_size=self._agent.observation_space, n_actions=self._agent.n_actions)

        self.net.to(self._device)
        self.target_net.to(self._device)

        self._timesteps = 0
        self._total_reward = 0
        self._episode_reward = 0

        self._eps = self.config.eps_init
        self._k = (config.eps_init - config.eps_final) / config.eps_decay_timesteps
        self._optimizer_fn = optimizer_fn

        self.init_replay_buffer(steps=config.init_buffer_steps)

    @property
    def eps(self):
        if self._timesteps < self.config.eps_decay_timesteps:
            return self.config.eps_init - self._k * self._timesteps

        return self.config.eps_final

    def init_replay_buffer(self, steps: int):
        for _ in range(steps):
            self._agent.perform_step(self.net, eps=1.0, device=self._device)

    def compute_loss(self, batch):
        states, actions, next_states, rewards, next_valid_actions, are_terminal = batch
        state_action_values = self.net(states.float()).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states.float()).max(1)[0]
            next_state_values[are_terminal] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.config.gamma + rewards

        return torch.nn.functional.mse_loss(state_action_values, expected_state_action_values)

    def training_step(self, batch, batch_idx: int):
        eps: float = self.eps

        reward, is_terminal = self._agent.perform_step(self.net, eps=eps, device=self._device)
        self._episode_reward += reward
        self.log('episode_reward', self._episode_reward)

        loss = self.compute_loss(batch)
        if is_terminal:
            self._total_reward = self._episode_reward
            self._episode_reward = 0

        self.log('eps', eps)

        self._timesteps += 1

        if self._timesteps % self.config.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict({
            'train_loss': loss,
            'reward': self._total_reward,
        })

        self.log('total_reward', self._total_reward, prog_bar=True)
        self.log('timesteps', self._timesteps, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return self._optimizer_fn(self.net.parameters(), lr=self.config.lr)

    def train_dataloader(self):
        return self.__dataloader()

    def __dataloader(self):
        dataset = ExperienceDataset(self._replay_buffer)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
        )

        return dataloader


LOG_WANDB = True

# https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/reinforce-learning-DQN.html
if __name__ == '__main__':
    environment = CartPole()
    config = DQNConfig(
        eps_init=1.0,
        eps_final=.01,
        eps_decay_timesteps=1000,
        beta=.3,
        gamma=.99,
        init_buffer_steps=1000,
        batch_size=16,
        sync_rate=50,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DQNLightning(environment, config, device, optimizer_fn=torch.optim.Adam)

    if LOG_WANDB:
        wandb_logger = WandbLogger(project='dqn', name='dqn')
        wandb_logger.watch(model)

    trainer = pl.Trainer(
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else 0,
        max_epochs=10000,
        val_check_interval=50,
        logger=wandb_logger if LOG_WANDB else None,
    )

    trainer.fit(model)
