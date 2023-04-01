import torch
import pytorch_lightning as pl
import torch.utils.data
import wandb
import os
import numpy as np

from collections import deque
from dataclasses import dataclass

from agent.deep_qlearning_agent.deep_qlearning_agent import DeepQLearningAgent
from agent.deep_qlearning_agent.replay_buffer import ReplayBuffer, ExperienceDataset
from src.environment import Environment


class DeepQNetwork(torch.nn.Module):
    """
    Network architecture for Deep Q-Learning
    """

    def __init__(self, input_size: int, n_actions: int, hidden_size_1: int = 256, hidden_size_2=128,
                 activation_fn=torch.nn.Mish):
        """
        MLP with 2 hidden layers
        Args:
            input_size: Size of the input layer
            n_actions: Number of actions
            hidden_size_1: Size of the first hidden layer
            hidden_size_2: Size of the second hidden layer
            activation_fn: Activation function
        """
        super().__init__()

        self.input_layer = torch.nn.Linear(in_features=input_size, out_features=hidden_size_1)
        self.hidden_layer = torch.nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2)
        self.output_layer = torch.nn.Linear(in_features=hidden_size_2, out_features=n_actions)
        self.activation = activation_fn()

    def forward(self, x):
        out = self.input_layer(x)
        out = self.activation(out)
        out = self.hidden_layer(out)
        out = self.activation(out)
        out = self.output_layer(out)
        return out


@dataclass(frozen=True)
class DeepQLearningConfig:
    batch_size: int = 16  # number of samples processed per each training step
    lr: float = 5e-4  # learning rate for the optimizer
    replay_buffer_size: int = 10_000  # size of the replay buffer
    init_buffer_steps: int = 1000  # number of steps to fill the replay buffer with random actions
    eps_init: float = 1.0  # initial value of epsilon
    eps_final: float = .05  # final value of epsilon
    eps_decay_timesteps: int = 15_000  # number of timesteps to decay epsilon from eps_init to eps_final
    gamma: float = .99  # discount factor
    max_score: int = 1000 # maximum score to reach before stopping the training. 1k is enough to "solve" CartPole


class DeepQModuleLightning(pl.LightningModule):

    def __init__(self, env: Environment, config: DeepQLearningConfig, device, log_wandb: bool = False,
                 optimizer_fn=torch.optim.AdamW):
        """
        PyTorch Lightning module for Deep Q-Learning
        Args:
            env: Environment
            config: Configuration
            device: Device to use
            log_wandb
            optimizer_fn: Optimizer function e.g. torch.optim.RMSProp
        """
        super().__init__()
        self.hparams['config'] = config
        self._log_wandb = log_wandb
        self.save_hyperparameters()

        self.config = config
        self._device = device

        self._replay_buffer = ReplayBuffer(config.replay_buffer_size)
        self._agent = DeepQLearningAgent(
            env=env,
            replay_buffer=self._replay_buffer,
            device=device,
        )

        self.candidate_net = DeepQNetwork(
            input_size=self._agent.observation_space,
            n_actions=self._agent.n_actions,
        )

        self.target_net = DeepQNetwork(
            input_size=self._agent.observation_space,
            n_actions=self._agent.n_actions,
        )

        self._optimizer_fn = optimizer_fn
        self._eps = self.config.eps_init
        self._k = (config.eps_init - config.eps_final) / config.eps_decay_timesteps
        self._running_values = deque([0 for _ in range(100)], maxlen=100)

        self._timesteps = 0
        self._max_reward = 0
        self._models_saved = 0

        # Move the networks to the device - this needs to be done explicitly
        # to ensure init_replay_buffer() works as intended on CUDA/MPS
        self.candidate_net.to(self._device)
        self.target_net.to(self._device)

        self._init_replay_buffer(self.config.init_buffer_steps)

    @property
    def eps(self):
        if self._timesteps < self.config.eps_decay_timesteps:
            return self.config.eps_init - self._k * self._timesteps

        return self.config.eps_final

    def _init_replay_buffer(self, steps: int):
        """
        Fill the replay buffer with random actions
        """
        for _ in range(steps):
            self._agent.step(self.candidate_net, eps=1.0, device=self._device)

    def compute_loss(self, batch):
        """
        Compute the loss for the given batch
        """
        states, actions, next_states, rewards, next_valid_actions, are_terminal = batch

        # Compute state action values
        state_action_values = self.candidate_net(states.float()).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states.float()).max(1)[0]
            next_state_values[are_terminal] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.config.gamma + rewards

        return torch.nn.functional.mse_loss(state_action_values, expected_state_action_values)

    def increment_timestep(self):
        """
        Increment the timestep counter - injected to the agent
        """
        self._timesteps += 1

    def training_step(self, batch, batch_idx: int):
        """
        Training step
        Args:
            batch: Batch of samples
            batch_idx: Batch index
        """
        total_reward = self._agent.run_episode(
            net=self.candidate_net,
            get_eps=lambda: self.eps,
            increment_timestep=lambda: self.increment_timestep(),
            device=self._device,
        )

        self._running_values.append(total_reward)
        self._max_reward = max(self._max_reward, total_reward)
        loss = self.compute_loss(batch)

        # 95th percentile of the last 1000 rewards
        reward95 = np.quantile(self._running_values, .95)
        running_avg = sum(self._running_values) / len(self._running_values)

        self.log('episode_reward', total_reward, prog_bar=True)
        self.log('eps', self.eps, prog_bar=True)
        self.log('max_reward', self._max_reward, prog_bar=False)
        self.log('loss', loss, prog_bar=True)
        self.log('timesteps', self._timesteps, prog_bar=False)
        self.log('running_avg_reward', running_avg, prog_bar=True)
        self.log('reward95', reward95, prog_bar=True)

        # Update the target network
        self.target_net.load_state_dict(self.candidate_net.state_dict())

        if reward95 > self.config.max_score:
            self._models_saved += 1
            run_name = wandb.run.name if self._log_wandb else 'model'
            os.makedirs('weights', exist_ok=True)
            self.trainer.save_checkpoint(
                f'weights/{run_name}_reward95={reward95}-episodes={self.trainer.current_epoch}.ckpt')

            if self._models_saved >= 10:
                print(f'Found 10 models with reward > {self.config.max_score}. Stopping training...')
                self.trainer.should_stop = True

        return loss

    def configure_optimizers(self):
        return self._optimizer_fn(self.candidate_net.parameters(), lr=self.config.lr)

    def train_dataloader(self):
        return self.__dataloader()

    def __dataloader(self):
        """
        Dataloader override to provide data to training_step
        """
        dataset = ExperienceDataset(self._replay_buffer)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
        )

        return dataloader
