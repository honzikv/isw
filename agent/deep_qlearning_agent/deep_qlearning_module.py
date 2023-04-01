import torch
import pytorch_lightning as pl
import torch.utils.data
import wandb
import os

from collections import deque
from dataclasses import dataclass

from agent.deep_qlearning_agent.deep_qlearning_agent import DeepQLearningAgent
from agent.deep_qlearning_agent.replay_buffer import ReplayBuffer, ExperienceDataset
from src.environment import Environment


class DeepQNetwork(torch.nn.Module):

    def __init__(self, input_size: int, n_actions: int, hidden_size_1: int = 256, hidden_size_2=128,
                 activation_fn=torch.nn.Mish):
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
    batch_size: int = 16
    lr: float = 1e-2
    replay_buffer_size: int = 1000
    init_buffer_steps: int = 1000
    eps_init: float = 1.0
    eps_final: float = .05
    eps_decay_timesteps: int = 15_000
    gamma: float = .99
    max_score: int = 150


class DeepQModuleLightning(pl.LightningModule):

    def __init__(self, env: Environment, config: DeepQLearningConfig, device, optimizer_fn=torch.optim.AdamW):
        super().__init__()
        self.hparams['config'] = config
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
        self._running_values = deque([0 for _ in range(1000)], maxlen=100)

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
        for _ in range(steps):
            self._agent.step(self.candidate_net, eps=1.0, device=self._device)

    def compute_loss(self, batch):
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
        self._timesteps += 1

    def training_step(self, batch, batch_idx: int):
        total_reward = self._agent.run_episode(
            net=self.candidate_net,
            get_eps=lambda: self.eps,
            increment_timestep=lambda: self.increment_timestep(),
            device=self._device,
        )

        self._running_values.append(total_reward)
        self._max_reward = max(self._max_reward, total_reward)
        loss = self.compute_loss(batch)
        running_avg = sum(self._running_values) / len(self._running_values)

        self.log('episode_reward', total_reward, prog_bar=True)
        self.log('eps', self.eps, prog_bar=True)
        self.log('max_reward', self._max_reward, prog_bar=False)
        self.log('loss', loss, prog_bar=True)
        self.log('timesteps', self._timesteps, prog_bar=False)
        self.log('running_avg_reward', running_avg, prog_bar=True)

        # Update the target network
        self.target_net.load_state_dict(self.candidate_net.state_dict())

        if running_avg > self.config.max_score:
            self._models_saved += 1
            run_name = wandb.run.name
            print(f'Found model with reward {total_reward} at epoch {self.trainer.current_epoch}! Saving...')
            os.makedirs('weights', exist_ok=True)
            self.trainer.save_checkpoint(
                f'weights/{run_name}_reward={total_reward}-episodes={self.trainer.current_epoch}.ckpt')

            if self._models_saved >= 50:
                print('Found 50 models with reward > 150. Stopping training...')
                self.trainer.should_stop = True

        return loss

    def configure_optimizers(self):
        return self._optimizer_fn(self.candidate_net.parameters(), lr=self.config.lr)

    def train_dataloader(self):
        return self.__dataloader()

    def __dataloader(self):
        dataset = ExperienceDataset(self._replay_buffer)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
        )

        return dataloader
