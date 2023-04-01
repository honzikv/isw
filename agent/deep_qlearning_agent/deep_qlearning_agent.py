from functools import reduce
from typing import Callable, List, Union

import torch
import random

from agent.deep_qlearning_agent.replay_buffer import Experience, ReplayBuffer
from src.agent import Agent
from src.environment import Environment


class DeepQLearningAgent(Agent):

    def __init__(self, env: Environment, replay_buffer: ReplayBuffer, device):
        super().__init__(env)

        self._replay_buffer = replay_buffer
        self._net = None
        self._device = device

        # Initialize state, a bit garbage but function is reused elsewhere
        self.state, self.valid_actions = None, None
        self.reset_env()

    @classmethod
    def from_pretrained(cls, env: Environment, net: torch.nn.Module, device='cpu'):
        agent = cls(env, None, device)
        agent.net = net

        return agent

    def reset_env(self):
        state, _, valid_actions, _ = self._env.set_to_initial_state()
        self.state = state
        self.valid_actions = list(valid_actions)
        self.valid_actions.sort()

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

    @property
    def net(self):
        return self._net

    @net.setter
    def net(self, net):
        """
        This needs to be called after training for the best_action method
        to behave as expected. Otherwise, net must be injected via parameter
        """
        self._net = net

    @torch.no_grad()
    def best_action(self, state: Union[torch.Tensor, set], valid_actions: List[int], net=None, eps=None, device=None):
        """
        Perform best action.
        Args:
            state: state to perform action on
            valid_actions: list of valid actions - must be sorted
            net: network to use for action selection - injected during training
            eps: epsilon value for epsilon-greedy action selection
            device: device to use for action selection
        """
        if net is None:
            net = self._net
            assert net is not None, 'net must be injected via parameter or set as property'

        device = device or self._device
        eps = eps or 0.0
        if random.uniform(0, 1) > eps:
            # if isinstance(state, set):
            #     state = list(state)
            #     state.sort()

            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32, device=device)
            else:
                state = state.to(device)

            q_values = net(state)
            # Filter only valid actions
            q_values = q_values[valid_actions]
            action = torch.argmax(q_values, dim=0)
            action = valid_actions[int(action.item())]
        else:
            action = random.choice(valid_actions)

        return action

    @torch.no_grad()
    def step(self, net: torch.nn.Module, eps: float, device):
        """
        Perform a single step in the environment
        Args:
            net: network to use for action selection
            eps: epsilon value for epsilon-greedy policy
        """
        action = self.best_action(self.state, self.valid_actions, net, eps, device)

        new_state, reward, new_valid_actions, is_terminal = self._env.act(action)
        next_valid_actions = list(new_valid_actions)
        next_valid_actions.sort()

        # Push to the buffer
        self._replay_buffer.push(Experience(
            state=self.state,
            action=action,
            next_state=new_state,
            reward=reward,
            next_valid_actions=next_valid_actions,
            is_terminal=is_terminal,
        ))

        # Update state and terminal status
        self.state = new_state

        if is_terminal:
            self.reset_env()
        else:
            self.valid_actions = next_valid_actions

        return reward, is_terminal

    @torch.no_grad()
    def run_episode(self, net: torch.nn.Module,
                    get_eps: Callable[[], float],
                    increment_timestep: Callable[[], None],
                    device):
        """
        Run a single episode. This is called during training
        Args:
            net: network to use for action selection
            get_eps: function to get epsilon value for epsilon-greedy policy
            increment_timestep: function to increment timestep
            device: device to use for action selection
        """
        self.reset_env()

        total_reward = 0.0
        while True:
            reward, is_terminal = self.step(net, get_eps(), device)
            total_reward += reward
            increment_timestep()

            if is_terminal:
                break

        return total_reward
