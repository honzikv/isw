from collections import defaultdict
from pathlib import Path

import tqdm
import pickle

from src.agent import Agent
from src.environment import Environment
from dataclasses import dataclass, asdict
from collections import deque

import numpy as np
import random


@dataclass(frozen=True)
class Experience:
    """
    An experience tuple
    """
    current_state: np.ndarray
    current_action: int
    current_qvalue: float
    reward: float
    next_state: np.ndarray
    next_qvalue: float
    is_terminal: bool
    episode_timesteps: int


@dataclass(frozen=True)
class QLearningConfig:
    """
    A Q-Learning agent configuration
    """
    eps_init: float
    eps_final: float
    eps_decay_timesteps: int
    beta: float
    gamma: float
    experience_replay_buffer_size: int = None
    experience_replay_sample_size: int = None
    candidate_q_table_update_frequency: int = None


class QLearningAgent(Agent):
    """
    A Q-Learning agent implementation
    """

    def __init__(self, env: Environment, config: QLearningConfig):
        super().__init__(env)
        self.config = config

        self._q_table = defaultdict(lambda: np.zeros(env.num_actions))
        self._timesteps = 0
        self._episode_timesteps = 0
        self._successful_episodes = 0
        self._eval = False

        if config.candidate_q_table_update_frequency:
            self._candidate_qtable = defaultdict(lambda: np.zeros(env.num_actions))
            self._use_candidate_qtable = True
        else:
            self._use_candidate_qtable = False

        if config.experience_replay_buffer_size and config.experience_replay_sample_size:
            self._experience_buffer = deque(maxlen=config.experience_replay_buffer_size)
            self._use_exp_buffer = True
        else:
            self._use_exp_buffer = False

        # Training params
        self._gamma = config.gamma
        self._eps = config.eps_init  # current epsilon
        self._k = (config.eps_init - config.eps_final) / config.eps_decay_timesteps

    def _on_timestep(self):
        """
        Updates the agent on a timestep
        """
        self._timesteps += 1
        self._episode_timesteps += 1

        # Update epsilon
        if self._timesteps < self.config.eps_decay_timesteps:
            self._eps = self.config.eps_init - self._k * self._timesteps

    def train(self, n_episodes=100_000):
        """
        Trains the agent
        """
        for _ in tqdm.tqdm(range(n_episodes)):
            self._run_episode()
        print('Training finished')
        print(
            f'Number of successful episodes: {self._successful_episodes} '
            f'({self._successful_episodes / n_episodes * 100}%)'
        )
        print(self._q_table)
        print(len(self._q_table))
        self._eval = True

    def eval(self, n_episodes=1000):
        """
        Evaluates the agent
        """
        successes = 0
        for _ in tqdm.tqdm(range(n_episodes)):
            success = self._run_episode()
            if success:
                successes += 1

        print(f'Number of successful episodes: {successes} ({successes / n_episodes * 100}%)')

    def run_episode(self):
        self._episode_timesteps = 0
        state, _, valid_actions, is_terminal = self._env.set_to_initial_state()

    def _run_episode(self):
        """
        Trains the agent on a single episode
        """
        self._episode_timesteps = 0
        state, _, valid_actions, is_terminal = self._env.set_to_initial_state()
        action = self.best_action(state, valid_actions)
        cumulative_reward = .0

        while not is_terminal:
            # Perform the action
            next_state, reward, next_valid_actions, next_is_terminal = self._env.act(action)
            next_action = self.best_action(next_state, next_valid_actions)

            # Optimize the agent
            self._optimize_step(
                current_state=state,
                current_action=action,
                reward=reward,
                next_state=next_state,
                next_valid_actions=next_valid_actions,
                is_terminal=is_terminal,
            )

            action = next_action
            state, reward, valid_actions, is_terminal = next_state, reward, next_valid_actions, next_is_terminal
            cumulative_reward += reward
            self._on_timestep()

        # Replay experience
        if self.config.experience_replay_buffer_size:
            self._experience_replay()

        if cumulative_reward > 0:
            # print(f'Episode {episode_no} finished with reward {cumu_reward} in {self._episode_timesteps} timesteps')
            self._successful_episodes += 1
            return True
        return False

    def _get_max_qval(self, state: np.ndarray, valid_actions: list[int]) -> tuple[int, float]:
        max_action_val = -np.inf
        best_action = valid_actions[0]
        q_values = self._q_table[hash(state.data.tobytes())]
        for action_no in valid_actions[1:]:
            action_val = q_values[action_no]
            if action_val > max_action_val:
                max_action_val = action_val
                best_action = action_no

        return best_action, max_action_val

    def best_action(self, state: np.ndarray, valid_actions: set[int]) -> int:
        """
        Returns the best action for the given state
        """
        valid_actions = list(valid_actions)
        if random.uniform(0, 1) < self._eps:
            # Explore
            return random.choice(valid_actions)

        # Exploit
        best_action, _ = self._get_max_qval(state, valid_actions)

        # Add best action to the distribution
        self._action_distribution[best_action] += 1

        return best_action

    def _get_new_qval(self, qval: float, next_qval: float, reward: float) -> float:
        new_qval = self.config.beta * (reward + (self._gamma ** self._episode_timesteps) * next_qval)
        new_qval += (1 - self.config.beta) * qval
        return new_qval

    def _optimize_step(self, current_state: np.ndarray, current_action: int, reward: float,
                       next_state: np.ndarray, next_valid_actions, is_terminal: bool):
        qval = self._q_table[hash(current_state.data.tobytes())][current_action]

        _, next_qval = self._get_max_qval(next_state, list(next_valid_actions))
        if next_qval > 0:
            pass
        new_qval = self._get_new_qval(qval, next_qval, reward)

        if self._use_exp_buffer:
            self._experience_buffer.append(
                Experience(
                    current_state=current_state,
                    current_action=current_action,
                    current_qvalue=qval,
                    reward=reward,
                    next_state=next_state,
                    next_qvalue=next_qval,
                    is_terminal=is_terminal,
                    episode_timesteps=self._episode_timesteps,
                )
            )

        if not self._use_candidate_qtable:
            self._q_table[hash(current_state.data.tobytes())][current_action] = new_qval
            return

        self._candidate_qtable[hash(current_state.data.tobytes())][current_action] = new_qval
        if self._timesteps % self.config.candidate_q_table_update_frequency == 0:
            self._q_table = {key: value.copy() for key, value in self._candidate_qtable.items()}

    def _experience_replay(self):
        for experience in random.sample(self._experience_buffer, self.config.experience_replay_sample_size):
            new_qval = self._get_new_qval(**asdict(experience))

            if experience.is_terminal:
                raise NotImplementedError('Experience replay with terminal states not implemented')
