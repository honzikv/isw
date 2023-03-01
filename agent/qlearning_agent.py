from collections import defaultdict

import tqdm

from src.agent import Agent
from src.environment import Environment
from dataclasses import dataclass

import numpy as np


@dataclass
class QLearningConfig:
    """
    A Q-Learning agent configuration
    """
    eps_init: float
    eps_final: float
    eps_decay_timesteps: int
    eps_decay_type: str  # linear or exponential
    beta: float
    gamma: float
    use_experience_buffer: bool
    use_candidate_q_table: bool


class QLearningAgent(Agent):
    """
    A Q-Learning agent implementation
    """

    def __init__(self, env: Environment, config: QLearningConfig):
        super().__init__(env)

        self._q_table = defaultdict(lambda: np.zeros(env.num_actions))
        self._timesteps = 0
        self._episode_timesteps = 0
        self._config = config
        self._successful_episodes = 0
        self._eval = False

        # Params
        self._eps = config.eps_init  # current epsilon

        if self._config.eps_decay_type == 'linear':
            self._k = (config.eps_init - config.eps_final) / config.eps_decay_timesteps
        else:
            raise NotImplementedError('Not implemented yet')

        self._gamma = config.gamma

        if config.use_experience_buffer:
            raise NotImplementedError('Experience buffer not implemented yet')

        if config.use_candidate_q_table:
            raise NotImplementedError('Candidate Q-table not implemented yet')

    def _on_timestep(self):
        """
        Updates the agent on a timestep
        """
        self._timesteps += 1
        self._episode_timesteps += 1

        # Update epsilon
        if self._timesteps < self._config.eps_decay_timesteps:
            self._eps = self._config.eps_init - self._k * self._timesteps
        else:
            self._eps = self._config.eps_final

    def train(self, n_episodes=100_000):
        """
        Trains the agent
        """
        for _ in tqdm.tqdm(range(n_episodes)):
            self._run_episode()
        print('Training finished')
        print(
            f'Number of successful episodes: {self._successful_episodes} ({self._successful_episodes / n_episodes * 100}%)')
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

    def _run_episode(self):
        """
        Trains the agent on a single episode
        """
        self._episode_timesteps = 0
        state, reward, valid_actions, is_terminal = self._env.set_to_initial_state()
        action = self.best_action(state, valid_actions)
        cumu_reward = reward

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
                next_action=next_action,
                is_term=is_terminal,
            )

            action = next_action
            state, reward, valid_actions, is_terminal = next_state, reward, next_valid_actions, next_is_terminal
            cumu_reward += reward
            self._on_timestep()

        if cumu_reward > 0:
            # print(f'Episode {episode_no} finished with reward {cumu_reward} in {self._episode_timesteps} timesteps')
            self._successful_episodes += 1
            return True
        return False

    def best_action(self, state: np.ndarray, valid_actions: set[int]) -> int:
        """
        Returns the best action for the given state
        """
        valid_actions = list(valid_actions)
        if np.random.random() < self._eps:
            # Explore
            return np.random.choice(valid_actions)

        max_action_val = -np.inf
        best_action = valid_actions[0]
        q_values = self._q_table[hash(state.tobytes())]
        for action_no in valid_actions[1:]:
            action_val = q_values[action_no]
            if action_val > max_action_val:
                max_action_val = action_val
                best_action = action_no

        # Add best action to the distribution
        self._action_distribution[best_action] += 1

        return best_action

    def _optimize_step(self, current_state: np.ndarray, current_action: int, reward: float,
                       next_state: np.ndarray, next_action: int, is_term: bool):
        if self._config.use_experience_buffer:
            raise NotImplementedError('Experience buffer not implemented yet')

        if self._config.use_candidate_q_table:
            raise NotImplementedError('Candidate Q-table not implemented yet')

        current_state_hash = hash(current_state.tobytes())
        qval = self._q_table[current_state_hash][current_action]
        next_qval = self._q_table[hash(next_state.tobytes())][next_action]

        next_qval = self._config.beta * (reward + (self._gamma ** self._episode_timesteps) * next_qval)
        next_qval += (1 - self._config.beta) * qval
        self._q_table[current_state_hash][current_action] = next_qval
