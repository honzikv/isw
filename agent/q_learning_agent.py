from src.agent import Agent
from src.environment import Environment
from dataclasses import dataclass

import tqdm

import numpy as np


class QLearningAgent(Agent):
    """
    A Q-Learning agent implementation
    """

    def __init__(self, environment: Environment, eps_init: float, eps_final: float, eps_decay_timesteps: int,
                 beta: float = .2, gamma: float = .999, use_experience_buffer: bool = False,
                 use_candidate_q_table: bool = False):
        """
        Initializes a Q-Learning agent instance
        Args:
            environment: The environment to train the agent in
            eps_init: The initial epsilon value
            eps_final: The final epsilon value
            eps_decay_timesteps: The number of timesteps to decay epsilon over
            beta: The learning rate
            gamma: The discount factor
            use_experience_buffer: Whether to use an experience buffer
            use_candidate_q_table: Whether to use candidate Q-tables
        """
        super().__init__(environment)
        self._q_table = {}
        self._use_experience_buffer = use_experience_buffer
        self._use_candidate_q_table = use_candidate_q_table

        self._eps = eps_init  # current epsilon
        self._eps_init = eps_init
        self._eps_final = eps_final
        self._eps_decay_timesteps = eps_decay_timesteps
        self._timesteps = 0
        self._k = (eps_init - eps_final) / eps_decay_timesteps

        self._beta = beta
        self._initial_gamma = gamma
        self._gamma = gamma

        if self._use_experience_buffer:
            raise NotImplementedError('Experience buffer not implemented yet')

        if self._use_candidate_q_table:
            raise NotImplementedError('Candidate Q-table not implemented yet')

    def _on_timestep(self):
        """
        Updates the agent on a timestep
        """
        self._timesteps += 1

        # Update gamma
        self._gamma = self._initial_gamma ** self._timesteps

        # Update epsilon
        if self._timesteps <= self._eps_decay_timesteps:
            self._eps = self._eps_init - self._k * self._timesteps
        else:
            self._eps = self._eps_final

    def train(self, n_episodes=1_000):
        """
        Trains the agent in the environment
        """

        for i in tqdm.tqdm(range(n_episodes)):
            self._run_episode(i+1)

    def _best_action(self, state, valid_actions):
        """
        Gets the best action index and value from the list of valid actions
        """

        state_bytes = state.tobytes()
        if state_bytes not in self._q_table:
            self._q_table[state_bytes] = np.zeros(len(valid_actions))

        # Flip a skewed coin to see if we should explore or exploit
        if np.random.random() < self._eps:
            # Explore
            action_idx = np.random.choice(len(valid_actions))
        else:
            # Otherwise exploit - i.e. get argmax over all valid actions
            # If there are no initialized valid actions init to zeros
            action_idx = np.argmax(
                self._q_table.get(state_bytes, np.zeros(len(valid_actions)))
            )

        action = list(valid_actions)[action_idx]
        return action_idx, action

    def _update_q_table(self, current_state: np.ndarray,
                        current_reward: float,
                        current_action_idx: int,
                        next_state: np.ndarray,
                        next_action_idx: int,
                        next_state_valid_actions: np.ndarray
                        ):
        """
        Updates the Q-table
        Args:
            current_state: The current state
            current_reward: The current reward
            current_action_idx: The current action index
            next_state: The next state
            next_action_idx: The next action index
            next_state_valid_actions: The next state's valid actions
        """
        if self._use_experience_buffer:
            raise NotImplementedError('Experience buffer not implemented yet')

        if self._use_candidate_q_table:
            raise NotImplementedError('Candidate Q-table not implemented yet')

        current_state_hash = current_state.tobytes()
        current_qval = self._q_table[current_state_hash][current_action_idx]
        next_qval = self._q_table.get(hash(next_state.tobytes()), np.zeros(len(next_state_valid_actions)))[
            next_action_idx]

        # Update the Q-table
        new_qval = self._beta * (current_reward + self._gamma * next_qval) + (1 - self._beta) * current_qval
        self._q_table[current_state.tobytes()][current_action_idx] = new_qval

    def _run_episode(self, episode_id: int):
        """
        Runs a single episode
        Args:
            episode_id: The episode ID
        """
        print(f'Running episode {episode_id}')
        current_state, current_reward, current_valid_actions, is_terminal = self._env.set_to_initial_state()
        cumu_reward = 0

        while not is_terminal:
            # Get the best action
            action_idx, action = self._best_action(current_state, current_valid_actions)
            curr_state_bytes = hash(current_state.data.tobytes())

            # Take the action
            next_state, next_state_reward, next_valid_actions, is_terminal = self._env.act(action)

            current_state_hash = hash(current_state.data.tobytes())
            print(curr_state_bytes == current_state_hash)

            # Update the Q-table
            self._update_q_table(
                current_state=current_state,
                current_reward=current_reward,
                current_action_idx=action_idx,
                next_state=next_state,
                next_action_idx=action_idx,
                next_state_valid_actions=np.array(list(next_valid_actions)),
            )

            # Tick - i.e. decay epsilon and change gamma
            self._on_timestep()

            cumu_reward += current_reward

        print(f'Episode {episode_id} finished with cumulative reward {cumu_reward}')
