import numpy as np

from agent.qlearning_agent import QLearningAgent, QLearningConfig
from src.agent import RandomAgent, BalancingAgent
from src.environment import Connect4, TicTacToe, FrozenLake, CartPole, MountainCar, LectureExample, MultiArmedBandit
from src.play import Simulator
from collections import Counter

if __name__ == '__main__':
    env = FrozenLake()

    agent = QLearningAgent(
        env=env,
        config=QLearningConfig(
            eps_init=1.0,
            eps_final=0.0,
            eps_decay_timesteps=2500,
            beta=0.2,
            gamma=.9,
            # candidate_q_table_update_frequency=10,
        )
    )

    agent.train(n_episodes=15_000)
    agent.eval(n_episodes=10_000)

    # n_tries = 0
    # while True:
    #     solved, _, _ = env.evaluate(agent=agent)
    #     if solved:
    #         break
    #     n_tries += 1
    #
    # print(f'Solved after {n_tries} tries')

    sim = Simulator(env, agent, fps=0)
    sim.show()
    sim.run()
